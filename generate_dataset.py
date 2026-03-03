from __future__ import annotations

import itertools
import json
import logging
import os
import hashlib

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import typer

from tqdm import tqdm

from qqe.backend import PennylaneBackend, QuimbBackend
from qqe.circuit.families import (
    CliffordBrickwork,
    HaarBrickwork,
    QuansistorBrickwork,
    RandomCircuit,
)
from qqe.circuit.patterns import TdopingRules, to_qasm
from qqe.experiments.core import run_experiment
from qqe.GNN.encoder import qasm_to_pyg_graph
from qqe.utils import FileCache, configure_logger

if TYPE_CHECKING:
    from qqe.circuit.spec import GateSpec

logger = logging.getLogger(__name__)


n_seeds = 50
n_bins = 50

qubits_range = np.arange(4, 11, 2)
layers_range = np.arange(1, 101, 2)

# Backend and family registries
BACKEND_REGISTRY = {
    "pennylane": PennylaneBackend,
    "quimb": QuimbBackend,
}

FAMILY_REGISTRY = {
    "haar": HaarBrickwork,
    "clifford": CliffordBrickwork,
    "quansistor": QuansistorBrickwork,
    "random": RandomCircuit,
}

# Initialize cache for results
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "outputs" / "gnn_graphs"
DATASET_DIR.mkdir(parents=True, exist_ok=True)
cache = FileCache(PROJECT_ROOT / "outputs" / "cache")


# def make_seed(family: str, n_qubits: int, n_layers: int, rep: int) -> int:
#     return hash((family, n_qubits, n_layers, rep)) % (2**32)
def make_seed(family: str, n_qubits: int, n_layers: int, rep: int) -> int:
    s = f"{family}|{n_qubits}|{n_layers}|{rep}".encode()
    return int.from_bytes(hashlib.blake2b(s, digest_size=4).digest(), "little")

def make_cid(family: str, n_qubits: int, n_layers: int, seed: int) -> str:
    return f"{family}_Q{n_qubits}_L{n_layers}_S{seed}"


def generate_dataset_params(
    families: list[str],
    qubits_values: np.ndarray,
    layers_values: np.ndarray,
    num_seeds: int,
) -> list[dict[str, Any]]:
    params: list[dict[str, Any]] = []
    for family, n_qubits, n_layers in itertools.product(
        families,
        qubits_values,
        layers_values,
    ):
        for rep in range(num_seeds):
            seed = make_seed(family, int(n_qubits), int(n_layers), rep)
            cid = make_cid(family, int(n_qubits), int(n_layers), int(seed))
            params.append(
                {
                    "cid": cid,
                    "family": family,
                    "n_qubits": int(n_qubits),
                    "n_layers": int(n_layers),
                    "seed": int(seed),
                },
            )
    return params


def _safe_gate_counts(gate_counts: Any) -> dict[str, int]:
    if not isinstance(gate_counts, dict):
        return {}
    safe: dict[str, int] = {}
    for key, value in gate_counts.items():
        safe[str(key)] = int(value)
    return safe


def compute_entry_for_config(
    family: str,
    n_qubits: int,
    n_layers: int,
    seed: int,
    backend: str = "quimb",
    method: str = "fwht",
    representation: str = "dense",
    n_bins_value: int = 50,
) -> dict[str, Any] | None:
    from qqe.experiments.core import ExperimentConfig
    from qqe.properties.compute import PropertyRequest
    from qqe.states.types import BackendConfig

    try:
        family_cls = FAMILY_REGISTRY[family]
        family_obj = family_cls()

        tdoping = TdopingRules(count=2 * n_layers, per_layer=2)
        spec = family_obj.make_spec(
            n_qubits=int(n_qubits),
            n_layers=int(n_layers),
            d=2,
            seed=int(seed),
            tdoping=tdoping if family == "clifford" else None,
        )

        gates = cast("tuple[GateSpec] | None", spec.gates)
        qasm = to_qasm(spec, gates)

        graph_data, gate_counts = qasm_to_pyg_graph(
            qasm_str=qasm,
            n_bins=n_bins_value,
            family=family,
            global_feature_variant="binned",
        )

        x_np = graph_data.x.detach().cpu().numpy()
        edge_index_np = graph_data.edge_index.detach().cpu().numpy()

        backend_config = BackendConfig(
            name=backend,
            representation=representation,
            params={},
        )
        property_request = PropertyRequest(
            name="SRE",
            method=method,
            params={},
        )
        exp_config = ExperimentConfig(
            spec=spec,
            backend=backend_config,
            properties=[property_request],
        )

        backend_factory = BACKEND_REGISTRY[backend]
        backend_instance = backend_factory() if callable(backend_factory) else backend_factory
        state = backend_instance.simulate(
            spec,
            representation=representation,
            **backend_config.params,
        )

        result = run_experiment(
            exp_config,
            backend_registry=BACKEND_REGISTRY,
            state=state,
            cache=cache,
        )

        sre_result = result.results.get("SRE")
        sre_value = float(sre_result.value) if sre_result is not None else None

        graph_path = DATASET_DIR / f"{cid}.pt"
        torch.save(graph_data, graph_path)   # graph_data already has x, edge_index, global_features

        entry = {
            "sre": sre_value,
            "graph_path": str(graph_path),
            "gate_counts": _safe_gate_counts(gate_counts),
        }

        if hasattr(graph_data, "u") and graph_data.u is not None:
            entry["data"]["u"] = graph_data.u.detach().cpu().numpy().tolist()

    except Exception:
        logger.exception(
            "Error computing entry for family=%s Q%s L%s S%s",
            family,
            n_qubits,
            n_layers,
            seed,
        )
        return None
    else:
        return entry


def compute_all_entries(
    params: list[dict[str, Any]],
    backend: str = "quimb",
    method: str = "fwht",
    *,
    use_dask: bool = False,
    n_bins_value: int = 50,
    dask_n_workers: int = 20,
) -> list[dict[str, Any]]:
    if use_dask:
        return compute_all_entries_parallel(
            params,
            backend=backend,
            method=method,
            n_bins_value=n_bins_value,
            dask_n_workers=dask_n_workers,
        )
    return compute_all_entries_sequential(
        params,
        backend=backend,
        method=method,
        n_bins_value=n_bins_value,
    )


def compute_all_entries_sequential(
    params: list[dict[str, Any]],
    backend: str,
    method: str,
    n_bins_value: int,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    for row in tqdm(params, desc="Computing dataset entries"):
        cid = row["cid"]
        family = row["family"]
        n_qubits = row["n_qubits"]
        n_layers = row["n_layers"]
        seed = row["seed"]

        entry = compute_entry_for_config(
            family=family,
            n_qubits=n_qubits,
            n_layers=n_layers,
            seed=seed,
            backend=backend,
            method=method,
            n_bins_value=n_bins_value,
        )
        if entry is None:
            continue

        entries.append({"cid": cid, **entry})
        logger.info("Computed %s: SRE=%s", cid, entry.get("sre"))

    return entries


def compute_all_entries_parallel(
    params: list[dict[str, Any]],
    backend: str,
    method: str,
    n_bins_value: int,
    dask_n_workers: int,
) -> list[dict[str, Any]]:
    from dask.distributed import as_completed

    from qqe.parallel import dask_client

    updated_rows: list[dict[str, Any] | None] = [None] * len(params)
    cpu_count = os.cpu_count() or 2
    safe_workers = max(1, min(dask_n_workers, cpu_count))

    with dask_client(
        mode="slurm",
        n_workers=safe_workers,
        threads_per_worker=1,
        memory_per_worker="64GiB",
        dashboard=True,
        walltime="0-2:30:00",
    ) as client:
        client.wait_for_workers(1)
        logger.info("Dask dashboard: %s", client.dashboard_link)
        logger.info("Workers connected: %s", len(client.scheduler_info()["workers"]))

        inflight: dict[Any, tuple[int, dict[str, Any]]] = {}
        ac = as_completed()
        max_inflight = min(max(8, 4 * safe_workers), 64)
        it = iter(enumerate(params))

        def submit_one(i: int, row: dict[str, Any]) -> None:
            fut = client.submit(
                compute_entry_for_config,
                family=row["family"],
                n_qubits=row["n_qubits"],
                n_layers=row["n_layers"],
                seed=row["seed"],
                backend=backend,
                method=method,
                n_bins_value=n_bins_value,
                pure=False,
            )
            inflight[fut] = (i, row)
            ac.add(fut)

        for _ in range(min(max_inflight, len(params))):
            i, row = next(it, (None, None))
            if row is None or i is None:
                break
            submit_one(i, row)

        for fut in tqdm(ac, total=len(params), desc="Computing entries (parallel)"):
            i, row = inflight.pop(fut)
            try:
                entry = fut.result()
                if entry is not None:
                    updated_rows[i] = {"cid": row["cid"], **entry}
            except Exception as exc:
                logger.error("Failed row %s (%s): %s", i, row["cid"], exc)

            j, next_row = next(it, (None, None))
            if next_row is not None and j is not None:
                submit_one(j, next_row)

    return [row for row in updated_rows if row is not None]


def main(
    backend: str = typer.Option("quimb", help="Backend to use (quimb or pennylane)"),
    method: str = typer.Option("fwht", help="SRE method (exact, fwht, or sampling)"),
    use_dask: bool = typer.Option(default=True, help="Use Dask for parallel computation"),
    output_file: str = typer.Option(
        "qqe/data/",
        help="Output folder for results",
    ),
    n_bins_option: int = typer.Option(50, help="Number of bins for graph encoding"),
    families: str = typer.Option(
        "haar,clifford,quansistor,random",
        help="Comma-separated families to include",
    ),
    n_seeds_option: int = typer.Option(
        15,
        help="Number of seeds per (family, qubits, layers)",
    ),
    qubits_min: int = typer.Option(4, help="Minimum number of qubits"),
    qubits_max: int = typer.Option(10, help="Maximum number of qubits (inclusive)"),
    qubits_step: int = typer.Option(2, help="Step for qubits range"),
    layers_min: int = typer.Option(1, help="Minimum number of layers"),
    layers_max: int = typer.Option(99, help="Maximum number of layers (inclusive)"),
    layers_step: int = typer.Option(2, help="Step for layers range"),
    max_configs: int | None = typer.Option(
        None,
        help="Optional cap on number of generated configurations (useful for local smoke tests)",
    ),
    dask_n_workers: int = typer.Option(4, help="Number of local Dask workers when --use-dask"),
):
    selected_families = [part.strip() for part in families.split(",") if part.strip()]
    invalid_families = [name for name in selected_families if name not in FAMILY_REGISTRY]
    if invalid_families:
        msg = f"Unknown families requested: {invalid_families}. Valid options: {list(FAMILY_REGISTRY.keys())}"
        raise typer.BadParameter(msg)

    qubits_values = np.arange(qubits_min, qubits_max + 1, qubits_step)
    layers_values = np.arange(layers_min, layers_max + 1, layers_step)

    params = generate_dataset_params(
        selected_families,
        qubits_values,
        layers_values,
        n_seeds_option,
    )
    if max_configs is not None:
        params = params[:max_configs]
    logger.info("Generated %s circuit configurations", len(params))

    entries = compute_all_entries(
        params,
        backend=backend,
        method=method,
        use_dask=use_dask,
        n_bins_value=n_bins_option,
        dask_n_workers=dask_n_workers,
    )

    output_path = PROJECT_ROOT / output_file / f"encoding_data_{backend}_{method}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "backend": backend,
            "method": method,
            "n_bins": n_bins_option,
            "n_seeds": n_seeds_option,
            "families": selected_families,
            "qubits_range": qubits_values.tolist(),
            "layers_range": layers_values.tolist(),
            "entries": len(entries),
            "use_dask": use_dask,
        },
        "data": entries,
    }

    with output_path.with_suffix(".json").open("w") as f:
        json.dump(payload, f)

    logger.info("Results saved to %s", output_path.with_suffix(".json"))
    logger.info("Completed %s dataset entries", len(entries))


if __name__ == "__main__":
    configure_logger(logging.INFO, logging.INFO)
    logger.info("Starting data generation...")
    typer.run(main)
