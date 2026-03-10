from __future__ import annotations

import ctypes
import gc
import hashlib
import itertools
import json
import logging
import os

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
import contextlib

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


def _trim_process_memory() -> None:
    """Best-effort memory release to keep Dask worker RSS under control."""
    with contextlib.suppress(Exception):
        gc.collect()

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    try:
        if os.name == "nt":
            process_handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.psapi.EmptyWorkingSet(process_handle)
        else:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
    except Exception:
        # APIs are platform-dependent; ignore if unavailable.
        pass


def compute_entry_for_config(
    cid: str,
    family: str,
    n_qubits: int,
    n_layers: int,
    seed: int,
    backend: str = "quimb",
    method: str = "fwht",
    representation: str = "dense",
    n_bins_value: int = 50,
    output_dir: Path | None = None,
) -> dict[str, Any] | None:
    from qqe.experiments.core import ExperimentConfig
    from qqe.properties.compute import PropertyRequest
    from qqe.states.types import BackendConfig

    try:
        # Resolve output path + ensure dir exists
        base_dir = output_dir if output_dir is not None else DATASET_DIR
        base_dir.mkdir(parents=True, exist_ok=True)
        path = base_dir / f"{cid}.pt"
        tmp_path = path.with_suffix(".pt.tmp")

        if path.exists():
            return {"cid": cid, "path": str(path), "cached": True}

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

        # ---- compact storage ----
        x = graph_data.x.detach().cpu()
        # Only uint8 if binary
        if x.numel() > 0 and x.min().item() >= 0 and x.max().item() <= 1:
            x = x.to(torch.uint8)
        else:
            x = x.to(torch.float16)

        edge_index = graph_data.edge_index.detach().cpu().to(torch.int32)
        global_features = graph_data.global_features.detach().cpu().to(torch.float32)

        payload = {
            "x": x,
            "edge_index": edge_index,
            "global_features": global_features,
            "gate_counts": _safe_gate_counts(gate_counts),
            "sre": sre_value,
            "meta": {
                "cid": cid,
                "family": family,
                "n_qubits": int(n_qubits),
                "n_layers": int(n_layers),
                "seed": int(seed),
                "backend": backend,
                "method": method,
                "representation": representation,
                "n_bins": int(n_bins_value),
            },
        }

        torch.save(payload, tmp_path)
        tmp_path.replace(path)

        return {"cid": cid, "sre": sre_value, "path": str(path)}

    except Exception:
        logger.exception(
            "Error computing entry for family=%s Q%s L%s S%s",
            family,
            n_qubits,
            n_layers,
            seed,
        )
        return None
    finally:
        _trim_process_memory()


def compute_all_entries(
    params: list[dict[str, Any]],
    backend: str = "quimb",
    method: str = "fwht",
    *,
    use_dask: bool = False,
    n_bins_value: int = 50,
    dask_n_workers: int = 20,
    dask_memory_per_worker: str | None = "auto",
    # NEW: where individual .pt samples + index files go
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    if use_dask:
        return compute_all_entries_parallel(
            params,
            backend=backend,
            method=method,
            n_bins_value=n_bins_value,
            dask_n_workers=dask_n_workers,
            dask_memory_per_worker=dask_memory_per_worker,
            output_dir=output_dir,
        )
    return compute_all_entries_sequential(
        params,
        backend=backend,
        method=method,
        n_bins_value=n_bins_value,
        output_dir=output_dir,
    )


def compute_all_entries_sequential(
    params: list[dict[str, Any]],
    backend: str,
    method: str,
    n_bins_value: int,
    output_dir: Path | None,
) -> list[dict[str, Any]]:
    """Sequential version.

    - Writes each sample to disk inside compute_entry_for_config(...)
    - Appends results to index.jsonl to keep RAM flat
    - Returns a *small* list of {"cid","sre","path"} rows (optional)
    """
    base_dir = output_dir if output_dir is not None else DATASET_DIR
    base_dir.mkdir(parents=True, exist_ok=True)

    family = params[0]["family"] if params else "unknown"
    index_path = base_dir / f"index_{family}.jsonl"
    entries: list[dict[str, Any]] = []

    with index_path.open("a", encoding="utf-8") as f:
        for row in tqdm(params, desc="Computing dataset entries (sequential)"):
            cid = row["cid"]
            entry = compute_entry_for_config(
                cid=cid,
                family=row["family"],
                n_qubits=row["n_qubits"],
                n_layers=row["n_layers"],
                seed=row["seed"],
                backend=backend,
                method=method,
                n_bins_value=n_bins_value,
                output_dir=base_dir,
            )
            if entry is None:
                continue

            # stream to disk immediately (low RAM)
            f.write(json.dumps(entry) + "\n")
            f.flush()

            entries.append(entry)
            logger.info("Computed %s: SRE=%s", cid, entry.get("sre"))

    return entries


def compute_all_entries_parallel(
    params: list[dict[str, Any]],
    backend: str,
    method: str,
    n_bins_value: int,
    dask_n_workers: int,
    output_dir: Path | None,
    dask_memory_per_worker: str | None = "auto",
) -> list[dict[str, Any]]:
    """Parallel version (Dask).

    - Caps in-flight tasks aggressively to avoid worker OOM
    - Each task writes its own .pt file to output_dir
    - The driver appends small results to index.jsonl as futures finish
    - Returns only small rows (cid/sre/path), not the whole dataset content.
    """
    from dask.distributed import as_completed

    from qqe.parallel import dask_client

    base_dir = output_dir if output_dir is not None else DATASET_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    family = params[0]["family"] if params else "unknown"
    index_path = base_dir / f"index_{family}.jsonl"

    cpu_count = os.cpu_count() or 2
    safe_workers = max(1, min(int(dask_n_workers), cpu_count))

    # IMPORTANT: keep this LOW for memory-heavy Quimb dense sims
    # Start conservative; increase only if stable.
    max_inflight = max(1, safe_workers//2)  # NOT 4*workers

    rows_out: list[dict[str, Any]] = []

    with dask_client(
        mode="local",
        n_workers=safe_workers,
        threads_per_worker=1,
        memory_per_worker=dask_memory_per_worker or "2GB",
        dashboard=True,
        walltime="0-2:30:00",
    ) as client:
        client.wait_for_workers(1)
        logger.info("Dask dashboard: %s", client.dashboard_link)
        logger.info("Workers connected: %s", len(client.scheduler_info()["workers"]))

        ac = as_completed()
        inflight: dict[Any, dict[str, Any]] = {}
        it = iter(params)

        def submit_one(row: dict[str, Any]) -> None:
            fut = client.submit(
                compute_entry_for_config,
                cid=row["cid"],
                family=row["family"],
                n_qubits=row["n_qubits"],
                n_layers=row["n_layers"],
                seed=row["seed"],
                backend=backend,
                method=method,
                n_bins_value=n_bins_value,
                output_dir=base_dir,
                pure=False,
            )
            inflight[fut] = row
            ac.add(fut)

        # Prime the queue
        for _ in range(min(max_inflight, len(params))):
            row = next(it, None)
            if row is None:
                break
            submit_one(row)

        # Stream results to disk as they complete
        with index_path.open("a", encoding="utf-8") as f:
            for fut in tqdm(ac, total=len(params), desc="Computing entries (parallel)"):
                row = inflight.pop(fut, None)
                try:
                    entry = fut.result()
                    if entry is not None:
                        f.write(json.dumps(entry) + "\n")
                        f.flush()
                        rows_out.append(entry)
                except Exception as exc:
                    cid = row["cid"] if row else "unknown"
                    logger.error("Failed (%s): %s", cid, exc)
                finally:
                    fut.release()

                # Keep the pipeline full but capped
                next_row = next(it, None)
                if next_row is not None:
                    submit_one(next_row)

    return rows_out


def main(
    backend: str = typer.Option("pennylane", help="Backend to use (quimb or pennylane)"),
    method: str = typer.Option("fwht", help="SRE method (exact, fwht, or sampling)"),
    use_dask: bool = typer.Option(default=True, help="Use Dask for parallel computation"),
    output_file: str = typer.Option(
        "qqe/data/",
        help="Output folder for results",
    ),
    n_bins_option: int = typer.Option(50, help="Number of bins for graph encoding"),
    families: str = typer.Option(
        "random",
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
    dask_memory_per_worker: str = typer.Option(
        "64GiB",
        help="Per-worker memory limit for Dask (for example: 6GB, 8000MB, auto)",
    ),
):
    selected_families = [part.strip() for part in families.split(",") if part.strip()]
    invalid_families = [name for name in selected_families if name not in FAMILY_REGISTRY]
    if invalid_families:
        msg = f"Unknown families requested: {invalid_families}. Valid options: {list(FAMILY_REGISTRY.keys())}"
        raise typer.BadParameter(msg)

    qubits_values = np.arange(qubits_min, qubits_max + 1, qubits_step)
    layers_values = np.arange(layers_min, layers_max + 1, layers_step)

    output_dir = PROJECT_ROOT / output_file / f"encoding_data_{backend}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for family in selected_families:
        logger.info("Selected family: %s", family)
        family_output_dir = output_dir / family
        family_output_dir.mkdir(parents=True, exist_ok=True)

        params = generate_dataset_params(
            [family],
            qubits_values,
            layers_values,
            n_seeds_option,
        )
        if max_configs is not None:
            params = params[:max_configs]
        logger.info("Generated %s circuit configurations for family %s", len(params), family)

        entries = compute_all_entries(
            params,
            backend=backend,
            method=method,
            use_dask=use_dask,
            n_bins_value=n_bins_option,
            dask_n_workers=dask_n_workers,
            dask_memory_per_worker=dask_memory_per_worker,
            output_dir=family_output_dir,
        )

        meta_path = family_output_dir / f"metadata_{family}.json"
        metadata = {
            "backend": backend,
            "method": method,
            "n_bins": n_bins_option,
            "n_seeds": n_seeds_option,
            "families": selected_families,
            "qubits_range": qubits_values.tolist(),
            "layers_range": layers_values.tolist(),
            "entries_completed": len(entries),
            "use_dask": use_dask,
            "index_file": f"index_{family}.jsonl",
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        logger.info("Completed family %s: %s entries", family, len(entries))
        logger.info("  Metadata: %s", meta_path)
        logger.info("  Index: %s", family_output_dir / f"index_{family}.jsonl")
        logger.info("  Samples: %s", family_output_dir)

    logger.info("All families processed successfully")

    # with output_path.with_suffix(".json").open("w") as f:
    #     json.dump(payload, f)

    # logger.info("Results saved to %s", output_path.with_suffix(".json"))
    # logger.info("Completed %s dataset entries", len(entries))


if __name__ == "__main__":
    configure_logger(logging.INFO, logging.INFO)
    logger.info("Starting data generation...")
    typer.run(main)
