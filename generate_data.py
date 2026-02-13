from __future__ import annotations

import itertools
import json
import logging

from pathlib import Path
from typing import Any

import numpy as np
import typer

from dask.distributed import as_completed
from tqdm import tqdm

from qqe.backend import PennylaneBackend, QuimbBackend
from qqe.circuit.families import (
    CliffordBrickwork,
    HaarBrickwork,
    QuansistorBrickwork,
)
from qqe.experiments.core import run_experiment
from qqe.parallel import dask_client
from qqe.utils import FileCache, configure_logger

logger = logging.getLogger(__name__)


data = [
    [
        "cid",
        "family",
        "n_qubits",
        "n_layers",
        "seed",
        "state_vector",
        "SRE",
        "SRE_std",
    ],  # state_vector
]

n_seeds = 50

qubits_range = np.arange(4, 11, 2)
layers_range = np.arange(1, 51, 2)
tcount_range = np.arange(0, 100, 2)

# Backend and family registries
BACKEND_REGISTRY = {
    "pennylane": PennylaneBackend,
    "quimb": QuimbBackend,
}

FAMILY_REGISTRY = {
    "haar": HaarBrickwork,
    "clifford": CliffordBrickwork,
    "quansistor": QuansistorBrickwork,
}

# Initialize cache for results
PROJECT_ROOT = Path(__file__).resolve().parent
cache = FileCache(PROJECT_ROOT / "outputs" / "cache")


def make_seed(n_qubits, n_layers, rep) -> int:
    return hash((n_qubits, n_layers, rep)) % (2**32)


def calculate_tcount(n_layers: int, per_layer: int = 2) -> int:
    """Calculate the number of t gates for a given number of layers.

    Args:
        n_layers: Number of layers in the circuit.
        per_layer: Number of t gates per layer (default: 2).

    Returns:
        Total t gate count: per_layer × (n_layers - 1)
        (Last layer is excluded from t-gate placement)
    """
    return per_layer * max(0, n_layers - 1)


def is_valid_config(n_qubits, n_layers, tcount) -> bool:
    max_tcount = calculate_tcount(n_layers, per_layer=2)
    return tcount <= max_tcount


def make_cid(family, n_qubits, n_layers, seed) -> str:
    return f"{family}_Q{n_qubits}_L{n_layers}_S{seed}"


def generate_data_params(qubits_range, layers_range, n_seeds):
    params = list(itertools.product(qubits_range, layers_range))
    data = []
    for rep in range(n_seeds):
        for n_qubits, n_layers in params:
            seed = make_seed(n_qubits, n_layers, rep)
            cid = make_cid("haar", n_qubits, n_layers, seed)
            data.append([cid, "haar", n_qubits, n_layers, seed, None, None, None])
    return data


def compute_sre_for_row(
    family: str,
    n_qubits: int,
    n_layers: int,
    seed: int,
    d: int = 2,
    backend: str = "quimb",
    method: str = "exact",
    representation: str = "dense",
) -> tuple[float | None, float | None, list | None]:
    """Compute SRE and state_vector for a single configuration.

    Args:
        family: Circuit family name (e.g., 'haar', 'clifford', 'quansistor')
        n_qubits: Number of qubits
        n_layers: Number of layers
        seed: Random seed
        d: Local dimension (default: 2 for qubits)
        backend: Backend to use ('quimb' or 'pennylane')
        method: SRE computation method ('exact', 'fwht', or 'sampling')
        representation: State representation ('dense' or 'mps')

    Returns:
        Tuple of (SRE_value, SRE_std, state_vector) where SRE_std is None for deterministic methods
    """
    from qqe.experiments.core import ExperimentConfig
    from qqe.properties.compute import PropertyRequest
    from qqe.states.types import DenseState

    try:
        # Get circuit specification
        family_cls = FAMILY_REGISTRY[family]
        family_obj = family_cls()
        spec = family_obj.make_spec(
            n_qubits=n_qubits,
            n_layers=n_layers,
            d=d,
            seed=seed,
        )

        # Configure backend and properties
        from qqe.states.types import BackendConfig

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

        # Create experiment configuration
        exp_config = ExperimentConfig(
            spec=spec,
            backend=backend_config,
            properties=[property_request],
        )

        # Run experiment
        result = run_experiment(
            exp_config,
            backend_registry=BACKEND_REGISTRY,
            cache=cache,
        )

        # Extract SRE value
        sre_result = result.results.get("SRE")
        sre_value = sre_result.value if sre_result else None
        sre_std = sre_result.meta.get("std") if sre_result else None

        # Extract state vector (only for dense representation)
        state_vector_list = None
        if representation == "dense":
            try:
                # Try to access state vector through public interface
                if hasattr(result, "state_info") and result.state_info is not None:
                    state_vec = result.state_info.get("state_vector")
                    logger.info("Accessed state vector")
                    if isinstance(state_vec, DenseState):
                        state_vector_list = state_vec.vector.tolist()
                    else:
                        state_vector_list = np.asarray(state_vec).tolist()
            except (AttributeError, TypeError, ValueError):
                pass  # If state extraction fails, just skip it

        # Convert numpy types to native Python floats for JSON serialization
        if sre_value is not None:
            sre_value = float(sre_value)
        if sre_std is not None:
            sre_std = float(sre_std)

        return sre_value, sre_std, state_vector_list

    except Exception:
        logger.exception("Error computing SRE for Q%s_L%s_S%s", n_qubits, n_layers, seed)
        return None, None, None


def compute_all_sre(
    data_rows: list[list[Any]],
    backend: str = "quimb",
    method: str = "exact",
    use_dask: bool = False,
) -> list[list[Any]]:
    """Compute SRE for all rows in data.

    Args:
        data_rows: List of data rows (excluding header)
        backend: Backend to use ('quimb' or 'pennylane')
        method: SRE computation method ('exact', 'fwht', or 'sampling')
        use_dask: Whether to use Dask for parallel computation

    Returns:
        Updated data rows with SRE and SRE_std columns filled
    """
    if use_dask:
        return compute_all_sre_parallel(data_rows, backend, method)
    return compute_all_sre_sequential(data_rows, backend, method)


def compute_all_sre_sequential(
    data_rows: list[list[Any]],
    backend: str,
    method: str,
) -> list[list[Any]]:
    """Compute SRE sequentially for all rows."""
    updated_rows = []

    for row in tqdm(data_rows, desc="Computing SRE"):
        cid, family, n_qubits, n_layers, seed, state_vector, sre, sre_std = row

        # Compute SRE, SRE_std, and state_vector
        sre_value, sre_std_value, state_vec = compute_sre_for_row(
            family=family,
            n_qubits=n_qubits,
            n_layers=n_layers,
            seed=seed,
            backend=backend,
            method=method,
        )
        logger.info(f"Computed SRE for {cid}: SRE={sre_value}, SRE_std={sre_std_value}")

        # Update row with results
        updated_row = [
            cid,
            family,
            n_qubits,
            n_layers,
            seed,
            state_vec,
            sre_value,
            sre_std_value,
        ]
        updated_rows.append(updated_row)

        logger.info(f"{cid}: SRE={sre_value}")

    return updated_rows


def compute_all_sre_parallel(data_rows, backend, method) -> list[list[Any]]:
    updated_rows: list[list[Any] | None] = [None] * len(data_rows)

    with dask_client(
        mode="slurm",
        n_workers=30,
        threads_per_worker=1,  # important (see next section)
        memory_per_worker="64GiB",
        dashboard=True,
        walltime="0-0:30:00",
    ) as client:

        client.wait_for_workers(1)
        logger.info(f"Dask dashboard: {client.dashboard_link}")
        logger.info(f"Workers connected: {len(client.scheduler_info()['workers'])}")

        inflight = {}
        max_inflight = 200  # tune this (start 50–200)

        it = iter(enumerate(data_rows))

        def submit_one(i, row):
            cid, family, n_qubits, n_layers, seed, *_ = row
            fut = client.submit(
                compute_sre_for_row,
                family=family,
                n_qubits=n_qubits,
                n_layers=n_layers,
                seed=seed,
                backend=backend,
                method=method,
                pure=False,
            )
            inflight[fut] = (i, row)

        # prime
        for _ in range(min(max_inflight, len(data_rows))):
            i, row = next(it, (None, None))
            if row is None:
                break
            submit_one(i, row)

        ac = as_completed(inflight)

        for fut in tqdm(ac, total=len(data_rows), desc="Computing SRE (parallel)"):
            i, row = inflight.pop(fut)
            try:
                sre_value, sre_std_value, state_vec = fut.result()
                cid, family, n_qubits, n_layers, seed, *_ = row
                updated_rows[i] = [
                    cid,
                    family,
                    n_qubits,
                    n_layers,
                    seed,
                    state_vec,
                    sre_value,
                    sre_std_value,
                ]
            except Exception as e:
                logger.error(f"Failed row {i}: {e}")
                updated_rows[i] = row

            # keep pipeline full
            j, next_row = next(it, (None, None))
            if next_row is not None:
                submit_one(j, next_row)

        return [row for row in updated_rows if row is not None]


def main(
    backend: str = typer.Option("quimb", help="Backend to use (quimb or pennylane)"),
    method: str = typer.Option("fwht", help="SRE method (exact, fwht, or sampling)"),
    use_dask: bool = typer.Option(True, help="Use Dask for parallel computation"),
    output_file: str = typer.Option("qqe/data/sre_data_quimb_exact.json", help="Output file for results"),
):
    """Generate and compute SRE data for Haar circuits."""
    # Generate parameter configurations
    data_rows = generate_data_params(qubits_range, layers_range, n_seeds)
    logger.info(f"Generated {len(data_rows)} configurations")

    # Compute SRE for all rows
    logger.info(f"Computing SRE using backend={backend}, method={method}, parallel={use_dask}")
    updated_rows = compute_all_sre(
        data_rows, backend=backend, method=method, use_dask=use_dask
    )

    # Save results
    output_path = PROJECT_ROOT / output_file
    results_data = {
        "header": [
            "cid",
            "family",
            "n_qubits",
            "n_layers",
            "seed",
            "state_vector",
            "SRE",
            "SRE_std",
        ],
        "data": updated_rows,
        "metadata": {
            "backend": backend,
            "method": method,
            "n_seeds": n_seeds,
            "qubits_range": qubits_range.tolist(),
            "layers_range": layers_range.tolist(),
        },
    }

    with output_path.open("w") as f:
        json.dump(results_data, f, indent=2, default=str)

    logger.info(f"Results saved to {output_path}")
    logger.info(f"Completed {len(updated_rows)} SRE computations")


if __name__ == "__main__":
    configure_logger(logging.INFO, logging.INFO)
    logger.info("Starting data generation...")
    typer.run(main)
