from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import typer

from qqe.GNN.dataset_builder import DataGenConfig, run_dataset_pipeline
from qqe.utils import configure_logger

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent


def main(
    backend: str = typer.Option("pennylane", help="Backend to use (quimb or pennylane)"),
    method: str = typer.Option("fwht", help="SRE method (exact, fwht, or sampling)"),
    use_dask: bool = typer.Option(default=True, help="Use Dask for parallel computation"),
    output_file: str = typer.Option(
        "outputs/data/",
        help="Output folder for results",
    ),
    n_bins_option: int = typer.Option(50, help="Number of bins for graph encoding"),
    families: str = typer.Option(
        "haar,random,clifford,quansistor",
        help="Comma-separated families to include",
    ),
    n_seeds_option: int = typer.Option(
        50,
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
    selected_families = [f.strip() for f in families.split(",") if f.strip()]
    qubits_values = np.arange(qubits_min, qubits_max + 1, qubits_step)
    layers_values = np.arange(layers_min, layers_max + 1, layers_step)

    output_dir = Path(output_file)

    config = DataGenConfig(
        backend=backend,
        method=method,
        families=selected_families,
        qubits_values=qubits_values,
        layers_values=layers_values,
        n_seeds=n_seeds_option,
        n_bins=n_bins_option,
        compute_sre=True,
        representation="dense",
        use_dask=use_dask,
        dask_n_workers=dask_n_workers,
        dask_memory_per_worker=dask_memory_per_worker,
        output_dir=output_dir,
        max_configs=max_configs,
    )

    run_dataset_pipeline(
        config=config,
        families=selected_families,
        qubits_values=qubits_values,
        layers_values=layers_values,
        n_seeds=n_seeds_option,
        use_dask=use_dask,
        max_configs=max_configs,
        dask_n_workers=dask_n_workers,
        dask_memory_per_worker=dask_memory_per_worker,
    )


if __name__ == "__main__":
    configure_logger(logging.INFO, logging.INFO)
    logger.info("Starting data generation...")
    typer.run(main)
