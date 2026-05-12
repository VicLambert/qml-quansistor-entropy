from __future__ import annotations

import logging

from pathlib import Path

import numpy as np
import typer

from qqe.src.GNN.dataset_builder import DataGenConfig, run_dataset_pipeline, SamplingConfig, RegimeDistribution
from qqe.src.utils import configure_logger

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main(
    backend: str = typer.Option("pennylane", help="Backend to use (quimb or pennylane)"),
    method: str = typer.Option("none", help="SRE method (exact, fwht, or sampling)"),
    use_dask: bool = typer.Option(True, help="Use Dask for parallel computation"),
    output_dir: str = typer.Option(
        "/outputs/data/prediction_data",
        help="Output folder for results",
    ),
    n_bins_option: int = typer.Option(50, help="Number of bins for graph encoding"),
    families: str = typer.Option(
        "random",
        help="Comma-separated families to include",
    ),
    sampling_config: SamplingConfig | None = None,
    n_seeds_option: int = typer.Option(
        75,
        help="Number of seeds per (family, qubits, layers)",
    ),
    qubits_min: int = typer.Option(12, help="Minimum number of qubits"),
    qubits_max: int = typer.Option(30, help="Maximum number of qubits (inclusive)"),
    qubits_step: int = typer.Option(2, help="Step for qubits range"),
    layers_min: int = typer.Option(2, help="Minimum number of layers"),
    layers_max: int = typer.Option(100, help="Maximum number of layers (inclusive)"),
    layers_step: int = typer.Option(2, help="Step for layers range"),
    max_configs: int | None = typer.Option(
        None,
        help="Optional cap on number of generated configurations (useful for local smoke tests)",
    ),
    dask_n_workers: int = typer.Option(4, help="Number of local Dask workers when --use-dask"),
    dask_memory_per_worker: str = typer.Option(
        "32GiB",
        help="Per-worker memory limit for Dask (for example: 6GB, 8000MB, auto)",
    ),
):
    selected_families = [f.strip() for f in families.split(",") if f.strip()]
    qubits_values = np.arange(qubits_min, qubits_max + 1, qubits_step)
    layers_values =  np.concatenate(([1], np.arange(layers_min, layers_max + 1, layers_step)))

    output_path = Path(str(PROJECT_ROOT) + output_dir)

    if sampling_config is None:
        sampling_config = SamplingConfig(
            clifford=RegimeDistribution(
                regimes=["zero", "low", "medium", "high"],
                probabilities=[0.15, 0.15, 0.25, 0.45],
            ),
            random=RegimeDistribution(
                regimes=["identity_like", "clifford_like", "small_angles", "generic"],
                probabilities=[0.15, 0.20, 0.2, 0.45],
            ),
            quansistor=RegimeDistribution(
                regimes=["identity_like", "weak", "moderate", "structured_equal_ab", "structured_opposite_ab", "generic_uniform"],
                probabilities=[0.15, 0.15, 0.15, 0.15, 0.15, 0.25],
            ),
            haar=RegimeDistribution(
                regimes=["none", "sparse_weak", "dense_weak", "sparse_full", "medium", "full"],
                probabilities=[0.15, 0.15, 0.15, 0.15, 0.15, 0.25],
            ),
        )

    config = DataGenConfig(
        backend=backend,
        method=method,
        families=selected_families,
        qubits_values=qubits_values,
        layers_values=layers_values,
        n_seeds=n_seeds_option,
        n_bins=n_bins_option,
        compute_sre=False,
        compute_EE=False,
        representation="dense",
        use_dask=use_dask,
        dask_n_workers=dask_n_workers,
        dask_memory_per_worker=dask_memory_per_worker,
        output_dir=output_path,
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
        sampling_config=sampling_config,
    )


if __name__ == "__main__":
    configure_logger(logging.INFO, logging.INFO)
    logger.info("Starting prediction data generation...")
    typer.run(main)
