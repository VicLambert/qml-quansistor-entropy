from __future__ import annotations

import logging

from pathlib import Path

import numpy as np
import typer
from GNN.dataset_builder import (
    DataGenConfig,
    RegimeDistribution,
    SamplingConfig,
    run_dataset_pipeline,
)
from utils import configure_logger

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def default_sampling_config() -> SamplingConfig:
    return SamplingConfig(
        clifford=RegimeDistribution(
            regimes=[
                "zero",
                "tiny",
                "very_low",
                "low",
                "medium_low",
                "medium",
                "medium_high",
                "high",
            ],
            probabilities=[
                0.05,
                0.05,
                0.10,
                0.10,
                0.20,
                0.20,
                0.20,
                0.10,
            ],
        ),
        random=RegimeDistribution(
            regimes=[
                "identity_like",
                "near_clifford",
                "small_angles",
                "medium_angles",
                "generic_sparse",
                "generic_dense",
            ],
            probabilities=[0.005, 0.155, 0.3, 0.25, 0.15, 0.14],
        ),
        quansistor=RegimeDistribution(
            regimes=[
                "identity_like",
                "weak",
                "moderate",
                "structured_equal_ab",
                "structured_opposite_ab",
                "generic_uniform",
            ],
            probabilities=[0.1, 0.4, 0.4, 0.1, 0.1, 0.1],
        ),
        haar=RegimeDistribution(
            regimes=[
                "identity_like",
                "very_weak",
                "sparse_weak",
                "medium_weak",
                "dense_weak",
                "sparse_full",
                "medium",
                "dense_medium",
                "sparse_full",
                "medium_full",
                "full",
            ],
            probabilities=[0.01, 0.01, 0.05, 0.1, 0.1, 0.12, 0.1, 0.12, 0.12, 0.12, 0.15],
        ),
    )


def main(
    backend: str = typer.Option("pennylane", help="Backend to use"),
    target: str = typer.Option("SRE", help="Target to compute: SRE, EE, or none"),
    method: str = typer.Option("fwht", help="SRE/EE computation method"),
    use_dask: bool = typer.Option(True, help="Use Dask"),
    output_dir: str = typer.Option(
        "/outputs/data/dataset_SRE",
        help="Output folder",
    ),
    n_bins_option: int = typer.Option(50, help="Number of bins for graph encoding"),
    families: str = typer.Option(
        "quansistor",
        help="Comma-separated circuit families",
    ),
    n_seeds_option: int = typer.Option(
        250,
        help="Seeds per (family, qubits, layers)",
    ),
    prediction_n_seeds_option: int | None = typer.Option(
        100,
        help="Optional different number of seeds for qubits without target",
    ),
    qubits_min: int = typer.Option(4),
    qubits_max: int = typer.Option(30),
    qubits_step: int = typer.Option(2),
    layers_min: int = typer.Option(2),
    layers_max: int = typer.Option(100),
    layers_step: int = typer.Option(2),
    target_qubits: str = typer.Option(
        "4,6,8,10",
        help="Comma-separated qubit values for which the target is computed",
    ),
    max_shards: int | None = typer.Option(None),
    dask_n_workers: int = typer.Option(4),
    dask_memory_per_worker: str = typer.Option("32GiB"),
    block_size: int = typer.Option(10, help="Number of layers per block for layer-wise processing"),
):
    selected_families = [f.strip() for f in families.split(",") if f.strip()]

    qubits_values = np.arange(qubits_min, qubits_max + 1, qubits_step)
    layers_values = np.concatenate(
        ([1], np.arange(layers_min, layers_max + 1, layers_step)),
    )

    selected_target_qubits = tuple(
        int(q.strip()) for q in target_qubits.split(",") if q.strip()
    )

    target_norm = target.strip().lower()

    compute_sre = target_norm == "sre"
    compute_EE = target_norm == "ee"

    if target_norm not in {"sre", "ee", "none"}:
        raise ValueError("target must be 'SRE', 'EE', or 'none'")

    output_path = PROJECT_ROOT / output_dir.strip("/")

    config = DataGenConfig(
        backend=backend,
        method=method,
        families=selected_families,
        qubits_values=qubits_values,
        layers_values=layers_values,
        n_seeds=n_seeds_option,
        prediction_n_seeds=prediction_n_seeds_option,
        n_bins=n_bins_option,
        compute_sre=compute_sre,
        compute_EE=compute_EE,
        target_qubits=selected_target_qubits,
        representation="dense",
        use_dask=use_dask,
        dask_n_workers=dask_n_workers,
        dask_memory_per_worker=dask_memory_per_worker,
        output_dir=output_path,
        max_configs=max_shards,
    )

    run_dataset_pipeline(
        config=config,
        families=selected_families,
        qubits_values=qubits_values,
        layers_values=layers_values,
        n_seeds=n_seeds_option,
        use_dask=use_dask,
        max_shards=max_shards,
        dask_n_workers=dask_n_workers,
        dask_memory_per_worker=dask_memory_per_worker,
        sampling_config=default_sampling_config(),
        layer_block_size = block_size,
    )


if __name__ == "__main__":
    configure_logger(logging.INFO, logging.INFO)
    logger.info("Starting general dataset generation...")
    typer.run(main)
