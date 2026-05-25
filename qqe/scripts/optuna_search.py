from __future__ import annotations

import logging

import numpy as np
import optuna
import torch
import typer

from GNN.training.datasets import (
    prepare_datasets,
)
from GNN.training.runners import MODEL_REGISTRY
from GNN.training.utils import (
    collect_dataset_paths,
    collect_files_path,
)
from utils import configure_logger

logger = logging.getLogger(__name__)

_AMP_DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"


def main(
    model_type: str = "gnn",
    training_mode: str = "global",
    family: str | None = "random",
    data_dir: str = "../outputs/data/general_dataset",
    target: str = "sre",
    epochs: int = 25,
    n_trials: int = 10,
    study_name: str = typer.Option("gnn_optuna_study", help="Optuna study name"),
    storage_url: str = typer.Option("sqlite:///gnn_optuna_study.db", help="Optuna storage URL, e.g. 'sqlite:///nn_optuna_study.db' or 'postgresql://user:pass@host/db'"),
):
    seed = np.random.randint(0, 10000)
    logger.info(f"Starting Optuna hyperparameter search...")
    data_paths = collect_dataset_paths(
        data_dir,
        family=family if training_mode == "per_family" else None,
        split="target",
    )

    logger.info(f"Found {len(data_paths)} dataset files for target={target} in data_dir={data_dir}.")

    prepared_data = prepare_datasets(
        data_paths,
        loader_kind=model_type,
        seed=seed,
        train_split=0.8,
        val_split=0.1,
        global_feature_variant="binned",
        node_feature_variant=None,
        family_projection=family if training_mode == "per_family" else None,
    )

    logger.info(f"Prepared datasets with loader_kind={model_type}. Starting Optuna study with name='{study_name}' and storage='{storage_url}'...")
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=storage_url,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5, n_startup_trials=10),
    )

    def objective_fn(trial):
        return MODEL_REGISTRY[model_type]["objective_fn"](
            trial,
            prepared_data=prepared_data,
            epochs=epochs,
        )
    logger.info(f"Running Optuna optimization for {n_trials} trials...")
    study.optimize(objective_fn, n_trials=n_trials, gc_after_trial=True)


if __name__ == "__main__":
    configure_logger(console_level=logging.INFO, file_level=logging.INFO)
    typer.run(main)
