from __future__ import annotations

import csv
import logging
import math
import sys
import time
import warnings

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer

from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice,
)
from torch.amp import GradScaler, autocast
from torch.amp.autocast_mode import autocast
from torch.optim import Adam
from torch.utils.data import DataLoader as TorchDataLoader, random_split
from torch_geometric.data import Data, Dataset as PyGDataset
from torch_geometric.loader import DataLoader, DataLoader as PyGDataLoader
from tqdm import tqdm

from qqe.experiments.plotting import plot_training_curves
from qqe.GNN.physics_aware_NN import GNN, NN, QuantumCircuitGraphDataset, Regressor
from qqe.GNN.training.datasets import (
    GlobalTargetDatasetWrapper,
    PaddedGraphDatasetWrapper,
)
from qqe.GNN.training.train import build_loss, train_model
from qqe.GNN.training.train_config import TrainConfig
from qqe.GNN.training.utils import (
    FamilyFeatureProjector,
    ProjectedDatasetWrapper,
    cache_root_paths,
    collect_files_path,
    evaluate_loss,
    unpack_supervised_batch,
)
from qqe.utils import configure_logger

logger = logging.getLogger(__name__)

_AMP_DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"


def main(
    model_type: str = "nn",
    training_mode: str = "global",
    family: str | None = None,
    target: str = "sre",
    epochs: int = 15,
    n_trials: int = 3,
    study_name: str = typer.Option("nn_optuna_study", help="Optuna study name"),
    storage_url: str = typer.Option("sqlite:///nn_optuna_study.db", help="Optuna storage URL, e.g. 'sqlite:///nn_optuna_study.db' or 'postgresql://user:pass@host/db'"),
):
    seed = np.random.randint(0, 10000)
    data_paths = collect_files_path("data/training_data", family=family if training_mode == "per_family" else None)

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
    study.optimize(objective_fn, n_trials=n_trials, gc_after_trial=True)


if __name__ == "__main__":
    configure_logger(console_level=logging.INFO, file_level=logging.INFO)
    typer.run(main)
