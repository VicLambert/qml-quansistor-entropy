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


@dataclass
class PreparedData:
    train_ds: object
    val_ds: object
    test_ds: object
    node_in_dim: int | None
    global_in_dim: int
    base_dataset: object
    loader_kind: str  # "gnn" or "nn"


def prepare_datasets(
    pt_paths: list[str],
    *,
    loader_kind: str,  # "gnn" | "nn"
    seed: int = 42,
    train_split: float = 0.8,
    val_split: float = 0.1,
    global_feature_variant: str = "binned",
    node_feature_variant: str | None = None,
    family_projection: str | None = None,
) -> PreparedData:
    suffix = (
        f"{global_feature_variant}"
        f"__backend_{node_feature_variant or 'none'}"
        f"_familyproj_{family_projection or 'none'}"
    )
    root = cache_root_paths(pt_paths, suffix=suffix)

    base_dataset = QuantumCircuitGraphDataset(
        root=root,
        pt_paths=pt_paths,
        global_feature_variant=global_feature_variant,
        node_feature_backend_variant=node_feature_variant,
    )

    if len(base_dataset) < 3:
        raise RuntimeError("Dataset too small for train/val/test splitting.")

    working_dataset = base_dataset

    if family_projection is not None:
        projector = FamilyFeatureProjector(
            family=family_projection,
            all_gate_keys=base_dataset.all_gate_keys,
        )
        working_dataset = ProjectedDatasetWrapper(
            working_dataset,
            transform=projector,
        )

    if loader_kind == "gnn":
        final_dataset = PaddedGraphDatasetWrapper(working_dataset)
        sample0 = final_dataset[0]
        node_in_dim = int(sample0.x.shape[1])
        global_in_dim = int(sample0.global_features.numel())

    elif loader_kind == "nn":
        final_dataset = GlobalTargetDatasetWrapper(working_dataset)
        sample0_g, _ = final_dataset[0]
        node_in_dim = None
        global_in_dim = int(sample0_g.numel())

    else:
        raise ValueError("loader_kind must be 'gnn' or 'nn'")

    generator = torch.Generator().manual_seed(seed)

    primary_train_len = max(1, int(len(final_dataset) * train_split))
    test_len = max(1, len(final_dataset) - primary_train_len)

    while primary_train_len + test_len > len(final_dataset):
        primary_train_len -= 1

    primary_train_dataset, test_ds = random_split(
        final_dataset,
        [primary_train_len, test_len],
        generator=generator,
    )

    val_len = max(1, int(len(primary_train_dataset) * val_split))
    real_train_len = max(1, len(primary_train_dataset) - val_len)

    train_ds, val_ds = random_split(
        primary_train_dataset,
        [real_train_len, val_len],
        generator=generator,
    )

    return PreparedData(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        node_in_dim=node_in_dim,
        global_in_dim=global_in_dim,
        base_dataset=base_dataset,
        loader_kind=loader_kind,
    )


def make_loaders(
    prepared: PreparedData,
    *,
    batch_size: int,
    num_workers: int = 0,
):
    pin_mem = torch.cuda.is_available()

    if prepared.loader_kind == "gnn":
        Loader = PyGDataLoader
    elif prepared.loader_kind == "nn":
        Loader = TorchDataLoader
    else:
        raise ValueError(f"Unknown loader_kind: {prepared.loader_kind}")

    return (
        Loader(
            prepared.train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_mem,
        ),
        Loader(
            prepared.val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_mem,
        ),
        Loader(
            prepared.test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_mem,
        ),
    )

def build_loaders(
    pt_paths: list[str],
    *,
    batch_size: int = 32,
    seed: int = 42,
    train_split: float = 0.8,
    val_split: float = 0.1,
    global_feature_variant: str = "binned",
    node_feature_variant: str | None = None,
    family_projection: str | None = None,
):
    prepared = prepare_datasets(
        pt_paths,
        loader_kind="gnn",
        seed=seed,
        train_split=train_split,
        val_split=val_split,
        global_feature_variant=global_feature_variant,
        node_feature_variant=node_feature_variant,
        family_projection=family_projection,
    )

    train_loader, val_loader, test_loader = make_loaders(
        prepared,
        batch_size=batch_size,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        prepared.node_in_dim,
        prepared.global_in_dim,
        prepared.base_dataset,
    )


def build_loaders_NN(
    pt_paths: list[str],
    *,
    batch_size: int = 32,
    seed: int = 42,
    train_split: float = 0.8,
    val_split: float = 0.1,
    global_feature_variant: str = "binned",
    node_feature_variant: str | None = None,
    family_projection: str | None = None,
):
    prepared = prepare_datasets(
        pt_paths,
        loader_kind="nn",
        seed=seed,
        train_split=train_split,
        val_split=val_split,
        global_feature_variant=global_feature_variant,
        node_feature_variant=node_feature_variant,
        family_projection=family_projection,
    )

    train_loader, val_loader, test_loader = make_loaders(
        prepared,
        batch_size=batch_size,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        prepared.global_in_dim,
        prepared.base_dataset,
    )

def _resolve_model_save_path(base_path: str, allow_overwrite: bool = False) -> str:
    """Return a non-colliding checkpoint path unless overwrite is explicitly allowed."""
    path = Path(base_path)
    if allow_overwrite or not path.exists():
        return str(path)

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 1

    while True:
        candidate = parent / f"{stem}_v{counter}{suffix}"
        if not candidate.exists():
            print(
                "Model checkpoint already exists at %s. Saving to %s instead.",
                path,
                candidate,
            )
            return str(candidate)
        counter += 1

def train(
    *,
    model_type: str = "gnn",
    epochs: int = 150,
    lr: float = 1e-3,
    loss_type: str = "huber",   # "mse" | "huber"
    batch_size: int = 32,
    training_mode: str = "global",  # "global" | "per_family"
    family: str | None = None,  # required if training_mode == "per_family"
    target: str = "sre",
    model_hparams: dict[str, int | float] | None = None,
    train_hparams: dict[str, int | float] | None = None,
    training_data_dir: str = "outputs/data",
    allow_overwrite: bool = False,
    save_checkpoint: bool = True,
    save_fig: bool = True,
    show_progress: bool = True,
    show_val_progress: bool = False,
    log_every_n_batches: int = 10,
    heartbeat_secs: float = 60.0,
    epoch_time_warning_secs: float = 500.0,
):
    VALID_FAMILIES = {"haar", "clifford", "quansistor", "random"}
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model_type: {model_type}. Must be one of {sorted(MODEL_REGISTRY)}")

    if training_mode not in {"global", "per_family"}:
        raise ValueError("training_mode must be 'global' or 'per_family'")

    if training_mode == "per_family":
        if family is None:
            raise ValueError("family must be provided when training_mode='per_family'")
        if family not in VALID_FAMILIES:
            raise ValueError(
                f"Invalid family: {family}. Must be one of {sorted(VALID_FAMILIES)}"
            )

    print(f"Starting training | model_type={model_type} | training_mode={training_mode} | family={family} | loss_type={loss_type}")
    cfg = TrainConfig(
        epochs=epochs,
        lr=lr,
        loss_type=loss_type,
        batch_size=batch_size,
        training_mode=training_mode,
        family=family,
        target=target,
        show_progress=show_progress,
        show_val_progress=show_val_progress,
        log_batch_loss_every=log_every_n_batches,
        heartbeat=heartbeat_secs,
        epoch_warning=epoch_time_warning_secs,
    )
    print("Training configuration done.")

    model_hparams = {} if model_hparams is None else dict(model_hparams)
    train_hparams = {} if train_hparams is None else dict(train_hparams)

    family_filter = family if training_mode == "per_family" else None
    family_projection = family if training_mode == "per_family" else None

    print("Collecting data paths...")
    data_paths = collect_files_path(training_data_dir, family=family_filter)
    if not data_paths:
        raise RuntimeError("No data paths found.")
    print(f"Found {len(data_paths)} data paths.")
    print("Data paths collected.")

    spec = MODEL_REGISTRY[model_type]
    print(f"Building loaders and model for model_type={model_type}...")

    loader_fn = spec["build_loaders"]
    if spec["returns_nodes_dim"]:
        train_loader, val_loader, test_loader, node_in_dim, global_in_dim, base_dataset = loader_fn(
            data_paths,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            train_split=cfg.train_split,
            val_split=cfg.val_split,
            global_feature_variant=cfg.global_feature_variant,
            node_feature_variant=cfg.node_feature_backend_variant,
            family_projection=family_projection,
        )
    else:
        train_loader, val_loader, test_loader, global_in_dim, base_dataset = loader_fn(
            data_paths,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            train_split=cfg.train_split,
            val_split=cfg.val_split,
            global_feature_variant=cfg.global_feature_variant,
            node_feature_variant=cfg.node_feature_backend_variant,
            family_projection=family_projection,
        )
        node_in_dim = global_in_dim

    model = spec["build_model"](node_in_dim, global_in_dim, model_hparams)
    print("Loaders and model built.")

    print("Starting training...")
    model, hist, dev = train_model(
        model,
        train_loader,
        val_loader,
        epochs=cfg.epochs,
        lr=cfg.lr,
        loss_type=cfg.loss_type,
        scheduler="plateau",
        show_progress=cfg.show_progress,
        show_val_progress=cfg.show_val_progress,
        log_every_n_batches=cfg.log_batch_loss_every,
        heartbeat_secs=cfg.heartbeat,
        epoch_time_warning_secs=cfg.epoch_warning,
        weight_decay=train_hparams.get("weight_decay", 0.0),
        grad_clip=train_hparams.get("grad_clip", None),
        early_stopping_patience=train_hparams.get("early_stopping_patience", 30),
        early_stopping_min_delta=train_hparams.get("early_stopping_min_delta", 0.0),
    )

    loss_fn = build_loss(cfg.loss_type, huber_delta=1.0)

    test_loss = evaluate_loss(
        model,
        test_loader,
        dev,
        loss_fn,
        use_amp=True,
        show_progress=show_progress,
    )
    print("Training complete.")

    run_name = f"{model_type}_{loss_type}_{family if training_mode == 'per_family' else 'global'}"

    plot_training_curves(
        hist,
        title=f"{model_type.upper()} SRE regression",
        save_fig=save_fig,
        fig_path=f"outputs/figures/training_curves/training_curves_{run_name}.png",
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_type": model_type,
        "model_config": {
            "node_in_dim": node_in_dim or None,
            "global_in_dim": global_in_dim,
            "hidden_dim": model_hparams.get("hidden_dim", 64),
            "gnn_hidden": model_hparams["gnn_hidden"],
            "gnn_heads": model_hparams["gnn_heads"],
            "global_hidden": model_hparams["global_hidden"],
            "reg_hidden": model_hparams["reg_hidden"],
            "num_layers": model_hparams["num_layers"],
            "dropout_rate": model_hparams["dropout_rate"],
        },
        "train_config": asdict(cfg),
        "train_hparams": train_hparams,
        "feature_config": {
            "global_feature_variant": cfg.global_feature_variant,
            "node_feature_backend_variant": cfg.node_feature_backend_variant,
            "all_gate_keys": getattr(base_dataset, "all_gate_keys", None),
            "family_projection": family_projection,
        },
        "final_metrics": {
            "test_loss": float(test_loss),
        },
        "history": hist,
    }

    if save_checkpoint:
        model_save_path = _resolve_model_save_path(
            f"models/{run_name}.pt",
            allow_overwrite=allow_overwrite,
        )
        torch.save(checkpoint, model_save_path)
        print(f"Saved model checkpoint to {model_save_path}")

    return model, float(test_loss), hist, checkpoint



def objective_GNN(trial: optuna.Trial, prepared_data, epochs=10):
    batch_size=trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    train_loader, val_loader, test_loader = make_loaders(
        prepared_data,
        batch_size=batch_size,
    )
    train_hparams = {
        "weight_decay": trial.suggest_float(
            "weight_decay", 1e-5, 1e-2, log=True,
        ),
        "grad_clip": trial.suggest_float(
            "grad_clip", 1e-2, 1e2, log=True,
        ),
        "early_stopping_patience": 20,
        "early_stopping_min_delta": 1e-5,
    }

    model_hparams = {
        "gnn_hidden": trial.suggest_categorical("gnn_hidden", [16, 32, 64, 128]),
        "gnn_heads": trial.suggest_categorical("gnn_heads", [2, 4, 8]),
        "global_hidden": trial.suggest_categorical("global_hidden", [16, 32, 64, 128]),
        "reg_hidden": trial.suggest_categorical("reg_hidden", [16, 32, 64, 128]),
        "num_layers": trial.suggest_int("num_layers", 2, 6),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
    }
    model = GNN(
        node_in_dim=prepared_data.node_in_dim,
        global_in_dim=prepared_data.global_in_dim,
        gnn_hidden=model_hparams["gnn_hidden"],
        gnn_heads=model_hparams["gnn_heads"],
        global_hidden=model_hparams["global_hidden"],
        reg_hidden=model_hparams["reg_hidden"],
        num_layers=model_hparams["num_layers"],
        dropout_rate=model_hparams["dropout_rate"],
    )

    model, hist, dev = train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        loss_type=trial.suggest_categorical("loss_type", ["mse", "huber"]),
        scheduler="plateau",
        show_progress=False,
        show_val_progress=False,
        weight_decay=train_hparams["weight_decay"],
        grad_clip=train_hparams["grad_clip"],
        early_stopping_patience=train_hparams["early_stopping_patience"],
        early_stopping_min_delta=train_hparams["early_stopping_min_delta"],
    )

    return min(hist.val_loss)

def objective_NN(trial: optuna.Trial, prepared_data, epochs=10):
    batch_size=trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    train_loader, val_loader, test_loader = make_loaders(
        prepared_data,
        batch_size=batch_size,
    )

    train_hparams = {
        "weight_decay": trial.suggest_float(
            "weight_decay", 1e-5, 1e-2, log=True,
        ),
        "grad_clip": trial.suggest_float(
            "grad_clip", 1e-2, 1e2, log=True,
        ),
        "early_stopping_patience": 20,
        "early_stopping_min_delta": 1e-5,
    }

    model_hparams = {
        "nn_hidden": trial.suggest_categorical("nn_hidden", [32, 64, 128, 256]),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
    }
    model = NN(
        in_dim=prepared_data.global_in_dim,
        hidden_dim=model_hparams["nn_hidden"],
        dropout_rate=model_hparams["dropout_rate"],
    )

    model, hist, dev = train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        loss_type=trial.suggest_categorical("loss_type", ["mse", "huber"]),
        scheduler="plateau",
        show_progress=False,
        show_val_progress=False,
        weight_decay=train_hparams["weight_decay"],
        grad_clip=train_hparams["grad_clip"],
        early_stopping_patience=train_hparams["early_stopping_patience"],
        early_stopping_min_delta=train_hparams["early_stopping_min_delta"],
    )

    return min(hist.val_loss)

MODEL_REGISTRY = {
    "gnn": {
        "build_loaders": build_loaders,
        "build_model": lambda node_in_dim, global_in_dim, hparams: GNN(
            node_in_dim=node_in_dim,
            global_in_dim=global_in_dim,
            gnn_hidden=hparams.get("gnn_hidden", 32),
            gnn_heads=hparams.get("gnn_heads", 8),
            global_hidden=hparams.get("global_hidden", 16),
            reg_hidden=hparams.get("reg_hidden", 16),
            num_layers=hparams.get("num_layers", 5),
            dropout_rate=hparams.get("dropout_rate", 0.1),
        ),
        "returns_nodes_dim": True,
        "objective_fn": objective_GNN,
    },
    "nn": {
        "build_loaders": build_loaders_NN,
        "build_model": lambda node_in_dim, global_in_dim, hparams: NN(
            in_dim=global_in_dim,
            hidden_dim=hparams.get("hidden_dim", 64),
            dropout_rate=hparams.get("dropout_rate", 0.1),
        ),
        "returns_nodes_dim": False,
        "objective_fn": objective_NN,
    },
    "regressor": {
        "build_loaders": build_loaders_NN,
        "build_model": lambda node_in_dim, global_in_dim, hparams: Regressor(
            in_dim=global_in_dim,
            hidden_dim=hparams.get("hidden_dim", 64),
            dropout_rate=hparams.get("dropout_rate", 0.1),
        ),
        "returns_nodes_dim": False,
    },
}

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
