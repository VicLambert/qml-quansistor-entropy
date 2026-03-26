from __future__ import annotations

import hashlib
import json
import logging
import sys
import time

from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import typer

from dask.graph_manipulation import checkpoint
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.data import Data, Dataset as PyGDataset
from torch_geometric.loader import DataLoader

from qqe.experiments.plotting import plot_training_curves
from qqe.GNN.physics_aware_NN import Regressor, GNN, QuantumCircuitGraphDataset
from qqe.GNN.training.train import build_loss, train_model
from qqe.GNN.training.train_config import TrainConfig
from qqe.GNN.training.utils import collect_files_path, evaluate_loss
from qqe.GNN.training.datasets import build_loaders
from qqe.utils import configure_logger

logger = logging.getLogger(__name__)

def run_training(cfg: TrainConfig):
    family_filter = cfg.family if cfg.training_mode == "per_family" else None
    family_projection = cfg.family if cfg.training_mode == "per_family" else None

    VALID_FAMILIES = {"haar", "clifford", "quansistor", "random"}
    if cfg.training_mode == "per_family" and cfg.family not in VALID_FAMILIES:
        raise ValueError(f"Invalid family: {cfg.family}. Must be one of {sorted(VALID_FAMILIES)}")
    if cfg.training_mode not in {"global", "per_family"}:
        raise ValueError("training_mode must be 'global' or 'per_family'")
    if cfg.training_mode == "per_family" and cfg.family is None:
        raise ValueError("family must be provided when training_mode='per_family'")

    data_paths = collect_files_path("outputs/data", family=family_filter)
    if not data_paths:
        raise RuntimeError("No data paths found.")

    train_loader, val_loader, test_loader, node_in_dim, global_in_dim, base_dataset = build_loaders(
        data_paths,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        train_split=cfg.train_split,
        val_split=cfg.val_split,
        global_feature_variant=cfg.global_feature_variant,
        node_feature_variant=cfg.node_feature_backend_variant,
        family_projection=family_projection,
    )

    model = GNN(
        node_in_dim=node_in_dim,
        global_in_dim=global_in_dim,
        gnn_hidden=32,
        gnn_heads=8,
        global_hidden=16,
        reg_hidden=16,
        num_layers=5,
        dropout_rate=0.1,
    )

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
    )

    loss_fn = build_loss(cfg.loss_type, huber_delta=1.0)
    test_loss = evaluate_loss(model, test_loader, dev, loss_fn, use_amp=True, show_progress=True)

    return model, hist, test_loss, node_in_dim, global_in_dim, base_dataset

def run_training_NN(
    cfg: TrainConfig,
):
    family_filter = cfg.family if cfg.training_mode == "per_family" else None
    family_projection = cfg.family if cfg.training_mode == "per_family" else None

    VALID_FAMILIES = {"haar", "clifford", "quansistor", "random"}
    if cfg.training_mode == "per_family" and cfg.family not in VALID_FAMILIES:
        raise ValueError(f"Invalid family: {cfg.family}. Must be one of {sorted(VALID_FAMILIES)}")
    if cfg.training_mode not in {"global", "per_family"}:
        raise ValueError("training_mode must be 'global' or 'per_family'")
    if cfg.training_mode == "per_family" and cfg.family is None:
        raise ValueError("family must be provided when training_mode='per_family'")

    data_paths = collect_files_path("outputs/data", family=family_filter)
    if not data_paths:
        raise RuntimeError("No data paths found.")

    train_loader, val_loader, test_loader, node_in_dim, global_in_dim, base_dataset = build_loaders(
        data_paths,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        train_split=cfg.train_split,
        val_split=cfg.val_split,
        global_feature_variant=cfg.global_feature_variant,
        node_feature_variant=cfg.node_feature_backend_variant,
        family_projection=family_projection,
    )
    model = Regressor(
        in_dim=global_in_dim,
        hidden_dim=128,
    )

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
    )

    loss_fn = build_loss(cfg.loss_type, huber_delta=1.0)
    test_loss = evaluate_loss(model, test_loader, dev, loss_fn, use_amp=True, show_progress=True)

    return model, hist, test_loss, node_in_dim, global_in_dim, base_dataset


def main(
    epochs: int = 30,
    lr: float = 0.001,
    loss_type: str = "mse",             # "mse" | "huber" | "l1"
    training_mode: str = "global",     # "global" | "per_family"
    family: str | None = None,
    show_progress: bool = typer.Option(True, help="Show progress bars during training"),
    show_val_progress: bool = typer.Option(False, help="Show progress bar during validation"),
    log_every_n_batches: int = typer.Option(5, help="Log training stats every N batches (0=disable)"),
    heartbeat_secs: float = typer.Option(60.0, help="Heartbeat log interval in seconds (0=disable)"),
    epoch_time_warning_secs: float = typer.Option(300.0, help="Warn if epoch exceeds N seconds (0=disable)"),
):
    train_config = TrainConfig(
        epochs=epochs,
        lr=lr,
        loss_type=loss_type,
        training_mode=training_mode,
        family=family,
        show_progress=show_progress,
        show_val_progress=show_val_progress,
        log_batch_loss_every=log_every_n_batches,
        heartbeat=heartbeat_secs,
        epoch_warning=epoch_time_warning_secs,
    )
    model, hist, test_loss, node_in_dim, global_in_dim, base_dataset = run_training(train_config)

    plot_training_curves(
        hist,
        title="GNN SRE regression",
        save_fig=True,
        fig_path=f"outputs/figures/training_curves/training_curves_{loss_type}_{family if training_mode == 'per_family' else 'global'}.png",
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "node_in_dim": node_in_dim,
            "global_in_dim": global_in_dim,
            "gnn_hidden": 32,
            "gnn_heads": 8,
            "global_hidden": 16,
            "reg_hidden": 16,
            "num_layers": 5,
            "dropout_rate": 0.1,
        },
        "train_config": asdict(train_config),
        "feature_config": {
            "global_feature_variant": train_config.global_feature_variant,
            "node_feature_backend_variant": train_config.node_feature_backend_variant,
            "all_gate_keys": getattr(base_dataset, "all_gate_keys", None),
            "family_projection": train_config.family if train_config.training_mode == "per_family" else None,
        },
        "final_metrics": {
            "test_loss": float(test_loss),
        },
    }

    model_save_path = f"models/gnn_model_{family if training_mode == 'per_family' else 'global'}.pt"
    torch.save(checkpoint, model_save_path)


if __name__ == "__main__":
    configure_logger(logging.INFO, logging.INFO)
    logger.info("Starting GNN training...")
    typer.run(main)
