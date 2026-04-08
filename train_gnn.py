from __future__ import annotations

import logging

from dataclasses import asdict

import torch
import typer

from qqe.experiments.plotting import plot_training_curves
from qqe.GNN.physics_aware_NN import GNN, Regressor
from qqe.GNN.training.datasets import build_loaders, build_loaders_NN
from qqe.GNN.training.train import build_loss, train_model
from qqe.GNN.training.train_config import TrainConfig
from qqe.GNN.training.utils import collect_files_path, evaluate_loss
from qqe.utils import configure_logger

logger = logging.getLogger(__name__)


def run_training(
    cfg: TrainConfig,
    model_hparams: dict | None = None,
    train_hparams: dict | None = None,
):
    family_filter = cfg.family if cfg.training_mode == "per_family" else None
    family_projection = cfg.family if cfg.training_mode == "per_family" else None

    VALID_FAMILIES = {"haar", "clifford", "quansistor", "random"}
    if cfg.training_mode == "per_family" and cfg.family not in VALID_FAMILIES:
        raise ValueError(
            f"Invalid family: {cfg.family}. Must be one of {sorted(VALID_FAMILIES)}"
        )
    if cfg.training_mode not in {"global", "per_family"}:
        raise ValueError("training_mode must be 'global' or 'per_family'")
    if cfg.training_mode == "per_family" and cfg.family is None:
        raise ValueError("family must be provided when training_mode='per_family'")

    data_paths = collect_files_path("outputs/data", family=family_filter)
    if not data_paths:
        raise RuntimeError("No data paths found.")

    train_loader, val_loader, test_loader, node_in_dim, global_in_dim, base_dataset = (
        build_loaders(
            data_paths,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            train_split=cfg.train_split,
            val_split=cfg.val_split,
            global_feature_variant=cfg.global_feature_variant,
            node_feature_variant=cfg.node_feature_backend_variant,
            family_projection=family_projection,
        )
    )

    default_model_hparams = {
        "gnn_hidden": 32,
        "gnn_heads": 8,
        "global_hidden": 16,
        "reg_hidden": 16,
        "num_layers": 5,
        "dropout_rate": 0.1,
    }
    if model_hparams:
        default_model_hparams.update(model_hparams)

    model = GNN(
        node_in_dim=node_in_dim,
        global_in_dim=global_in_dim,
        gnn_hidden=default_model_hparams["gnn_hidden"],
        gnn_heads=default_model_hparams["gnn_heads"],
        global_hidden=default_model_hparams["global_hidden"],
        reg_hidden=default_model_hparams["reg_hidden"],
        num_layers=default_model_hparams["num_layers"],
        dropout_rate=default_model_hparams["dropout_rate"],
    )

    default_train_hparams = {
        "weight_decay": 0.0,
        "grad_clip": 5.0,
        "early_stopping_patience": 15,
        "early_stopping_min_delta": 0.0,
    }
    if train_hparams:
        default_train_hparams.update(train_hparams)

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
        weight_decay=default_train_hparams["weight_decay"],
        grad_clip=default_train_hparams["grad_clip"],
        early_stopping_patience=default_train_hparams["early_stopping_patience"],
        early_stopping_min_delta=default_train_hparams["early_stopping_min_delta"],
    )

    loss_fn = build_loss(cfg.loss_type, huber_delta=1.0)
    test_loss = evaluate_loss(
        model, test_loader, dev, loss_fn, use_amp=True, show_progress=True
    )

    return (
        model,
        hist,
        test_loss,
        node_in_dim,
        global_in_dim,
        base_dataset,
        default_model_hparams,
    )


def run_training_NN(
    cfg: TrainConfig,
):
    family_filter = cfg.family if cfg.training_mode == "per_family" else None
    family_projection = cfg.family if cfg.training_mode == "per_family" else None

    VALID_FAMILIES = {"haar", "clifford", "quansistor", "random"}
    if cfg.training_mode == "per_family" and cfg.family not in VALID_FAMILIES:
        raise ValueError(
            f"Invalid family: {cfg.family}. Must be one of {sorted(VALID_FAMILIES)}"
        )
    if cfg.training_mode not in {"global", "per_family"}:
        raise ValueError("training_mode must be 'global' or 'per_family'")
    if cfg.training_mode == "per_family" and cfg.family is None:
        raise ValueError("family must be provided when training_mode='per_family'")

    data_paths = collect_files_path("outputs/data", family=family_filter)
    if not data_paths:
        raise RuntimeError("No data paths found.")

    train_loader, val_loader, test_loader, global_in_dim, base_dataset = build_loaders_NN(
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
    test_loss = evaluate_loss(
        model, test_loader, dev, loss_fn, use_amp=True, show_progress=True
    )

    return model, hist, test_loss, global_in_dim, base_dataset


def main(
    epochs: int = 30,
    lr: float = 0.001,
    loss_type: str = "huber",  # "mse" | "huber" | "l1"
    training_mode: str = "global",  # "global" | "per_family"
    model_type: str = "gnn",  # "gnn" | "nn"
    family: str | None = None,
    target: str = "sre",  # "sre" | "ee"
    show_progress: bool = typer.Option(
        default=True, help="Show progress bars during training"
    ),
    show_val_progress: bool = typer.Option(
        default=False, help="Show progress bar during validation"
    ),
    log_every_n_batches: int = typer.Option(
        5, help="Log training stats every N batches (0=disable)"
    ),
    heartbeat_secs: float = typer.Option(
        60.0, help="Heartbeat log interval in seconds (0=disable)"
    ),
    epoch_time_warning_secs: float = typer.Option(
        300.0, help="Warn if epoch exceeds N seconds (0=disable)"
    ),
):
    train_config = TrainConfig(
        epochs=epochs,
        lr=lr,
        loss_type=loss_type,
        training_mode=training_mode,
        family=family,
        target=target,
        show_progress=show_progress,
        show_val_progress=show_val_progress,
        log_batch_loss_every=log_every_n_batches,
        heartbeat=heartbeat_secs,
        epoch_warning=epoch_time_warning_secs,
    )
    if model_type == "gnn":
        (
            model,
            hist,
            test_loss,
            node_in_dim,
            global_in_dim,
            base_dataset,
            model_hparams,
        ) = run_training(train_config)
    else:
        model, hist, test_loss, global_in_dim, base_dataset = run_training_NN(train_config)
        node_in_dim = None
        model_hparams = {
            "gnn_hidden": None,
            "gnn_heads": None,
            "global_hidden": None,
            "reg_hidden": None,
            "num_layers": None,
            "dropout_rate": None,
        }

    plot_training_curves(
        hist,
        title=f"{model_type.upper()} SRE regression",
        save_fig=True,
        fig_path=f"outputs/figures/training_curves/training_curves_{model_type}_{loss_type}_{family if training_mode == 'per_family' else 'global'}.png",
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "node_in_dim": node_in_dim or None,
            "global_in_dim": global_in_dim,
            "gnn_hidden": model_hparams["gnn_hidden"],
            "gnn_heads": model_hparams["gnn_heads"],
            "global_hidden": model_hparams["global_hidden"],
            "reg_hidden": model_hparams["reg_hidden"],
            "num_layers": model_hparams["num_layers"],
            "dropout_rate": model_hparams["dropout_rate"],
        },
        "train_config": asdict(train_config),
        "feature_config": {
            "global_feature_variant": train_config.global_feature_variant,
            "node_feature_backend_variant": train_config.node_feature_backend_variant,
            "all_gate_keys": getattr(base_dataset, "all_gate_keys", None),
            "family_projection": (
                train_config.family if train_config.training_mode == "per_family" else None
            ),
        },
        "final_metrics": {
            "test_loss": float(test_loss),
        },
    }

    model_save_path = f"models/{model_type}_model_{loss_type}_{family if training_mode == 'per_family' else 'global'}.pt"
    torch.save(checkpoint, model_save_path)


if __name__ == "__main__":
    configure_logger(logging.INFO, logging.INFO)
    logger.info("Starting GNN training...")
    typer.run(main)
