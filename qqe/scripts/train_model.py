from __future__ import annotations

import logging
import os
import typer

from GNN.training.runners import train
from utils import configure_logger

logger = logging.getLogger(__name__)


default_model_hparams = {
    "gnn_hidden": 64,
    "gnn_heads": 2,
    "global_hidden": 32,
    "reg_hidden": 128,
    "num_layers": 3,
    "dropout_rate": 0.10,
}

default_train_hparams = {
    "weight_decay": 3e-3,
    "grad_clip": 1.0,
    "early_stopping_patience": 15,
    "early_stopping_min_delta": 0.0,
    "num_workers": 0,
}

PARAMS = {
    "random" : {
        "lr" : 0.0009860340204413903,
        "batch_size" : 128,
        "model_hparams": {
            "gnn_hidden" : 64,
            "gnn_heads": 2,
            "global_hidden": 32,
            "reg_hidden": 128,
            "num_layers": 6,
            "dropout_rate": 0.06640302989664926,
        },
        "train_params": {
            "weight_decay": 3.0500428108369453e-05,
            "grad_clip": 4.379744711312854,
            "early_stopping_patience": 15,
            "early_stopping_min_delta": 0.0,
            "num_workers": 0,
        },
    },
    "clifford" : {
        "lr" : 2.5223176427539664e-05,
        "batch_size" : 32,
        "model_hparams": {
            "gnn_hidden" : 128,
            "gnn_heads": 2,
            "global_hidden": 16,
            "reg_hidden": 32,
            "num_layers": 6,
            "dropout_rate": 0.014842423657881243,
        },
        "train_params": {
            "weight_decay": 1.560010639264171e-05,
            "grad_clip": 7.2440146231033875,
            "early_stopping_patience": 15,
            "early_stopping_min_delta": 0.0,
            "num_workers": 0,
        },
    },
    "haar" : {
        "lr" : 0.0005456850011484297,
        "batch_size" : 16,
        "model_hparams": {
            "gnn_hidden" : 64,
            "gnn_heads": 2,
            "global_hidden": 128,
            "reg_hidden": 16,
            "num_layers": 2,
            "dropout_rate": 0.004127592869557634,
        },
        "train_params": {
            "weight_decay": 1.0439900428164368e-05,
            "grad_clip":  0.011242628935673588,
            "early_stopping_patience": 15,
            "early_stopping_min_delta": 0.0,
            "num_workers": 0,
        },
    },
    "quansistor" : {
        "lr" : 1.0600038808287122e-05,
        "batch_size" : 32,
        "model_hparams": {
            "gnn_hidden" : 128,
            "gnn_heads": 8,
            "global_hidden": 16,
            "reg_hidden": 64,
            "num_layers": 3,
            "dropout_rate": 0.4050959484660077,
        },
        "train_params": {
            "weight_decay": 0.0006626449388616718,
            "grad_clip":  4.779019014790367,
            "early_stopping_patience": 15,
            "early_stopping_min_delta": 0.0,
            "num_workers": 0,
        },
    },
}


def main(
    epochs: int = 40,
    lr: float = 1.5131621801102364e-05,
    loss_type: str = "huber",  # "mse" | "huber" | "l1"
    batch_size: int = 16,
    training_mode: str = "per_family",  # "global" | "per_family"
    model_type: str = "gnn",  # "gnn" | "nn"
    allow_overwrite: bool = typer.Option(
        False,
        help="Allow overwriting an existing model checkpoint with the same name",
    ),
    family: str | None = "random",
    target: str = "sre",  # "sre" | "ee"
    target_variant: str = "sre_density",  # "sre" | "sre_density" ...
    model_hparams = None,
    train_hparams = None,
    training_data_dir = "outputs/data/datasets_SRE",
    split: str = "target",
    model_save_path: str | None = None,
    save_fig_path: str = typer.Option(
        "outputs/figures/training_curves/training_curves",
    ),
    show_progress: bool = typer.Option(
        default=True,
        help="Show progress bars during training",
    ),
    show_val_progress: bool = typer.Option(
        default=False,
        help="Show progress bar during validation",
    ),
    log_every_n_batches: int = typer.Option(
        5,
        help="Log training stats every N batches (0=disable)",
    ),
    heartbeat_secs: float = typer.Option(
        60.0,
        help="Heartbeat log interval in seconds (0=disable)",
    ),
    epoch_time_warning_secs: float = typer.Option(
        300.0,
        help="Warn if epoch exceeds N seconds (0=disable)",
    ),
):
    lr = PARAMS.get(family, {}).get("lr", lr)
    batch_size = PARAMS.get(family, {}).get("batch_size", batch_size)
    model_hparams = PARAMS.get(family, {}).get("model_hparams", default_model_hparams) if model_hparams is None else default_model_hparams
    train_hparams = PARAMS.get(family, {}).get("train_params", default_train_hparams) if train_hparams is None else default_train_hparams

    model, loss, hist, chkpt = train(
        model_type=model_type,
        epochs = epochs,
        lr = lr,
        loss_type = loss_type,   # "mse" | "huber"
        batch_size = batch_size,
        training_mode = training_mode,  # "global" | "per_family"
        family = family,  # required if training_mode == "per_family"
        target = target,
        target_variant = target_variant,
        model_hparams = model_hparams,
        train_hparams = train_hparams,
        training_data_dir = training_data_dir,
        split = split,
        allow_overwrite = allow_overwrite,
        save_checkpoint = True,
        model_save_path=model_save_path,
        save_fig = True,
        save_fig_path = save_fig_path,
        show_progress = show_progress,
        show_val_progress = show_val_progress,
        log_every_n_batches = log_every_n_batches,
        heartbeat_secs = heartbeat_secs,
        epoch_time_warning_secs = epoch_time_warning_secs,
    )

    logger.info(f"Final test loss: {loss:.6f}")
    logger.info(f"Test R2 score: {chkpt['final_metrics'].get('test_r2_score', 0):.4f}")
    logger.info(f"Validation R2 score: {chkpt['final_metrics'].get('val_r2_score', 0):.4f}")
    logger.info(f"Training R2 score: {chkpt['final_metrics'].get('train_r2_score', 0):.4f}")

if __name__ == "__main__":
    configure_logger(logging.INFO, logging.INFO)
    logger.info("Starting GNN training...")
    typer.run(main)
