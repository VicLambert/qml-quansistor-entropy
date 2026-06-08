from __future__ import annotations

import logging
import os
import typer

from GNN.training.runners import train
from utils import configure_logger

logger = logging.getLogger(__name__)


default_model_hparams = {
    "gnn_hidden": 32,
    "gnn_heads": 4,
    "global_hidden": 128,
    "reg_hidden": 128,
    "num_layers": 3,
    "dropout_rate": 0.13173830279748305,
}

default_train_hparams = {
    "weight_decay": 0.0003324725858640221,
    "grad_clip": 1.0289214665544766,
    "early_stopping_patience": 10,
    "early_stopping_min_delta": 0.0,
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
        model_hparams = default_model_hparams,
        train_hparams = default_train_hparams,
        training_data_dir = training_data_dir,
        split = split,
        allow_overwrite = allow_overwrite,
        save_checkpoint = True,
        model_save_path=model_save_path,
        save_fig = True,
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
