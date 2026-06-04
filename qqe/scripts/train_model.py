from __future__ import annotations

import logging

import typer

from GNN.training.runners import train
from utils import configure_logger

logger = logging.getLogger(__name__)


def main(
    epochs: int = 40,
    lr: float = 0.001,
    loss_type: str = "mse",  # "mse" | "huber" | "l1"
    batch_size: int = 64,
    training_mode: str = "per_family",  # "global" | "per_family"
    model_type: str = "nn",  # "gnn" | "nn"
    allow_overwrite: bool = typer.Option(
        False,
        help="Allow overwriting an existing model checkpoint with the same name",
    ),
    family: str | None = "clifford",
    target: str = "sre",  # "sre" | "ee"
    target_variant: str = "sre_density",  # "sre" | "sre_density" ...
    model_hparams: dict[str, int | float] | None = None,
    train_hparams: dict[str, int | float] | None = None,
    training_data_dir = "../outputs/data",
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
        model_hparams = model_hparams,
        train_hparams = train_hparams,
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



if __name__ == "__main__":
    configure_logger(logging.INFO, logging.INFO)
    logger.info("Starting GNN training...")
    typer.run(main)
