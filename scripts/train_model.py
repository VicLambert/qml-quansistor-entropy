from __future__ import annotations

import logging

from dataclasses import asdict

import torch
import typer

from qqe.experiments.plotting import plot_training_curves
from qqe.GNN.training.runners import (
    _resolve_model_save_path,
    run_training,
    run_training_NN,
)
from qqe.GNN.training.train_config import TrainConfig
from qqe.utils import configure_logger

logger = logging.getLogger(__name__)


def main(
    epochs: int = 40,
    lr: float = 0.001,
    loss_type: str = "mse",  # "mse" | "huber" | "l1"
    training_mode: str = "per_family",  # "global" | "per_family"
    model_type: str = "nn",  # "gnn" | "nn"
    nn_type: str = "MLP",  # "MLP" | "regressor"
    allow_overwrite: bool = typer.Option(
        False,
        help="Allow overwriting an existing model checkpoint with the same name",
    ),
    family: str | None = "clifford",
    target: str = "sre",  # "sre" | "ee"
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
        model, hist, test_loss, global_in_dim, base_dataset = run_training_NN(train_config, model_type=nn_type, model_params={"hidden_dim": 128, "dropout_rate": 0.1})
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
        fig_path=f"outputs/figures/training_curves/new/training_curves_{model_type}_{loss_type}_{family if training_mode == 'per_family' else 'global'}.png",
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

    model_save_path = _resolve_model_save_path(
        f"models/{model_type}_model_{loss_type}_{family if training_mode == 'per_family' else 'global'}.pt",
        allow_overwrite=allow_overwrite,
    )
    torch.save(checkpoint, model_save_path)
    logger.info("Saved model checkpoint to %s", model_save_path)


if __name__ == "__main__":
    configure_logger(logging.INFO, logging.INFO)
    logger.info("Starting GNN training...")
    typer.run(main)
