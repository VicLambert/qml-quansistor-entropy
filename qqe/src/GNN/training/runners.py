
from __future__ import annotations

import logging

from dataclasses import asdict
from pathlib import Path

import torch

from qqe.src.experiments.plotting import plot_training_curves
from qqe.src.GNN.parameter_search.helpers import objective_GNN, objective_NN
from qqe.src.GNN.physics_aware_NN import GNN, NN, Regressor
from qqe.src.GNN.training.datasets import build_loaders, build_loaders_NN
from qqe.src.GNN.training.train import build_loss, train_model
from qqe.src.GNN.training.train_config import TrainConfig
from qqe.src.GNN.training.utils import collect_files_path, evaluate_loss

logger = logging.getLogger(__name__)

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
            logger.info(
                "Model checkpoint already exists at %s. Saving to %s instead.",
                path,
                candidate,
            )
            return str(candidate)
        counter += 1


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
        msg = f"Unsupported model_type: {model_type}. Must be one of {sorted(MODEL_REGISTRY)}"
        raise ValueError(msg)

    if training_mode not in {"global", "per_family"}:
        msg = "training_mode must be 'global' or 'per_family'"
        raise ValueError(msg)

    if training_mode == "per_family":
        if family is None:
            raise ValueError("family must be provided when training_mode='per_family'")
        if family not in VALID_FAMILIES:
            msg = f"Invalid family: {family}. Must be one of {sorted(VALID_FAMILIES)}"
            raise ValueError(
                msg,
            )

    logger.info(f"Starting training | model_type={model_type} | training_mode={training_mode} | family={family} | loss_type={loss_type}")
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
    logger.info("Training configuration done.")

    model_hparams = {} if model_hparams is None else dict(model_hparams)
    train_hparams = {} if train_hparams is None else dict(train_hparams)

    family_filter = family if training_mode == "per_family" else None
    family_projection = family if training_mode == "per_family" else None

    logger.info("Collecting data paths...")
    data_paths = collect_files_path(training_data_dir, family=family_filter)
    if not data_paths:
        raise RuntimeError("No data paths found.")
    logger.info(f"Found {len(data_paths)} data paths.")
    logger.info("Data paths collected.")

    spec = MODEL_REGISTRY[model_type]
    logger.info(f"Building loaders and model for model_type={model_type}...")

    loader_fn: object = spec["build_loaders"]
    returns_nodes_dim: bool = spec.get("returns_nodes_dim", False)
    if returns_nodes_dim:
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
    logger.info("Loaders and model built.")

    logger.info("Starting training...")
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
    logger.info("Training complete.")

    run_name = f"{model_type}_{loss_type}_{family if training_mode == 'per_family' else 'global'}"

    plot_training_curves(
        hist,
        title=f"{model_type.upper()} SRE regression",
        save_fig=save_fig,
        fig_path=f"../outputs/figures/training_curves/training_curves_{run_name}.png",
    )

    # Build model config safely, providing defaults for missing hparams depending on model
    model_config = {
        "node_in_dim": node_in_dim or None,
        "global_in_dim": global_in_dim,
        "hidden_dim": model_hparams.get("hidden_dim", 64),
        "dropout_rate": model_hparams.get("dropout_rate", 0.1),
    }
    # GNN specific params (only meaningful for gnn models)
    model_config.update({
        "gnn_hidden": model_hparams.get("gnn_hidden", None),
        "gnn_heads": model_hparams.get("gnn_heads", None),
        "global_hidden": model_hparams.get("global_hidden", None),
        "reg_hidden": model_hparams.get("reg_hidden", None),
        "num_layers": model_hparams.get("num_layers", None),
    })

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_type": model_type,
        "model_config": model_config,
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
            f"../outputs/models/{run_name}.pt",
            allow_overwrite=allow_overwrite,
        )
        torch.save(checkpoint, model_save_path)
        logger.info(f"Saved model checkpoint to {model_save_path}")

    return model, float(test_loss), hist, checkpoint
