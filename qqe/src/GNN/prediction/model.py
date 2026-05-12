
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from qqe.src.GNN.physics_aware_NN import GNN, Regressor, NN

import logging

logger = logging.getLogger(__name__)


def _extract_state_dict(payload):
    if isinstance(payload, nn.Module):
        return payload.state_dict()
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    if isinstance(payload, dict) and all(torch.is_tensor(v) for v in payload.values()):
        return payload
    raise RuntimeError("Unsupported model file format.")


def _get_model_type(family: str | None, model_type: str):
    if family is not None and model_type == "per_family":
        MODEL_STATE_PATH = f"models/gnn_model_{family}.pt"
        logger.info(f"Using per-family model for family '{family}' at {MODEL_STATE_PATH}")
    else:
        MODEL_STATE_PATH = "models/gnn_model_global.pt"
        logger.info(f"Using global model at {MODEL_STATE_PATH}")

def checkpoint_path(model_kind: str, training_scope: str, family: str | None = None, loss_type: str = "mse") -> Path:
    if model_kind not in {"gnn", "nn"}:
        raise ValueError("model_kind must be 'gnn' or 'nn'")
    if training_scope not in {"global", "family"}:
        raise ValueError("training_scope must be 'global' or 'family'")

    if training_scope == "family":
        if family is None:
            raise ValueError("family must be provided when training_scope='family'")
        return Path(f"models/{model_kind}_model_{loss_type}_{family}.pt")

    return Path(f"models/{model_kind}_model_{loss_type}_global.pt")


def load_checkpoint(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(payload, dict):
        raise RuntimeError("Checkpoint must be a dict.")

    if "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        model_config = payload.get("model_config", {}) or {}
        feature_config = payload.get("feature_config", {}) or {}
    else:
        # older raw state_dict format
        state_dict = payload
        model_config = {}
        feature_config = {}

    return state_dict, model_config, feature_config


def build_model(model_kind: str, model_config: dict[str, Any]) -> torch.nn.Module:
    if model_kind == "gnn":
        return GNN(
            node_in_dim=int(model_config["node_in_dim"]),
            global_in_dim=int(model_config["global_in_dim"]),
            gnn_hidden=int(model_config.get("gnn_hidden", 32)),
            gnn_heads=int(model_config.get("gnn_heads", 8)),
            global_hidden=int(model_config.get("global_hidden", 16)),
            reg_hidden=int(model_config.get("reg_hidden", 16)),
            num_layers=int(model_config.get("num_layers", 5)),
            dropout_rate=float(model_config.get("dropout_rate", 0.1)),
        )

    if model_kind == "regressor":
        return Regressor(
            in_dim=int(model_config["global_in_dim"]),
            hidden_dim=int(model_config.get("hidden_dim", 128)),
        )

    if model_kind == "nn":
        return NN(
            in_dim=int(model_config["global_in_dim"]),
            hidden_dim=int(model_config.get("hidden_dim", 128)),
        )

    raise ValueError(f"Unsupported model_kind: {model_kind}")

