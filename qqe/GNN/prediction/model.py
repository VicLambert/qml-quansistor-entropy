
from __future__ import annotations

import torch
import torch.nn as nn


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
