
from __future__ import annotations

from tqdm import tqdm
import torch
import numpy as np
from typing import TYPE_CHECKING, Any

from GNN.prediction.utils import extract_target_value

if TYPE_CHECKING:
    from .pred_config import PredConfig

# =========================================================
# Prediction
# =========================================================
def _to_python_scalar(value, default=None):
    if value is None:
        return default

    if torch.is_tensor(value):
        value = value.detach().cpu().flatten()
        if value.numel() == 0:
            return default
        return value[0].item()

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return default
        return value[0]

    return value


def get_sample_field(sample, name: str, default=None):
    """
    Read metadata from either:
      - new sharded Data attributes: sample.n_qubits, sample.seed, ...
      - old format: sample.meta["n_qubits"], sample.meta["seed"], ...
    """
    if hasattr(sample, name):
        return _to_python_scalar(getattr(sample, name), default)

    meta = getattr(sample, "meta", {}) or {}
    if isinstance(meta, dict):
        return meta.get(name, default)

    return default


def denormalize_prediction(pred: float, sample, target_variant: str) -> float:
    """
    Convert model output back to raw SRE if needed.
    """
    if target_variant == "sre":
        return float(pred)

    if target_variant == "sre_density":
        n_qubits = get_sample_field(sample, "n_qubits")
        if n_qubits is None:
            raise ValueError(
                "Cannot convert SRE density prediction to raw SRE: "
                "sample is missing n_qubits."
            )
        return float(pred) * float(n_qubits)

    if target_variant == "log_sre":
        return float(torch.expm1(torch.tensor(float(pred))).item())

    if target_variant == "sqrt_sre":
        return float(pred) ** 2

    raise ValueError(f"Unsupported target_variant={target_variant}")


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader,
    *,
    model_kind: str,
    device: torch.device,
    target_variant: str = "sre",
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    total_batches = len(loader) if hasattr(loader, "__len__") else None

    if model_kind == "gnn":
        for batch in tqdm(
            loader,
            total=total_batches,
            desc="Predicting (gnn)",
            unit="batch",
            disable=not show_progress,
        ):
            samples = batch.to_data_list()
            batch = batch.to(device)
            preds = model(batch).view(-1).cpu().tolist()

            for sample, pred_model_output in zip(samples, preds):
                target_raw_sre = extract_target_value(sample)

                pred_model_output = float(pred_model_output)
                pred_raw_sre = denormalize_prediction(
                    pred_model_output,
                    sample,
                    target_variant,
                )

                if target_raw_sre is not None:
                    error_raw_sre = abs(pred_raw_sre - float(target_raw_sre))
                else:
                    error_raw_sre = None

                rows.append(
                    {
                        "cid": get_sample_field(sample, "cid"),
                        "family": get_sample_field(sample, "family"),
                        "regime": get_sample_field(sample, "regime"),
                        "seed": get_sample_field(sample, "seed"),
                        "n_qubits": get_sample_field(sample, "n_qubits"),
                        "n_layers": get_sample_field(sample, "n_layers"),

                        "target_variant": target_variant,
                        "prediction_model_output": pred_model_output,

                        "target_sre": target_raw_sre,
                        "predicted_sre": pred_raw_sre,
                        "error_sre": error_raw_sre,
                    },
                )
        return rows

    if model_kind == "nn" or model_kind == "regressor":
        for x, metas, targets in tqdm(
            loader,
            total=total_batches,
            desc="Predicting (nn)",
            unit="batch",
            disable=not show_progress,
        ):
            x = x.to(device)
            preds = model(x).view(-1).cpu().tolist()

            for meta, pred, target in zip(metas, preds, targets):
                pred = pred * meta.get("n_qubits")
                rows.append(
                    {
                        "cid": meta.get("cid"),
                        "family": meta.get("family"),
                        "seed": int(meta.get("seed")),
                        "n_qubits": int(meta.get("n_qubits")),
                        "n_layers": int(meta.get("n_layers")),
                        "target": target,
                        "prediction": float(pred),
                        "error": abs(float(pred - target)) if target is not None else None,
                    },
                )
        return rows

    raise ValueError(f"Unsupported model_kind: {model_kind}")
