from __future__ import annotations

from typing import Any

import hashlib
import json
import logging
import sys
import time
import csv

from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer

from dask.graph_manipulation import checkpoint
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.data import Data, Dataset as PyGDataset
from torch_geometric.loader import DataLoader

from qqe.experiments.plotting import plot_training_curves
from qqe.GNN.physics_aware_NN import GNN, QuantumCircuitGraphDataset
from qqe.GNN.training.train import build_loss, train_model
from qqe.GNN.training.train_config import TrainConfig
from qqe.GNN.training.utils import collect_files_path, evaluate_loss
from qqe.GNN.training.datasets import build_loaders
from qqe.utils import configure_logger

logger = logging.getLogger(__name__)


# -----------------------------
# basic path helpers
# -----------------------------

def collect_pred_paths(dataset_dir: str, family: str | None = None) -> list[str]:
    d = Path(dataset_dir)
    pred_root = d / "predictions"

    if family is not None:
        paths = sorted((pred_root / family).glob("*.pt"))
    else:
        paths = []
        if pred_root.exists():
            for family_dir in sorted(pred_root.iterdir()):
                if family_dir.is_dir():
                    paths.extend(sorted(family_dir.glob("*.pt")))

    if not paths:
        paths = sorted(d.glob("*.pt"))

    return [str(p.resolve()) for p in paths]


def cache_root_for_paths(paths: list[str], suffix: str = "") -> str:
    canonical = "|".join(sorted(str(Path(p).resolve()) for p in paths))
    digest = hashlib.md5(canonical.encode("utf-8")).hexdigest()[:10]
    tag = f"_{suffix}" if suffix else ""
    cache_dir = Path("qqe") / "cache" / f"pyg_cache_{digest}{tag}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir.resolve())


# -----------------------------
# dataset wrapper
# -----------------------------

class PaddedGraphDatasetWrapper:
    """Pad or truncate node/global feature widths to the trained model dimensions."""

    def __init__(
        self,
        dataset,
        target_node_dim: int | None = None,
        target_global_dim: int | None = None,
    ):
        self.dataset = dataset
        self.target_node_dim = target_node_dim
        self.target_global_dim = target_global_dim

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        data = self.dataset[idx].clone()

        if self.target_node_dim is not None:
            cur_node_dim = int(data.x.shape[1])
            if cur_node_dim < self.target_node_dim:
                data.x = F.pad(data.x, (0, self.target_node_dim - cur_node_dim), value=0.0)
            elif cur_node_dim > self.target_node_dim:
                data.x = data.x[:, : self.target_node_dim]

        if hasattr(data, "global_features") and data.global_features is not None:
            g = data.global_features
            if g.dim() == 0:
                g = g.view(1)
            elif g.dim() > 1:
                g = g.view(-1)

            if self.target_global_dim is not None:
                cur_global_dim = int(g.shape[0])
                if cur_global_dim < self.target_global_dim:
                    g = F.pad(g, (0, self.target_global_dim - cur_global_dim), value=0.0)
                elif cur_global_dim > self.target_global_dim:
                    g = g[: self.target_global_dim]

            data.global_features = g

        return data


# -----------------------------
# checkpoint/model helpers
# -----------------------------

def extract_state_dict(payload):
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    if isinstance(payload, dict) and all(torch.is_tensor(v) for v in payload.values()):
        return payload
    raise RuntimeError("Unsupported checkpoint format.")


def load_checkpoint(model_path: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = torch.load(model_path, map_location="cpu", weights_only=False)

    if not isinstance(payload, dict):
        raise RuntimeError("Expected checkpoint payload to be a dict.")

    state_dict = extract_state_dict(payload)
    model_config = payload.get("model_config", {}) or {}
    feature_config = payload.get("feature_config", {}) or {}

    return state_dict, model_config, feature_config


def build_model(model_config: dict[str, Any]) -> GNN:
    required_keys = ["node_in_dim", "global_in_dim"]
    for key in required_keys:
        if key not in model_config:
            raise RuntimeError(f"Missing required model_config key: {key}")

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


# -----------------------------
# prediction dataset/loader
# -----------------------------

def build_prediction_loader(
    pt_paths: list[str],
    *,
    batch_size: int,
    global_feature_variant: str,
    node_feature_backend_variant: str | None,
    fixed_all_gate_keys: list[str] | None,
    target_node_dim: int | None,
    target_global_dim: int | None,
) -> tuple[QuantumCircuitGraphDataset, DataLoader]:
    suffix = f"{global_feature_variant}_backend_{node_feature_backend_variant or 'none'}"
    root = cache_root_for_paths(pt_paths, suffix=suffix)

    dataset = QuantumCircuitGraphDataset(
        root=root,
        pt_paths=pt_paths,
        global_feature_variant=global_feature_variant,
        node_feature_backend_variant=node_feature_backend_variant,
        fixed_all_gate_keys=fixed_all_gate_keys,
    )

    wrapped = PaddedGraphDatasetWrapper(
        dataset,
        target_node_dim=target_node_dim,
        target_global_dim=target_global_dim,
    )

    loader = DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return dataset, loader


# -----------------------------
# inference
# -----------------------------

@torch.no_grad()
def run_predictions(
    model: GNN,
    loader: DataLoader,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    records: list[dict[str, Any]] = []

    for batch in loader:
        samples = batch.to_data_list()
        batch = batch.to(device)
        pred = model(batch).view(-1).cpu().tolist()

        for sample, pred_value in zip(samples, pred):
            meta = getattr(sample, "meta", {}) or {}
            records.append(
                {
                    "cid": meta.get("cid"),
                    "prediction": float(pred_value),
                    "family": meta.get("family"),
                    "n_qubits": meta.get("n_qubits"),
                    "n_layers": meta.get("n_layers"),
                }
            )

    return records


def save_predictions_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["cid", "prediction", "family", "n_qubits", "n_layers"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


# -----------------------------
# CLI
# -----------------------------

def main(
    family: str | None = typer.Option(None, help="Family to predict on for per-family data selection."),
    model_type: str = typer.Option("global", help="'global' or 'per_family'"),
    dataset_dir: str = typer.Option("outputs/data", help="Root directory containing prediction .pt files."),
    batch_size: int = typer.Option(32, help="Prediction batch size."),
    global_feature_variant: str = typer.Option("binned", help="Global feature variant."),
    node_feature_backend_variant: str | None = typer.Option(None, help="Optional node feature backend variant."),
    output_csv: str = typer.Option("outputs/predictions/predictions.csv", help="Where to save predictions."),
):
    if model_type not in {"global", "per_family"}:
        raise ValueError("model_type must be 'global' or 'per_family'")

    if model_type == "per_family" and family is None:
        raise ValueError("family must be provided when model_type='per_family'")

    model_path = (
        f"models/gnn_model_{family}.pt"
        if model_type == "per_family"
        else "models/gnn_model_global.pt"
    )

    state_dict, model_config, feature_config = load_checkpoint(model_path)

    pt_paths = collect_pred_paths(dataset_dir, family=family if model_type == "per_family" else None)
    if not pt_paths:
        raise RuntimeError("No prediction .pt files found.")

    logger.info("Found %d prediction files.", len(pt_paths))

    dataset, loader = build_prediction_loader(
        pt_paths,
        batch_size=batch_size,
        global_feature_variant=global_feature_variant,
        node_feature_backend_variant=node_feature_backend_variant,
        fixed_all_gate_keys=feature_config.get("all_gate_keys"),
        target_node_dim=model_config.get("node_in_dim"),
        target_global_dim=model_config.get("global_in_dim"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_config).to(device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    logger.info("Loaded model from %s", model_path)
    if missing_keys:
        logger.info("Missing keys: %s", missing_keys)
    if unexpected_keys:
        logger.info("Unexpected keys: %s", unexpected_keys)

    records = run_predictions(model, loader, device)
    save_predictions_csv(records, Path(output_csv))

    logger.info("Saved %d predictions to %s", len(records), output_csv)

    preview = min(10, len(records))
    for i, row in enumerate(records[:preview], start=1):
        logger.info(
            "[%d] cid=%s | pred=%.6f | family=%s | n_qubits=%s | n_layers=%s",
            i,
            str(row["cid"]),
            float(row["prediction"]),
            str(row["family"]),
            str(row["n_qubits"]),
            str(row["n_layers"]),
        )


if __name__ == "__main__":
    configure_logger(logging.INFO, logging.INFO)
    logger.info("Starting prediction...")
    typer.run(main)
