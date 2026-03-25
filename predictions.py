from __future__ import annotations

import csv
import hashlib
import logging
import re
import time

from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import typer

from torch_geometric.loader import DataLoader

from qqe.GNN.physics_aware_NN import GNN, QuantumCircuitGraphDataset
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


def _safe_tag(value: str | None, fallback: str = "none") -> str:
    """Return a filesystem-friendly tag for file naming."""
    raw = (value or fallback).strip()
    cleaned = [ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in raw]
    tag = "".join(cleaned).strip("-")
    return tag or fallback


def resolve_output_csv_path(
    output_csv: str,
    *,
    model_type: str,
    family: str | None,
    model_path: str,
    global_feature_variant: str,
    node_feature_backend_variant: str | None,
    batch_size: int,
) -> Path:
    """Build a descriptive default filename while preserving explicit custom paths."""
    if "{family}" in output_csv:
        output_csv = output_csv.format(family=family or "all")

    out = Path(output_csv)

    # Only auto-expand the default output file into a run-specific file name.
    if out.name == "predictions.csv":
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_tag = _safe_tag(Path(model_path).stem.replace("gnn_model_", ""), fallback="model")
        family_tag = _safe_tag(family, fallback="all")
        global_tag = _safe_tag(global_feature_variant, fallback="binned")
        backend_tag = _safe_tag(node_feature_backend_variant, fallback="none")
        model_type_tag = _safe_tag(model_type, fallback="global")

        filename = (
            f"pred_{model_type_tag}"
            f"__model-{model_tag}"
            f"__family-{family_tag}"
            f"__gf-{global_tag}"
            f"__nb-{backend_tag}"
            f"__bs-{batch_size}"
            f"__{timestamp}.csv"
        )
        return out.parent / filename

    return out


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
                    "seed": meta.get("seed"),
                    "n_qubits": meta.get("n_qubits"),
                    "n_layers": meta.get("n_layers"),
                },
            )

    return records


def save_predictions_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["cid", "prediction", "family", "seed", "n_qubits", "n_layers"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def aggregate_predictions_by_size(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute one seed-averaged prediction for each (n_qubits, n_layers) pair."""
    # First level: collect values per (n_qubits, n_layers, seed)
    per_seed: dict[tuple[int, int, int], list[float]] = {}

    for row in records:
        n_qubits = row.get("n_qubits")
        n_layers = row.get("n_layers")
        pred = row.get("prediction")
        seed = row.get("seed")

        if n_qubits is None or n_layers is None or pred is None:
            continue

        if seed is None:
            cid = str(row.get("cid") or "")
            match = re.search(r"_S(\d+)$", cid)
            if match is not None:
                seed = int(match.group(1))
            else:
                # Fallback: if seed is not available, treat each row as its own pseudo-seed.
                seed = int(hashlib.md5(cid.encode("utf-8")).hexdigest()[:8], 16)

        per_seed[(int(n_qubits), int(n_layers), int(seed))] = per_seed.get(
            (int(n_qubits), int(n_layers), int(seed)),
            [],
        ) + [float(pred)]

    # Second level: average seed means per (n_qubits, n_layers)
    by_size: dict[tuple[int, int], list[float]] = {}
    for (n_qubits, n_layers, _seed), values in per_seed.items():
        seed_mean = sum(values) / len(values)
        by_size.setdefault((n_qubits, n_layers), []).append(seed_mean)

    aggregated: list[dict[str, Any]] = []
    for (n_qubits, n_layers), seed_means in sorted(by_size.items()):
        aggregated.append(
            {
                "n_qubits": n_qubits,
                "n_layers": n_layers,
                "n_seeds": len(seed_means),
                "prediction_avg": sum(seed_means) / len(seed_means),
            },
        )

    return aggregated


def save_aggregated_predictions_csv(aggregated: list[dict[str, Any]], output_path: Path) -> None:
    """Save per-size aggregated predictions to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["n_qubits", "n_layers", "n_seeds", "prediction_avg"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregated)


def plot_aggregated_predictions(aggregated: list[dict[str, Any]], output_path: Path) -> None:
    """Plot average prediction by size and save to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not aggregated:
        logger.info("No aggregated points available; skipping plot save to %s", output_path)
        return

    x = [int(row["n_qubits"]) for row in aggregated]
    y = [int(row["n_layers"]) for row in aggregated]
    z = [float(row["prediction_avg"]) for row in aggregated]
    n_seeds = [int(row["n_seeds"]) for row in aggregated]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x, y, c=z, s=[30 + 10 * c for c in n_seeds], cmap="viridis", alpha=0.9)
    plt.colorbar(scatter, label="Average prediction")
    plt.xlabel("n_qubits")
    plt.ylabel("n_layers")
    plt.title("Seed-averaged prediction per (n_qubits, n_layers)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_predictions_vs_qubits_for_layer(
    records: list[dict[str, Any]],
    *,
    n_layers: int,
    output_path: Path,
) -> None:
    """Plot all prediction points for a fixed n_layers against n_qubits."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    filtered = [row for row in records if row.get("n_layers") is not None and int(row["n_layers"]) == n_layers]
    if not filtered:
        logger.info("No prediction rows found for n_layers=%d; skipping plot %s", n_layers, output_path)
        return

    x = [int(row["n_qubits"]) for row in filtered if row.get("n_qubits") is not None]
    y = [float(np.mean(row["prediction"])) for row in filtered if row.get("n_qubits") is not None]

    if not x:
        logger.info("No valid n_qubits values for n_layers=%d; skipping plot %s", n_layers, output_path)
        return

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, alpha=0.75, s=28)
    plt.xlabel("n_qubits")
    plt.ylabel("prediction")
    plt.title(f"All predictions vs n_qubits (n_layers={n_layers})")
    plt.grid(visible=True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def derive_aggregate_output_paths(predictions_output_csv: Path) -> tuple[Path, Path]:
    """Derive aggregate CSV and plot paths from the prediction CSV path."""
    stem = predictions_output_csv.stem
    parent = predictions_output_csv.parent
    aggregate_csv = parent / f"{stem}__avg_by_size.csv"
    aggregate_plot = parent / f"{stem}__avg_by_size.png"
    return aggregate_csv, aggregate_plot


def derive_layer_plot_output_path(predictions_output_csv: Path, n_layers: int) -> Path:
    """Derive fixed-layer plot path from the prediction CSV path."""
    stem = predictions_output_csv.stem
    parent = predictions_output_csv.parent
    return parent / f"{stem}__pred_vs_qubits__L{n_layers}.png"


# -----------------------------
# CLI
# -----------------------------

def main(
    family: str | None = typer.Option(None, help="Family to predict on for per-family data selection."),
    model_type: str = typer.Option("global", help="'global' or 'per_family'"),
    dataset_dir: str = typer.Option("outputs/data/predictions", help="Root directory containing prediction .pt files."),
    batch_size: int = typer.Option(32, help="Prediction batch size."),
    global_feature_variant: str = typer.Option("binned", help="Global feature variant."),
    node_feature_backend_variant: str | None = typer.Option(None, help="Optional node feature backend variant."),
    plot_n_layers: int | None = typer.Option(None, help="If set, save a plot of all predictions vs n_qubits for this n_layers value."),
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

    _dataset, loader = build_prediction_loader(
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
    output_csv_path = resolve_output_csv_path(
        output_csv,
        model_type=model_type,
        family=family,
        model_path=model_path,
        global_feature_variant=global_feature_variant,
        node_feature_backend_variant=node_feature_backend_variant,
        batch_size=batch_size,
    )
    save_predictions_csv(records, output_csv_path)

    aggregate_csv_path, aggregate_plot_path = derive_aggregate_output_paths(output_csv_path)
    aggregated = aggregate_predictions_by_size(records)
    save_aggregated_predictions_csv(aggregated, aggregate_csv_path)
    plot_aggregated_predictions(aggregated, aggregate_plot_path)

    if plot_n_layers is not None:
        layer_plot_path = derive_layer_plot_output_path(output_csv_path, plot_n_layers)
        plot_predictions_vs_qubits_for_layer(records, n_layers=plot_n_layers, output_path=layer_plot_path)
        logger.info("Saved fixed-layer plot (n_layers=%d) to %s", plot_n_layers, layer_plot_path)

    logger.info("Saved %d predictions to %s", len(records), output_csv_path)
    logger.info("Saved %d aggregated rows to %s", len(aggregated), aggregate_csv_path)
    logger.info("Saved aggregate plot to %s", aggregate_plot_path)

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
