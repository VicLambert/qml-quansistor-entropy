
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import typer

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from qqe.GNN.physics_aware_NN import GNN, QuantumCircuitGraphDataset
from qqe.utils import configure_logger

logger = logging.getLogger(__name__)

MASTER_GATE_TYPES = [
    "input", "measurement",
    "h", "s", "t", "id",
    "rx", "ry", "rz",
    "cx",
    "qx", "qy", "haar",
]

FAMILY_GATE_TYPES = {
    "random": ["input", "measurement", "rx", "ry", "rz", "cx"],
    "clifford": ["input", "measurement", "h", "s", "t", "id", "cx"],
    "haar": ["input", "measurement", "haar"],
    "quansistor": ["input", "measurement", "qx", "qy"],
}

class FamilyNodeProjector:
    def __init__(self, family: str):
        self.family = family
        self.master_gate_types = MASTER_GATE_TYPES
        self.family_gate_types = FAMILY_GATE_TYPES[family]
        self.keep_gate_idx = [
            self.master_gate_types.index(name) for name in self.family_gate_types
        ]
        self.n_gate_master = len(self.master_gate_types)

    def __call__(self, data: Data) -> Data:
        gate_part = data.x[:, :self.n_gate_master]
        qubit_part = data.x[:, self.n_gate_master:]

        out = data.clone()
        out.x = torch.cat([gate_part[:, self.keep_gate_idx], qubit_part], dim=1)
        return out

class PaddedGraphDatasetWrapper:
    """Wrapper that pads/truncates graph features to consistent dimensions."""

    def __init__(
        self,
        dataset,
        target_node_dim: int | None = None,
        target_global_dim: int | None = None,
        target_dim: int | None = None,  # backwards compatibility
    ):
        self.dataset = dataset

        # Backwards compatibility with older call sites using target_dim.
        if target_node_dim is None and target_dim is not None:
            target_node_dim = target_dim

        self.target_dim = target_node_dim if target_node_dim is not None else self._compute_max_node_dim()
        self.target_global_dim = (
            target_global_dim if target_global_dim is not None else self._compute_max_global_dim()
        )

    def _compute_max_node_dim(self) -> int:
        """Find max node feature width across all samples."""
        max_dim = 0
        for i in range(len(self.dataset)):
            data = self.dataset[i]
            x = getattr(data, "x", None)
            if x is not None and x.dim() > 1:
                max_dim = max(max_dim, int(x.shape[1]))
        return max_dim

    def _compute_max_global_dim(self) -> int:
        """Find max global feature width across all samples."""
        max_dim = 0
        for i in range(len(self.dataset)):
            data = self.dataset[i]
            g = getattr(data, "global_features", None)
            if g is None:
                continue
            if g.dim() == 0:
                width = 1
            elif g.dim() == 1:
                width = int(g.shape[0])
            else:
                width = int(g.shape[-1])
            max_dim = max(max_dim, width)
        return max_dim

    def _fit_node_dim(self, data):
        x = getattr(data, "x", None)
        if x is None or x.dim() <= 1:
            return data
        current = int(x.shape[1])
        if current == self.target_dim:
            return data
        out = data.clone()
        if current < self.target_dim:
            pad_size = self.target_dim - current
            out.x = torch.nn.functional.pad(out.x, (0, pad_size), mode="constant", value=0.0)
        else:
            out.x = out.x[:, : self.target_dim]
        return out

    def _fit_global_dim(self, data):
        g = getattr(data, "global_features", None)
        if g is None or self.target_global_dim <= 0:
            return data

        out = data.clone() if out_is_same(data, g) else data
        g = out.global_features

        # Ensure graph-level vector shape.
        if g.dim() == 0:
            g = g.view(1)
        elif g.dim() > 1:
            g = g.view(-1)

        current = int(g.shape[0])
        if current < self.target_global_dim:
            g = torch.nn.functional.pad(
                g, (0, self.target_global_dim - current), mode="constant", value=0.0,
            )
        elif current > self.target_global_dim:
            g = g[: self.target_global_dim]

        out.global_features = g
        return out

    def __getitem__(self, idx: int):
        data = self.dataset[idx]
        data = self._fit_node_dim(data)
        data = self._fit_global_dim(data)
        return data

    def __len__(self) -> int:
        return len(self.dataset)


def out_is_same(data, g):
    # Clone lazily only when we actually need to edit global features.
    return hasattr(data, "global_features") and data.global_features is g


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


def _cache_root_for_paths(paths: list[str], suffix: str = "") -> str:
    # Include full resolved paths in the digest to avoid collisions across folders/families.
    canonical = "|".join(sorted(str(Path(p).resolve()) for p in paths))
    digest = hashlib.md5(canonical.encode("utf-8")).hexdigest()[:10]
    tag = f"_{suffix}" if suffix else ""
    cache_dir = Path("..") / "qqe" / "cache" / f"pyg_cache_{digest}{tag}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir.resolve())

def build_pred_loaders_two_stage(
    pt_paths: list[str],
    batch_size: int = 32,
    seed: int = 42,
    global_feature_variant: str = "baseline",
    node_feature_backend_variant: str | None = None,
) -> tuple[QuantumCircuitGraphDataset, PaddedGraphDatasetWrapper,DataLoader, int, int]:
    suffix = f"{global_feature_variant}_backend_{node_feature_backend_variant or 'none'}"
    root = _cache_root_for_paths(pt_paths, suffix=suffix)

    dataset = QuantumCircuitGraphDataset(
        root=root,
        pt_paths=pt_paths,
        global_feature_variant=global_feature_variant,
        node_feature_backend_variant=node_feature_backend_variant,
    )

    if len(dataset) < 3:
        raise RuntimeError("Dataset too small for train/val/test splitting.")

    padded_dataset = PaddedGraphDatasetWrapper(dataset)
    node_in_dim = padded_dataset.target_dim
    global_in_dim = dataset.global_feature_dim

    pred_ds = padded_dataset
    pin_mem = torch.cuda.is_available()
    return (
        dataset,
        padded_dataset,
        DataLoader(pred_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_mem),
        node_in_dim,
        global_in_dim,
    )

def _extract_state_dict(payload):
    if isinstance(payload, nn.Module):
        return payload.state_dict()
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    if isinstance(payload, dict) and all(torch.is_tensor(v) for v in payload.values()):
        return payload
    raise RuntimeError("Unsupported model file format.")


def _rank_tensor(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, values.numel() + 1, dtype=torch.float32)
    return ranks


def _spearman_corr(y_true: list[float], y_pred: list[float]) -> float:
    if len(y_true) < 2:
        return float("nan")

    true_t = torch.tensor(y_true, dtype=torch.float32)
    pred_t = torch.tensor(y_pred, dtype=torch.float32)

    r_true = _rank_tensor(true_t)
    r_pred = _rank_tensor(pred_t)

    r_true = r_true - r_true.mean()
    r_pred = r_pred - r_pred.mean()

    denom = torch.norm(r_true) * torch.norm(r_pred)
    if float(denom) == 0.0:
        return float("nan")
    return float((r_true * r_pred).sum() / denom)


def _compute_regression_metrics(records: list[dict]) -> dict[str, float | int]:
    labeled = [
        r for r in records
        if r.get("target") is not None
        and math.isfinite(float(r["target"]))
        and math.isfinite(float(r["prediction"]))
    ]

    n = len(labeled)
    if n == 0:
        return {
            "count_labeled": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "spearman": float("nan"),
        }

    y_true = torch.tensor([float(r["target"]) for r in labeled], dtype=torch.float32)
    y_pred = torch.tensor([float(r["prediction"]) for r in labeled], dtype=torch.float32)

    residuals = y_pred - y_true
    mae = float(torch.mean(torch.abs(residuals)))
    rmse = float(torch.sqrt(torch.mean(residuals ** 2)))

    ss_res = float(torch.sum((y_true - y_pred) ** 2))
    y_mean = float(torch.mean(y_true))
    ss_tot = float(torch.sum((y_true - y_mean) ** 2))
    r2 = float("nan") if ss_tot == 0.0 else 1.0 - (ss_res / ss_tot)

    spearman = _spearman_corr(y_true.tolist(), y_pred.tolist())

    return {
        "count_labeled": n,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "spearman": spearman,
    }


def _compute_grouped_metrics(records: list[dict], group_key: str) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[dict]] = {}
    for record in records:
        key = record.get(group_key)
        key_name = "unknown" if key is None else str(key)
        groups.setdefault(key_name, []).append(record)

    return {group: _compute_regression_metrics(group_records) for group, group_records in groups.items()}


def _write_prediction_records_csv(records: list[dict], csv_path: Path) -> None:
    fieldnames = [
        "index",
        "prediction",
        "target",
        "target_source",
        "abs_error",
        "n_qubits",
        "n_layers",
        "family",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, record in enumerate(records, start=1):
            pred_val = float(record["prediction"])
            target_val = record.get("target")
            abs_error = None
            if target_val is not None and math.isfinite(float(target_val)):
                abs_error = abs(pred_val - float(target_val))

            writer.writerow(
                {
                    "index": i,
                    "prediction": pred_val,
                    "target": target_val,
                    "target_source": record.get("target_source"),
                    "abs_error": abs_error,
                    "n_qubits": record.get("n_qubits"),
                    "n_layers": record.get("n_layers"),
                    "family": record.get("family"),
                },
            )


def _aggregate_by_circuit_shape(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int | None, int | None], list[dict]] = {}
    for record in records:
        key = (
            str(record.get("family") or "unknown"),
            record.get("n_qubits"),
            record.get("n_layers"),
        )
        grouped.setdefault(key, []).append(record)

    summary_rows: list[dict] = []
    for (fam, n_qubits, n_layers), recs in sorted(
        grouped.items(), key=lambda x: (x[0][0], str(x[0][1]), str(x[0][2])),
    ):
        pred_vals = [float(r["prediction"]) for r in recs if math.isfinite(float(r["prediction"]))]
        if not pred_vals:
            continue
        pred_t = torch.tensor(pred_vals, dtype=torch.float32)

        target_vals = [
            float(r["target"])
            for r in recs
            if r.get("target") is not None and math.isfinite(float(r["target"]))
        ]

        target_mean = None
        target_std = None
        abs_err_mean = None
        if target_vals:
            target_t = torch.tensor(target_vals, dtype=torch.float32)
            target_mean = float(target_t.mean())
            target_std = float(target_t.std(unbiased=False)) if target_t.numel() > 1 else 0.0
            abs_err_mean = abs(float(pred_t.mean()) - target_mean)

        summary_rows.append(
            {
                "family": fam,
                "n_qubits": n_qubits,
                "n_layers": n_layers,
                "count": len(recs),
                "prediction_mean": float(pred_t.mean()),
                "prediction_std": float(pred_t.std(unbiased=False)) if pred_t.numel() > 1 else 0.0,
                "target_mean": target_mean,
                "target_std": target_std,
                "abs_error_of_means": abs_err_mean,
            },
        )

    return summary_rows


def _write_grouped_summary_csv(summary_rows: list[dict], csv_path: Path) -> None:
    fieldnames = [
        "family",
        "n_qubits",
        "n_layers",
        "count",
        "prediction_mean",
        "prediction_std",
        "target_mean",
        "target_std",
        "abs_error_of_means",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def _save_eval_plots(records: list[dict], output_dir: Path, run_tag: str) -> None:
    labeled = [
        r for r in records
        if r.get("target") is not None
        and math.isfinite(float(r["target"]))
        and math.isfinite(float(r["prediction"]))
    ]
    if not labeled:
        return

    y_true = [float(r["target"]) for r in labeled]
    y_pred = [float(r["prediction"]) for r in labeled]
    y_min = min(min(y_true), min(y_pred))
    y_max = max(max(y_true), max(y_pred))

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.65)
    plt.plot([y_min, y_max], [y_min, y_max], "r--", linewidth=1.0)
    plt.xlabel("True SRE")
    plt.ylabel("Predicted SRE")
    plt.title("Predicted vs True SRE")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"pred_vs_true_{run_tag}.png", dpi=160)
    plt.close()

    x_layers = [r.get("n_layers") for r in labeled]
    residuals = [float(r["prediction"]) - float(r["target"]) for r in labeled]
    layer_points = [(x, y) for x, y in zip(x_layers, residuals, strict=False) if x is not None]
    if layer_points:
        lx, ly = zip(*layer_points, strict=False)
        plt.figure(figsize=(7, 4))
        plt.scatter(lx, ly, alpha=0.65)
        plt.axhline(0.0, color="r", linestyle="--", linewidth=1.0)
        plt.xlabel("n_layers")
        plt.ylabel("Residual (pred - true)")
        plt.title("Residuals vs n_layers")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"residual_vs_layers_{run_tag}.png", dpi=160)
        plt.close()

    x_qubits = [r.get("n_qubits") for r in labeled]
    qubit_points = [(x, y) for x, y in zip(x_qubits, residuals, strict=False) if x is not None]
    if qubit_points:
        qx, qy = zip(*qubit_points, strict=False)
        plt.figure(figsize=(7, 4))
        plt.scatter(qx, qy, alpha=0.65)
        plt.axhline(0.0, color="r", linestyle="--", linewidth=1.0)
        plt.xlabel("n_qubits")
        plt.ylabel("Residual (pred - true)")
        plt.title("Residuals vs n_qubits")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"residual_vs_qubits_{run_tag}.png", dpi=160)
        plt.close()


def _collect_label_paths(dataset_dir: str, family: str | None = None) -> list[str]:
    d = Path(dataset_dir)
    label_root = d / "encoding_data_pennylane"

    if family is not None:
        paths = sorted((label_root / family).glob("*.pt"))
    else:
        paths = []
        if label_root.exists():
            for family_dir in sorted(label_root.iterdir()):
                if family_dir.is_dir():
                    paths.extend(sorted(family_dir.glob("*.pt")))

    if not paths:
        paths = sorted(d.glob("*.pt"))

    return [str(p.resolve()) for p in paths]


def _build_target_lookup(dataset_dir: str, family: str | None = None) -> dict[str, float]:
    lookup: dict[str, float] = {}
    label_paths = _collect_label_paths(dataset_dir=dataset_dir, family=family)
    for path in label_paths:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        meta = obj.get("meta", {}) or {}
        cid = meta.get("cid", None)
        sre = obj.get("sre", None)
        if cid is None or sre is None:
            continue
        sre_val = float(sre)
        if math.isfinite(sre_val):
            lookup[str(cid)] = sre_val
    return lookup


def _build_shape_target_lookup(dataset_dir: str, family: str | None = None) -> dict[tuple[str, int, int], dict[str, float | int]]:
    grouped: dict[tuple[str, int, int], list[float]] = {}
    label_paths = _collect_label_paths(dataset_dir=dataset_dir, family=family)
    for path in label_paths:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        meta = obj.get("meta", {}) or {}
        fam = str(meta.get("family") or family or "unknown")
        q = meta.get("n_qubits", None)
        l = meta.get("n_layers", None)
        sre = obj.get("sre", None)
        if q is None or l is None or sre is None:
            continue
        sre_val = float(sre)
        if not math.isfinite(sre_val):
            continue
        key = (fam, int(q), int(l))
        grouped.setdefault(key, []).append(sre_val)

    summary: dict[tuple[str, int, int], dict[str, float | int]] = {}
    for key, vals in grouped.items():
        t = torch.tensor(vals, dtype=torch.float32)
        summary[key] = {
            "mean": float(t.mean()),
            "std": float(t.std(unbiased=False)) if t.numel() > 1 else 0.0,
            "count": int(t.numel()),
        }
    return summary


def _summarize_unmatched_shapes(records: list[dict]) -> list[tuple[tuple[str, int | None, int | None], int]]:
    counts: dict[tuple[str, int | None, int | None], int] = {}
    for r in records:
        if r.get("target") is not None:
            continue
        key = (
            str(r.get("family") or "unknown"),
            r.get("n_qubits"),
            r.get("n_layers"),
        )
        counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)


def _parse_qubit_list(spec: str) -> set[int]:
    out: set[int] = set()
    for token in (spec or "").split(","):
        token = token.strip()
        if not token:
            continue
        out.add(int(token))
    return out


def _load_sim_target_cache(cache_path: Path) -> dict[str, float]:
    if not cache_path.exists():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(payload, dict):
        return {}

    out: dict[str, float] = {}
    for k, v in payload.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            out[str(k)] = fv
    return out


def _save_sim_target_cache(cache_path: Path, cache_data: dict[str, float]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache_data, indent=2, sort_keys=True), encoding="utf-8")


def _simulate_missing_targets(
    prediction_records: list[dict],
    *,
    allowed_qubits: set[int],
    backend: str,
    method: str,
    representation: str,
    max_simulations: int,
    cache_path: Path,
) -> tuple[int, int]:
    # Late import to avoid loading heavy simulation stack unless explicitly requested.
    from generate_data import compute_sre_for_row

    sim_cache = _load_sim_target_cache(cache_path)
    updated_from_cache = 0
    simulated_now = 0

    candidates: list[dict[str, Any]] = []
    for r in prediction_records:
        if r.get("target") is not None:
            continue
        nq = r.get("n_qubits")
        nl = r.get("n_layers")
        fam = r.get("family")
        cid = r.get("cid")

        if cid is None or nq is None or nl is None or fam is None:
            continue
        if int(nq) not in allowed_qubits:
            continue

        if str(cid) in sim_cache:
            r["target"] = float(sim_cache[str(cid)])
            r["target_source"] = "sim_cache"
            updated_from_cache += 1
            continue

        # Parse seed from CID suffix ..._S<seed>
        seed = None
        cid_str = str(cid)
        s_idx = cid_str.rfind("_S")
        if s_idx != -1:
            try:
                seed = int(cid_str[s_idx + 2:])
            except ValueError:
                seed = None
        if seed is None:
            continue

        candidates.append(
            {
                "record": r,
                "family": str(fam),
                "n_qubits": int(nq),
                "n_layers": int(nl),
                "seed": int(seed),
                "cid": cid_str,
            },
        )

    if max_simulations > 0:
        candidates = candidates[:max_simulations]

    if candidates:
        logger.info(
            "Running simulation fallback for %d records (backend=%s, method=%s, representation=%s)",
            len(candidates),
            backend,
            method,
            representation,
        )

    for item in candidates:
        sre = compute_sre_for_row(
            family=item["family"],
            n_qubits=item["n_qubits"],
            n_layers=item["n_layers"],
            seed=item["seed"],
            backend=backend,
            method=method,
            representation=representation,
        )
        if sre is None:
            continue
        sre_val = float(sre)
        if not math.isfinite(sre_val):
            continue

        rec = item["record"]
        rec["target"] = sre_val
        rec["target_source"] = "simulated"
        sim_cache[item["cid"]] = sre_val
        simulated_now += 1

    if simulated_now > 0:
        _save_sim_target_cache(cache_path, sim_cache)

    return updated_from_cache, simulated_now

def main(
    family: str | None = "random",
    model_type: str = "global",     # "global" | "per_family"
    global_feature_variant: str = "binned",
    node_feature_backend_variant: str | None = None,
    evaluate: bool = True,
    eval_output_dir: str = "outputs/predictions_eval",
    labels_dataset_dir: str = "outputs/data",
    simulate_missing_targets: bool = False,
    simulate_qubits: str = "12,14",
    simulate_method: str = "fwht",
    simulate_backend: str = "quimb",
    simulate_representation: str = "dense",
    simulate_max_records: int = 0,
    sim_cache_path: str = "outputs/predictions_eval/simulated_targets_cache.json",
):
    if family is not None and model_type == "per_family":
        MODEL_STATE_PATH = f"models/gnn_model_{family}.pt"
        logger.info(f"Using per-family model for family '{family}' at {MODEL_STATE_PATH}")
    else:
        MODEL_STATE_PATH = "models/gnn_model_global.pt"
        logger.info(f"Using global model at {MODEL_STATE_PATH}")

    pred_data_paths = collect_pred_paths("outputs/data", family=family)
    logger.info(f"Collected {len(pred_data_paths)} .pt files for dataset.")

    # dataset, padded_dataset, pred_loader, node_in_dim, global_in_dim = build_pred_loaders_two_stage(
    #     pred_data_paths,
    #     batch_size=32,
    #     seed=42,
    #     global_feature_variant=global_feature_variant,
    #     node_feature_backend_variant=node_feature_backend_variant,
    # )

    # logger.info(f"Dataset size: {len(dataset)}")
    # logger.info(f"Node feature dimension: {node_in_dim}")
    # logger.info(f"Global feature dimension: {global_in_dim}")

    if not pred_data_paths:
        raise RuntimeError("No prediction .pt files found. Check outputs/data/predictions/<family>.")

    checkpoint = None
    model_config = {}
    fixed_all_gate_keys = None

    if Path(MODEL_STATE_PATH).exists():
        checkpoint = torch.load(MODEL_STATE_PATH, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict):
            model_config = checkpoint.get("model_config", {}) or {}
            feature_config = checkpoint.get("feature_config", {}) or {}
            fixed_all_gate_keys = feature_config.get("all_gate_keys", None)

    suffix = f"{global_feature_variant}_backend_{node_feature_backend_variant or 'none'}"
    root = _cache_root_for_paths(pred_data_paths, suffix=suffix)

    pred_dataset = QuantumCircuitGraphDataset(
        root=root,
        pt_paths=pred_data_paths,
        global_feature_variant=global_feature_variant,
        node_feature_backend_variant=node_feature_backend_variant,
        fixed_all_gate_keys=fixed_all_gate_keys,
    )

    trained_node_in_dim = model_config.get("node_in_dim", None)
    trained_global_in_dim = model_config.get("global_in_dim", None)

    padded_pred_dataset = PaddedGraphDatasetWrapper(
        pred_dataset,
        target_node_dim=trained_node_in_dim if trained_node_in_dim is not None else None,
        target_global_dim=trained_global_in_dim if trained_global_in_dim is not None else None,
        target_dim=trained_node_in_dim if trained_node_in_dim is not None else None,
    )

    pred_loader = DataLoader(
        padded_pred_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    node_in_dim = padded_pred_dataset.target_dim
    global_in_dim = padded_pred_dataset.target_global_dim

    logger.info(f"Prediction graphs: {len(pred_dataset)}")
    logger.info(f"Prediction node feature dim: {node_in_dim}")
    logger.info(f"Prediction global feature dim: {global_in_dim}")
    if trained_node_in_dim is not None or trained_global_in_dim is not None:
        logger.info(f"Trained node/global dims from checkpoint: {trained_node_in_dim}/{trained_global_in_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    if checkpoint is not None and isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        if not Path(MODEL_STATE_PATH).exists():
            raise FileNotFoundError(f"Could not find model weights at {MODEL_STATE_PATH}")
        raw_payload = torch.load(MODEL_STATE_PATH, map_location="cpu")
        state_dict = _extract_state_dict(raw_payload)

    model_kwargs = {
        "node_in_dim": int(trained_node_in_dim if trained_node_in_dim is not None else node_in_dim),
        "global_in_dim": int(trained_global_in_dim if trained_global_in_dim is not None else global_in_dim),
        "gnn_hidden": int(model_config.get("gnn_hidden", 32)),
        "gnn_heads": int(model_config.get("gnn_heads", 8)),
        "global_hidden": int(model_config.get("global_hidden", 16)),
        "reg_hidden": int(model_config.get("reg_hidden", 16)),
        "num_layers": int(model_config.get("num_layers", 5)),
        "dropout_rate": float(model_config.get("dropout_rate", 0.1)),
    }

    model = GNN(**model_kwargs).to(device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    model.eval()

    logger.info("Loaded model config: %s", model_kwargs)
    if missing_keys:
        logger.info("Missing keys: %s", missing_keys)
    if unexpected_keys:
        logger.info("Unexpected keys: %s", unexpected_keys)

    predictions = []
    prediction_records = []
    expected_node_dim = model_kwargs["node_in_dim"]
    expected_global_dim = model_kwargs["global_in_dim"]
    target_lookup: dict[str, float] = {}
    shape_target_lookup: dict[tuple[str, int, int], dict[str, float | int]] = {}

    if evaluate:
        target_lookup = _build_target_lookup(dataset_dir=labels_dataset_dir, family=family)
        shape_target_lookup = _build_shape_target_lookup(dataset_dir=labels_dataset_dir, family=family)
        logger.info("Loaded %d labeled targets from %s", len(target_lookup), labels_dataset_dir)
        logger.info("Loaded %d labeled (family,n_qubits,n_layers) groups", len(shape_target_lookup))

    matched_exact = 0
    matched_shape = 0
    unmatched = 0

    with torch.no_grad():
        for batch in pred_loader:
            batch_samples = batch.to_data_list()

            # Adapt node features to the model input dimension.
            if batch.x.size(1) < expected_node_dim:
                pad_size = expected_node_dim - batch.x.size(1)
                batch.x = torch.nn.functional.pad(batch.x, (0, pad_size), mode="constant", value=0.0)
            elif batch.x.size(1) > expected_node_dim:
                batch.x = batch.x[:, :expected_node_dim]

            # Adapt global features to the model input dimension.
            g = batch.global_features
            if g.dim() == 1:
                g = g.view(batch.num_graphs, -1)
            if g.size(1) < expected_global_dim:
                g = torch.nn.functional.pad(g, (0, expected_global_dim - g.size(1)), mode="constant", value=0.0)
            elif g.size(1) > expected_global_dim:
                g = g[:, :expected_global_dim]
            batch.global_features = g

            batch = batch.to(device)
            pred = model(batch).view(-1)
            pred_values = pred.cpu().tolist()
            predictions.extend(pred_values)

            for sample, pred_value in zip(batch_samples, pred_values):
                meta = getattr(sample, "meta", {}) or {}
                cid = meta.get("cid", None)
                n_qubits = meta.get("n_qubits", getattr(sample, "num_qubits", None))
                n_layers = meta.get("n_layers", meta.get("layers", None))
                family_value = str(meta.get("family") or family or "unknown")
                sample_y = getattr(sample, "y", None)
                target = None
                target_source = "none"
                if torch.is_tensor(sample_y) and sample_y.numel() > 0:
                    target_val = float(sample_y.view(-1)[0].item())
                    if math.isfinite(target_val):
                        target = target_val
                        target_source = "sample_y"
                if target is None and cid is not None and str(cid) in target_lookup:
                    target = float(target_lookup[str(cid)])
                    target_source = "exact_cid"
                if (
                    target is None
                    and n_qubits is not None
                    and n_layers is not None
                ):
                    shape_key = (family_value, int(n_qubits), int(n_layers))
                    if shape_key in shape_target_lookup:
                        target = float(shape_target_lookup[shape_key]["mean"])
                        target_source = "shape_mean"

                if target_source in ("sample_y", "exact_cid"):
                    matched_exact += 1
                elif target_source == "shape_mean":
                    matched_shape += 1
                else:
                    unmatched += 1

                prediction_records.append(
                    {
                        "cid": str(cid) if cid is not None else None,
                        "prediction": float(pred_value),
                        "target": target,
                        "target_source": target_source,
                        "n_qubits": int(n_qubits) if n_qubits is not None else None,
                        "n_layers": int(n_layers) if n_layers is not None else None,
                        "family": family_value,
                    },
                )

    if evaluate and simulate_missing_targets:
        allowed_qubits = _parse_qubit_list(simulate_qubits)
        from_cache, simulated_now = _simulate_missing_targets(
            prediction_records,
            allowed_qubits=allowed_qubits,
            backend=simulate_backend,
            method=simulate_method,
            representation=simulate_representation,
            max_simulations=int(simulate_max_records),
            cache_path=Path(sim_cache_path),
        )
        logger.info(
            "Simulation fallback updates | from_cache=%d | simulated_now=%d | qubits=%s",
            from_cache,
            simulated_now,
            sorted(allowed_qubits),
        )

        # Recompute coverage counters after potential simulation backfill.
        matched_exact = 0
        matched_shape = 0
        unmatched = 0
        for r in prediction_records:
            src = str(r.get("target_source") or "none")
            if src in ("sample_y", "exact_cid"):
                matched_exact += 1
            elif src in ("shape_mean", "sim_cache", "simulated"):
                matched_shape += 1
            else:
                unmatched += 1

    # logger.info("First 10 predictions: %s", predictions[:10])
    # logger.info("Total predictions: %d", len(predictions))

    preview_count = min(10, len(prediction_records))
    logger.info("First %d prediction records with parameters:", preview_count)
    for i, record in enumerate(prediction_records[:preview_count], start=1):
        logger.info(
            "[%d] pred=%.6f | target=%s | n_qubits=%s | n_layers=%s",
            i,
            record["prediction"],
            str(record["target"]),
            str(record["n_qubits"]),
            str(record["n_layers"]),
        )

    logger.info(
        "Target assignment coverage | exact=%d | shape_mean=%d | unmatched=%d",
        matched_exact,
        matched_shape,
        unmatched,
    )

    if unmatched > 0 and evaluate:
        available_qubits = sorted({k[1] for k in shape_target_lookup.keys()})
        available_layers = sorted({k[2] for k in shape_target_lookup.keys()})
        logger.info(
            "Label support summary | available n_qubits=%s | layer range=%s..%s",
            available_qubits,
            str(available_layers[0]) if available_layers else "N/A",
            str(available_layers[-1]) if available_layers else "N/A",
        )

        unmatched_shapes = _summarize_unmatched_shapes(prediction_records)
        top_k = min(10, len(unmatched_shapes))
        logger.info("Top %d unmatched (family, n_qubits, n_layers) groups:", top_k)
        for i, (shape_key, count) in enumerate(unmatched_shapes[:top_k], start=1):
            fam, nq, nl = shape_key
            logger.info(
                "[%d] family=%s | n_qubits=%s | n_layers=%s | count=%d",
                i,
                fam,
                str(nq),
                str(nl),
                count,
            )

    if evaluate:
        overall = _compute_regression_metrics(prediction_records)
        logger.info(
            "Evaluation | labeled=%d | MAE=%.6f | RMSE=%.6f | R2=%s | Spearman=%s",
            int(overall["count_labeled"]),
            float(overall["mae"]),
            float(overall["rmse"]),
            str(overall["r2"]),
            str(overall["spearman"]),
        )

        if int(overall["count_labeled"]) == 0:
            logger.info("No finite labels were found in prediction files; skipped eval artifacts.")
            return

        grouped_qubits = _compute_grouped_metrics(prediction_records, "n_qubits")
        grouped_layers = _compute_grouped_metrics(prediction_records, "n_layers")
        grouped_circuit_shape = _aggregate_by_circuit_shape(prediction_records)

        output_dir = Path(eval_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        run_tag = family if (model_type == "per_family" and family is not None) else "global"

        _write_prediction_records_csv(prediction_records, output_dir / f"prediction_records_{run_tag}.csv")
        _write_grouped_summary_csv(
            grouped_circuit_shape,
            output_dir / f"prediction_grouped_by_nq_nl_{run_tag}.csv",
        )
        _save_eval_plots(prediction_records, output_dir, run_tag)

        preview_groups = min(10, len(grouped_circuit_shape))
        logger.info("First %d grouped rows by (n_qubits, n_layers):", preview_groups)
        for i, row in enumerate(grouped_circuit_shape[:preview_groups], start=1):
            logger.info(
                "[%d] family=%s | n_qubits=%s | n_layers=%s | count=%d | pred_mean=%.6f | pred_std=%.6f | target_mean=%s | abs_err_mean=%s",
                i,
                str(row["family"]),
                str(row["n_qubits"]),
                str(row["n_layers"]),
                int(row["count"]),
                float(row["prediction_mean"]),
                float(row["prediction_std"]),
                str(row["target_mean"]),
                str(row["abs_error_of_means"]),
            )

        summary = {
            "run_tag": run_tag,
            "overall": overall,
            "by_n_qubits": grouped_qubits,
            "by_n_layers": grouped_layers,
            "by_circuit_shape": grouped_circuit_shape,
        }
        with (output_dir / f"evaluation_summary_{run_tag}.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info("Saved evaluation artifacts to %s", str(output_dir.resolve()))


if __name__ == "__main__":
    configure_logger(logging.INFO, logging.INFO)
    logger.info("Starting prediction data loading...")
    typer.run(main)
