from __future__ import annotations

from pathlib import Path
import numpy as np
import csv
from pathlib import Path

import math
import torch
import torch.nn.functional as F
from typing import Any

from torch_geometric.data import Data

from .pred_config import FAMILY_GATE_TYPES, MASTER_GATE_TYPES


def _amp_device_type() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _family_global_gate_keys(family: str, all_gate_keys: list[str]) -> list[str]:
    """Return the subset of global gate-feature keys relevant for a given family.

    all_gate_keys contains only the gate-count part of the binned global vector,
    i.e. it does NOT include the leading metadata entries [n_qubits, n_bins].
    """
    if family == "random":
        keep = [
            k
            for k in all_gate_keys
            if (k.startswith(("rx_bin_", "ry_bin_", "rz_bin_")) or k == "CNOT_count")
        ]
    elif family == "clifford":
        wanted = {"I_count", "H_count", "S_count", "T_count", "CNOT_count"}
        keep = [k for k in all_gate_keys if k in wanted]
    elif family == "haar":
        keep = [k for k in all_gate_keys if k == "haar_count" or k.startswith("haar_eig_bin_")]
    elif family == "quansistor":
        keep = [k for k in all_gate_keys if k.startswith("qx_") or k.startswith("qy_")]
    else:
        raise ValueError(f"Unknown family '{family}'")

    return keep


class FamilyNodeProjector:
    def __init__(self, family: str):
        self.family = family
        self.keep_gate_idx = [
            MASTER_GATE_TYPES.index(name) for name in FAMILY_GATE_TYPES[family]
        ]
        self.n_gate_master = len(MASTER_GATE_TYPES)

    def __call__(self, data: Data) -> Data:
        gate = data.x[:, : self.n_gate_master]
        qubit = data.x[:, self.n_gate_master :]

        out = data.clone()
        out.x = torch.cat([gate[:, self.keep_gate_idx], qubit], dim=1)
        return out


class FamilyGlobalProjector:
    """Projects data.global_features from the master binned schema to a family-specific one.

    Assumes the global feature layout is:
        [n_qubits, n_bins] + all_gate_keys
    where all_gate_keys is the same ordering used by QuantumCircuitGraphDataset.
    """

    def __init__(self, family: str, all_gate_keys: list[str]):
        self.family = family
        self.all_gate_keys = list(all_gate_keys)

        keep_gate_keys = _family_global_gate_keys(family, self.all_gate_keys)

        # First two positions are metadata: [n_qubits, n_bins]
        self.keep_idx = [0, 1] + [2 + self.all_gate_keys.index(k) for k in keep_gate_keys]

    def __call__(self, data: Data) -> Data:
        out = data.clone()

        g = out.global_features
        if g.dim() == 1:
            g = g.unsqueeze(0)

        out.global_features = g[:, self.keep_idx]
        return out


class FamilyFeatureProjector:
    """Combined projector for both node features and global features."""

    def __init__(self, family: str, all_gate_keys: list[str]):
        self.node_projector = FamilyNodeProjector(family)
        self.global_projector = FamilyGlobalProjector(family, all_gate_keys)

    def __call__(self, data: Data) -> Data:
        out = self.node_projector(data)
        out = self.global_projector(out)
        return out


def out_is_same(data, g):
    # Clone lazily only when we actually need to edit global features.
    return hasattr(data, "global_features") and data.global_features is g

def extract_target_value(sample: Any) -> float | None:
    y = getattr(sample, "y", None)
    if y is None:
        return None

    if torch.is_tensor(y):
        if y.numel() == 0:
            return None
        value = float(y.flatten()[0].item())
    else:
        value = float(y)

    if not np.isfinite(value):
        return None
    if math.isnan(value):
        return None

    return value


from pathlib import Path


def collect_dataset_indices(
    dataset_root: str | Path,
    *,
    family: str | None = None,
) -> list[str]:
    root = Path(dataset_root)

    if family is not None:
        search_dirs = [root / family]
    else:
        search_dirs = [
            p for p in sorted(root.iterdir())
            if p.is_dir()
        ] if root.exists() else []

    index_paths: list[Path] = []

    for search_dir in search_dirs:
        if search_dir.exists():
            index_paths.extend(sorted(search_dir.glob("index_*.jsonl")))

    if not index_paths:
        index_paths = sorted(root.glob("index_*.jsonl"))

    return [str(p.resolve()) for p in index_paths]

# =========================================================
# Dataset wrappers
# =========================================================

class PredictionGraphWrapper:
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
            cur = int(data.x.shape[1])
            if cur < self.target_node_dim:
                data.x = F.pad(data.x, (0, self.target_node_dim - cur))
            elif cur > self.target_node_dim:
                data.x = data.x[:, : self.target_node_dim]

        if hasattr(data, "global_features"):
            g = data.global_features
            if g.dim() > 1:
                g = g.view(-1)

            if self.target_global_dim is not None:
                cur = int(g.shape[0])
                if cur < self.target_global_dim:
                    g = F.pad(g, (0, self.target_global_dim - cur))
                elif cur > self.target_global_dim:
                    g = g[: self.target_global_dim]

            data.global_features = g

        return data


class PredictionTensorWrapper:
    def __init__(self, dataset, target_global_dim: int | None = None):
        self.dataset = dataset
        self.target_global_dim = target_global_dim

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        data = self.dataset[idx]
        g = data.global_features
        if not torch.is_tensor(g):
            g = torch.as_tensor(g, dtype=torch.float32)
        g = g.flatten().to(torch.float32)

        if self.target_global_dim is not None:
            cur = int(g.shape[0])
            if cur < self.target_global_dim:
                g = F.pad(g, (0, self.target_global_dim - cur))
            elif cur > self.target_global_dim:
                g = g[: self.target_global_dim]

        meta = getattr(data, "meta", {}) or {}
        target = extract_target_value(data)
        return g, meta, target


# =========================================================
# Saving + aggregation
# =========================================================

def save_predictions_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No prediction rows to save.")

    preferred_order = [
        "cid",
        "family",
        "regime",
        "seed",
        "n_qubits",
        "n_layers",

        # model-space values
        "target",
        "prediction_model_output",

        # raw SRE values
        "target_sre",
        "prediction",
        "error",
    ]

    # Add any extra keys that were not listed above.
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    fieldnames = [
        key for key in preferred_order
        if key in all_keys
    ]

    fieldnames += sorted(all_keys - set(fieldnames))

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def aggregate_mean_std(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    fixed_key: str | None = None,
    fixed_value: int | None = None,
) -> list[dict[str, Any]]:
    filtered = rows
    if fixed_key is not None and fixed_value is not None:
        filtered = [r for r in rows if int(r[fixed_key]) == int(fixed_value)]

    groups: dict[int, list[float]] = {}
    for r in filtered:
        x = int(r[x_key])
        groups.setdefault(x, []).append(float(r["prediction"]))

    out = []
    for x in sorted(groups):
        vals = np.asarray(groups[x], dtype=float)
        out.append(
            {
                x_key: x,
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=0)),
                "n": len(vals),
            },
        )
    return out
