
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from pathlib import Path
import hashlib

from qqe.GNN.physics_aware_NN import GNN, QuantumCircuitGraphDataset
from .utils import FamilyFeatureProjector, out_is_same


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

def build_prediction_loader(
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
        DataLoader(pred_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_mem),
        node_in_dim,
        global_in_dim,
    )
