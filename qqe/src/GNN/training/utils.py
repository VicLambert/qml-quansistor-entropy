from __future__ import annotations

import hashlib
import logging

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

from torch.amp import autocast
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from .train_config import FAMILY_GATE_TYPES, MASTER_GATE_TYPES

logger = logging.getLogger(__name__)
DatasetSplit = Literal["all", "target", "prediction"]

import ast


def to_scalar(x):
    # Already numeric
    if isinstance(x, (int, float)):
        return x

    # torch / numpy scalar
    if hasattr(x, "item"):
        return x.item()

    # Strings
    if isinstance(x, str):
        x = x.strip()

        # Handle tensor(...) by stripping wrapper FIRST
        if x.startswith("tensor(") and x.endswith(")"):
            x = x[len("tensor("):-1].strip()

        try:
            val = ast.literal_eval(x)
        except Exception:
            # fallback: plain float string
            return float(x)

        # If it's a list/tuple like [10]
        if isinstance(val, (list, tuple)):
            if len(val) == 1:
                return float(val[0])
            raise ValueError(f"Unexpected list length: {val}")

        return float(val)

    raise ValueError(f"Unsupported type: {type(x)}")

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
        keep = [k for k in all_gate_keys if k.startswith("haar_eig_bin_")]
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
        qubit_mask = data.x[:, self.n_gate_master:]

        out = data.clone()
        out.x = torch.cat([gate[:, self.keep_gate_idx], qubit_mask], dim=1)
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

        out.global_features = g[:, self.keep_idx].squeeze(0).to(torch.float32)
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


class ProjectedDatasetWrapper:
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[Data, torch.Tensor]:
        data = self.dataset[idx]
        return self.transform(data)


def _amp_device_type() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _safe_y(batch) -> torch.Tensor:
    if not hasattr(batch, "y") or batch.y is None:
        raise ValueError("Batch does not have 'y' attribute or it is None")
    y = batch.y
    if y.dim() == 0:
        y = y.view(1)
    if y.dim() == 2 and y.size(1) == 1:
        y = y.view(-1)

    return y.float()

def _move_to_device(x, device: torch.device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if hasattr(x, "to"):
        return x.to(device, non_blocking=True)
    return x


def unpack_supervised_batch(
    batch,
    device: torch.device,
) -> tuple[object, torch.Tensor, int]:
    """Normalize batches from either:
    - PyG loaders yielding Data/Batch objects with .y and .num_graphs
    - Torch loaders yielding (features, target) tuples/lists
    """
    if isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise ValueError("Tuple/list batch must contain at least (inputs, targets).")

        x = _move_to_device(batch[0], device)
        y_raw = _move_to_device(batch[1], device)

        if torch.is_tensor(y_raw):
            y = y_raw.float().view(-1)
        else:
            y = torch.as_tensor(y_raw, dtype=torch.float32, device=device).view(-1)

        batch_size = int(y.numel()) if y.numel() > 0 else 1
        return x, y, batch_size

    moved_batch = _move_to_device(batch, device)
    y = _safe_y(moved_batch)
    batch_size = int(getattr(moved_batch, "num_graphs", y.numel()))
    return moved_batch, y, max(1, batch_size)


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

def collect_dataset_paths(
    dataset_root: str | Path,
    *,
    family: str | None = None,
    split: DatasetSplit = "all",
) -> list[str]:
    root = Path(dataset_root)

    if family is not None:
        search_dirs = [root / family]
    else:
        search_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()] if root.exists() else []

    paths: list[Path] = []

    for search_dir in search_dirs:
        if search_dir.exists():
            paths.extend(sorted(search_dir.glob("*.pt")))

    if not paths:
        paths = sorted(root.glob("*.pt"))

    if split == "all":
        return [str(p.resolve()) for p in paths]

    import torch

    filtered: list[Path] = []

    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        has_target = bool(payload.get("meta", {}).get("has_target", "sre" in payload or "ee" in payload))

        if split == "target" and has_target or (split == "prediction" and not has_target):
            filtered.append(path)

    return [str(p.resolve()) for p in filtered]


def cache_root_paths(paths: list[str], suffix: str = "") -> str:
    """Given a list of file paths, compute a unique hash and return the corresponding cache path."""
    # Create a unique hash based on the sorted list of paths
    canonical = "|".join(sorted(Path(p).name for p in paths))
    digest = hashlib.md5(canonical.encode("utf-8")).hexdigest()[:10]
    tag = f"_{suffix}" if suffix else ""
    cache_dir = Path("qqe") / "cache" / f"pyg_cache_{digest}{tag}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    use_amp: bool = True,
    show_progress: bool = False,
) -> float:
    model.eval()
    total_loss = 0.0
    total_graphs = 0

    loader_iter = tqdm(loader, desc="Validation", leave=False) if show_progress else loader

    for batch in loader_iter:
        model_input, y, batch_size = unpack_supervised_batch(batch, device)

        with autocast(
            device_type=_amp_device_type(),
            enabled=(use_amp and device.type == "cuda"),
        ):
            pred = model(model_input).view(-1).float()

            # (Optional) ignore NaN labels if you ever have any
            mask = torch.isfinite(y)
            if mask.sum() == 0:
                continue
            loss = loss_fn(pred[mask], y[mask])

        total_loss += float(loss.item()) * batch_size
        total_graphs += batch_size

    return total_loss / max(1, total_graphs)

@torch.no_grad()
def evaluate_r2(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
    show_progress: bool = False,
) -> float:
    model.eval()
    total_ss_res = 0.0
    total_ss_tot = 0.0
    total_graphs = 0

    loader_iter = tqdm(loader, desc="R² Evaluation", leave=False) if show_progress else loader

    for batch in loader_iter:
        model_input, y, batch_size = unpack_supervised_batch(batch, device)

        with autocast(
            device_type=_amp_device_type(),
            enabled=(use_amp and device.type == "cuda"),
        ):
            pred = model(model_input).view(-1).float()

            mask = torch.isfinite(y)
            if mask.sum() == 0:
                continue

            y_true = y[mask]
            y_pred = pred[mask]

            ss_res = torch.sum((y_true - y_pred) ** 2).item()
            ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2).item()

        total_ss_res += ss_res
        total_ss_tot += ss_tot
        total_graphs += batch_size

    r2_score = 1 - (total_ss_res / total_ss_tot) if total_ss_tot > 0 else float("nan")
    return r2_score

