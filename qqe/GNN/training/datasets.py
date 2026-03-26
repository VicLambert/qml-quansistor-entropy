from __future__ import annotations

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader as TorchDataLoader, random_split
from torch_geometric.loader import DataLoader

from qqe.GNN.physics_aware_NN import QuantumCircuitGraphDataset

from .utils import FamilyFeatureProjector, ProjectedDatasetWrapper, cache_root_paths


class PaddedGraphDatasetWrapper:
    def __init__(self, dataset, target_dim: int | None = None):
        self.dataset = dataset
        self.target_dim = target_dim or self._compute_max_dim()

    def _compute_max_dim(self) -> int:
        max_dim = 0
        for i in range(len(self.dataset)):
            data = self.dataset[i]
            if hasattr(data, "x") and data.x.dim() > 1:
                max_dim = max(max_dim, data.x.shape[1])
        return max_dim

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        data = self.dataset[idx]

        if hasattr(data, "x") and data.x.shape[1] < self.target_dim:
            # Pad the feature matrix with zeros
            out = data.clone()
            pad_size = self.target_dim - data.x.shape[1]
            out.x = F.pad(out.x, (0, pad_size), value=0)
            return out
        return data


class GlobalTargetDatasetWrapper:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.dataset[idx]

        g = getattr(data, "global_features", None)
        if g is None:
            raise ValueError("Sample is missing 'global_features'.")
        if not torch.is_tensor(g):
            g = torch.as_tensor(g, dtype=torch.float32)
        g = g.flatten().to(torch.float32)

        y = getattr(data, "y", None)
        if y is None:
            y_tensor = torch.tensor(float("nan"), dtype=torch.float32)
        elif torch.is_tensor(y):
            y_tensor = y.flatten()[0].to(torch.float32)
        else:
            y_tensor = torch.tensor(float(y), dtype=torch.float32)

        return g, y_tensor


def build_loaders(
    pt_paths: list[str],
    *,
    batch_size: int = 32,
    seed: int = 42,
    train_split: float = 0.8,
    val_split: float = 0.1,
    global_feature_variant: str = "binned",
    node_feature_variant: str | None = None,
    family_projection: str | None = None,
):
    suffix = (
        f"{global_feature_variant}"
        f"__backend_{node_feature_variant or 'none'}"
        f"_familyproj_{family_projection or 'none'}"
    )
    root = cache_root_paths(pt_paths, suffix=suffix)

    base_dataset = QuantumCircuitGraphDataset(
        root=root,
        pt_paths=pt_paths,
        global_feature_variant=global_feature_variant,
        node_feature_backend_variant=node_feature_variant,
    )

    if len(base_dataset) < 3:
        raise RuntimeError("Dataset too small for train/val/test splitting.")

    working_dataset = base_dataset
    if family_projection is not None:
        projector = FamilyFeatureProjector(
            family=family_projection,
            all_gate_keys=base_dataset.all_gate_keys,
        )
        working_dataset = ProjectedDatasetWrapper(
            working_dataset,
            transform=projector,
        )

    padded_dataset = PaddedGraphDatasetWrapper(working_dataset)
    sample0 = padded_dataset[0]

    node_in_dim = sample0.x.shape[1]
    global_in_dim = sample0.global_features.numel()

    generator = torch.Generator().manual_seed(seed)
    primary_train_len = max(1, int(len(padded_dataset) * train_split))
    test_len = max(1, len(padded_dataset) - primary_train_len)

    while primary_train_len + test_len > len(padded_dataset):
        primary_train_len -= 1

    primary_train_dataset, test_ds = random_split(
        padded_dataset,
        [primary_train_len, test_len],
        generator=generator,
    )

    val_len = max(1, int(len(primary_train_dataset) * val_split))
    real_train_len = max(1, len(primary_train_dataset) - val_len)

    train_ds, val_ds = random_split(
        primary_train_dataset,
        [real_train_len, val_len],
        generator=generator,
    )

    pin_mem = torch.cuda.is_available()

    return (
        DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_mem
        ),
        DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_mem
        ),
        DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_mem
        ),
        node_in_dim,
        global_in_dim,
        base_dataset,
    )


def build_loaders_NN(
    pt_paths: list[str],
    *,
    batch_size: int = 32,
    seed: int = 42,
    train_split: float = 0.8,
    val_split: float = 0.1,
    global_feature_variant: str = "binned",
    node_feature_variant: str | None = None,
    family_projection: str | None = None,
):
    suffix = (
        f"{global_feature_variant}"
        f"__backend_{node_feature_variant or 'none'}"
        f"_familyproj_{family_projection or 'none'}"
    )
    root = cache_root_paths(pt_paths, suffix=suffix)

    base_dataset = QuantumCircuitGraphDataset(
        root=root,
        pt_paths=pt_paths,
        global_feature_variant=global_feature_variant,
        node_feature_backend_variant=node_feature_variant,
    )

    if len(base_dataset) < 3:
        raise RuntimeError("Dataset too small for train/val/test splitting.")

    working_dataset = base_dataset
    if family_projection is not None:
        projector = FamilyFeatureProjector(
            family=family_projection,
            all_gate_keys=base_dataset.all_gate_keys,
        )
        working_dataset = ProjectedDatasetWrapper(
            working_dataset,
            transform=projector,
        )

    nn_dataset = GlobalTargetDatasetWrapper(working_dataset)
    sample0_g, _ = nn_dataset[0]

    global_in_dim = int(sample0_g.numel())

    generator = torch.Generator().manual_seed(seed)
    primary_train_len = max(1, int(len(nn_dataset) * train_split))
    test_len = max(1, len(nn_dataset) - primary_train_len)

    while primary_train_len + test_len > len(nn_dataset):
        primary_train_len -= 1

    primary_train_dataset, test_ds = random_split(
        nn_dataset,
        [primary_train_len, test_len],
        generator=generator,
    )

    val_len = max(1, int(len(primary_train_dataset) * val_split))
    real_train_len = max(1, len(primary_train_dataset) - val_len)

    train_ds, val_ds = random_split(
        primary_train_dataset,
        [real_train_len, val_len],
        generator=generator,
    )

    pin_mem = torch.cuda.is_available()

    return (
        TorchDataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_mem
        ),
        TorchDataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_mem
        ),
        TorchDataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_mem
        ),
        global_in_dim,
        base_dataset,
    )
