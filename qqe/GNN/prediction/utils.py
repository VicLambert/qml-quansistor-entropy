
from __future__ import annotations

import hashlib

from pathlib import Path
from tqdm import tqdm

import torch
from torch_geometric.data import Data
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch_geometric.loader import DataLoader

from .pred_config import FAMILY_GATE_TYPES, MASTER_GATE_TYPES


def _amp_device_type() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _family_global_gate_keys(family: str, all_gate_keys: list[str]) -> list[str]:
    """
    Return the subset of global gate-feature keys relevant for a given family.

    all_gate_keys contains only the gate-count part of the binned global vector,
    i.e. it does NOT include the leading metadata entries [n_qubits, n_bins].
    """
    if family == "random":
        keep = [
            k for k in all_gate_keys
            if (
                k.startswith(("rx_bin_", "ry_bin_", "rz_bin_")) or k == "CNOT_count"
            )
        ]
    elif family == "clifford":
        wanted = {"I_count", "H_count", "S_count", "T_count", "CNOT_count"}
        keep = [k for k in all_gate_keys if k in wanted]
    elif family == "haar":
        keep = [k for k in all_gate_keys if k == "haar_count"]
    elif family == "quansistor":
        keep = [
            k for k in all_gate_keys
            if k.startswith("qx_") or k.startswith("qy_")
        ]
    else:
        raise ValueError(f"Unknown family '{family}'")

    return keep


class FamilyNodeProjector:
    def __init__(self, family: str):
        self.family = family
        self.keep_gate_idx = [
            MASTER_GATE_TYPES.index(name)
            for name in FAMILY_GATE_TYPES[family]
        ]
        self.n_gate_master = len(MASTER_GATE_TYPES)

    def __call__(self, data: Data) -> Data:
        gate = data.x[:, :self.n_gate_master]
        qubit = data.x[:, self.n_gate_master:]

        out = data.clone()
        out.x = torch.cat([gate[:, self.keep_gate_idx], qubit], dim=1)
        return out


class FamilyGlobalProjector:
    """
    Projects data.global_features from the master binned schema to a family-specific one.

    Assumes the global feature layout is:
        [n_qubits, n_bins] + all_gate_keys
    where all_gate_keys is the same ordering used by QuantumCircuitGraphDataset.
    """
    def __init__(self, family: str, all_gate_keys: list[str]):
        self.family = family
        self.all_gate_keys = list(all_gate_keys)

        keep_gate_keys = _family_global_gate_keys(family, self.all_gate_keys)

        # First two positions are metadata: [n_qubits, n_bins]
        self.keep_idx = [0, 1] + [
            2 + self.all_gate_keys.index(k) for k in keep_gate_keys
        ]

    def __call__(self, data: Data) -> Data:
        out = data.clone()

        g = out.global_features
        if g.dim() == 1:
            g = g.unsqueeze(0)

        out.global_features = g[:, self.keep_idx]
        return out


class FamilyFeatureProjector:
    """
    Combined projector for both node features and global features.
    """
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

