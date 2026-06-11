
from __future__ import annotations

import warnings
from pathlib import Path
from collections import OrderedDict

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.amp.autocast_mode import autocast
from torch_geometric.data import Data, Dataset as PyGDataset
from torch_geometric.data.separate import separate

_AMP_DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"


class Parameters_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 1)

    def forward(self, x):
        h = self.linear1(x)
        h = self.relu(h)
        h = self.linear2(h)
        h = self.relu(h)
        h = self.linear3(h)
        return h


GNN_HIDDEN = 32
GNN_HEADS = 8
GLOBAL_HIDDEN = 16
REG_HIDDEN = 16
NUM_LAYERS = 3


class GlobalMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.0):
        super().__init__()
        dr = float(dropout_rate) if dropout_rate is not None else 0.0
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dr),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dr),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dr),
        )

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return self.net(g)


class Regressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout_rate: float = 0.0):
        super().__init__()
        dr = float(dropout_rate) if dropout_rate is not None else 0.0
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dr),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dr),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)

def _get_activation(activation: str) -> nn.Module:
    if activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    else:
        raise ValueError(f"Unsupported activation function: {activation}")


class NN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.0):
        super().__init__()
        dr = float(dropout_rate) if dropout_rate is not None else 0.0
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dr),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dr),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dr),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return self.net(g)

class GNN(nn.Module):
    def __init__(
        self,
        node_in_dim: int = 23,
        gnn_hidden: int = GNN_HIDDEN,
        gnn_heads: int = GNN_HEADS,
        global_in_dim: int = 8,
        global_hidden: int = GLOBAL_HIDDEN,
        reg_hidden: int = REG_HIDDEN,
        num_layers: int = NUM_LAYERS,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        from torch_geometric.nn import TransformerConv, global_mean_pool

        self.global_mean_pool = global_mean_pool

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.num_layers = int(num_layers)
        self.gnn_hidden = gnn_hidden
        self.gnn_heads = gnn_heads
        self.dropout_rate = float(dropout_rate) if dropout_rate is not None else 0.0

        conv_layers = [
            TransformerConv(
                node_in_dim, gnn_hidden, heads=gnn_heads, dropout=self.dropout_rate, beta=False
            ),
        ]
        conv_layers.extend(
            [
                TransformerConv(
                    gnn_hidden * gnn_heads,
                    gnn_hidden,
                    heads=gnn_heads,
                    dropout=self.dropout_rate,
                    beta=False,
                )
                for _ in range(1, self.num_layers)
            ],
        )
        self.conv_layers = nn.ModuleList(conv_layers)
        self.global_mlp = GlobalMLP(global_in_dim, global_hidden, dropout_rate=self.dropout_rate)
        concat_dim = gnn_hidden * gnn_heads + global_hidden
        self.regressor = Regressor(concat_dim, reg_hidden, dropout_rate=self.dropout_rate)


    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        num_graphs = int(batch.max().item() + 1) if batch.numel() else 1

        # GNN branch
        if x is None or x.size(0) == 0:
            x_pool = x.new_zeros((num_graphs, self.gnn_hidden * self.gnn_heads), dtype=torch.float32)
        else:
            with autocast(_AMP_DEVICE_TYPE, enabled=False):
                h = x.float()
                for conv in self.conv_layers:
                    h = F.relu(conv(h, edge_index))
                    if self.dropout_rate:
                        h = F.dropout(h, p=self.dropout_rate, training=self.training)
                x_pool = self.global_mean_pool(h, batch)

        # Global branch
        g = data.global_features
        if g.dim() == 1:
            if g.numel() % num_graphs != 0:
                raise RuntimeError(
                    f"Inconsistent global_features in batch: total={g.numel()}, "
                    f"num_graphs={num_graphs}. Expected fixed per-graph feature length."
                )
            g = g.view(num_graphs, -1)
        elif g.dim() == 2 and g.size(0) == num_graphs:
            pass
        else:
            g = g.reshape(num_graphs, -1)
        g_feat = self.global_mlp(g.float())

        # Combine
        out = self.regressor(torch.cat([x_pool, g_feat], dim=-1))
        return out.view(-1)


def normalize_gate_count(dataset):
    all_keys = set()
    for data in dataset:
        if hasattr(data, "gate_count") or isinstance(data.gate_count, dict):
            all_keys.update(data.gate_count.keys())
    all_keys = sorted(all_keys)

    for data in dataset:
        if hasattr(data, "gate_count") and isinstance(data.gate_count, dict):
            for key in all_keys:
                if key not in data.gate_count:
                    data.gate_count[key] = 0

    return dataset, all_keys


class QuantumCircuitGraphDataset(PyGDataset):
    """Loads per-circuit .pt files produced by compute_entry_for_config.

    Each sample file is expected to contain:
      - x: Tensor [N, node_dim]
      - edge_index: Tensor [2, E]
      - global_features: Tensor [G]
      - sre: float (label)
      - gate_counts: dict (optional, carried along)
      - meta: dict (optional)
    """
    def __init__(
        self,
        root: str,
        pt_paths: list[str],
        global_feature_variant: str = "baseline",
        node_feature_backend_variant: str | None = None,
        fixed_all_gate_keys: list[str] | None = None,
        transform=None,
        pre_transform=None,
        target_variant: str = "sre",
    ):
        self.pt_paths = [str(p) for p in pt_paths]
        self.global_feature_variant = global_feature_variant
        self.node_feature_backend_variant = node_feature_backend_variant

        if fixed_all_gate_keys is None:
            self.all_gate_keys = self._collect_all_gate_keys()
        else:
            self.all_gate_keys = list(fixed_all_gate_keys)

        self.global_feature_dim = self._collect_global_feature_dim()
        self.target_variant = target_variant

        super().__init__(root=root, transform=transform, pre_transform=pre_transform)

    def _collect_all_gate_keys(self) -> list[str]:
        """Scan all .pt files to collect all unique gate count keys."""
        all_keys = set()
        for pt_path in self.pt_paths:
            obj = torch.load(pt_path, map_location="cpu", weights_only=True)
            gate_counts = obj.get("gate_counts", {})
            if isinstance(gate_counts, dict):
                all_keys.update(gate_counts.keys())
        return sorted(all_keys)

    def _build_uniform_binned_global(self, normalized_gate_counts: dict[str, int], meta: dict) -> torch.Tensor:
        """Build uniform binned global feature vector from normalized gate counts."""
        feature_list = [
            float(meta.get("n_qubits", 0)),
            float(meta.get("n_bins", 50)),
        ]
        feature_list.extend(float(normalized_gate_counts.get(k, 0)) for k in self.all_gate_keys)
        return torch.tensor(feature_list, dtype=torch.float32)

    def _collect_global_feature_dim(self) -> int:
        # For binned variant, dimension is fixed: 2 (metadata) + num_gate_keys
        if self.global_feature_variant == "binned":
            return 2 + len(self.all_gate_keys)

        # For other variants, scan actual dimensions
        dim_counts: dict[int, int] = {}
        for pt_path in self.pt_paths:
            obj = torch.load(pt_path, map_location="cpu", weights_only=True)
            g = obj.get("global_features", None)
            if g is None:
                continue
            if not torch.is_tensor(g):
                g = torch.as_tensor(g)
            d = int(g.numel())
            dim_counts[d] = dim_counts.get(d, 0) + 1

        if not dim_counts:
            return 0

        if len(dim_counts) > 1:
            warnings.warn(
                f"Inconsistent global_features dims found: {dim_counts}. "
                f"Will pad/truncate to max dim={max(dim_counts)}."
            )
        return max(dim_counts)

    def _make_target(self, data):
        sre = float(data.get("sre", float("nan")))
        meta = data.get("meta", {})
        n = int(meta.get("n_qubits", 0))

        if self.target_variant == "sre":
            y = sre
        elif self.target_variant == "sre_density":
            y = sre / n if n > 0 else float("nan")

        elif self.target_variant == "log_sre":
            y = np.log1p(sre)

        elif self.target_variant == "sqrt_sre":
            y = np.sqrt(max(sre, 0.0))

        else:
            raise ValueError(f"Unsupported target_variant: {self.target_variant}")
        return torch.tensor([y], dtype=torch.float32)

    @property
    def raw_file_names(self) -> list[str]:
        # Not used (we bypass PyG raw/processed system), but required by base class.
        return []

    @property
    def processed_file_names(self) -> list[str]:
        return []

    def len(self) -> int:
        return len(self.pt_paths)

    def get(self, idx: int) -> Data:
        obj = torch.load(self.pt_paths[idx], map_location="cpu", weights_only=True)

        x = obj["x"]
        edge_index = obj["edge_index"]
        g = obj["global_features"]
        y_val = self._make_target(obj)

        # Normalize gate_counts: add missing keys with value 0
        gate_counts = obj.get("gate_counts", {})
        if isinstance(gate_counts, dict):
            normalized_gate_counts = {key: gate_counts.get(key, 0) for key in self.all_gate_keys}
        else:
            normalized_gate_counts = {key: 0 for key in self.all_gate_keys}

        raw_meta = obj.get("meta", {}) or {}
        meta = {
            "cid": "" if raw_meta.get("cid") is None else str(raw_meta.get("cid")),
            "family": "" if raw_meta.get("family") is None else str(raw_meta.get("family")),
            "seed": 0 if raw_meta.get("seed") is None else int(raw_meta.get("seed")),
            "n_qubits": 0 if raw_meta.get("n_qubits") is None else int(raw_meta.get("n_qubits")),
            "n_layers": 0 if raw_meta.get("n_layers") is None else int(raw_meta.get("n_layers")),
        }

        # Build global features uniformly from normalized gate counts (binned) or pad/truncate (other)
        if self.global_feature_variant == "binned":
            g = self._build_uniform_binned_global(normalized_gate_counts, meta)
        else:
            # Dtypes for model
            if not torch.is_tensor(g):
                g = torch.as_tensor(g, dtype=torch.float32)
            g = g.flatten().to(torch.float32)

            if self.global_feature_dim > 0 and g.numel() != self.global_feature_dim:
                if g.numel() < self.global_feature_dim:
                    g = F.pad(g, (0, self.global_feature_dim - g.numel()))
                else:
                    g = g[: self.global_feature_dim]

        # Dtypes for model
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        if edge_index.dtype != torch.long:
            edge_index = edge_index.to(torch.long)

        # label
        if y_val is None:
            y = torch.tensor([float("nan")], dtype=torch.float32)
        else:
            y = torch.tensor([float(y_val)], dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.global_features = g.unsqueeze(0)
        data.num_qubits = int(meta.get("n_qubits", 0))
        data.gate_counts = normalized_gate_counts
        data.meta = meta

        return data

class ShardedQuantumCircuitGraphDataset(PyGDataset):
    def __init__(
        self,
        index_paths,
        *,
        target_variant="sre",
        split = "target",
        cache_size = 4,
        fixed_all_gate_keys = None,
    ):
        self.index_paths = [Path(p) for p in index_paths]
        self.target_variant = target_variant

        self.split = split
        self.cache_size = cache_size
        self._cache = OrderedDict()
        self.rows = self._load_index_rows()

        self.all_gate_keys = self._collect_all_gate_keys_from_shards()

    def _load_index_rows(self):
        rows = []
        for index_path in self.index_paths:
            index_path = Path(index_path)
            if not index_path.exists():
                raise FileNotFoundError(f"Index file not found: {index_path}")
            with index_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    row = json.loads(line)

                    if self.split == "target":
                        if not bool(row.get("has_target", False)):
                            continue

                    elif self.split == "prediction":
                        if bool(row.get("has_target", False)):
                            continue

                    elif self.split == "all":
                        pass

                    else:
                        raise ValueError(f"Unknown split={self.split}")
                    if "shard_path" not in row:
                        raise KeyError(f"Missing 'shard_path' in row: {row}")

                    if "local_idx" not in row:
                        raise KeyError(f"Missing 'local_idx' in row: {row}")
                    raw_shard_path = Path(row["shard_path"])

                    if raw_shard_path.is_absolute():
                        shard_path = raw_shard_path
                    else:
                        # Important: resolve relative to the index file location
                        shard_path = index_path.parent / raw_shard_path

                    if not shard_path.exists():
                        raise FileNotFoundError(
                            "Shard file not found.\n"
                            f"Index file: {index_path}\n"
                            f"Stored shard_path: {row['shard_path']}\n"
                            f"Resolved shard_path: {shard_path}"
                        )

                    row["shard_path"] = str(shard_path.resolve())
                    rows.append(row)

        return rows

    def _collect_all_gate_keys_from_shards(self) -> list[str]:
        keys = set()
        seen = set()

        for row in self.rows:
            shard_path = row["shard_path"]

            if shard_path in seen:
                continue

            seen.add(shard_path)

            _, _, shard_meta = torch.load(
                shard_path,
                map_location="cpu",
                weights_only=False,
            )

            keys.update(shard_meta.get("all_gate_keys", []))

        return sorted(keys)

    def __len__(self):
        return len(self.rows)

    def _load_shard(self, shard_path):
        shard_path = str(shard_path)
        if shard_path in self._cache:
            self._cache.move_to_end(shard_path)
            return self._cache[shard_path]
        data, slices, shard_meta = torch.load(shard_path, map_location="cpu", weights_only=False)
        self._cache[shard_path] = (data, slices, shard_meta)

        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

        return data, slices, shard_meta

    def _apply_target_transform(self, data: Data) -> Data:
        nan = torch.tensor([float("nan")], dtype=torch.float32)
        if not hasattr(data, "sre"):
            data.sre = nan.clone()
            data.raw_sre = nan.clone()
            data.y = nan.clone()
            return data
        sre = data.sre.float().view(-1)

        if self.target_variant == "sre":
            y = sre
        elif self.target_variant == "sre_density":
            n = data.n_qubits.float().view(-1)
            y = sre / n
        elif self.target_variant == "log_sre":
            y = torch.log1p(torch.clamp(sre, min=0.0))
        else:
            raise ValueError(f"Unsupported target_variant: {self.target_variant}")

        data.raw_sre = sre.clone()
        data.y = y.float().view(1)

        return data

    def __getitem__(self, idx: int):
        row = self.rows[idx]

        data, slices, shard_meta = self._load_shard(row["shard_path"])

        sample = separate(
            cls=data.__class__,
            batch=data,
            idx=int(row["local_idx"]),
            slice_dict=slices,
            decrement=False,
        )

        sample = self._apply_target_transform(sample)

        return sample
