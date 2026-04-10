
from __future__ import annotations
from typing import Sequence

import warnings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast

from torch_geometric.data import Data, Dataset as PyGDataset

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
    def __init__(
        self,
        global_in_dim: int = 8,
        global_hidden: Sequence[int] = (64, 64, 64),
        activation: str = "relu",
        use_batchnorm: bool = False,
        dropout_rate: float = 0.1,
    ):
        self.global_in_dim = global_in_dim
        self.hidden_layers = tuple(int(h) for h in global_hidden)
        self.use_batchnorm = use_batchnorm

        activation_layer = _get_activation(activation)

        feature_layers: list[nn.Module] = []

        previous_dim = self.global_in_dim

        for hidden_dim in self.hidden_layers:
            feature_layers.append(nn.Linear(previous_dim, hidden_dim))
            if self.use_batchnorm:
                feature_layers.append(nn.BatchNorm1d(hidden_dim))
            feature_layers.append(activation_layer)
            if dropout_rate:
                feature_layers.append(nn.Dropout(p=dropout_rate))
            previous_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*feature_layers) if feature_layers else nn.Identity()

        last_dim = self.hidden_layers[-1] if self.hidden_layers else previous_dim
        self.regressor = nn.Linear(last_dim, 1)

        self._initialize_weights(activation_layer)

    def _initialize_weights(self, activation_layer: nn.Module) -> None:
        nonlinearity = "relu"
        negative_slope = 0.01 if isinstance(activation_layer, nn.LeakyReLU) else 0.0

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight,
                    a=negative_slope,
                    mode="fan_in",
                    nonlinearity=nonlinearity,
                )
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                    nn.init.uniform_(module.bias, -bound, bound)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute regression prediction.

        Expects x with shape [batch_size, input_dim]. Returns a tensor with
        shape [batch_size], squeezing the final singleton dimension.
        """
        if x.dim() != 2 or x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input of shape [batch, {self.input_dim}], got {tuple(x.shape)}"
            )

        features = self.feature_extractor(x)
        output = self.regressor(features)
        return output.squeeze(-1)

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
    ):
        self.pt_paths = [str(p) for p in pt_paths]
        self.global_feature_variant = global_feature_variant
        self.node_feature_backend_variant = node_feature_backend_variant

        if fixed_all_gate_keys is None:
            self.all_gate_keys = self._collect_all_gate_keys()
        else:
            self.all_gate_keys = list(fixed_all_gate_keys)

        self.global_feature_dim = self._collect_global_feature_dim()

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
        y_val = obj.get("sre", None)

        # Normalize gate_counts: add missing keys with value 0
        gate_counts = obj.get("gate_counts", {})
        if isinstance(gate_counts, dict):
            normalized_gate_counts = {key: gate_counts.get(key, 0) for key in self.all_gate_keys}
        else:
            normalized_gate_counts = {key: 0 for key in self.all_gate_keys}

        meta = obj.get("meta", {})

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


