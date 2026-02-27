import torch
import torch.nn as nn


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
NUM_LAYERS = 5


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


class GNN(nn.Module):
    def __init__(
        self,
        node_in_dim: int = 13,
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

        elf.global_mean_pool = global_mean_pool

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
        x, edge_index, batch = 

