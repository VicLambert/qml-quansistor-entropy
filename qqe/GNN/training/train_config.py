
from dataclasses import dataclass

FAMILY_REGISTRY = {
    "haar": True,
    "random": True,
    "clifford": True,
    "quansistor": True,
}

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


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    loss_type: str = "mse"
    batch_size: int = 32
    training_mode: str = "global"   # "global" or "per_family"
    family: str | None = None       # Only used if training_mode is "per_family"
    global_feature_variant: str = "binned"
    node_feature_backend_variant: str | None = None
    seed: int = 42
    train_split: float = 0.8
    val_split: float = 0.1
    show_progress: bool = True
    show_val_progress: bool = False
    log_batch_loss_every: int = 10
    heartbeat: float = 60.0
    epoch_warning: float = 300.0

