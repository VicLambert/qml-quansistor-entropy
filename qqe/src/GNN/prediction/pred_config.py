
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
class PredConfig:
    family: str | None = None
    model_type: str = "global"
    global_feature_variant: str = "binned"
    node_feature_backend_variant: str | None = None
    evaluate: bool = True
    eval_output_path: str = "outputs/predictions_eval"
    labels_dataset_dir: str = "outputs/data"
    simulate_qubits: str = "4,6,8,10,12"
    simulate_method: str = "fwht"
    simulate_backend: str = "quimb"
    simulate_representation: str = "dense"
    simulate_max_records: int = 0
    sim_cache_path: str = "outputs/predictions_eval/simulated_targets_cache.json"
    batch_size: int = 32
    seed: int = 42
