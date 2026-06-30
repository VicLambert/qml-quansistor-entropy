from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import gc
import hashlib
import itertools
import json
import logging
import os

from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from tqdm import tqdm

from backend import PennylaneBackend, QuimbBackend
from circuit.families import (
    CliffordBrickwork,
    HaarBrickwork,
    QuansistorBrickwork,
    RandomCircuit,
)

# Keep this import path only if it is the correct one in your project.
# If your real function lives in circuit.matrix_factory, switch it there.
from circuit.gates import gate_unitary
from circuit.patterns import TdopingRules, to_qasm
from experiments.core import run_experiment
from GNN.encoder import eigenvalue_phase_histogram_features, qasm_to_pyg_graph
from utils import FileCache

if TYPE_CHECKING:
    from circuit.spec import GateSpec

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "outputs" / "gnn_graphs"
DATASET_DIR.mkdir(parents=True, exist_ok=True)
cache = FileCache(PROJECT_ROOT / "outputs" / "cache")

BACKEND_REGISTRY = {
    "pennylane": PennylaneBackend,
    "quimb": QuimbBackend,
}

FAMILY_REGISTRY = {
    "haar": HaarBrickwork,
    "clifford": CliffordBrickwork,
    "quansistor": QuansistorBrickwork,
    "random": RandomCircuit,
}

@dataclass(frozen=True)
class RegimeDistribution:
    regimes: list[str]
    probabilities: list[float]

    def sample(self, rng: np.random.Generator) -> str:
        p = np.asarray(self.probabilities, dtype=float)
        p = p / p.sum()
        return str(rng.choice(self.regimes, p=p))

@dataclass(frozen=True)
class SamplingConfig:
    clifford: RegimeDistribution = RegimeDistribution(
        regimes=[
                "zero",
                "tiny",
                "very_low",
                "low",
                "medium_low",
                "medium",
                "medium_high",
                "high",
            ],
        probabilities=[
                0.05,
                0.05,
                0.10,
                0.10,
                0.20,
                0.20,
                0.20,
                0.10,
            ],
    )
    random: RegimeDistribution = RegimeDistribution(
        regimes=[
                "identity_like",
                "near_clifford",
                "small_angles",
                "medium_angles",
                "generic_sparse",
                "generic_dense",
            ],
        probabilities=[0.005, 0.155, 0.3, 0.25, 0.15, 0.14],
    )
    haar: RegimeDistribution = RegimeDistribution(
        regimes=[
                "identity_like",
                "very_weak",
                "sparse_weak",
                "medium_weak",
                "dense_weak",
                "sparse_full",
                "medium",
                "dense_medium",
                "sparse_full",
                "medium_full",
                "full",
            ],
        probabilities=[0.01, 0.01, 0.05, 0.1, 0.1, 0.12, 0.1, 0.12, 0.12, 0.12, 0.15],
    )
    quansistor: RegimeDistribution = RegimeDistribution(
        regimes=[
                "identity_like",
                "weak",
                "moderate",
                "structured_equal_ab",
                "structured_opposite_ab",
                "generic_uniform",
            ],
        probabilities=[0.1, 0.4, 0.4, 0.1, 0.1, 0.1],
    )

@dataclass(frozen=True)
class ShardConfig:
    shard_id: str
    family: str
    n_qubits: int
    layers: tuple[int, ...]
    configs: tuple[CircuitConfig, ...]

@dataclass
class DataGenConfig:
    backend: str
    method: str
    families: list[str]
    qubits_values: np.ndarray
    layers_values: np.ndarray
    n_seeds: int
    prediction_n_seeds: int | None
    n_bins: int
    compute_sre: bool
    compute_EE: bool
    target_qubits: tuple[int, ...]
    representation: str
    use_dask: bool
    dask_n_workers: int
    dask_memory_per_worker: str
    output_dir: Path
    max_configs: int | None


@dataclass
class CircuitConfig:
    cid: str
    family: str
    n_qubits: int
    n_layers: int
    seed: int


def make_seed(family: str, n_qubits: int, n_layers: int, rep: int) -> int:
    s = f"{family}|{n_qubits}|{n_layers}|{rep}".encode()
    return int.from_bytes(hashlib.blake2b(s, digest_size=4).digest(), "little")


def make_cid(family: str, n_qubits: int, n_layers: int, seed: int) -> str:
    return f"{family}_Q{n_qubits}_L{n_layers}_S{seed}"


def build_op_descriptors_from_spec(
    gates: tuple[GateSpec, ...] | None,
    family: str,
) -> list[dict[str, Any]] | None:
    """Build per-gate descriptor metadata aligned with the parsed QASM op order.

    Only Haar gates need a nonzero fixed-size node descriptor block.
    All other gates may still keep metadata, but should not carry a Haar descriptor.
    """
    if gates is None:
        return None

    op_descriptors: list[dict[str, Any]] = []

    for gate in gates:
        kind = str(gate.kind).lower()
        kind_norm = "haar" if kind == "haar2" else kind

        descriptor: dict[str, Any] = {
            "kind": kind_norm,
            "wires": tuple(gate.wires),
            "params": gate.params,
            "seed": gate.seed,
            "d": gate.d,
        }

        if family == "haar" and kind_norm == "haar":
            U = gate_unitary(gate)
            descriptor["haar_descriptor"] = eigenvalue_phase_histogram_features(U)

        op_descriptors.append(descriptor)

    return op_descriptors


def generate_dataset_params(
    families: list[str],
    qubits_values: np.ndarray,
    layers_values: np.ndarray,
    num_seeds: int,
) -> list[CircuitConfig]:
    config = []
    for family, n_qubits, n_layers in itertools.product(
        families,
        qubits_values,
        layers_values,
    ):
        for rep in range(num_seeds):
            seed = make_seed(family, n_qubits, n_layers, rep)
            cid = make_cid(family, n_qubits, n_layers, seed)
            config.append(
                CircuitConfig(
                    cid=cid,
                    family=family,
                    n_qubits=int(n_qubits),
                    n_layers=int(n_layers),
                    seed=int(seed),
                ),
            )
    return config


def generate_pyg_shard(
    families: list[str],
    qubits_values: np.ndarray,
    layers_values: np.ndarray,
    n_seeds: int,
    layer_blocks_size: int=10,
) -> list[ShardConfig]:
    shards: list[ShardConfig] = []

    circuit_configs = generate_dataset_params(
        families=families,
        qubits_values=qubits_values,
        layers_values=layers_values,
        num_seeds=n_seeds, 
    )
    layers_list = [int(x) for x in layers_values.tolist()]

    layer_to_block: dict[int, tuple[int, tuple[int, ...]]] = {}

    for block_idx, start in enumerate(range(0, len(layers_list), layer_blocks_size)):
        layer_block = tuple(layers_list[start:start + layer_blocks_size])
        for layer in layer_block:
            layer_to_block[int(layer)] = (block_idx, layer_block)

    grouped: dict[tuple[str, int, int], list[CircuitConfig]] = defaultdict(list)

    for cfg in circuit_configs:
        block_idx, _ = layer_to_block[int(cfg.n_layers)]
        key = (cfg.family, int(cfg.n_qubits), block_idx)
        grouped[key].append(cfg)

    shards: list[ShardConfig] = []

    for (family, n_qubits, block_idx), configs in sorted(grouped.items()):
        _, layer_block = layer_to_block[int(configs[0].n_layers)]

        layer_min = min(layer_block)
        layer_max = max(layer_block)

        shard_id = (
            f"{family}"
            f"_q{int(n_qubits):03d}"
            f"_layers_{layer_min:03d}_{layer_max:03d}"
        )

        configs_sorted = tuple(
            sorted(configs, key=lambda c: (int(c.n_layers), int(c.seed)))
        )

        shards.append(
            ShardConfig(
                shard_id=shard_id,
                family=family,
                n_qubits=int(n_qubits),
                layers=layer_block,
                configs=configs_sorted,
            )
        )

    return shards


def sanitize_gate_counts(gate_counts: dict[str, int]) -> dict[str, int]:
    if not isinstance(gate_counts, dict):
        return {}
    safe: dict[str, int] = {}
    for key, value in gate_counts.items():
        # Handle Tensors by calling .item() if needed
        if isinstance(value, torch.Tensor):
            safe[str(key)] = int(value.item())
        else:
            safe[str(key)] = int(value)
    return safe


def trim_memory() -> None:
    with contextlib.suppress(Exception):
        gc.collect()

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    try:
        if os.name == "nt":
            process_handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.psapi.EmptyWorkingSet(process_handle)
        else:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
    except Exception:
        pass

def sample_t_count(n_layers: int, rng: np.random.Generator, regime: str) -> tuple[str, int]:
    max_t = max(0, 2 * n_layers)
    transistion_cap = min(max_t, 100)

    if max_t == 0 or regime == "zero":
        return regime, 0
    if max_t < 40:
        if regime == "tiny":
            low, high = 2, max(2, int(0.15 * max_t))

        elif regime == "very_low":
            low, high = max(2, int(0.15 * max_t)), max(4, int(0.30 * max_t))

        elif regime == "low":
            low, high = max(2, int(0.30 * max_t)), max(4, int(0.50 * max_t))

        elif regime == "medium_low":
            low, high = max(2, int(0.50 * max_t)), max(4, int(0.70 * max_t))

        elif regime == "medium":
            low, high = max(2, int(0.70 * max_t)), max(4, int(0.90 * max_t))

        elif regime in {"medium_high", "high"}:
            low, high = max(2, int(0.90 * max_t)), max_t

        else:
            raise ValueError(f"Unknown t-count regime: {regime}")

        if low > high:
            low = high
        N_T = int(rng.integers(low, high + 1))
        return regime, N_T

    if regime == "tiny":
        low, high = 2, 6

    elif regime == "very_low":
        low, high = 6, 10

    elif regime == "low":
        low, high = 10, 15

    elif regime == "medium_low":
        low, high = 15, 25

    elif regime == "medium":
        low, high = 25, 41

    elif regime == "medium_high":
        low, high = 40, transistion_cap+1

    elif regime == "high":
        low, high = transistion_cap, max_t + 1

    else:
        raise ValueError(f"Unknown t-count regime: {regime}")

    high = min(high, max_t)
    if low > high:
        low = high

    N_T = int(rng.integers(low, high + 1))
    return (regime, N_T)

def sample_haar_controls(rng: np.random.Generator, regime: str) -> dict[str, Any]:
    if regime == "identity_like":
        p_haar = 0.0
        haar_strength = 0.0
        haar_mode = "identity"

    elif regime == "very_weak":
        p_haar = rng.uniform(0.01, 0.10)
        haar_strength = rng.uniform(0.01, 0.10)
        haar_mode = "exp_hermitian"

    elif regime == "sparse_weak":
        p_haar = rng.uniform(0.05, 0.25)
        haar_strength = rng.uniform(0.02, 0.20)
        haar_mode = "exp_hermitian"

    elif regime == "medium_weak":
        p_haar = rng.uniform(0.25, 0.55)
        haar_strength = rng.uniform(0.02, 0.20)
        haar_mode = "exp_hermitian"

    elif regime == "dense_weak":
        p_haar = rng.uniform(0.60, 1.0)
        haar_strength = rng.uniform(0.02, 0.20)
        haar_mode = "exp_hermitian"

    elif regime == "sparse_full":
        p_haar = rng.uniform(0.05, 0.25)
        haar_strength = 1.0
        haar_mode = "full_haar"

    elif regime == "medium":
        p_haar = rng.uniform(0.15, 0.75)
        haar_strength = rng.uniform(0.25, 0.80)
        haar_mode = "exp_hermitian"

    elif regime == "dense_medium":
        p_haar = float(rng.uniform(0.60, 1.00))
        haar_strength = float(rng.uniform(0.20, 0.70))
        haar_mode = "exp_hermitian"

    elif regime == "sparse_full":
        p_haar = float(rng.uniform(0.05, 0.25))
        haar_strength = 1.0
        haar_mode = "full_haar"

    elif regime == "medium_full":
        p_haar = float(rng.uniform(0.25, 0.60))
        haar_strength = 1.0
        haar_mode = "full_haar"

    elif regime == "full":
        p_haar = float(rng.uniform(0.70, 1.00))
        haar_strength = 1.0
        haar_mode = "full_haar"
    else:
        raise ValueError(f"Unknown regime: {regime}")
    return {
        "sampling_regime": regime,
        "gate_probability": p_haar,
        "haar_probability": p_haar,
        "haar_strength": haar_strength,
        "haar_mode": haar_mode,
    }

def sample_random_controls(rng: np.random.Generator, regime: str) -> dict[str, Any]:

    if regime == "identity_like":
        gate_probability = float(rng.uniform(0.05, 0.20))
        angle_regime = "identity_like"
        angle_scale = float(rng.uniform(0.00, 0.03))

    elif regime == "near_clifford":
        gate_probability = float(rng.uniform(0.25, 0.65))
        angle_regime = "clifford_like"
        angle_scale = float(rng.uniform(0.005, 0.07))

    elif regime == "small_angles":
        gate_probability = float(rng.uniform(0.30, 0.70))
        angle_regime = "small_angles"
        angle_scale = float(rng.uniform(0.10, np.pi/4))

    elif regime == "medium_angles":
        gate_probability = float(rng.uniform(0.60, 0.90))
        angle_regime = "small_angles"
        angle_scale = float(rng.uniform(np.pi/4, np.pi/2))

    elif regime == "generic_sparse":
        gate_probability = float(rng.uniform(0.10, 0.40))
        angle_regime = "generic"
        angle_scale = None

    elif regime == "generic_dense":
        gate_probability = float(rng.uniform(0.60, 1.00))
        angle_regime = "generic"
        angle_scale = None

    else:
        raise ValueError(f"Unknown random regime={regime}")

    return {
        "sampling_regime": regime,
        "gate_probability": gate_probability,
        "angle_regime": angle_regime,
        "angle_scale": angle_scale,
    }

def sample_quansistor_controls(rng: np.random.Generator, regime: str) -> dict[str, Any]:
    if regime == "identity_like":
        gate_probability = float(rng.uniform(0.00, 0.10))
        param_scale = 0.02
    elif regime == "weak":
        gate_probability = float(rng.uniform(0.10, 0.45))
        param_scale = float(rng.uniform(0.05, 0.30))
    elif regime == "moderate":
        gate_probability = float(rng.uniform(0.40, 0.85))
        param_scale = float(rng.uniform(0.30, 0.90))
    elif regime == "structured_equal_ab" or regime == "structured_opposite_ab":
        gate_probability = float(rng.uniform(0.40, 0.80))
        param_scale = float(rng.uniform(0.30, 1.0))
    elif regime == "generic_uniform":
        gate_probability = float(rng.uniform(0.60, 1.00))
        param_scale = None
    else:
        raise ValueError(f"Unknown quansistor regime={regime}")

    return {
        "sampling_regime": regime,
        "gate_probability": gate_probability,
        "param_scale": param_scale,
    }

def sample_generation_controls(
    family: str,
    n_layers: int,
    seed: int,
    sampling_config: SamplingConfig | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    if sampling_config is None:
        sampling_config = SamplingConfig()

    controls: dict[str, Any] = {
        "sampling_regime": "default",
        "tdoping": None,
        "t_count": None,
        "gate_probability": None,
        "angle_regime": None,
        "angle_scale": None,
        "haar_probability": None,
        "param_regime": None,
        "param_scale": None,
    }

    if family == "clifford":
        regime = sampling_config.clifford.sample(rng)
        regime, t_count = sample_t_count(int(n_layers), rng, regime)

        controls["sampling_regime"] = str(regime)
        controls["t_count"] = int(t_count)
        controls["tdoping"] = TdopingRules(count=int(t_count), per_layer=2)

    elif family == "haar":
        regime = sampling_config.haar.sample(rng)
        controls.update(sample_haar_controls(rng, regime))

    elif family == "random":
        regime = sampling_config.random.sample(rng)
        controls.update(sample_random_controls(rng, regime))

    elif family == "quansistor":
        regime = sampling_config.quansistor.sample(rng)
        controls.update(sample_quansistor_controls(rng, regime))

    return controls

def build_data_object(
    config: DataGenConfig,
    cid: str,
    family: str,
    n_qubits: int,
    n_layers: int,
    seed: int,
    sampling_config: SamplingConfig | None = None,
) -> Data | None:
    from experiments.core import ExperimentConfig
    from properties.compute import PropertyRequest
    from states.types import BackendConfig

    try:
        base_dir = config.output_dir or DATASET_DIR
        base_dir.mkdir(parents=True, exist_ok=True)
        # path = base_dir / f"{cid}.pt"
        # tmp_path: Path = path.with_suffix(".pt.tmp")

        # if path.exists():
        #     return {"cid": cid, "path": str(path), "cached": True}

        controls = sample_generation_controls(
            family=family,
            n_layers=int(n_layers),
            seed=int(seed),
            sampling_config=sampling_config,
        )

        make_spec_kwargs = {
            "d": 2,
            "seed": int(seed),
        }
        family_cls = FAMILY_REGISTRY[family]
        family_obj = family_cls()

        if family == "clifford":
            make_spec_kwargs["tdoping"] = controls["tdoping"]

        elif family == "random":
            make_spec_kwargs["angle_regime"] = controls["angle_regime"]
            make_spec_kwargs["angle_scale"] = controls.get("angle_scale")
            make_spec_kwargs["gate_probability"] = controls["gate_probability"]

        elif family == "haar":
            make_spec_kwargs["gate_probability"] = controls["gate_probability"]
            make_spec_kwargs["haar_probability"] = controls["haar_probability"]
            make_spec_kwargs["haar_strength"] = controls["haar_strength"]
            make_spec_kwargs["haar_mode"] = controls["haar_mode"]

        elif family == "quansistor":
            make_spec_kwargs["param_regime"] = controls.get("sampling_regime")
            make_spec_kwargs["param_scale"] = controls.get("param_scale")
            make_spec_kwargs["gate_probability"] = controls.get("gate_probability")


        spec = family_obj.make_spec(
            int(n_qubits),
            int(n_layers),
            **make_spec_kwargs,
        )

        gates = cast("tuple[GateSpec, ...] | None", spec.gates)
        qasm = to_qasm(spec, gates)
        op_descriptors = build_op_descriptors_from_spec(gates, family)

        graph_data, gate_counts = qasm_to_pyg_graph(
            qasm_str=qasm,
            n_bins=config.n_bins,
            family=family,
            global_feature_variant="binned",
            op_descriptors=op_descriptors,
        )

        sre_value = None
        EE_value = None

        should_compute_target = int(n_qubits) in set(config.target_qubits)

        property_requests = []

        if should_compute_target and config.compute_sre:
            property_requests.append(PropertyRequest(name="SRE", method=config.method, params={}))

        if should_compute_target and config.compute_EE:
            property_requests.append(PropertyRequest(name="entanglement_entropy", method=config.method, params={}))

        if property_requests:
            backend_config = BackendConfig(
                name=config.backend,
                representation=config.representation,
                params={},
            )

            exp_config = ExperimentConfig(
                spec=spec,
                backend=backend_config,
                properties=property_requests,
            )

            result = run_experiment(
                exp_config,
                backend_registry=BACKEND_REGISTRY,
                cache=cache,
            )
        if should_compute_target and config.compute_sre:
            sre_key = f"SRE:{config.method.lower()}"
            sre_result = result.results.get(sre_key)
            sre_value = float(sre_result.value) if sre_result else None

        if should_compute_target and config.compute_EE:
            ee_key = f"entanglement_entropy:{config.method.lower()}"
            ee_result = result.results.get(ee_key)
            EE_value = float(ee_result.value) if ee_result else None

        x = graph_data.x.detach().cpu().to(torch.float32)
        # if x.numel() > 0 and x.min().item() >= 0 and x.max().item() <= 1:
        #     x = x.to(torch.uint8)
        # else:
        #     x = x.to(torch.float16)

        edge_index = graph_data.edge_index.detach().cpu().to(torch.long)
        global_features = graph_data.global_features.detach().cpu().to(torch.float32).flatten().view(1,-1)

        data = Data(
            x = x,
            edge_index=edge_index,
            global_features=global_features,
        )
        if should_compute_target and config.compute_sre:
            sre = float(sre_value) if sre_value is not None else float("nan")
            data.y = torch.tensor([sre], dtype=torch.float32)
            data.sre = torch.tensor([sre], dtype=torch.float32)

        if should_compute_target and config.compute_EE:
            ee = float(EE_value) if EE_value is not None else float("nan")
            data.ee = torch.tensor([ee], dtype=torch.float32)

        if not hasattr(data, "y"):
            data.y = torch.tensor([float("nan")], dtype=torch.float32)

        has_valid_sre = (
            should_compute_target
            and config.compute_sre
            and sre_value is not None
            and np.isfinite(float(sre_value))
        )

        has_valid_ee = (
            should_compute_target
            and config.compute_EE
            and EE_value is not None
            and np.isfinite(float(EE_value))
        )
        data.cid = cid
        data.family = family
        data.regime = str(controls["sampling_regime"])

        data.n_qubits = torch.tensor([int(n_qubits)], dtype=torch.long)
        data.n_layers = torch.tensor([int(n_layers)], dtype=torch.long)
        data.seed = torch.tensor([int(seed)], dtype=torch.long)
        has_valid_target = has_valid_sre or has_valid_ee

        data.has_target = torch.tensor([has_valid_target], dtype=torch.bool)

        data.backend = config.backend
        data.method = config.method
        data.representation = config.representation
        data.n_bins = torch.tensor([int(config.n_bins)], dtype=torch.long)

        clean_counts = sanitize_gate_counts(gate_counts)
        
        # Ensure all values are Python ints (not Tensors)
        data.gate_counts = {str(k): int(v) for k, v in clean_counts.items()}
        
        # Also keep individual count_* attributes for backward compatibility
        for key, value in clean_counts.items():
            if isinstance(value, (int, float)):
                setattr(
                    data,
                    f"count_{key}",
                    torch.tensor([value], dtype=torch.float32),
                )
        return data

    except Exception:
        logger.exception(
            "Error computing entry for family=%s Q%s L%s S%s",
            family,
            n_qubits,
            n_layers,
            seed,
        )
        return None

    finally:
        trim_memory()


def compute_pyg_shard(
    shard: ShardConfig,
    config: DataGenConfig,
    *,
    sampling_config: SamplingConfig | None = None,
) -> dict[str, Any] | None:
    base_dir = config.output_dir or DATASET_DIR
    processed_dir = base_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    shard_path = processed_dir / f"{shard.shard_id}.pt"
    tmp_path = shard_path.with_suffix(".pt.tmp")

    if shard_path.exists():
        _, _, old_meta = torch.load(shard_path, map_location="cpu", weights_only=False)
        if int(old_meta.get("num_failures", 0)) == 0:
            return {
                "shard_id": shard.shard_id,
                "path": str(shard_path),
                "cached": True,
            }
        logger.warning(
            "Recomputing shard %s due to previous failures: %d failures out of %d samples",
            shard.shard_id,
            int(old_meta.get("num_failures", 0)),
            int(old_meta.get("num_samples", 0)),
        )
        shard_path.unlink()

    data_list: list[Data] = []
    index_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    all_gate_keys_collected: set[str] = set()

    for cfg in shard.configs:
        cid = (
            f"{shard.family}"
            f"_q{shard.n_qubits:03d}"
            f"_L{int(cfg.n_layers):03d}"
            f"_s{int(cfg.seed):05d}"
        )

        data = build_data_object(
            config=config,
            cid=cid,
            family=shard.family,
            n_qubits=shard.n_qubits,
            n_layers=int(cfg.n_layers),
            seed=int(cfg.seed),
            sampling_config=sampling_config,
        )

        if data is None:
            failures.append(
                {
                    "cid": cid,
                    "family": shard.family,
                    "n_qubits": shard.n_qubits,
                    "n_layers": int(cfg.n_layers),
                    "seed": int(cfg.seed),
                },
            )
            continue
        if hasattr(data, "gate_counts") and isinstance(data.gate_counts, dict):
            all_gate_keys_collected.update(data.gate_counts.keys())

        local_idx = len(data_list)
        data_list.append(data)

        row = {
            "cid": cid,
            "family": shard.family,
            "n_qubits": shard.n_qubits,
            "n_layers": int(cfg.n_layers),
            "seed": int(cfg.seed),
            "shard_id": shard.shard_id,
            "shard_path": str(Path("processed") / f"{shard.shard_id}.pt"),
            "local_idx": local_idx,
            "regime": getattr(data, "regime", None),
            "has_target": bool(data.has_target.item()),
        }

        if hasattr(data, "sre"):
            row["sre"] = float(data.sre.item())

        if hasattr(data, "sre_density"):
            row["sre_density"] = float(data.sre_density.item())

        if hasattr(data, "ee"):
            row["ee"] = float(data.ee.item())

        index_rows.append(row)

    if not data_list:
        logger.warning("Shard %s produced no valid samples", shard.shard_id)
        return None

    data, slices = InMemoryDataset.collate(data_list)

    shard_meta = {
        "format": "qqe_pyg_inmemory_shard_v1",
        "shard_id": shard.shard_id,
        "cids": [c.cid for c in shard.configs],
        "family": shard.family,
        "n_qubits": int(shard.n_qubits),
        "layers": [int(x) for x in shard.layers],
        "seeds": sorted({int(c.seed) for c in shard.configs}),
        "num_samples": len(data_list),
        "num_failures": len(failures),
        "backend": config.backend,
        "method": config.method,
        "representation": config.representation,
        "n_bins": int(config.n_bins),
        "compute_sre": bool(config.compute_sre),
        "compute_EE": bool(config.compute_EE),
        "target_qubits": list(config.target_qubits),
        "all_gate_keys": sorted(all_gate_keys_collected),  # NEW
        "index_rows": index_rows,
        "failures": failures,
    }

    if failures:
        logger.warning(
            "Shard %s had %d failures out of %d samples",
            shard.shard_id,
            len(failures),
            len(data_list) + len(failures),
            "First 5 failures: %s",
            failures[:5],
        )
    torch.save((data, slices, shard_meta), tmp_path)
    tmp_path.replace(shard_path)

    return {
        "shard_id": shard.shard_id,
        "path": str(shard_path),
        "num_samples": len(data_list),
        "num_failures": len(failures),
    }


def compute_entry(
    config: DataGenConfig,
    cid: str,
    family: str,
    n_qubits: int,
    n_layers: int,
    seed: int,
    sampling_config: SamplingConfig | None = None,
) -> dict[str, Any] | None:
    from experiments.core import ExperimentConfig
    from properties.compute import PropertyRequest
    from states.types import BackendConfig

    try:
        base_dir = config.output_dir or DATASET_DIR
        base_dir.mkdir(parents=True, exist_ok=True)
        path = base_dir / f"{cid}.pt"
        tmp_path = path.with_suffix(".pt.tmp")

        if path.exists():
            return {"cid": cid, "path": str(path), "cached": True}

        controls = sample_generation_controls(
            family=family,
            n_layers=int(n_layers),
            seed=int(seed),
            sampling_config=sampling_config,
        )

        make_spec_kwargs = {
            "d": 2,
            "seed": int(seed),
        }
        family_cls = FAMILY_REGISTRY[family]
        family_obj = family_cls()

        if family == "clifford":
            make_spec_kwargs["tdoping"] = controls["tdoping"]

        elif family == "random":
            make_spec_kwargs["angle_regime"] = controls["angle_regime"]
            make_spec_kwargs["angle_scale"] = controls.get("angle_scale")
            make_spec_kwargs["gate_probability"] = controls["gate_probability"]

        elif family == "haar":
            make_spec_kwargs["gate_probability"] = controls["gate_probability"]
            make_spec_kwargs["haar_probability"] = controls["haar_probability"]
            make_spec_kwargs["haar_strength"] = controls["haar_strength"]
            make_spec_kwargs["haar_mode"] = controls["haar_mode"]

        elif family == "quansistor":
            make_spec_kwargs["param_regime"] = controls.get("sampling_regime")
            make_spec_kwargs["param_scale"] = controls.get("param_scale")
            make_spec_kwargs["gate_probability"] = controls.get("gate_probability")


        spec = family_obj.make_spec(
            int(n_qubits),
            int(n_layers),
            **make_spec_kwargs,
        )

        gates = cast("tuple[GateSpec, ...] | None", spec.gates)
        qasm = to_qasm(spec, gates)
        op_descriptors = build_op_descriptors_from_spec(gates, family)

        graph_data, gate_counts = qasm_to_pyg_graph(
            qasm_str=qasm,
            n_bins=config.n_bins,
            family=family,
            global_feature_variant="binned",
            op_descriptors=op_descriptors,
        )

        sre_value = None
        EE_value = None

        should_compute_target = int(n_qubits) in set(config.target_qubits)

        property_requests = []

        if should_compute_target and config.compute_sre:
            property_requests.append(PropertyRequest(name="SRE", method=config.method, params={}))

        if should_compute_target and config.compute_EE:
            property_requests.append(PropertyRequest(name="entanglement_entropy", method=config.method, params={}))

        if property_requests:
            backend_config = BackendConfig(
                name=config.backend,
                representation=config.representation,
                params={},
            )

            exp_config = ExperimentConfig(
                spec=spec,
                backend=backend_config,
                properties=property_requests,
            )

            result = run_experiment(
                exp_config,
                backend_registry=BACKEND_REGISTRY,
                cache=cache,
            )
        if should_compute_target and config.compute_sre:
            sre_key = f"SRE:{config.method.lower()}"
            sre_result = result.results.get(sre_key)
            sre_value = float(sre_result.value) if sre_result else None

        if should_compute_target and config.compute_EE:
            ee_key = f"entanglement_entropy:{config.method.lower()}"
            ee_result = result.results.get(ee_key)
            EE_value = float(ee_result.value) if ee_result else None

        x = graph_data.x.detach().cpu()
        if x.numel() > 0 and x.min().item() >= 0 and x.max().item() <= 1:
            x = x.to(torch.uint8)
        else:
            x = x.to(torch.float16)

        edge_index = graph_data.edge_index.detach().cpu().to(torch.int64)
        global_features = graph_data.global_features.detach().cpu().to(torch.float64)

        payload = {
            "x": x,
            "edge_index": edge_index,
            "global_features": global_features,
            "gate_counts": sanitize_gate_counts(gate_counts),
            "meta": {
                "cid": cid,
                "family": family,
                "n_qubits": int(n_qubits),
                "n_layers": int(n_layers),
                "seed": int(seed),
                "has_target": bool(should_compute_target),
                "target_qubits": list(config.target_qubits),
                "backend": config.backend,
                "method": config.method,
                "representation": config.representation,
                "n_bins": int(config.n_bins),
                "regime": controls["sampling_regime"],
                "sampling_controls": {
                    k: str(v) if isinstance(v, TdopingRules) else v
                    for k, v in controls.items()
                },
            },
        }

        if should_compute_target and config.compute_sre:
            payload["sre"] = float(sre_value) if sre_value is not None else float("nan")

        if should_compute_target and config.compute_EE:
            payload["ee"] = float(EE_value) if EE_value is not None else float("nan")


        return payload

    except Exception:
        logger.exception(
            "Error computing entry for family=%s Q%s L%s S%s",
            family,
            n_qubits,
            n_layers,
            seed,
        )
        return None

    finally:
        trim_memory()


def compute_all_entries(
    params: list[CircuitConfig],
    config: DataGenConfig,
    *,
    use_dask: bool = False,
    dask_n_workers: int = 4,
    dask_memory_per_worker: str | None = "auto",
    sampling_config: SamplingConfig | None = None,
) -> list[dict[str, Any]]:
    if use_dask:
        return compute_all_entries_parallel(
            params,
            config,
            dask_n_workers=dask_n_workers,
            dask_memory_per_worker=dask_memory_per_worker,
            sampling_config=sampling_config,
        )
    return compute_all_entries_sequential(params, config, sampling_config=sampling_config)

def compute_all_shards(
    shards: list[ShardConfig],
    config: DataGenConfig,
    *,
    use_dask: bool = False,
    dask_n_workers: int = 4,
    dask_memory_per_worker: str | None = "auto",
    sampling_config: SamplingConfig | None = None,
) -> list[dict[str, Any]]:
    if use_dask:
        return compute_all_shards_parallel(
            shards,
            config,
            dask_n_workers=dask_n_workers,
            dask_memory_per_worker=dask_memory_per_worker,
            sampling_config=sampling_config,
        )
    return compute_all_shards_sequential(shards, config, sampling_config=sampling_config)

def compute_all_entries_sequential(
    params: list[CircuitConfig],
    config: DataGenConfig,
    sampling_config: SamplingConfig | None = None,
) -> list[dict[str, Any]]:
    base_dir = config.output_dir or DATASET_DIR
    base_dir.mkdir(parents=True, exist_ok=True)

    family = params[0].family if params else "unknown"
    index_path = base_dir / f"index_{family}.jsonl"

    entries: list[dict[str, Any]] = []

    with index_path.open("a", encoding="utf-8") as f:
        for row in tqdm(params, desc="Computing dataset entries"):
            entry = compute_entry(
                config,
                cid=row.cid,
                family=row.family,
                n_qubits=row.n_qubits,
                n_layers=row.n_layers,
                seed=row.seed,
                sampling_config=sampling_config,
            )

            if entry is None:
                continue

            f.write(json.dumps(entry) + "\n")
            f.flush()
            entries.append(entry)

            if config.compute_sre:
                logger.info("Computed %s: SRE=%s", row.cid, entry.get("sre"))
            else:
                logger.info("Computed %s", row.cid)

    return entries

def compute_all_shards_sequential(
    shards: list[ShardConfig],
    config: DataGenConfig,
    sampling_config: SamplingConfig | None = None,
) -> list[dict[str, Any]]:
    base_dir = config.output_dir or DATASET_DIR
    base_dir.mkdir(parents=True, exist_ok=True)

    family = shards[0].family if shards else "unknown"
    index_path = base_dir / f"index_{family}.jsonl"

    if index_path.exists():
        index_path.unlink()
    entries: list[dict[str, Any]] = []

    with index_path.open("w", encoding="utf-8") as f:
        for shard in tqdm(shards, desc="Computing dataset entries"):
            entry = compute_pyg_shard(
                shard=shard,
                config=config,
                sampling_config=sampling_config,
            )

            if entry is None:
                continue

            entries.append(entry)

            shard_data, shard_slice, shard_meta = torch.load(
                entry["path"],
                map_location="cpu",
                weights_only=False,
            )

            for row in shard_meta.get("index_rows", []):
                f.write(json.dumps(row) + "\n")
            f.flush()

            logger.info(
                "Computed shard %s: %s samples, %s failures",
                entry.get("shard_id"),
                entry.get("num_samples"),
                entry.get("num_failures"),
            )

    return entries

def compute_all_entries_parallel(
    params: list[CircuitConfig],
    config: DataGenConfig,
    dask_n_workers: int,
    dask_memory_per_worker: str | None = "auto",
    sampling_config: SamplingConfig | None = None,
) -> list[dict[str, Any]]:
    from dask.distributed import as_completed

    from parallel import dask_client

    base_dir = config.output_dir or DATASET_DIR
    base_dir.mkdir(parents=True, exist_ok=True)

    family = params[0].family if params else "unknown"
    index_path = base_dir / f"index_{family}.jsonl"

    cpu_count = os.cpu_count() or 2
    safe_workers = max(1, min(int(dask_n_workers), cpu_count))
    # Increase max_inflight to better utilize all workers
    # Use 3x workers to keep pipeline full while managing memory
    max_inflight = max(safe_workers, safe_workers * 3)

    rows_out: list[dict[str, Any]] = []

    with dask_client(
        mode="local",
        n_workers=safe_workers,
        threads_per_worker=1,
        memory_per_worker=dask_memory_per_worker or "2GB",
        dashboard=True,
        walltime="0-2:30:00",
    ) as client:
        client.wait_for_workers(1)

        ac = as_completed()
        inflight = {}
        it = iter(params)

        def submit_one(row: CircuitConfig) -> None:
            fut = client.submit(
                compute_entry,
                config,
                cid=row.cid,
                family=row.family,
                n_qubits=row.n_qubits,
                n_layers=row.n_layers,
                seed=row.seed,
                sampling_config=sampling_config,
                pure=False,
            )
            inflight[fut] = row
            ac.add(fut)

        for _ in range(min(max_inflight, len(params))):
            row = next(it, None)
            if row is None:
                break
            submit_one(row)

        with index_path.open("a", encoding="utf-8") as f:
            for fut in tqdm(ac, total=len(params), desc="Parallel dataset generation"):
                row = inflight.pop(fut, None)

                try:
                    entry = fut.result()
                    if entry is not None:
                        f.write(json.dumps(entry) + "\n")
                        f.flush()
                        rows_out.append(entry)
                except Exception as exc:
                    cid = row.cid if row else "unknown"
                    logger.error("Failed (%s): %s", cid, exc)
                finally:
                    fut.release()

                next_row = next(it, None)
                if next_row is not None:
                    submit_one(next_row)

    return rows_out

def compute_all_shards_parallel(
    shards: list[ShardConfig],
    config: DataGenConfig,
    dask_n_workers: int,
    dask_memory_per_worker: str | None = "auto",
    sampling_config: SamplingConfig | None = None,
) -> list[dict[str, Any]]:
    from dask.distributed import as_completed

    from parallel import dask_client

    base_dir = config.output_dir or DATASET_DIR
    base_dir.mkdir(parents=True, exist_ok=True)

    family = shards[0].family if shards else "unknown"
    index_path = base_dir / f"index_{family}.jsonl"

    cpu_count = os.cpu_count() or 2
    safe_workers = max(1, min(int(dask_n_workers), cpu_count))
    max_inflight = max(safe_workers, safe_workers * 3)

    rows_out: list[dict[str, Any]] = []

    with dask_client(
        mode="local",
        n_workers=safe_workers,
        threads_per_worker=1,
        memory_per_worker=dask_memory_per_worker or "2GB",
        dashboard=True,
        walltime="0-2:30:00",
    ) as client:
        client.wait_for_workers(1)

        ac = as_completed()
        inflight = {}
        it = iter(shards)

        def submit_one(shard: ShardConfig) -> None:
            fut = client.submit(
                compute_pyg_shard,
                shard,
                config,
                sampling_config=sampling_config,
                pure=False,
            )
            inflight[fut] = shard
            ac.add(fut)

        for _ in range(min(max_inflight, len(shards))):
            shard = next(it, None)
            if shard is None:
                break
            submit_one(shard)

        with index_path.open("w", encoding="utf-8") as f:
            for fut in tqdm(ac, total=len(shards), desc="Parallel dataset generation"):
                shard = inflight.pop(fut, None)

                try:
                    entry = fut.result()
                    if entry is not None:
                        rows_out.append(entry)

                        shard_data, shard_slice, shard_meta = torch.load(
                            entry["path"],
                            map_location="cpu",
                            weights_only=False,
                        )

                        for row in shard_meta.get("index_rows", []):
                            f.write(json.dumps(row) + "\n")
                        f.flush()

                except Exception as exc:
                    sid = shard.shard_id if shard else "unknown"
                    logger.exception("Failed (%s): %s", sid, exc)
                finally:
                    fut.release()

                next_shard = next(it, None)
                if next_shard is not None:
                    submit_one(next_shard)

    return rows_out

def run_dataset_pipeline(
    *,
    config: DataGenConfig,
    families: list[str],
    qubits_values: np.ndarray,
    layers_values: np.ndarray,
    n_seeds: int,
    use_dask: bool = False,
    max_shards: int | None = None,
    dask_n_workers: int = 4,
    dask_memory_per_worker: str | None = None,
    sampling_config: SamplingConfig | None = None,
    layer_block_size: int = 10,
    use_sharded: bool = True,
) -> None:
    invalid = [f for f in families if f not in FAMILY_REGISTRY]
    if invalid:
        raise ValueError(
            f"Unknown families: {invalid}. Valid: {list(FAMILY_REGISTRY.keys())}",
        )
    


    base_output_dir: Path = config.output_dir
    base_output_dir.mkdir(parents=True, exist_ok=True)

    for family in families:
        logger.info("Processing family: %s", family)

        family_output_dir = base_output_dir / family
        family_output_dir.mkdir(parents=True, exist_ok=True)

        if use_sharded:
            shards = generate_pyg_shard(
                [family],
                qubits_values,
                layers_values,
                n_seeds,
                layer_blocks_size=layer_block_size,
            )

            if max_shards is not None:
                shards = shards[:max_shards]

            logger.info("Generated %d shards for %s", len(shards), family)

            family_config = dataclasses.replace(config, output_dir=family_output_dir)

            entries = compute_all_shards(
                shards,
                family_config,
                use_dask=use_dask,
                dask_n_workers=dask_n_workers,
                dask_memory_per_worker=dask_memory_per_worker,
                sampling_config=sampling_config,
            )
        else:
            params = generate_dataset_params(
                [family],
                qubits_values,
                layers_values,
                n_seeds,
            )

            logger.info("Generated %d entries for %s", len(params), family)

            family_config = dataclasses.replace(config, output_dir=family_output_dir)

            entries = compute_all_entries(
                params,
                family_config,
                use_dask=use_dask,
                dask_n_workers=dask_n_workers,
                dask_memory_per_worker=dask_memory_per_worker,
                sampling_config=sampling_config,
            )

        meta_path = family_output_dir / f"metadata_{family}.json"
        metadata = {
            "format": "qqe_pyg_inmemory_shard_v1",
            "backend": family_config.backend,
            "method": family_config.method,
            "n_bins": family_config.n_bins,
            "n_seeds": n_seeds,
            "families": families,
            "qubits_range": qubits_values.tolist(),
            "layers_range": layers_values.tolist(),
            "entries_completed": len(entries),
            "use_dask": use_dask,
            "compute_sre": family_config.compute_sre,
            "compute_EE": family_config.compute_EE,
            "index_file": f"index_{family}.jsonl",
            "processed_dir": "processed",
            "layer_block_size": int(layer_block_size),
        }

        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        logger.info("Completed %s: %d entries", family, len(entries))
        logger.info("Metadata: %s", meta_path)
        logger.info("Samples: %s", family_output_dir)

    logger.info("All families processed successfully")
