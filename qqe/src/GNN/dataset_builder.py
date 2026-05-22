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
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from tqdm import tqdm

from qqe.src.backend import PennylaneBackend, QuimbBackend
from qqe.src.circuit.families import (
    CliffordBrickwork,
    HaarBrickwork,
    QuansistorBrickwork,
    RandomCircuit,
)

# Keep this import path only if it is the correct one in your project.
# If your real function lives in circuit.matrix_factory, switch it there.
from qqe.src.circuit.gates import gate_unitary
from qqe.src.circuit.patterns import TdopingRules, to_qasm
from qqe.src.experiments.core import run_experiment
from qqe.src.GNN.encoder import eigenvalue_phase_histogram_features, qasm_to_pyg_graph
from qqe.src.utils import FileCache

if TYPE_CHECKING:
    from qqe.src.circuit.spec import GateSpec

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
        regimes = ["zero", "low", "medium", "high"],
        probabilities = [0.15, 0.35, 0.30, 0.20],
    )
    haar: RegimeDistribution = RegimeDistribution(
        regimes = ["none", "sparse_weak", "dense_weak", "sparse_full", "medium", "full"],
        probabilities = [0.10, 0.20, 0.20, 0.15, 0.20, 0.15],
    )
    random: RegimeDistribution = RegimeDistribution(
        regimes = ["identity_like", "clifford_like", "small_angles", "generic"],
        probabilities = [0.15, 0.20, 0.30, 0.35],
    )
    quansistor: RegimeDistribution = RegimeDistribution(
        regimes = ["identity_like", "weak", "moderate", "structured_equal_ab", "structured_opposite_ab", "generic_uniform"],
        probabilities = [0.10, 0.20, 0.25, 0.15, 0.15, 0.15],
    )

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


def sanitize_gate_counts(gate_counts: dict[str, int]) -> dict[str, int]:
    if not isinstance(gate_counts, dict):
        return {}
    safe: dict[str, int] = {}
    for key, value in gate_counts.items():
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
    max_t = 2 * n_layers

    if regime == "zero":
        N_T: int = 0

    elif regime == "low":
        N_T = int(rng.integers(0, max(2, max_t // 8 + 1)))

    elif regime == "medium":
        N_T = int(rng.integers(max(10, max_t // 8), max(2, max_t // 4 + 1)))

    elif regime == "high":
        N_T = int(rng.integers(max(10, max_t // 4), max_t + 1))

    else:
        raise ValueError(f"Unknown t-count regime: {regime}")

    return (regime, N_T)

def sample_haar_controls(rng: np.random.Generator, regime: str) -> dict[str, Any]:
    if regime == "none":
        p_haar = 0.0
        haar_strength = 0.0
        haar_mode = "identity"

    elif regime == "sparse_weak":
        p_haar = rng.uniform(0.05, 0.25)
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
        p_haar = rng.uniform(0.25, 0.70)
        haar_strength = rng.uniform(0.20, 0.80)
        haar_mode = "exp_hermitian"
    elif regime == "full":
        p_haar = rng.uniform(0.70, 1.0)
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
        angle_scale = 0.02

    elif regime == "clifford_like":
        gate_probability = float(rng.uniform(0.20, 0.60))
        angle_scale = 0.02

    elif regime == "small_angles":
        gate_probability = float(rng.uniform(0.30, 0.80))
        angle_scale = 0.25

    elif regime == "generic":
        gate_probability = float(rng.uniform(0.60, 1.00))
        angle_scale = None

    else:
        raise ValueError(f"Unknown random regime={regime}")

    return {
        "sampling_regime": regime,
        "gate_probability": gate_probability,
        "angle_regime": regime,
        "angle_scale": angle_scale,
    }

def sample_quansistor_controls(rng: np.random.Generator, regime: str) -> dict[str, Any]:
    if regime == "identity_like":
        gate_probability = float(rng.uniform(0.00, 0.10))
        param_scale = 0.02
    elif regime == "weak":
        gate_probability = float(rng.uniform(0.10, 0.40))
        param_scale = 0.20
    elif regime == "moderate":
        gate_probability = float(rng.uniform(0.40, 0.85))
        param_scale = 0.75
    elif regime == "structured_equal_ab" or regime == "structured_opposite_ab":
        gate_probability = float(rng.uniform(0.40, 0.80))
        param_scale = 1.00
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

def compute_entry(
    config: DataGenConfig,
    cid: str,
    family: str,
    n_qubits: int,
    n_layers: int,
    seed: int,
    sampling_config: SamplingConfig | None = None,
) -> dict[str, Any] | None:
    from qqe.src.experiments.core import ExperimentConfig
    from qqe.src.properties.compute import PropertyRequest
    from qqe.src.states.types import BackendConfig

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

        edge_index = graph_data.edge_index.detach().cpu().to(torch.int32)
        global_features = graph_data.global_features.detach().cpu().to(torch.float32)

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

        torch.save(payload, tmp_path)
        tmp_path.replace(path)

        return {
            "cid": cid,
            "path": str(path),
            **({"sre": sre_value} if config.compute_sre else {}),
            **({"ee": EE_value} if config.compute_EE else {}),
        }

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


def compute_all_entries_parallel(
    params: list[CircuitConfig],
    config: DataGenConfig,
    dask_n_workers: int,
    dask_memory_per_worker: str | None = "auto",
    sampling_config: SamplingConfig | None = None,
) -> list[dict[str, Any]]:
    from dask.distributed import as_completed

    from qqe.src.parallel import dask_client

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


def run_dataset_pipeline(
    *,
    config: DataGenConfig,
    families: list[str],
    qubits_values: np.ndarray,
    layers_values: np.ndarray,
    n_seeds: int,
    use_dask: bool = False,
    max_configs: int | None = None,
    dask_n_workers: int = 4,
    dask_memory_per_worker: str | None = None,
    sampling_config: SamplingConfig | None = None,
) -> None:
    invalid = [f for f in families if f not in FAMILY_REGISTRY]
    if invalid:
        raise ValueError(
            f"Unknown families: {invalid}. Valid: {list(FAMILY_REGISTRY.keys())}",
        )

    base_output_dir: Path = config.output_dir
    base_output_dir.mkdir(parents=True, exist_ok=True)
    target = "sre" if config.compute_sre else "ee" if config.compute_EE else None

    # data_dir = f"training_data_{target}_{config.backend}" if target else "prediction_data"

    for family in families:
        logger.info("Processing family: %s", family)

        family_output_dir = base_output_dir / family
        family_output_dir.mkdir(parents=True, exist_ok=True)

        params = generate_dataset_params(
            [family],
            qubits_values,
            layers_values,
            n_seeds,
        )

        if max_configs is not None:
            params = params[:max_configs]

        logger.info("Generated %d configs for %s", len(params), family)

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
            "index_file": f"index_{family}.jsonl",
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        logger.info("Completed %s: %d entries", family, len(entries))
        logger.info("Metadata: %s", meta_path)
        logger.info("Samples: %s", family_output_dir)

    logger.info("All families processed successfully")
