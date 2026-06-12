from __future__ import annotations

import logging

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
import typer

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.visualization import dag_drawer

from circuit.patterns import to_qasm
from GNN.dataset_builder import (
    FAMILY_REGISTRY,
    RegimeDistribution,
    SamplingConfig,
    build_op_descriptors_from_spec,
    sample_generation_controls,
)
from GNN.encoder import qasm_to_pyg_graph
from utils import configure_logger

if TYPE_CHECKING:
    from circuit.spec import GateSpec

logger = logging.getLogger(__name__)

@dataclass
class SimulationConfig:
    n_qubits: int
    n_layers: int
    backend: str
    n_bins: int = 50

sampling_config_saturated = SamplingConfig(
        clifford=RegimeDistribution(
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
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        ),
        random=RegimeDistribution(
            regimes=[
                "identity_like",
                "near_clifford",
                "small_angles",
                "medium_angles",
                "generic_sparse",
                "generic_dense",
            ],
            probabilities=[0.00, 0.0, 0., 0.0, 0.0, 1.0],
        ),
        quansistor=RegimeDistribution(
            regimes=[
                "identity_like",
                "weak",
                "moderate",
                "structured_equal_ab",
                "structured_opposite_ab",
                "generic_uniform",
            ],
            probabilities=[0., 0., 0., 0., 0., 1.0],
        ),
        haar=RegimeDistribution(
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
            probabilities=[0.0, 0.0, 0.0, 0., 0., 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ),
    )

sampling_config_identity_like = SamplingConfig(
        clifford=RegimeDistribution(
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
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        ),
        random=RegimeDistribution(
            regimes=[
                "identity_like",
                "near_clifford",
                "small_angles",
                "medium_angles",
                "generic_sparse",
                "generic_dense",
            ],
            probabilities=[1.00, 0.0, 0., 0.0, 0.0, 0.0],
        ),
        quansistor=RegimeDistribution(
            regimes=[
                "identity_like",
                "weak",
                "moderate",
                "structured_equal_ab",
                "structured_opposite_ab",
                "generic_uniform",
            ],
            probabilities=[1., 0., 0., 0., 0., 0.0],
        ),
        haar=RegimeDistribution(
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
            probabilities=[1.0, 0.0, 0.0, 0., 0., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
    )

def main(
    family: str = typer.Option("random", help="Circuit family to simulate."),
    n_qubits: int = typer.Option(4, help="Number of qubits in the circuit."),
    n_layers: int = typer.Option(4, help="Number of layers in the circuit."),
    backend: str = typer.Option("pennylane", help="Simulation backend to use."),
    seed = typer.Option(42, help="Random seed for reproducibility."),
    regime_type: str = typer.Option("saturated", help="Regime type to sample from (default, balanced)"),
):
    """Simulate quantum circuits and saves the circuit plot and the circuit DAG."""
    seed = seed
    selected_families = [f.strip() for f in family.split(",") if f.strip()]
    config = SimulationConfig(
        n_qubits=n_qubits,
        n_layers=n_layers,
        backend=backend,
    )
    logger.info(f"Simulating circuit family: {family}")

    if regime_type == "saturated":
        sampling_config = sampling_config_saturated
    elif regime_type == "identity_like":
        sampling_config = sampling_config_identity_like

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

    qc = QuantumCircuit.from_qasm_str(qasm)
    dag = circuit_to_dag(qc)

    img = dag_drawer(dag)
    return img, qc, graph_data, gate_counts, spec


if __name__ == "__main__":
    configure_logger(console_level=logging.INFO, file_level=logging.INFO)
    typer.run(main)