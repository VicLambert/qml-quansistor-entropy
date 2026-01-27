
"""
Example script demonstrating circuit visualization with real backends.
"""

from pathlib import Path

from src.circuit.spec import CircuitSpec
from src.circuit.families.clifford import CliffordBrickwork
from src.circuit.families.haar import HaarBrickwork
from src.circuit.families.quansistor import QuansistorBrickwork
from src.backend.pennylane_backend import PennylaneBackend
from src.backend.quimb_backend import QuimbBackend

from src.circuit.families.pattern.tdoping import TdopingRules

from src.experiments.visualizer import (
    plot_circuit_diagram,
    plot_state_probabilities_dense,
    plot_pennylane_circuit,
)


OUT = Path("example_outputs")
OUT.mkdir(exist_ok=True)
from dataclasses import replace

def materialize_spec(spec: CircuitSpec, family) -> CircuitSpec:
    return replace(spec, gates=tuple(family.gates(spec)))

def example_pennylane():
    print("Example 1: PennyLane Clifford Circuit")
    print("-" * 50)

    # family = CliffordBrickwork(tdoping=TdopingRules(
    #     count=48,
    #     placement="center_pair",
    #     per_layer=2,
    # ))
    # family = HaarBrickwork()
    family = QuansistorBrickwork()

    spec = family.make_spec(
        n_qubits=6,
        n_layers=25,
        d=2,
        seed=42,
    )
    spec = materialize_spec(spec, family)
    # IMPORTANT: materialize gates
    # spec.gates = list(family.gates(spec))

    backend = PennylaneBackend()
    # state = backend.simulate(spec)

    plot_pennylane_circuit(
        spec,
        save_path=OUT / "pennylane_quansistor_circuit.png",
    )
    # plot_circuit_diagram(
    #     spec,
    #     title="PennyLane Clifford Circuit",
    #     save_path=OUT / "pennylane_circuit.png",
    # )

    # plot_state_probabilities_dense(
    #     state,
    #     top_k=10,
    #     save_path=OUT / "pennylane_state.png",
    # )

    print("Saved PennyLane plots.\n")


def example_quimb_dense():
    print("Example 2: Quimb Clifford Circuit (Dense)")
    print("-" * 50)

    family = CliffordBrickwork(tdoping=TdopingRules(
        count=4,
        placement="center_pair",
        per_layer=2,
    ))

    spec = family.make_spec(
        n_qubits=4,
        n_layers=6,
        d=2,
        seed=42,
    )

    spec = materialize_spec(spec, family)

    backend = QuimbBackend()
    state = backend.simulate(spec, state_type="dense")

    plot_circuit_diagram(
        spec,
        title="Quimb Clifford Circuit (Dense)",
        save_path=OUT / "quimb_dense_circuit.png",
    )

    plot_state_probabilities_dense(
        state,
        top_k=10,
        save_path=OUT / "quimb_dense_state.png",
    )

    print("Saved Quimb dense plots.\n")


def example_quimb_mps():
    print("Example 3: Quimb Clifford Circuit (MPS)")
    print("-" * 50)

    family = CliffordBrickwork(tdoping=0.2)

    spec = family.make_spec(
        n_qubits=8,
        n_layers=4,
        d=2,
        seed=456,
    )

    spec.gates = list(family.gates(spec))

    backend = QuimbBackend()
    state = backend.simulate(
        spec,
        state_type="mps",
        max_bond=64,
    )

    plot_circuit_diagram(
        spec,
        title="Quimb Clifford Circuit (MPS)",
        save_path=OUT / "quimb_mps_circuit.png",
    )

    # Visualization module works on DenseState,
    # so explicitly convert
    plot_state_probabilities_dense(
        state.mps.to_dense(),
        top_k=10,
        save_path=OUT / "quimb_mps_state.png",
    )

    print("Saved Quimb MPS plots.\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Quantum Circuit Visualization Examples")
    print("=" * 60)

    example_pennylane()
    # example_quimb_dense()
    # example_quimb_mps()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)