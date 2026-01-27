
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


OUT = Path("circuit_outputs")
OUT.mkdir(exist_ok=True)
from dataclasses import replace

def materialize_spec(spec: CircuitSpec, family) -> CircuitSpec:
    return replace(spec, gates=tuple(family.gates(spec)))

_FAMILIES = {
    "clifford": CliffordBrickwork(tdoping=TdopingRules(
        count=48,
        placement="center_pair",
        per_layer=2,
    )),
    "haar": HaarBrickwork(),
    "quansistor": QuansistorBrickwork()
}

def example_pennylane():
    """Visulize circuits to test proper implementations."""
    

    type = "clifford"
    print(f"Example 1: PennyLane {type} Circuit")
    print("-" * 50)
    family = _FAMILIES[type]

    spec = family.make_spec(
        n_qubits=10,
        n_layers=25,
        d=2,
        seed=42,
    )
    spec = materialize_spec(spec, family)

    path_name = str(OUT / f"pennylane_{type}_circuit.png")
    plot_pennylane_circuit(
        spec,
        save_path=path_name,
    )

    print("Saved PennyLane plots.\n")


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