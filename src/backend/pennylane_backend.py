
from __future__ import annotations

from functools import singledispatchmethod
from typing import Any, Literal

import numpy as np
import quimb as qb
import quimb.tensor as qtn

from backend.base import BaseBackend
from circuit.families.gates import haar_unitary_gate, random_quansistor_gate
from circuit.spec import CircuitSpec, GateSpec
from states.types import DenseState, MPSState

import pennylane as qml

class PennylaneBackend(BaseBackend):
    name = "pennylane"

    def __init__(self, device_name: str = "lightning.qubit"):
        """Initialize the PennylaneBackend."""
        self.device_name = device_name
        self.dev = qml.device(self.device_name, wires=0)  # Placeholder, will be updated in simulate

    def _apply_gate(self, gate: GateSpec, d: int = 2):
        kind = gate.kind
        wire = gate.wires

        return self._apply_gate_dispatch(kind, wire, gate, d)

    @singledispatchmethod
    def _apply_gate_dispatch(self, kind: str, wire: tuple[int, ...], gate: GateSpec, d: int):
        msg = f"Gate kind '{kind}' is not implemented in PennylaneBackend."
        raise NotImplementedError(msg)

    @_apply_gate_dispatch.register
    def _(self, kind: Literal["T"], wire: tuple[int], gate: GateSpec, d: int):
        qml.T(wires=wire[0])

    @_apply_gate_dispatch.register
    def _(self, kind: Literal["haar"], wire: tuple[int, int], gate: GateSpec, d: int):
        seed = gate.seed
        rng = np.random.default_rng(seed)
        U = haar_unitary_gate(d=d**2, rng=rng)
        qml.QubitUnitary(U, wires=wire)

    @_apply_gate_dispatch.register
    def _(self, kind: Literal["clifford"], wire: tuple[int, int], gate: GateSpec, d: int):
        seed = gate.seed
        rng = np.random.default_rng(seed)
        oneq_cliffords = ["I", "H", "S"]
        a, b = wire
        U_a, U_b = rng.choice(oneq_cliffords, size=2)
        qml.QubitUnitary(U_a, wires=a)
        qml.QubitUnitary(U_b, wires=b)
        qml.CNOT(wires=[a, b])

    @_apply_gate_dispatch.register
    def _(self, kind: Literal["quansistor"], wire: tuple[int, int], gate: GateSpec, d: int):
        seed = gate.seed
        rng = np.random.default_rng(seed)
        U = random_quansistor_gate(rng)
        qml.QubitUnitary(U, wires=wire)

    def simulate(
        self,
        spec: CircuitSpec,
        state_type: str = "dense",
        max_bond: int | None = None,
        **kwargs: Any,
    ):
        self.dev = qml.device(self.device_name, wires=spec.n_qubits)

        @qml.qnode(self.dev)
        def circuit():
            for gate in spec.gates:
                self._apply_gate(gate, d=spec.d)
            return qml.state()

        vec = circuit()
        vec = np.asarray(vec).reshape(-1)
        return DenseState(vector=vec, n_qubits=spec.n_qubits, d=spec.d, backend=self.name)
