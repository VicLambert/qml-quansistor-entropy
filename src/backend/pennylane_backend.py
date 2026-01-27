
from __future__ import annotations

from functools import singledispatchmethod
from typing import Any, Literal

import numpy as np
import quimb as qb
import quimb.tensor as qtn

from backend.base import BaseBackend, StateType, State
from circuit.matrix_factory import gate_unitary
from circuit.spec import CircuitSpec, GateSpec
from states.types import DenseState, MPSState

import pennylane as qml

class PennylaneBackend(BaseBackend):
    name = "pennylane"

    def __init__(self, device_name: str = "lightning.qubit"):
        """Initialize the PennylaneBackend."""
        self.device_name = device_name

    def simulate(
        self,
        spec: CircuitSpec,
        *,
        state_type: StateType = "dense",
        max_bond: int | None = None,
        cutoff: int | None = None,
        **kwargs: Any,
    ) -> State:
        if state_type != "dense":
            msg = f"PennylaneBackend supports state_type='dense' only (requested {state_type!r})."
            raise NotImplementedError(msg)

        dev = qml.device(self.device_name, wires=spec.n_qubits)

        @qml.qnode(dev)
        def circuit():
            for gate in spec.gates:
                U = gate_unitary(gate)
                qml.QubitUnitary(U, wires=gate.wires)
            return qml.state()

        vec = np.asarray(circuit(), dtype=complex).reshape(-1)
        return DenseState(vector=vec, n_qubits=spec.n_qubits, d=spec.d, backend=self.name)
