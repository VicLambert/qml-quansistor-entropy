
"""PennyLane backend for quantum circuit simulation.

This module provides the PennylaneBackend class, which leverages PennyLane's
quantum computing framework to simulate quantum circuits with support for
various quantum devices through PennyLane's device interface.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pennylane as qml

from src.backend.base import BaseBackend, State, StateType
from src.circuit.matrix_factory import gate_unitary
from src.circuit.spec import CircuitSpec
from src.states.types import DenseState


class PennylaneBackend(BaseBackend):
    """PennylaneBackend for quantum circuit simulation using PennyLane.

    This backend leverages PennyLane's quantum computing framework to simulate
    quantum circuits. It provides support for various quantum devices through
    PennyLane's device interface.

    Attributes:
        name (str): The name identifier for this backend, set to "pennylane".
    """
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
        """Simulate a quantum circuit using PennyLane.

        Parameters
        ----------
        spec : CircuitSpec
            The circuit specification to simulate.
        state_type : StateType, optional
            The type of state representation, must be "dense" (default).
        max_bond : int | None, optional
            Maximum bond dimension (not used in this backend).
        cutoff : int | None, optional
            Cutoff parameter (not used in this backend).
        **kwargs : Any
            Additional keyword arguments (unused).

        Returns:
        -------
        State
            The resulting quantum state as a DenseState.

        Raises:
        ------
        NotImplementedError
            If state_type is not "dense".
        """
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
