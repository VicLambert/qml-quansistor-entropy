
"""Quimb tensor network backend for quantum circuit simulation.

This module provides the QuimbBackend class that implements quantum circuit
simulation using the quimb library, supporting both dense and MPS state types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import quimb.tensor as qtn

from qqe.backend.base import BaseBackend, State
from qqe.circuit.matrix_factory import gate_unitary
from qqe.states.types import DenseState, MPSState

if TYPE_CHECKING:
    from qqe.circuit.spec import CircuitSpec, GateSpec


class QuimbBackend(BaseBackend):
    """Quimb tensor network backend for quantum circuit simulation.

    Supports both dense and MPS state types for quantum circuit simulation
    using the quimb library.

    Attributes:
    ----------
    name : str
        The name of the backend ("quimb").
    """
    name = "quimb"

    def _make_circuit(
        self,
        n_qubits: int,
        *,
        state_type: str,
        max_bond: int | None = None,
        cutoff: int | None = None,
    ):
        if state_type == "mps":
            kwargs: dict[str, Any] = {}
            if max_bond is not None:
                kwargs["max_bond"] = max_bond
            if cutoff is not None:
                kwargs["cutoff"] = cutoff
            return qtn.CircuitMPS(n_qubits, **kwargs)
        if state_type == "dense":
            return qtn.Circuit(n_qubits)
        msg = f"QuimbBackend does not support state_type={state_type!r}"
        raise NotImplementedError(msg)

    def _apply_gate(self, circ: qtn.Circuit, gate: GateSpec) -> None:
        U = gate_unitary(gate)
        circ.apply_gate(U, *gate.wires)

    def simulate(
        self,
        spec: CircuitSpec,
        *,
        state_type: str = "dense",
        max_bond: int | None = None,
        cutoff: int | None = None,
        **kwargs: Any,
    ) -> State:
        """Simulate a quantum circuit using quimb.

        Parameters
        ----------
        spec : CircuitSpec
            The circuit specification to simulate.
        state_type : str, optional
            The state type to use ("dense" or "mps"), by default "dense".
        max_bond : int | None, optional
            Maximum bond dimension for MPS, by default None.
        cutoff : int | None, optional
            Cutoff for MPS truncation, by default None.
        **kwargs : Any
            Additional keyword arguments.

        Returns:
        -------
        State
            The resulting quantum state after simulation.

        Raises:
        ------
        NotImplementedError
            If the state_type is not supported.
        """
        self._validate_materialized(spec)

        circ = self._make_circuit(
            n_qubits=spec.n_qubits,
            state_type=state_type,
            max_bond=max_bond,
            cutoff=cutoff,
        )

        for gate in spec.gates:
            self._apply_gate(circ, gate)

        psi = circ.psi

        if state_type == "dense":
            vec = psi.to_dense().reshape(-1).astype(complex, copy=False)
            return DenseState(vector=vec, n_qubits=spec.n_qubits, d=spec.d, backend=self.name)

        return MPSState(mps=psi, n_qubits=spec.n_qubits, d=spec.d, backend=self.name, max_bond=max_bond)
