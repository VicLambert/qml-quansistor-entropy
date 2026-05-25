
"""Quimb tensor network backend for quantum circuit simulation.

This module provides the QuimbBackend class that implements quantum circuit
simulation using the quimb library, supporting both dense and MPS state types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import quimb.tensor as qtn

from backend.base import BaseBackend, State
from circuit.gates import gate_unitary
from states.types import DenseState, MPSState

if TYPE_CHECKING:
    from circuit.spec import CircuitSpec, GateSpec


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
        representation: str,
        max_bond: int | None = None,
        cutoff: float | None = None,
    ):
        if representation == "mps":
            kwargs: dict[str, Any] = {}
            if max_bond is not None:
                kwargs["max_bond"] = max_bond
            if cutoff is not None:
                kwargs["cutoff"] = cutoff
            return qtn.CircuitMPS(n_qubits, **kwargs)
        if representation == "dense":
            return qtn.Circuit(n_qubits)
        msg = f"QuimbBackend does not support representation={representation!r}"
        raise NotImplementedError(msg)

    def _apply_gate(self, circ: qtn.Circuit, gate: GateSpec) -> None:
        U = gate_unitary(gate)
        circ.apply_gate(U, *gate.wires)

    def simulate(
        self,
        spec: CircuitSpec,
        *,
        representation: str = "dense",
        max_bond: int | None = None,
        cutoff: float | None = None,
        **kwargs: Any,
    ) -> State:
        """Simulate a quantum circuit using quimb.

        Parameters
        ----------
        spec : CircuitSpec
            The circuit specification to simulate.
        representation : str, optional
            The state representation to use ("dense" or "mps"), by default "dense".
        max_bond : int | None, optional
            Maximum bond dimension for MPS, by default None.
        cutoff : float | None, optional
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
            If the representation is not supported.
        """
        self._validate_materialized(spec)

        circ = self._make_circuit(
            n_qubits=spec.n_qubits,
            representation=representation,
            max_bond=max_bond,
            cutoff=cutoff,
        )

        for gate in spec.gates:
            self._apply_gate(circ, gate)

        psi = circ.psi

        if representation == "dense":
            vec = psi.to_dense().reshape(-1).astype(complex, copy=False)
            return DenseState(vector=vec, n_qubits=spec.n_qubits, d=spec.d, backend=self.name)

        return MPSState(mps=psi, n_qubits=spec.n_qubits, d=spec.d, backend=self.name, max_bond=max_bond)
