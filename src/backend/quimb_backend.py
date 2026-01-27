
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


class QuimbBackend(BaseBackend):
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