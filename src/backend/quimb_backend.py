
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


class QuimbBackend(BaseBackend):
    name = "quimb"

    def _make_circuit(
        self,
        n_qubits: int,
        d: int,
        state_type: str,
        max_bond: int | None = None,
    ):
        if d != 2:
            raise NotImplementedError("QuimbBackend only supports qubit systems (d=2).")

        if state_type == "mps":
            kwargs: dict[str, Any] = {}
            if max_bond is not None:
                kwargs["max_bond"] = max_bond
            return qtn.CircuitMPS(n_qubits, **kwargs)

        return qtn.Circuit(n_qubits)

    def _apply_gate(self, circ: qtn.Circuit, gate: GateSpec, d: int = 2):
        kind = gate.kind
        wire = gate.wires

        return self._apply_gate_dispatch(kind, wire, circ, gate, d)

    @singledispatchmethod
    def _apply_gate_dispatch(self, kind: str, wire: tuple[int, ...], circ: qtn.Circuit, gate: GateSpec, d: int):
        msg = f"Gate kind '{kind}' is not implemented in QuimbBackend."
        raise NotImplementedError(msg)

    @_apply_gate_dispatch.register
    def _(self, kind: Literal["unitary1"], wire: tuple[int], circ: qtn.Circuit, gate: GateSpec, d: int):
        matrix = gate.meta["matrix"]
        circ.apply_gate(matrix, wire[0])

    @_apply_gate_dispatch.register
    def _(self, kind: Literal["unitary2"], wire: tuple[int, int], circ: qtn.Circuit, gate: GateSpec, d: int):
        matrix = gate.meta["matrix"]
        circ.apply_gate(matrix, wire[0], wire[1])

    @_apply_gate_dispatch.register
    def _(self, kind: Literal["T"], wire: tuple[int], circ: qtn.Circuit, gate: GateSpec, d: int):
        matrix = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
        circ.apply_gate(matrix, wire[0])

    @_apply_gate_dispatch.register
    def _(self, kind: Literal["haar"], wire: tuple[int, int], circ: qtn.Circuit, gate: GateSpec, d: int):
        seed = gate.seed
        rng = np.random.default_rng(seed)
        U = haar_unitary_gate(d=d**2, rng=rng)
        circ.apply_gate(U, wire[0], wire[1])

    @_apply_gate_dispatch.register
    def _(self, kind: Literal["clifford"], wire: tuple[int, int], circ: qtn.Circuit, gate: GateSpec, d: int):
        seed = gate.seed
        rng = np.random.default_rng(seed)
        oneq_cliffords = ["I", "H", "S"]
        a, b = wire
        U_a, U_b = rng.choice(oneq_cliffords, size=2)
        circ.apply_gate(U_a, a)
        circ.apply_gate(U_b, b)
        circ.apply_gate("CNOT", a, b)

    @_apply_gate_dispatch.register
    def _(self, kind: Literal["quansistor"], wire: tuple[int, int], circ: qtn.Circuit, gate: GateSpec, d: int):
        seed = gate.seed
        rng = np.random.default_rng(seed)
        U = random_quansistor_gate(rng)
        circ.apply_gate(U, wire[0], wire[1])
    
    def simulate(
        self,
        spec: CircuitSpec,
        state_type: str = "dense",
        max_bond: int | None = None,
        **kwargs: Any,
    ):
        circ = self._make_circuit(
            n_qubits=spec.n_qubits,
            d=spec.d,
            state_type=state_type,
            max_bond=max_bond,
        )

        for gate in spec.gates:
            self._apply_gate(circ, gate, d=spec.d)

        psi = circ.psi

        if state_type == "dense":
            vec = psi.to_dense().reshape(-1)
            return DenseState(vector=vec, n_qubits=spec.n_qubits, d=spec.d, backend=self.name)

        return MPSState(mps=psi, n_qubits=spec.n_qubits, d=spec.d, backend=self.name, max_bond=max_bond)