from __future__ import annotations

import numpy as np

from src.circuit.spec import GateSpec
from src.circuit.families.gates import haar_unitary_gate, random_quansistor_gate, clifford_recipe_unitary

def _T_matrix() -> np.ndarray:
    return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def gate_unitary(gate: GateSpec) -> np.ndarray:
    kind = gate.kind

    if kind in ("unitary1", "unitary2"):
        try:
            U = gate.meta["matrix"]
        except KeyError as e:
            raise KeyError(f"{kind} requires gate.meta['matrix']") from e
        U = np.asarray(U, dtype=complex)
        return U

    if kind == "T":
        return _T_matrix()

    if kind == "haar":
        if gate.seed is None:
            raise ValueError("Haar gate requires a seed.")
        rng = np.random.default_rng(gate.seed)
        return haar_unitary_gate(d=gate.d ** len(gate.wires), rng=rng)

    if kind == "quansistor":
        if gate.seed is None:
            raise ValueError("Quansistor gate requires a seed.")
        rng = np.random.default_rng(gate.seed)
        U = random_quansistor_gate(rng)
        U = np.asarray(U, dtype=complex)
        return U

    if kind == "clifford":
        if gate.seed is None:
            raise ValueError("Clifford 2x2 gate requires a seed")
        _, _, U = clifford_recipe_unitary(gate.seed)
        return U

    raise NotImplementedError(f"Unknown gate kind: {kind!r}")
