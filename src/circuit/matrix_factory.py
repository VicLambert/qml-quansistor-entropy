
"""Factory functions for generating quantum gate unitary matrices.

This module provides utilities to construct unitary matrices for various
quantum gates including T gates, Haar random gates, Quansistor gates, and
Clifford gates.
"""

from __future__ import annotations

import numpy as np

from src.circuit.families.gates import (
    clifford_recipe_unitary,
    haar_unitary_gate,
    random_quansistor_gate,
)
from src.circuit.spec import GateSpec


def _T_matrix() -> np.ndarray:
    return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def gate_unitary(gate: GateSpec) -> np.ndarray:
    """Generate the unitary matrix for a quantum gate.

    Parameters
    ----------
    gate : GateSpec
        The gate specification containing the gate kind, seed, and metadata.

    Returns:
    -------
    np.ndarray
        The unitary matrix for the specified gate.

    Raises:
    ------
    KeyError
        If a unitary gate is missing the 'matrix' key in metadata.
    ValueError
        If a Haar or Quansistor gate is missing a seed.
    NotImplementedError
        If the gate kind is not recognized.
    """
    kind = gate.kind

    if kind in ("unitary1", "unitary2"):
        try:
            U = gate.meta["matrix"]
        except KeyError as e:
            msg = f"{kind} requires gate.meta['matrix']"
            raise KeyError(msg) from e
        return np.asarray(U, dtype=complex)

    if kind == "T":
        return _T_matrix()

    if kind == "haar":
        if gate.seed is None:
            msg = "Haar gate requires a seed."
            raise ValueError(msg)
        rng = np.random.default_rng(gate.seed)
        return haar_unitary_gate(d=gate.d ** len(gate.wires), rng=rng)

    if kind == "quansistor":
        if gate.seed is None:
            msg = "Quansistor gate requires a seed."
            raise ValueError(msg)
        rng = np.random.default_rng(gate.seed)
        U = random_quansistor_gate(rng)
        return np.asarray(U, dtype=complex)

    if kind == "clifford":
        if gate.seed is None:
            msg = "Clifford 2x2 gate requires a seed"
            raise ValueError(msg)
        _, _, U = clifford_recipe_unitary(gate.seed)
        return U

    msg = f"Unknown gate kind: {kind!r}"
    raise NotImplementedError(msg)
