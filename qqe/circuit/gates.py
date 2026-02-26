
from __future__ import annotations
from typing import Literal

import numpy as np
from qqe.circuit.spec import GateSpec


def circ_X(a, b, g):
    """Creates a matrix of symmetry class X defined in section A of
    https://doi.org/10.1103/PhysRevA.106.062610.
    
    @param a: Parameter alpha defined in equation 18 of
    https://doi.org/10.1103/PhysRevA.106.062610.
    @param b: Parameter beta defined in equation 18 of
    https://doi.org/10.1103/PhysRevA.106.062610.
    @param g: Parameter gamma defined in equation 18 of
    https://doi.org/10.1103/PhysRevA.106.062610.

    @return: Matrix of symmetry class X.
    """
    # Matrix implementation in logical basis.
    # Obtained from Eqn.33 of paper.

    # Eigenvalues lambda_0,...,lambda_3. Note lambda_0=lambda_4 from paper.
    l0 = 2 * a + g
    l1 = -2 * b - g
    l2 = -2 * a + g
    l3 = 2 * b - g

    e0 = 0.25 * np.exp(-1.0j * l0)
    e1 = 0.25 * np.exp(-1.0j * l1)
    e2 = 0.25 * np.exp(-1.0j * l2)
    e3 = 0.25 * np.exp(-1.0j * l3)

    e11 = e0 + e1 + e2 + e3
    e12 = e0 - 1.0j * e1 - e2 + 1.0j * e3
    e13 = e0 - e1 + e2 - e3
    e14 = e0 + 1.0j * e1 - e2 - 1.0j * e3

    return np.array(
        [
            [e11, e12, e13, e14],
            [e14, e11, e12, e13],
            [e13, e14, e11, e12],
            [e12, e13, e14, e11],
        ],
    )


def circ_Y(a, b, g):
    """Creates a matrix of symmetry class Y defined in section B of
    https://doi.org/10.1103/PhysRevA.106.062610.
    
    @param a: Parameter alpha defined in equation 23 of
    https://doi.org/10.1103/PhysRevA.106.062610.
    @param b: Parameter beta defined in equation 23 of
    https://doi.org/10.1103/PhysRevA.106.062610.
    @param g: Parameter gamma defined in equation 23 of
    https://doi.org/10.1103/PhysRevA.106.062610.

    @return: Matrix of symmetry class Y.
    """
    # Matrix implementation in logical basis.
    # Obtained from Eqn.33 of paper.

    # Constants
    SQRT2 = np.sqrt(2)
    E_I_PI_OVER_4 = (1 + 1.0j) / SQRT2
    E_I_3_PI_OVER_4 = (-1 + 1.0j) / SQRT2

    # Eigenvalues lambda_0,...,lambda_3. Note lambda_0 = lambda_4 from paper.
    l0 = SQRT2 * a + SQRT2 * b + g
    l1 = SQRT2 * a - SQRT2 * b - g
    l2 = -SQRT2 * a - SQRT2 * b + g
    l3 = -SQRT2 * a + SQRT2 * b - g

    e0 = 0.25 * np.exp(-1.0j * l0)
    e1 = 0.25 * np.exp(-1.0j * l1)
    e2 = 0.25 * np.exp(-1.0j * l2)
    e3 = 0.25 * np.exp(-1.0j * l3)

    e11 = e0 + e1 + e2 + e3
    e12 = e0 - 1.0j * e1 - e2 + 1.0j * e3
    e13 = -e0 + e1 - e2 + e3
    e14 = e0 + 1.0j * e1 - e2 - 1.0j * e3
    e23a = (e0 - e2) * E_I_3_PI_OVER_4
    e23b = (e1 - e3) * E_I_PI_OVER_4
    e24 = e0 - e1 + e2 - e3
    e34a = -(e0 - e2) * E_I_PI_OVER_4
    e34b = -(e1 - e3) * E_I_3_PI_OVER_4

    return np.array(
        [
            [e11, E_I_PI_OVER_4 * e12, e13, E_I_PI_OVER_4 * e14],
            [-E_I_3_PI_OVER_4 * e14, e11, e23a + e23b, e24],
            [e13, e34a + e34b, e11, e34a - e34b],
            [-E_I_3_PI_OVER_4 * e12, e24, e23a - e23b, e11],
        ],
    )


def haar_unitary_gate(d, rng):
    a, b = rng.normal(size=(d, d)), rng.normal(size=(d, d))

    Z = a + 1j * b
    Q, R = np.linalg.qr(Z)

    Lambda = np.diag([R[i, i] / np.abs(R[i, i]) for i in range(d)])
    return np.dot(Q, Lambda)


def random_quansistor_gate(a, b, g, axis):
    """Generate a random quansistor gate from symmetry class X or Y.

    @param rng: Random number generator.
    @return: Random quansistor gate matrix.
    """
    #a, b, g = rng.standard_normal(3)
    return circ_X(a, b, g) if axis == "X" else circ_Y(a, b, g)


def _I() -> np.ndarray:
    return np.eye(2, dtype=complex)


def _H() -> np.ndarray:
    return (1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=complex)


def _S() -> np.ndarray:
    return np.array([[1, 0], [0, 1j]], dtype=complex)

def _Rx(theta: float) -> np.ndarray:
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([
                    [c, -1j * s],
                    [-1j * s, c]],
                dtype=complex)

def _Ry(theta: float) -> np.ndarray:
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([
                [c, -s],
                [s, c],
            ],
            dtype=complex)

def _Rz(theta: float) -> np.ndarray:
    return np.array([
                [np.exp(-1j * theta / 2), 0],
                [0, np.exp(1j * theta / 2)],
            ],
            dtype=complex)

def _CNOT() -> np.ndarray:
    # control = first qubit, target = second qubit, basis |00>,|01>,|10>,|11>
    return np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]],
            dtype=complex)


_ONEQ = {
    "I": _I(),
    "H": _H(),
    "S": _S(),
    "RX": _Rx,
    "RY": _Ry,
    "RZ": _Rz,
    "CNOT": _CNOT(),
}

def kron2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.kron(A, B)


def clifford_recipe_unitary(seed: int) -> tuple[str, str, np.ndarray]:
    """Using the given seed, choose two random 1-qubit gates from {I,H,S}.

    Apply a CNOT between them (control=wire0 -> target=wire1).
    Returns (U_a_name, U_b_name, U_4x4).
    """
    rng = np.random.default_rng(seed)
    labels = np.array(["I", "H", "S"], dtype=object)
    U_a_name, U_b_name = rng.choice(labels, size=2, replace=True)

    Ua = _ONEQ[str(U_a_name)]
    Ub = _ONEQ[str(U_b_name)]

    U = _CNOT() @ kron2(Ua, Ub)
    return str(U_a_name), str(U_b_name), U


def _T_matrix() -> np.ndarray:
    return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def random_rot_gate(seed: int | None = None, rot_set=("RX", "RY", "RZ")) -> np.ndarray:
    """Sample a random 1-qubit rotation gate unitary from rot_set with angle in [0, 2π)."""
    rng = np.random.default_rng(seed)
    gate = rng.choice(rot_set)
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    return _ONEQ[gate](theta)



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
    if kind == "I":
        return _I()

    if kind == "H":
        return _H()

    if kind == "S":
        return _S()

    if kind == "T":
        return _T_matrix()

    if kind == "haar":
        if gate.seed is None:
            msg = "Haar gate requires a seed."
            raise ValueError(msg)
        rng = np.random.default_rng(gate.seed)
        U = haar_unitary_gate(d=gate.d ** len(gate.wires), rng=rng)
        U = np.asarray(U, dtype=complex)
        expected = (gate.d ** len(gate.wires), gate.d ** len(gate.wires))
        if U.shape != expected:
            msg = (
                f"gate_unitary returned {U.shape} for kind={gate.kind} wires={gate.wires}, "
                f"expected {expected}"
            )
            raise ValueError(msg)
        return U

    if kind == "quansistor":
        if gate.seed is None:
            msg = "Quansistor gate requires a seed."
            raise ValueError(msg)
        rng = np.random.default_rng(gate.seed)
        a, b, g, _ = gate.params if gate.params else rng.standard_normal(3)
        axis = rng.choice(["X", "Y"])
        U = random_quansistor_gate(a, b, g, axis)
        U = np.asarray(U, dtype=complex)
        expected = (gate.d ** len(gate.wires), gate.d ** len(gate.wires))
        if U.shape != expected:
            msg = (
                f"gate_unitary returned {U.shape} for kind={gate.kind} wires={gate.wires}, "
                f"expected {expected}"
            )
            raise ValueError(msg)
        return U



    if kind == "clifford":
        if gate.seed is None:
            msg = "Clifford 2x2 gate requires a seed"
            raise ValueError(msg)
        u_a, u_b, U = clifford_recipe_unitary(gate.seed)
        U = np.asarray(U, dtype=complex)
        expected = (gate.d ** len(gate.wires), gate.d ** len(gate.wires))
        if U.shape != expected:
            msg = (
                f"gate_unitary returned {U.shape} for kind={gate.kind} wires={gate.wires}, "
                f"expected {expected}"
            )
            raise ValueError(msg)
        return U

    if kind in ("RX", "RY", "RZ"):
        # Prefer explicit rotation gates when params are provided
        if not gate.params or len(gate.params) != 1:
            msg = f"{kind} gate requires params=(theta,), got {gate.params}"
            raise ValueError(msg)
        theta = float(gate.params[0])
        return _ONEQ[kind](theta)

    if kind == "CNOT":
        return _CNOT()

    msg = f"Unknown gate kind: {kind!r}"
    raise NotImplementedError(msg)
