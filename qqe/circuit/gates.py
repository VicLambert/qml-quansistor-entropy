
from __future__ import annotations

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


def random_quansistor_gate(rng):
    """Generate a random quansistor gate from symmetry class X or Y.
    
    @param rng: Random number generator.
    @return: Random quansistor gate matrix.
    """
    a, b, g = rng.standard_normal(3)
    return circ_X(a, b, g) if rng.choice(["X", "Y"]) == "X" else circ_Y(a, b, g)


def _I() -> np.ndarray:
    return np.eye(2, dtype=complex)


def _H() -> np.ndarray:
    return (1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=complex)


def _S() -> np.ndarray:
    return np.array([[1, 0], [0, 1j]], dtype=complex)


def _CNOT() -> np.ndarray:
    # control = first qubit, target = second qubit, basis |00>,|01>,|10>,|11>
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


_ONEQ = {
    "I": _I(),
    "H": _H(),
    "S": _S(),
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
            U = gate.params["matrix"]
        except KeyError as e:
            msg = f"{kind} requires gate.params['matrix']"
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
