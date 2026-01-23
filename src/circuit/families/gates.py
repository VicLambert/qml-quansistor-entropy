import numpy as np


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

def haar_unitary_gate(dim, rng):
    a, b = rng.normal(size=(dim, dim)), rng.normal(size=(dim, dim))

    Z = a + 1j * b
    Q, R = np.linalg.qr(Z)

    Lambda = np.diag([R[i, i] / np.abs(R[i, i]) for i in range(dim)])
    return np.dot(Q, Lambda)
