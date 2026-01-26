
from __future__ import annotations

import itertools

from typing import TYPE_CHECKING, Any

import numpy as np

from properties.results import PropertyResult

if TYPE_CHECKING:
    from states.types import DenseState


def make_qudit_ops(dim):
    """Return {idx: W_{a,b}} where W_{a,b} = Z^a X^b, idx = a*dim + b."""
    # generalized X
    X = np.zeros((dim, dim), dtype=complex)
    for j in range(dim):
        X[(j + 1) % dim, j] = 1.0

    # generalized Z
    omega = np.exp(2j * np.pi / dim)
    Z = np.diag([omega**j for j in range(dim)])

    ops = {}
    idx = 0
    for a in range(dim):      # power of Z
        Za = np.linalg.matrix_power(Z, a)
        for b in range(dim):  # power of X
            Xb = np.linalg.matrix_power(X, b)
            ops[idx] = Za @ Xb          # single-site dim×dim matrix
            idx += 1
    return ops

def fast_kron(mats, vec):
    """Compute (M1 ⊗ M2 ⊗ ... ⊗ Mn) @ vec efficiently."""
    mats = [np.asarray(M, dtype=complex) for M in mats]
    dims = [M.shape[0] for M in mats]
    y = vec.reshape(dims)
    for axis, A in enumerate(mats):
        y = np.tensordot(A, y, axes=([1], [axis]))
        y = np.moveaxis(y, 0, axis)
    return y.reshape(-1)


def pauli_expval_fast_kron(state, label, dim):
    """Compute the expectation value of a multi-qudit Pauli operator."""
    pauli_ops = make_qudit_ops(dim=dim)
    mats = [pauli_ops[int(idx)] for idx in label]
    v = fast_kron(mats, state)
    return np.vdot(state, v)

def compute(state: DenseState, **kwargs: Any) -> PropertyResult:
    """Computes the exact SRE for a dense state.

    Args:
        state: The dense quantum state.
        **kwargs: Additional keyword arguments.

    Returns:
        PropertyResult: The computed SRE property result.
    """
    n_qubits = state.n_qubits
    pauli_list = list(range(state.d ** 2))  # All single-qudit Pauli operators

    res = 0.0
    for label in itertools.product(pauli_list, repeat=n_qubits):
        expval = pauli_expval_fast_kron(state.vector, label, state.d)
        res += np.abs(expval) ** 4

    phys_dim = state.d ** n_qubits
    sre = -np.log2(res * phys_dim)

    details={"method":"exact_dense", "n_qubits": n_qubits}
    return PropertyResult(name="SRE", value=sre, meta=details)
