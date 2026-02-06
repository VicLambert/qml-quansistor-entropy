
from __future__ import annotations

import itertools

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

# from quimb.tensor.tensor_arbgeom import tensor_network_apply_op_vec
from qqe.properties.results import PropertyResult
from qqe.states.types import DenseState, MPSState


@dataclass
class ReplicaMPSParams:
    renyi_index: int = 2
    chi_max: int = 32
    chiP_max: int = 64
    cutoff_P: float = 1e-6
    cutoff_W: float = 1e-6

def qubit_pauli_ops():
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return [I, X, Y, Z]

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


# ----------- Exact -----------

def fast_kron(mats, vec):
    """Compute (M1 ⊗ M2 ⊗ ... ⊗ Mn) @ vec efficiently."""
    mats = [np.asarray(M, dtype=complex) for M in mats]
    dims = [M.shape[0] for M in mats]
    y = vec.reshape(dims)
    for axis, A in enumerate(mats):
        y = np.tensordot(A, y, axes=([1], [axis]))
        y = np.moveaxis(y, 0, axis)
    return y.reshape(-1)


def pauli_expval_fast_kron(state, label, dim: int | None = None):
    """Compute the expectation value of a multi-qudit Pauli/Weyl operator."""
    # Use qubit Pauli operators by default or when dim == 2, to preserve
    # existing behavior. For higher dimensions, use generalized qudit ops.
    if dim is None or dim == 2:
        pauli_ops = qubit_pauli_ops()
    else:
        pauli_ops_dict = make_qudit_ops(dim=dim)
        # Convert dict {idx: op} to list ordered by idx for consistent indexing
        pauli_ops = [pauli_ops_dict[i] for i in range(dim * dim)]
    mats = [pauli_ops[int(idx)] for idx in label]
    v = fast_kron(mats, state)
    return np.vdot(state, v)

def sre_exact(state: DenseState, **kwargs: Any) -> PropertyResult:
    """Computes the exact SRE for a dense state.

    Args:
        state: The dense quantum state.
        **kwargs: Additional keyword arguments.

    Returns:
        PropertyResult: The computed SRE property result.
    """
    n_qubits = state.n_qubits
    ops_1 = qubit_pauli_ops()
    psi = np.asarray(state.vector, dtype=complex).reshape(-1)

    pauli_list = list(range(state.d ** 2))  # All single-qudit Pauli operators

    res = 0.0
    norm_factor = state.d ** (n_qubits / 2)
    for label in itertools.product(pauli_list, repeat=n_qubits):
        mats = [ops_1[idx] for idx in label]
        v = fast_kron(mats, psi)
        expval = np.vdot(psi, v) / norm_factor
        res += (np.abs(expval) ** 4)
        # expval = pauli_expval_fast_kron(state.vector, label, state.d)
        # res += np.abs(expval) ** 4

    phys_dim = state.d ** n_qubits
    sre = -np.log2(res * phys_dim)

    details={"method":"exact", "n_qubits": n_qubits}
    return PropertyResult(name="SRE", value=sre, meta=details)

# ----------- MPS ------------
# def pauli_mps_site(A: np.ndarray, pauli_op: list[tuple[int, int]]) -> np.ndarray:
#     """Apply single-site Pauli operator to MPS tensor A.

#     Args:
#         A: MPS tensor of shape (d, chi_left, chi_right).
#         pauli_op: Single-site Pauli operator of shape (d, d).

#     Returns:
#         np.ndarray: Transformed MPS tensor after applying Pauli operator.
#     """
#     dL, d, dR = A.shape
#     d2 = len(pauli_op)

#     assert d == d2, "Pauli operator dimension must match MPS physical dimension."

#     dL2 = dL * dL
#     dR2 = dR * dR

#     A_op = [A[:, s, :] for s in range(d)]
#     B_i = np.zeros(shape=(dL2, dR2), dtype=complex)

#     for id, pauli in enumerate(pauli_op):
#         acc = np.zeros(shape=(dL2, dR2), dtype=complex)
#         for i in range(d):
#             for j in range(d):
#                 coeff = pauli[i, j] / np.sqrt(d)
#                 if coeff != 0:
#                     Ai = A_op[i]
#                     Aj = A_op[j]
#                     kron_prod = np.kron(Ai, np.conj(Aj))
#                     acc += coeff * kron_prod
#         B_i[:, id, :] = acc
#     return B_i


# def W_from_mps(B: np.ndarray):
#     """Construct the W tensor from the MPS tensor B.

#     Args:
#         B: MPS tensor after applying Pauli operators, shape (dL^2, d2, dR^2).

#     Returns:
#         np.ndarray: W tensor of shape (dL^2, dR^2, d2).
#     """
#     dL2, d2, dR2 = B.shape
#     W = np.zeros(shape=(dL2, d2, d2, dR2), dtype=complex)
#     for id in range(d2):
#         W[:, id, id, :] = B[:, id, :]
#     return W

# def apply_W(psi_mps: MPSState, W: np.ndarray, site: int) -> MPSState:
#     """Apply W tensor to MPS state at specified site.

#     Args:
#         psi_mps: Original MPS state.
#         W: W tensor to apply.
#         site: Site index to apply W.

#     Returns:
#         MPSState: New MPS state after applying W.
#     """
#     new_mps = psi_mps.mps.copy()
#     for _ in range(psi_mps.n_qubits):
#         psi = tensor_network_apply_op_vec(
#             A=W,
#             x=psi_mps,
#             max_bond=psi_mps.max_bond,
#             contract=False,
#             compress=False,
#         )
#     return MPSState(mps=new_mps, n_qubits=psi_mps.n_qubits, d=psi_mps.d, backend=psi_mps.backend)


# def sre_mps(state: MPSState, params: ReplicaMPSParams) -> PropertyResult:
#     """Computes the SRE for an MPS state using replica method.

#     Args:
#         state: The MPS quantum state.
#         params: Parameters for the replica MPS computation.

#     Returns:
#         PropertyResult: The computed SRE property result.
#     """
#     #TODO Implement Replica MPS based method?
#     return PropertyResult(name="mps", value=0.0, meta={})

# ----------- FWHT -----------
def in_place_FHWT(arr: np.ndarray):
    """In-place Fast Walsh-Hadamard Transform of array a (length must be power of 2)."""
    n = arr.shape[0]
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            u = arr[i:i+h]
            v = arr[i+h:i+2*h]
            u[:], v[:] = u + v, u - v
        h *= 2

def sre_fwht(state: DenseState) -> PropertyResult:
    psi = np.asarray(state.vector, dtype=complex).reshape(-1)
    N = psi.size

    # qubits-only assumption:
    n = int(np.log2(N))
    if 2**n != N:
        raise ValueError(f"FWHT method assumes N=2^n. Got N={N}")

    idx = np.arange(N, dtype=np.int64)

    acc=0.0
    for k in range(N):
        A = np.conjugate(psi[idx ^ k]) * psi   # shape (N,)
        in_place_FHWT(A)
        acc += np.sum(np.abs(A)**4)
    sre = -np.log2(acc/(2**n))

    details = {"method":"FWHT", "n_qubits":n}
    return PropertyResult(name="SRE", value=sre, meta=details)

# ----------- MCMC -----------
def sre_mcmc(state: DenseState, *, seed: int, n_samples: int = 20000, batch_size: int = 500) -> PropertyResult:
    n = state.n_qubits
    psi = np.asarray(state.vector, dtype=complex).reshape(-1)

    norm = np.vdot(psi, psi).real
    if not np.isclose(norm, 1.0, atol=1e-10):
        psi = psi / np.sqrt(norm)

    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 4, size=(n_samples, n), dtype=np.int64)

    y = np.empty(n_samples, dtype=np.float64)
    for s in range(n_samples):
        expval = pauli_expval_fast_kron(psi, labels[s])
        y[s] = float(np.abs(expval) ** 4)

    mean = float(y.mean())
    acc = (4**n) * mean
    sre = float(-np.log2(acc / (2**n)))

    B = n_samples // batch_size
    se_sre = None
    if B >= 2:
        y_trunc = y[: B * batch_size].reshape(B, batch_size).mean(axis=1)  # batch means of y
        var_mean_y = float(np.var(y_trunc, ddof=1) / B)

        # delta-method for SRE = -log2( (4^n * mean_y)/2^n ) = -log2( 2^n * mean_y )
        # derivative wrt mean_y: dSRE/dm = -(1/(m ln 2))
        se_sre = float(np.sqrt(var_mean_y) / (mean * np.log(2)))

    details = {
        "method": "sampling",
        "n_qubits": n,
        "d": state.d,
        "n_samples": n_samples,
        "batch_size": batch_size,
        "acc_hat": acc,
        "mean_abs_expval4": mean,
        "se_sre_approx": se_sre,
    }
    return PropertyResult(name="SRE", value=sre, meta=details)
