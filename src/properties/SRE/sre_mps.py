
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from quimb.tensor.tensor_arbgeom import tensor_network_apply_op_vec

    from properties.results import PropertyResult
    from states.types import MPSState

@dataclass
class ReplicaMPSParams:
    renyi_index: int = 2
    chi_max: int = 32
    chiP_max: int = 64
    cutoff_P: float = 1e-6
    cutoff_W: float = 1e-6


def pauli_mps_site(A: np.ndarray, pauli_op: list[tuple[int, int]]) -> np.ndarray:
    """Apply single-site Pauli operator to MPS tensor A.

    Args:
        A: MPS tensor of shape (d, chi_left, chi_right).
        pauli_op: Single-site Pauli operator of shape (d, d).

    Returns:
        np.ndarray: Transformed MPS tensor after applying Pauli operator.
    """
    dL, d, dR = A.shape
    d2 = len(pauli_op)

    assert d == d2, "Pauli operator dimension must match MPS physical dimension."

    dL2 = dL * dL
    dR2 = dR * dR

    A_op = [A[:, s, :] for s in range(d)]
    B_i = np.zeros(shape=(dL2, dR2), dtype=complex)

    for id, pauli in enumerate(pauli_op):
        acc = np.zeros(shape=(dL2, dR2), dtype=complex)
        for i in range(d):
            for j in range(d):
                coeff = pauli[i, j] / np.sqrt(d)
                if coeff != 0:
                    Ai = A_op[i]
                    Aj = A_op[j]
                    kron_prod = np.kron(Ai, np.conj(Aj))
                    acc += coeff * kron_prod
        B_i[:, id, :] = acc
    return B_i


def W_from_mps(B: np.ndarray):
    """Construct the W tensor from the MPS tensor B.

    Args:
        B: MPS tensor after applying Pauli operators, shape (dL^2, d2, dR^2).

    Returns:
        np.ndarray: W tensor of shape (dL^2, dR^2, d2).
    """
    dL2, d2, dR2 = B.shape
    W = np.zeros(shape=(dL2, d2, d2, dR2), dtype=complex)
    for id in range(d2):
        W[:, id, id, :] = B[:, id, :]
    return W

def apply_W(psi_mps, W, site: int):
    """Apply W tensor to MPS state at specified site.

    Args:
        psi_mps: Original MPS state.
        W: W tensor to apply.
        site: Site index to apply W.

    Returns:
        MPSState: New MPS state after applying W.
    """
    new_mps = psi_mps.mps.copy()
    for _ in range(psi_mps.n_qubits):
        psi = tensor_network_apply_op_vec(
            A=W,
            x=psi_mps,
            max_bond=psi_mps.max_bond,
            contract=False,
            compress=False,
        )


def compute(state: MPSState, params: ReplicaMPSParams) -> PropertyResult:
    """Computes the SRE for an MPS state using replica method.

    Args:
        state: The MPS quantum state.
        params: Parameters for the replica MPS computation.

    Returns:
        PropertyResult: The computed SRE property result.
    """
    #TODO Implement Replica MPS based method?
    return PropertyResult(name="mps", value=0.0, meta = {})
