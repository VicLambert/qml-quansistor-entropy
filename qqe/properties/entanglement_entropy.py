
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from qqe.properties.compute import PropertyResult
from qqe.states.types import DenseState


# ----------- Rényi -----------

def renyi_ee(state: DenseState, subsysA_size: int | None = None) -> PropertyResult:
    n_qubits = state.n_qubits
    d = state.d
    psi = np.asarray(state.vector, dtype=complex).reshape(-1)

    if subsysA_size is None:
        subsysA_size = n_qubits // 2

    psi = psi.reshape((2,)*n_qubits)
    sys_A = list(range(subsysA_size))
    sys_B = list(range(subsysA_size, n_qubits))

    ax = sys_A + sys_B
    psi_AB = np.transpose(psi, axes=ax).reshape(2**subsysA_size, 2**(n_qubits - subsysA_size))
    ket = np.linalg.svd(psi_AB, compute_uv=False)
    ket = ket / max(np.linalg.norm(ket), 1e-16)
    rho_A = ket * ket
    rho_A = rho_A[rho_A > 0]
    rho_A = rho_A / rho_A.sum()

    trace = float(np.sum(rho_A ** 2))
    S = float(-(np.log(trace) / np.log(2)))

    details = {
        "n_qubits": n_qubits,
        "subsystem_size": subsysA_size,
    }
    return PropertyResult("entanglement_entropy", S, meta=details)

# ----------- Von Neumann -----------

def von_neumann_ee(state: DenseState, subsysA_size: int | None = None) -> PropertyResult:
    n_qubits = state.n_qubits
    d = state.d
    psi = np.asarray(state.vector, dtype=complex).reshape(-1)

    if subsysA_size is None:
        subsysA_size = n_qubits // 2

    psi = psi.reshape((2,)*n_qubits)
    sys_A = list(range(subsysA_size))
    sys_B = list(range(subsysA_size, n_qubits))

    ax = sys_A + sys_B
    psi_AB = np.transpose(psi, axes=ax).reshape(2**subsysA_size, 2**(n_qubits - subsysA_size))
    ket = np.linalg.svd(psi_AB, compute_uv=False)
    ket = ket / max(np.linalg.norm(ket), 1e-16)
    rho_A = ket * ket
    rho_A = rho_A[rho_A > 0]
    S = float(-np.sum(rho_A * np.log2(rho_A)))

    details = {
        "n_qubits": n_qubits,
        "subsystem_size": subsysA_size,
    }
    return PropertyResult("entanglement_entropy", S, meta=details)

