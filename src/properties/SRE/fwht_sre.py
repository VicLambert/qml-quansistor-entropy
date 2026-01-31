
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

from properties.results import PropertyResult

if TYPE_CHECKING:
    from states.types import DenseState

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

def compute(state: DenseState) -> PropertyResult:
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
