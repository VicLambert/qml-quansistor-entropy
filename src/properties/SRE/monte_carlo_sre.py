
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

from properties.results import PropertyResult

from .sre_exact_dense import pauli_expval_fast_kron

if TYPE_CHECKING:
    from states.types import DenseState

def compute(state: DenseState, *, seed:int, n_samples: int = 20000, batch_size: int = 500) -> PropertyResult:
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
        "method": "uniform_pauli_sampling",
        "n_qubits": n,
        "d": state.d,
        "n_samples": n_samples,
        "batch_size": batch_size,
        "acc_hat": acc,
        "mean_abs_expval4": mean,
        "se_sre_approx": se_sre,
    }
    return PropertyResult(name="SRE_uniform", value=sre, meta=details)



# PAULIS = {
#     "I": np.array([[1, 0], [0, 1]], dtype=complex),
#     "X": np.array([[0, 1], [1, 0]], dtype=complex),
#     "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
#     "Z": np.array([[1, 0], [0, -1]], dtype=complex),
# }

# def pauli_from_bits(a: int, b: int) -> str:
#     if a == 0 and b == 0:
#         return "I"
#     if a == 0 and b == 1:
#         return "X"
#     if a == 1 and b == 0:
#         return "Z"
#     return "Y"

# def random_pauli_label(n, dim, rng):
#     return rng.integers(0, dim**2, size=n, dtype=np.int64)

# def propose_update(label, dim, rng, n_sites=1):
#     n = len(label)
#     new = np.array(label, copy=True)
#     sites = rng.choice(n, size=n_sites, replace=False)
#     new[sites] = rng.integers(0, dim**2, size=n_sites, dtype=np.int64)
#     return new

# def propose_local_pauli(current, dim, rng):
#     new = current.copy()
#     j = rng.integers(0, len(current))
#     new[j] = rng.integers(0, dim**2)
#     return new

# def var_mean_batch_means(samples: np.ndarray, batch_size: int = 50) -> float:
#     x = np.asarray(samples, dtype=float)
#     n = x.size
#     B = n // batch_size
#     if B < 2:
#         # not enough data to estimate; fall back conservatively
#         return float(np.var(x, ddof=1) / max(n, 1))
#     x = x[:B * batch_size].reshape(B, batch_size)
#     batch_means = x.mean(axis=1)
#     # variance of the overall mean:
#     return float(np.var(batch_means, ddof=1) / B)

# def clean_prob(p, tol=1e-12):
#     return 0.0 if p < tol else p

# def compute(state: DenseState, seed: int, n_samples: int = 10000) -> PropertyResult:
#     n_qubits = state.n_qubits
#     d = state.d
#     psi = np.asarray(state.vector, dtype=complex).reshape(-1)

#     rng = np.random.default_rng(seed)

#     current = rng.integers(0, d**2, size=n_qubits, dtype=np.int64)
#     current_prob = float(np.abs(pauli_expval_fast_kron(psi, current, d))**2)

#     samples = []
#     n_acc = 0.0
#     burn_in = n_samples // 5
#     thin = 10

#     for s in range(n_samples):
#         # proposed = rng.integers(0, d**2, size = n_qubits, dtype=np.int64)
#         proposed = propose_local_pauli(current, d, rng)
#         proposed_prob = float(np.abs(pauli_expval_fast_kron(psi, proposed, d))**2)
#         alpha = min(1.0, (proposed_prob + 1e-30) / (current_prob + 1e-30))

#         if rng.random() < alpha:
#             current = proposed
#             current_prob = proposed_prob
#             n_acc += 1

#         if s >= burn_in and s % thin == 0:
#             samples.append(clean_prob(current_prob))

#     samples = np.asarray(samples)
#     mean = float(np.mean(samples))
#     var = var_mean_batch_means(samples, batch_size=50)

#     sre = float(-np.log2(mean))
#     corrected_sre = sre - (var / (2 * mean**2 * np.log(2)))

#     details={"method":"sampling", "n_qubits": n_qubits}
#     return PropertyResult(name="SRE_sampled", value=sre, meta=details)
