
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class BackendConfig:
    name: str               # "quimb" or "pennylane"
    representation: str     # "dense" or "mps"...
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DenseState:
    """Class representing a dense quantum state."""
    vector: np.ndarray
    n_qubits: int
    d: int
    backend: str

    def copy(self) -> DenseState:
        return DenseState(self.vector.copy(), self.n_qubits, self.d, self.backend)

@dataclass
class MPSState:
    """Class representing a matrix product state (MPS)."""
    mps: Any
    n_qubits: int
    d: int
    backend: str
    max_bond: int | None = None
