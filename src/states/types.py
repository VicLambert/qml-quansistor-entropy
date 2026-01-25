
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


@dataclass
class DenseState:
    """Class representing a dense quantum state."""
    vector: np.ndarray
    n_qubits: int
    d: int
    backend: str

@dataclass
class MPSState:
    """Class representing a matrix product state (MPS)."""
    mps: Any
    n_qubits: int
    d: int
    backend: str
    max_bond: int | None = None
