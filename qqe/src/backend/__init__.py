"""Backend package for quantum circuit simulation and entropy calculations.

This package provides interfaces to different quantum simulation backends,
including PennyLane and QuimB.
"""

from src.backend.base import BaseBackend
from src.backend.pennylane_backend import PennylaneBackend
from src.backend.quimb import QuimbBackend

__all__ = [
    "BaseBackend",
    "PennylaneBackend",
    "QuimbBackend",
]
