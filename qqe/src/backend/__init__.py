"""Backend package for quantum circuit simulation and entropy calculations.

This package provides interfaces to different quantum simulation backends,
including PennyLane and QuimB.
"""

from backend.base import BaseBackend
from backend.pennylane_backend import PennylaneBackend
from backend.quimb import QuimbBackend

__all__ = [
    "BaseBackend",
    "PennylaneBackend",
    "QuimbBackend",
]
