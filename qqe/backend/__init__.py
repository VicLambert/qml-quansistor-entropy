"""Backend package for quantum circuit simulation and entropy calculations.

This package provides interfaces to different quantum simulation backends,
including PennyLane and QuimB.
"""

from qqe.backend.base import BaseBackend
from qqe.backend.pennylane_backend import PennylaneBackend
from qqe.backend.quimb import QuimbBackend

__all__ = [
    "BaseBackend",
    "PennylaneBackend",
    "QuimbBackend",
]
