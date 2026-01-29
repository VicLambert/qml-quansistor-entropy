"""Backend package for quantum circuit simulation and entropy calculations.

This package provides interfaces to different quantum simulation backends,
including PennyLane and QuimB.
"""

from .pennylane_backend import PennylaneBackend
from .quimb_backend import QuimbBackend


__all__ = [
    "PennylaneBackend",
    "QuimbBackend",
]