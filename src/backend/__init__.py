"""Backend package for quantum circuit simulation and entropy calculations.

This package provides interfaces to different quantum simulation backends,
including PennyLane and QuimB.
"""

from src.backend import base, pennylane_backend, quimb_backend

__all__ = [
    "base",
    "pennylane_backend",
    "quimb_backend",
]