"""Data classes for quantum circuit specifications.

This module defines the core data structures for representing quantum circuits,
including gate specifications and circuit configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Wires = tuple[int, ...]

@dataclass(frozen=True)
class GateSpec:
    kind: str
    wires: Wires
    d: int
    seed: int | None = None
    tags: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CircuitSpec:
    n_qubits: int
    n_layers: int
    d: int
    family: str
    connectivity: str
    pattern: str
    global_seed: int
    gates: tuple[GateSpec, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)
