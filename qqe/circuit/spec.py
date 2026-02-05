"""Data classes for quantum circuit specifications.

This module defines the core data structures for representing quantum circuits,
including gate specifications and circuit configurations.
"""

from __future__ import annotations

import hashlib
import json

import dataclasses
from dataclasses import asdict, dataclass, field
from typing import Any

from qqe.utils.reading import _to_jsonable

Wires = tuple[int, ...]


@dataclass(frozen=True)
class GateSpec:
    """Specification for a quantum gate.

    Attributes:
    ----------
    kind : str
        The type or name of the quantum gate.
    wires : Wires
        The qubits on which the gate operates.
    d : int
        The dimension parameter for the gate.
    seed : int | None
        Optional random seed for gate initialization.
    tags : tuple[str, ...]
        Optional tags for categorizing or labeling the gate.
    meta : dict[str, Any]
        Optional metadata associated with the gate.
    """
    kind: str
    wires: Wires
    d: int
    seed: int | None = None
    tags: tuple[str, ...] = ()
    params: tuple[Any, ...] = ()


@dataclass(frozen=True)
class CircuitSpec:
    """Specification for a quantum circuit.

    Attributes:
    ----------
    n_qubits : int
        The number of qubits in the circuit.
    n_layers : int
        The number of layers in the circuit.
    d : int
        The dimension parameter for the circuit.
    family : str
        The family or type of the circuit.
    connectivity : str
        The connectivity pattern of the circuit.
    pattern : str
        The gate pattern used in the circuit.
    global_seed : int
        The global random seed for circuit initialization.
    gates : tuple[GateSpec, ...]
        The gates in the circuit.
    params : dict[str, Any]
        Optional parameters for the circuit.
    """
    n_qubits: int
    n_layers: int
    d: int
    family: str
    connectivity: str
    pattern: str
    global_seed: int
    gates: tuple[GateSpec, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)

    def spec_id(self) -> str:
        """Deterministic identifier for this circuit spec."""
        payload = {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "d": self.d,
            "family": self.family,
            "connectivity": self.connectivity,
            "pattern": self.pattern,
            "global_seed": self.global_seed,
            "params": _to_jsonable(self.params),
            "gates": [
                {
                    "kind": g.kind,
                    "wires": g.wires,
                    "params": _to_jsonable(g.params),
                    "seed": g.seed,
                    "tags": g.tags,
                }
                for g in self.gates
            ],
        }

        glob = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(glob.encode()).hexdigest()[:16]
