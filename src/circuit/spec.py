from dataclasses import dataclass, field
from typing import Tuple, dict, Any

Wires = Tuple[int, ...]

@dataclass(frozen=True)
class GateSpec:
    kind: str
    wires: Wires
    seed: int | None = None
    tags: Tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CircuitSpec:
    n_qubits: int
    n_layers: int
    d: int
    family: str
    topology: str
    global_seed: int
    gates: Tuple[GateSpec, ...]
    params: list[Any]