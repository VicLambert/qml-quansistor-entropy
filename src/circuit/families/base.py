from dataclasses import dataclass
from typing import Iterable, Protocol, Any
from ..spec import CircuitSpec, GateSpec

class Family(Protocol):
    name: str

    def make_spec(
            self,
            n_qubits: int,
            n_layers: int,
            d: int,
            seed: int | None,
            *,
            topology: str = "loop",
            **kwargs: Any,
    ) -> CircuitSpec:
        """Returns the circuit specification"""


    def gates(self, spec: CircuitSpec) -> Iterable[GateSpec]:
        """Returns the GatesSpec"""
