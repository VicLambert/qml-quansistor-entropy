"""Base protocol for circuit family implementations.

This module defines the Family protocol that specifies the interface for
circuit family implementations, including methods for generating circuit
specifications and retrieving gate specifications.
"""
from typing import Any, Iterable, Protocol

from ..spec import CircuitSpec, GateSpec


class Family(Protocol):
    name: str

    def make_spec(
            self,
            n_qubits: int,
            n_layers: int,
            d: int,
            seed: int,
            *,
            connectivity: str = "loop",
            pattern: str = "random",
            **kwargs: Any,
    ) -> CircuitSpec:
        """Returns the circuit specification"""
        ...

    def gates(self, spec: CircuitSpec) -> Iterable[GateSpec]:
        """Returns the GatesSpec"""
        ...
