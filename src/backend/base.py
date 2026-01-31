
"""Abstract base class and types for quantum circuit backends.

This module provides the base interface for implementing quantum circuit
simulators with support for different state representations (dense, MPS, TN).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, Union

from src.circuit.spec import CircuitSpec
from src.states.types import DenseState, MPSState

StateType = Literal["dense", "mps", "tn"]
State = Union[DenseState, MPSState]

class BaseBackend(ABC):
    """Abstract base class for quantum circuit backends."""
    name: str

    @abstractmethod
    def simulate(
        self,
        spec: CircuitSpec,
        *,
        state_type: StateType = "dense",
        max_bond: int | None = None,
        **kwargs: Any,
    ) -> State:
        """Simulate a quantum circuit based on the given specification.

        Args:
            spec: Circuit specification.
            state_type: Type of state representation to use.
            max_bond: Maximum bond dimension for MPS state representation.
            **kwargs: Additional backend-specific keyword arguments.

        Returns:
            The final state after simulation.
        """
        raise NotImplementedError

    def _validate_materialized(self, spec: CircuitSpec) -> None:
        if not spec.gates:
            msg = "CircuitSpec.gates is empty. Initialize the circuit first before simulating."
            raise ValueError(msg)
