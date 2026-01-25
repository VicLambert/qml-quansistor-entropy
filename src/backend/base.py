
from abc import ABC, abstractmethod
from typing import Any, Literal

from circuit.spec import CircuitSpec

StateType = Literal["dense", "mps", "tn"]

class BaseBackend(ABC):
    """Abstract base class for quantum circuit backends."""
    name: str

    @abstractmethod
    def simulate(
        self,
        spec: CircuitSpec,
        state_type: StateType = "dense",

    ):
        """Simulate a quantum circuit based on the given specification.

        Args:
            spec: Circuit specification.
            state_type: Type of state representation to use.

        Returns:
            The final state after simulation.
        """
        raise NotImplementedError
    