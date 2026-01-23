"""Haar circuit family specification and gate generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rng.seeds import gate_seed

from ..spec import CircuitSpec, GateSpec
from .pattern.brickwork import brickwork_pattern

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass(frozen=True)
class HaarBrickwork:
    """Haar random circuit family with brickwork pattern.

    Attributes:
        name: The name of the circuit family.
    """

    name: str = "haar"

    def make_spec(
        self,
        n_qubits: int,
        n_layers: int,
        d: int,
        seed: int,
        *,
        topology: str = "loop",
        **kwargs,
    ) -> CircuitSpec:
        return CircuitSpec(
            n_qubits=n_qubits,
            n_layers=n_layers,
            d=d,
            global_seed=seed,
            family=self.name,
            topology=topology,
        )

    def gates(self, spec: CircuitSpec) -> Generator[GateSpec]:
        for layer in range(spec.n_layers):
            pairs = brickwork_pattern(spec.n_qubits, layer, topology=spec.topology)

            for slot, (a, b) in enumerate(pairs):
                s = gate_seed(
                    spec.global_seed,
                    layer=layer,
                    slot=slot,
                    wires=(a, b),
                    kind="haar",
                )
                yield GateSpec(
                    kind="haar",
                    wires=(a, b),
                    d=spec.d,
                    seed=s,
                    tags=("layer", f"L{layer}", "haar"),
                )
