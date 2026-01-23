from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Any

from .pattern.brickwork import brickwork_pattern
from ..spec import CircuitSpec, GateSpec
from ...rng.seeds import gate_seed

@dataclass(frozen=True)
class HaarBrickwork:
    name: str = "haar"

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
        return CircuitSpec(
            n_qubits=n_qubits,
            n_layers=n_layers,
            d=d,
            global_seed=seed,
            family=self.name,
            topology=topology,
        )
    
    def gates(self, spec: CircuitSpec) -> Iterable[GateSpec]:
        for l in range(spec.n_layers):
            pairs = brickwork_pattern(spec.n_qubits, l, topology=spec.topology)

            for slot, (a, b) in enumerate(pairs):
                s = gate_seed(spec.seed, layer=l, slot=slot, wires=(a, b), kind="haar")
                yield GateSpec(
                    kind="haar",
                    wires=(a, b),
                    d=spec.d,
                    seed = s,
                    tags=("layer", f"L{l}", "haar")
                )