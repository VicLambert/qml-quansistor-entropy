from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from rng.seeds import gate_seed

from ..spec import CircuitSpec, GateSpec
from .pattern.brickwork import brickwork_pattern
from .pattern.tdoping import TdopingRules


@dataclass(frozen=True)
class CliffordBrickwork:
    name: str = "clifford"
    tdoping: TdopingRules | None = None

    def make_spec(
            self,
            n_qubits: int,
            n_layers: int,
            d: int,
            seed: int,
            *,
            connectivity: str = "loop",
            pattern: str = "brickwork",
            **kwargs: Any,
    ) -> CircuitSpec:
        params = dict(kwargs)
        params["tdoping"] = None if self.tdoping is None else self.tdoping
        return CircuitSpec(
            n_qubits=n_qubits,
            n_layers=n_layers,
            d=d,
            global_seed=seed,
            family=self.name,
            connectivity=connectivity,
            pattern=pattern,
            params=params,
        )

    def gates(self, spec: CircuitSpec) -> Iterable[GateSpec]:
        tdoping = spec.params.get("tdoping", None)
        tlocs = set()

        if tdoping is not None:
            tlocs = tdoping.locations(
                n_qubits=spec.n_qubits, layer=spec.n_layers, seed=spec.global_seed, connectivity=spec.connectivity,
            )

        for layer in range(spec.n_layers):
            pairs = brickwork_pattern(spec.n_qubits, layer, connectivity=spec.connectivity)

            for slot, (a, b) in enumerate(pairs):
                s = gate_seed(
                    spec.global_seed, layer=layer, slot=slot, wires=(a,b), kind="clifford",
                )
                yield GateSpec(
                    kind="clifford",
                    wires=(a,b),
                    d=spec.d,
                    seed=s,
                    tags=("layer", f"L{layer}", "clifford"),
                )

            for wire in range(spec.n_qubits):
                if (layer, wire) in tlocs:
                    yield GateSpec(
                        kind="T",
                        wires=(wire,),
                        d=spec.d,
                        seed=s,
                        tags=("layer", f"L{layer}", "T-gate"),
                    )
