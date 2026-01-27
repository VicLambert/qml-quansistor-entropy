from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Set, Tuple

from src.rng.seeds import gate_seed
from ..spec import CircuitSpec, GateSpec


def quansistor_block(
    n_qubits: int,
    n_layer: int,
) -> list[tuple[int, int, int, int]]:
    start = n_layer % 2
    blocks = []

    for i in range(start, n_qubits - 3, 4):
        blocks.append((i, i + 1, i + 2, i + 3))
    return blocks


def leftover_pairs(n_qubits: int, used: Set[int], topology: str) -> list[tuple[int, int]]:
    left = [i for i in range(n_qubits) if i not in used]
    # TODO implement neighbours logic
    return []


@dataclass(frozen=True)
class QuansistorBrickwork:
    name: str = "quansistor"

    def make_spec(
        self,
        n_qubits: int,
        n_layers: int,
        d: int,
        seed: int,
        *,
        connectivity: str = "line",
        pattern: str = "brickwork",
        **kwargs: Any,
    ) -> CircuitSpec:
        return CircuitSpec(
            n_qubits=n_qubits,
            n_layers=n_layers,
            d=d,
            global_seed=seed,
            family=self.name,
            connectivity=connectivity,
            pattern=pattern,
            params={},
        )

    def gates(self, spec: CircuitSpec) -> Iterable[GateSpec]:
        for layer in range(spec.n_layers):
            blocks = quansistor_block(spec.n_qubits, layer)
            used = set()

            for idx, (q0, q1) in enumerate(blocks):
                used.update((q0, q1))
                s = gate_seed(
                    spec.global_seed,
                    layer=layer,
                    slot=idx,
                    wires=(q0, q1),
                    kind="quansistor",
                )
                yield GateSpec(
                    kind="quansistor",
                    wires=(q0, q1),
                    d=spec.d,
                    seed=s,
                    tags=("layer", f"L{layer}", "block"),
                )

            pairs = leftover_pairs(spec.n_qubits, used, spec.connectivity)
            for j, (a, b) in enumerate(pairs):
                s = gate_seed(
                    spec.global_seed, layer=layer, slot=1000 + j, wires=(a, b), kind="quansistor2",
                )
                yield GateSpec(
                    kind="quansistor",
                    wires=(a, b),
                    d=spec.d,
                    seed=s,
                    tags=("layer", f"L{layer}", "leftover"),
                )
