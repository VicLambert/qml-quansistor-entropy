from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Set, Tuple

from src.rng.seeds import gate_seed

from ..spec import CircuitSpec, GateSpec


def quansistor_blocks(n_qubits: int, n_layer: int) -> list[tuple[int, int, int, int]]:
    start = (n_layer % 2) * 2  # shift by 2 wires each odd layer
    blocks: list[tuple[int, int, int, int]] = []
    for i in range(start, n_qubits - 3, 4):
        blocks.append((i, i + 1, i + 2, i + 3))
    return blocks


def leftover_pairs(n_qubits: int, used: Set[int], connectivity: str) -> list[tuple[int, int]]:
    left = [i for i in range(n_qubits) if i not in used]
    leftover = set(left)

    pairs : list[tuple[int, int]] = []
    used : set[int] = set()

    for i in range(n_qubits - 1):
        a, b = i, i + 1
        if a in leftover and b in leftover and a not in used and b not in used:
            pairs.append((a, b))
            used.update((a, b))
    if connectivity in ("loop", "ring"):
        a, b = n_qubits - 1, 0
        if a in leftover and b in leftover and a not in used and b not in used:
            pairs.append((a, b))
            used.update((a, b))
    return pairs


_BLOCK_STEPS = (
    (0, 1),  # q0 q1
    (2, 3),  # q2 q3
    (1, 2),  # q1 q2
    (0, 1),  # q0 q1
    (2, 3),  # q2 q3
)

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
        spec = CircuitSpec(
            n_qubits=n_qubits,
            n_layers=n_layers,
            d=d,
            global_seed=seed,
            family=self.name,
            connectivity=connectivity,
            pattern=pattern,
            params={},
        )
        return replace(spec, gates=tuple(self.gates(spec)))

    def gates(self, spec: CircuitSpec) -> Iterable[GateSpec]:
        for layer in range(spec.n_layers):
            blocks = quansistor_blocks(spec.n_qubits, layer)
            used: set[int] = set()

            # ---- 4-qubit blocks -> 5 two-qubit gates each
            for block_idx, (q0, q1, q2, q3) in enumerate(blocks):
                used.update((q0, q1, q2, q3))

                # One seed namespace per block (so 5 gates share a single RNG stream conceptually)
                # We derive per-step seeds deterministically from that block identity.
                block_seed = gate_seed(
                    spec.global_seed,
                    kind="quansistor_block",
                    layer=layer,
                    slot=block_idx,
                    wires=(q0, q1, q2, q3),
                    ordered_wires=True,
                )

                # Emit the 5 sequential 2-qubit quansistor gates
                wires4 = (q0, q1, q2, q3)
                for step_idx, (i, j) in enumerate(_BLOCK_STEPS):
                    a, b = wires4[i], wires4[j]

                    # step seed derived from block_seed (NOT spec.global_seed directly)
                    s = gate_seed(
                        block_seed,
                        kind="quansistor",
                        layer=step_idx,   # local to block
                        slot=0,
                        wires=(a, b),
                        ordered_wires=True,
                        extra=f"L{layer}_B{block_idx}_S{step_idx}",
                    )

                    yield GateSpec(
                        kind="quansistor",
                        wires=(a, b),
                        d=spec.d,
                        seed=s,
                        tags=("layer", f"L{layer}", "block", f"B{block_idx}", "step", f"S{step_idx}"),
                        params=(("block", (q0, q1, q2, q3)), ("step", step_idx)),
                    )

            # ---- leftover nearest-neighbor pairs
            pairs = leftover_pairs(spec.n_qubits, used, spec.connectivity)
            for j, (a, b) in enumerate(pairs):
                for _ in range(2):
                    s = gate_seed(
                        spec.global_seed,
                        kind="quansistor",
                        layer=layer,
                        slot=1000 + 10 * j + _,
                        wires=(a, b),
                        ordered_wires=True,
                    )
                    yield GateSpec(
                        kind="quansistor",
                        wires=(a, b),
                        d=spec.d,
                        seed=s,
                        tags=("layer", f"L{layer}", "leftover", f"p{j}"),
                        params=("leftover", True),
                    )
