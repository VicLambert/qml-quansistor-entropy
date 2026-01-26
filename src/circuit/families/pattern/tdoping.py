from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Placement = Literal["center_pair", "center_wire", "random_wires"]

@dataclass(frozen=True)
class TdopingRules:
    count: int
    placement: Placement = "center_pair"
    per_layer: int = 2

    def locations(
            self,
            *,
            n_qubits: int,
            n_layers: int,
            seed: int,
            connectivity: str,
    ) -> set[tuple[int, int]]:

        if n_qubits < 1:
            raise ValueError("Number of qubits must be >= 1.")

        if n_layers < 1 or self.count <= 0:
            return set()

        if self.per_layer < 1:
            raise ValueError("per_layer must be >= 1.")

        max_tgates = n_layers * min(self.per_layer, n_qubits)
        if self.count > max_tgates:
            raise ValueError(f"Requested T-gates ({self.count}) exceed maximum possible ({max_tgates}).")

        rng = np.random.default_rng(seed)
        remaining = self.count
        t_gate_per_layer = np.zeros(n_layers, dtype=int)

        n_distinct = min(remaining, n_layers)
        chosen_layers = rng.choice(n_layers, size=n_distinct, replace=False)

        t_gate_per_layer[chosen_layers] += 1
        remaining -= n_distinct

        while remaining > 0:
            elligible_layers = np.flatnonzero(t_gate_per_layer < min(self.per_layer, n_qubits))

            if len(elligible_layers) == 0:
                raise RuntimeError("No eligible layers left to place T-gates.")
            layer = int(rng.choice(elligible_layers))
            t_gate_per_layer[layer] += 1
            remaining -= 1

        locs: set[tuple[int,int]] = set()
        if n_qubits % 2 == 0:
            center_wires = (n_qubits // 2 - 1, n_qubits // 2)
        else:
            center_wires = (n_qubits // 2,)

        for layer in range(n_layers):
            n_tgates = t_gate_per_layer[layer]
            if n_tgates == 0:
                continue

            if self.placement == "center_pair":
                if n_tgates <= len(center_wires):
                    wires = rng.choice(list(center_wires), size=n_tgates, replace=False).tolist()
                else:
                    used = set(center_wires)
                    wires = list(center_wires)
                    remaining_wires = n_tgates - len(wires)
                    if remaining_wires > 0:
                        possible_wires = [w for w in range(n_qubits) if w not in used]
                        extra_wires = rng.choice(possible_wires, size=remaining_wires, replace=False).tolist()
                        wires.extend(extra_wires)

            elif self.placement == "center_wire":
                if len(center_wires) == 1:
                    wires = [center_wires[0]] * n_tgates
                else:
                    offset = int(rng.choice(0, 2))
                    wires = [center_wires[(offset + i) % 2] for i in range(n_tgates)]

            elif self.placement == "random_wires":
                possible_wires = rng.choice(n_qubits, size=n_tgates, replace=False).tolist()
            else:
                raise ValueError(f"Unknown placement strategy: {self.placement!r}")

            for w in wires:
                locs.add((layer, w))

        if len(locs) != self.count:
            raise RuntimeError("Mismatch in the number of T-gate locations generated.")
        return locs
