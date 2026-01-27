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

        if n_layers < 2 or self.count <= 0:
            return set()

        if self.per_layer < 1:
            raise ValueError("per_layer must be >= 1.")

        # Exclude the last layer from T-gate placement
        available_layers = n_layers - 1
        max_tgates = available_layers * min(self.per_layer, n_qubits)
        if self.count > max_tgates:
            raise ValueError(f"Requested T-gates ({self.count}) exceed maximum possible ({max_tgates}).")

        rng = np.random.default_rng(seed)
        if n_qubits % 2 == 0:
            center_wires = (n_qubits // 2 - 1, n_qubits // 2)
        else:
            center_wires = (n_qubits // 2,)

        def pick_wires_for_layer(k: int) -> list[int]:
            if k <= 0:
                return []

            if self.placement == "random_wires":
                return [int(x) for x in rng.choice(n_qubits, size=k, replace=False)]

            if self.placement == "center_wire":
                c = center_wires[0]
                wires = [c]
                if k == 1:
                    return wires
                step = 1

                while len(wires) < k:
                    left = c - step
                    right = c + step
                    if left >= 0:
                        wires.append(left)
                        if len(wires) == k:
                            break
                    if right < n_qubits:
                        wires.append(right)
                        if len(wires) == k:
                            break
                    step += 1
                return wires[:k]
            if self.placement == "center_pair":
                base = list(center_wires)
                if k <= len(base):
                    return [int(x) for x in rng.choice(base, size=k, replace=False)]
                used = set(base)
                wires = base[:]
                mid = center_wires[0]
                step = 1

                while len(wires) < k:
                    for w in (mid - step, mid + step, (center_wires[-1] + step) if len(center_wires) == 2 else None):
                        if w is None:
                            continue
                        if 0 <= w < n_qubits and w not in used:
                            used.add(w)
                            wires.append(w)
                            if len(wires) == k:
                                break
                    step += 1
                return wires[:k]
            raise ValueError(f"Unknown placement strategy")


        remaining = self.count
        t_gate_per_layer = np.zeros(n_layers, dtype=int)

        # Only choose from layers excluding the last one
        available_layers = n_layers - 1
        n_distinct = min(remaining, available_layers)
        chosen_layers = rng.choice(available_layers, size=n_distinct, replace=False)

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

        for layer in range(n_layers):
            n_tgates = int(t_gate_per_layer[layer])
            if n_tgates == 0:
                continue
            wires = pick_wires_for_layer(n_tgates)
            if len(set(wires)) != len(wires):
                raise RuntimeError("Internal Error")
            for w in wires:
                locs.add((layer, w))

        if len(locs) != self.count:
            raise RuntimeError("Mismatch in the number of T-gate locations generated.")
        return locs
