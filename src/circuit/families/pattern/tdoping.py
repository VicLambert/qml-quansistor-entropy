from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Set, Tuple

Placement = Literal["center_pair", "center_wire", "random_wires"]

@dataclass(frozen=True)
class TdopingRules:
    count: int
    placement: Placement = "center_pair"
    per_layer: int = 2

    def location(
            self,
            *,
            n_qubit: int,
            layer: int,
            seed: int,
            topology: str,
    ) -> Set[Tuple[int, int]]:

        locs: set[tuple[int,int]] = set()
