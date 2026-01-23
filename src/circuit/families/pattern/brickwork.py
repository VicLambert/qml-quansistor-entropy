from __future__ import annotations

from typing import Tuple, list


def brickwork_pattern(
        n_qubits: int,
        layer: int,
        *,
        topology: str = "line",
) -> list[Tuple[int, int]]:
    """Returns the nearest-neighbour pairs in brickwork pattern.
    Parity of layers sets the offset
    """
    start = layer % 2
    pairs = [(i, i + 1) for i in range(start, n_qubits - 1 , 2)]

    if topology == "loop":
        raise NotImplementedError

    return pairs
