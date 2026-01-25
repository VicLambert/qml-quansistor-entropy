from __future__ import annotations


def brickwork_pattern(
        n_qubits: int,
        layer: int,
        *,
        topology: str = "line",
) -> list[tuple[int, int]]:
    """Returns the nearest-neighbour pairs in brickwork pattern.

    Args:
        n_qubits: Number of qubits in the circuit.
        layer: Layer index (0-indexed).
        topology: Topology of the qubits, either "line" or "loop".
    Parity of layers sets the offset.
    """
    start = layer % 2
    pairs = [(i, i + 1) for i in range(start, n_qubits - 1 , 2)]

    if topology == "loop":
        raise NotImplementedError

    return pairs
