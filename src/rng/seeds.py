from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Union

import numpy as np

WireLike = Union[int, Iterable[int]]

def _u64_to_bytes(x: int) -> bytes:
    if x < 0:
        raise ValueError("Only non-negative integers can be converted to bytes.")
    return (x & ((1 << 64) - 1)).to_bytes(8, byteorder="little", signed=False)

def _encode_str(s: str) -> bytes:
    return s.encode("utf-8")

def _normalize_wires(wires: WireLike | None, *, ordered: bool) -> tuple[int, ...]:
    if wires is None:
        return tuple()
    w = (wires,) if isinstance(wires, int) else tuple(int(x) for x in wires)
    return w if ordered else tuple(sorted(w))

def gate_seed(
        global_seed: int,
        *,
        kind: str,
        layer: int,
        slot: int = 0,
        wires: WireLike | None = None,
        extra: str = "",
        ordered_wires: bool = False,
    ) -> int:
    """Generates a seed for a specific gate based on the global seed and gate properties.

    Args:
        global_seed: The global seed for the circuit.
        kind: The kind of gate (e.g., "haar", "unitary1", etc.).
        layer: The layer index of the gate.
        slot: The slot index of the gate within the layer.
        wires: The wires (qubits) the gate acts on.
        extra: An optional extra string to differentiate seeds.
        ordered_wires: Whether to consider wire order in seed generation.

    Returns:
        An integer seed for the gate.
    """
    w = _normalize_wires(wires, ordered=ordered_wires)
    payload = b"|".join(
        [
            b"v1",
            _u64_to_bytes(int(global_seed)),
            _encode_str(kind),
            _u64_to_bytes(int(layer)),
            _u64_to_bytes(int(slot)),
            b"|".join(_u64_to_bytes(int(x)) for x in w),
            _encode_str(extra),
        ]
    )
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)

def rng_from_gate(**kwargs) -> np.random.Generator:
    """Creates a NumPy random number generator for a specific gate."""
    seed = gate_seed(**kwargs)
    return np.random.default_rng(seed)

def spawn_seed(global_seed: int, *, name: str) -> int:
    """Generates a new seed spawned from the global seed and a name."""
    return gate_seed(global_seed, kind="spawn", layer=0, extra=name)

@dataclass(frozen=True)
class SeedSchedule:
    """A schedule for generating seeds for gates based on a global seed and settings."""
    global_seed: int
    ordered_wires: bool = False

    def seed(
        self,
        *,
        kind: str,
        layer: int,
        slot: int = 0,
        wires: WireLike | None = None,
        extra: str = "",
        ordered_wires: bool | None = None,
    ) -> int:
        """Generates a seed for a specific gate using the schedule's global seed and settings."""
        return gate_seed(
            self.global_seed,
            kind=kind,
            layer=layer,
            slot=slot,
            wires=wires,
            extra=extra,
            ordered_wires=self.ordered_wires if ordered_wires is None else ordered_wires,
    )

    def rng(
        self,
        *,
        kind: str,
        layer: int,
        slot: int = 0,
        wires: WireLike | None = None,
        extra: str = "",
        ordered_wires: bool | None = None,
    ) -> np.random.Generator:
        """Creates a NumPy random number generator for a specific gate using the schedule's global seed and settings."""
        seed = self.seed(
            kind=kind,
            layer=layer,
            slot=slot,
            wires=wires,
            extra=extra,
            ordered_wires=ordered_wires,
        )
        return np.random.default_rng(seed)