
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Any

import numpy as np

from qqe.circuit.spec import CircuitSpec, GateSpec

GATES = [
    "IN", "OUT",
    "CNOT",                             # random + clifford
    "RX", "RY", "RZ",                   # random
    "H", "S", "T",                      # clifford
    "haar",                             # haar
    "quansistor_X", "quansistor_Y",     # quansistor
]

def bin_tetha(theta: float, n_bins:int = 50) -> int:
    t = theta % (2 * np.pi)
    idx = int(np.floor(n_bins * t / (2 * np.pi)))

    return min(max(idx, 0), n_bins - 1)


