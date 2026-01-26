
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    import numpy as np

Value = Union[float, np.ndarray]

@dataclass(frozen=True)
class PropertyResult:
    name: str
    value: Value
    error: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)
