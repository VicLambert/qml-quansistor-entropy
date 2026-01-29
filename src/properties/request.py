
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PropertyRequest:
    name: str
    method: str = "exact"
    params: dict[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        return f"{self.name}: {self.method}"
