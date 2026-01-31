
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PropertyRequest:
    name: str
    method: str = "exact"
    params: dict[str, Any] = field(default_factory=dict)

    def normalized_method(self) -> str:
        return (self.method or "default").strip().lower()

    def normalized_name(self) -> str:
        return (self.name or "").strip()

    def key(self) -> str:
        return f"{self.normalized_name()}:{self.normalized_method()}"
