from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Union, Callable
import numpy as np

Value = Union[float, np.ndarray, np.floating]

@dataclass(frozen=True)
class PropertyResult:
    name: str
    value: Value
    error: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

ComputeFn = Callable[..., PropertyResult]
PropertyFn = Callable[..., PropertyResult]


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

_REGISTRY : dict[tuple[str, str], ComputeFn] = {}

def registry(name: str, method: str):
    def _decorator(fn: ComputeFn) -> ComputeFn:
        key = (name, method)
        if key in _REGISTRY:
            msg = "Property already registered"
            raise KeyError(msg)
        _REGISTRY[key] = fn
        return fn
    return _decorator

def get(name: str, method: str) -> ComputeFn:
    try:
        return _REGISTRY[(name, method)]
    except KeyError as e:
        available = ", ".join([f"{n}:{m}" for (n, m) in sorted(_REGISTRY.keys())])
        msg = f"Unknown property '{name}:{method}'. Available: {available}"
        raise KeyError(msg) from e
