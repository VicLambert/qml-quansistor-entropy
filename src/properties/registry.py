from __future__ import annotations

from typing import Callable

from .request import PropertyRequest
from .results import PropertyResult

ComputeFn = Callable[..., PropertyResult]

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
