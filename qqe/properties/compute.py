from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Any, Union

import numpy as np

from qqe.properties.SRE import sre_fwht
from qqe.properties.SRE import sre_exact
from qqe.properties.SRE import sre_mcmc

from qqe.properties.entanglement_entropy import renyi_ee
from qqe.properties.entanglement_entropy import von_neumann_ee


PROPERTY_REGISTRY: dict[str, dict[str, Any]] = {
    "SRE": {
        "fwht": sre_fwht,
        "exact": sre_exact,
        "sampling": sre_mcmc,
        # add others here
    },
    "entanglement_entropy": {
        "renyi": renyi_ee,
        "von_neumann": von_neumann_ee,
    }
}

@dataclass(frozen=True)
class PropertyResult:
    name: str
    value: Value
    error: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


ComputeFn = Callable[..., PropertyResult]
Value = Union[float, np.ndarray, np.floating]
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


def available_properties() -> list[str]:
    out = []
    for name, methods in PROPERTY_REGISTRY.items():
        for method in methods:
            out.append(f"{name}:{method}")
    return out

def compute_property(state, req: PropertyRequest) -> PropertyResult:
    name = req.normalized_name()
    method = req.normalized_method()

    try:
        fn = PROPERTY_REGISTRY[name][method]
    except KeyError as e:
        msg = f"Unknown property '{name}:{method}'. Available: {available_properties()}"
        raise KeyError(msg) from e

    return fn(state, **(req.params or {}))
