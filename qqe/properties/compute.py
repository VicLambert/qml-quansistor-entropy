"""Compute properties for quantum states.

This module provides functions to compute various quantum properties such as
entanglement entropy and subsystem Rényi entropy (SRE) using different methods.
"""

from __future__ import annotations

from typing import Any, Callable, Union

import numpy as np

from qqe.properties.entanglement_entropy import renyi_ee, von_neumann_ee
from qqe.properties.results import PropertyRequest, PropertyResult
from qqe.properties.SRE import sre_exact, sre_fwht, sre_mcmc

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
    },
}

ComputeFn = Callable[..., PropertyResult]
Value = Union[float, np.ndarray, np.floating]
PropertyFn = Callable[..., PropertyResult]


def available_properties() -> list[str]:
    return [
        f"{name}:{method}" for name, methods in PROPERTY_REGISTRY.items() for method in methods
    ]


def compute_property(state, req: PropertyRequest) -> PropertyResult:
    name = req.normalized_name()
    method = req.normalized_method()

    try:
        fn = PROPERTY_REGISTRY[name][method]
    except KeyError as e:
        msg = f"Unknown property '{name}:{method}'. Available: {available_properties()}"
        raise KeyError(msg) from e

    return fn(state, **(req.params or {}))
