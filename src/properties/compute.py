from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from .registry import get
from .request import PropertyRequest
from .results import PropertyResult

from src.properties.SRE.fwht_sre import compute as sre_fwht
from src.properties.SRE.sre_exact_dense import compute as sre_exact
from src.properties.SRE.monte_carlo_sre import compute as sre_sampled

PropertyFn = Callable[..., PropertyResult]

PROPERTY_REGISTRY: dict[str, dict[str, PropertyFn]] = {
    "SRE": {
        "fwht": sre_fwht,
        "exact": sre_exact,
        "sampling": sre_sampled,
        # add others here
    },
    # "entanglement_entropy": {...}
}

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

# def compute_property(state, request: PropertyRequest) -> PropertyResult:
#     fn = get(request.name, request.method)
#     # pass params as keyword args
#     return fn(state, **request.params)