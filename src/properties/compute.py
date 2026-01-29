from __future__ import annotations

from .registry import get
from .request import PropertyRequest
from .results import PropertyResult


def compute_property(state, request: PropertyRequest) -> PropertyResult:
    fn = get(request.name, request.method)
    # pass params as keyword args
    return fn(state, **request.params)