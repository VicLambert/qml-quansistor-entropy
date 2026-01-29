from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.circuit.spec import CircuitSpec
from src.properties.request import PropertyRequest
from src.properties.results import PropertyResult


@dataclass(frozen=True)
class BackendConfig:
    name: str               # "quimb" or "pennylane"
    representation: str     # "dense" or "mps"...
    params: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ExperimentConfig:
    """Defines the parameters for the run."""
    spec: CircuitSpec
    backend: BackendConfig
    properties: list[PropertyRequest]
    meta_data: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ExperimentRun:
    """Defines what to run."""
    spec_id: str
    backend: BackendConfig
    results: dict[str, PropertyResult]
    state_info: dict[str, Any] = field(default_factory=dict)
    meta_data: dict[str, Any] = field(default_factory=dict)