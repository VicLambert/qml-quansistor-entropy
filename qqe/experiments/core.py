from __future__ import annotations

import contextlib
import time

from dataclasses import dataclass, field
from typing import Any

# from qqe.circuit.DAG import circuit_spec_to_dag
from qqe.circuit.spec import CircuitSpec
from qqe.properties.compute import PropertyRequest, PropertyResult, compute_property
from qqe.states.types import DenseState, MPSState
from qqe.utils.reading import FileCache, cache_lock, make_property_cache_key


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


def _summarize_state(state: Any) -> dict[str, Any]:
    """Collect lightweight info about the produced state without depending on the backend."""
    info: dict[str, Any] = {"state_type": type(state).__name__}

    # Your wrappers:
    if isinstance(state, DenseState):
        info.update(
            {
                "n": state.n_qubits,
                "d": state.d,
                "dtype": str(state.vector.dtype),
                "shape": tuple(state.vector.shape),
            },
        )
    elif isinstance(state, MPSState):
        info.update({"n": state.n_qubits, "d": state.d})
        # Optional: try to record bonds if available
        mps = state.mps
        if hasattr(mps, "bond_sizes"):
            with contextlib.suppress(Exception):
                info["bond_sizes"] = list(mps.bond_sizes())
    else:
        # unknown state type, store minimal info only
        pass

    return info


def run_experiment(
    cfg: ExperimentConfig,
    *,
    backend_registry: dict[str, Any],
    state: Any | None = None,
    raise_property_error: bool = False,
    cache: FileCache | None = None,
) -> ExperimentRun:
    t0 = time.time()
    spec_id = cfg.spec.spec_id()

    if cfg.backend.name not in backend_registry:
        msg = (
            f"Unknown backend '{cfg.backend.name}'. "
            f"Available: {list(backend_registry.keys())}"
        )
        raise KeyError(msg)
    backend_obj = backend_registry[cfg.backend.name]
    backend = backend_obj() if callable(backend_obj) else backend_obj

    sim_second = None
    if state is None:
        t_sim = time.time()
        state = backend.simulate(
            cfg.spec,
            representation=cfg.backend.representation,
            **cfg.backend.params,
        )
        sim_second = time.time() - t_sim

    # dag = circuit_spec_to_dag(cfg.spec)

    state_info = _summarize_state(state)
    if sim_second is not None:
        state_info["simulate_seconds"] = sim_second

    results: dict[str, Any] = {}
    for req in cfg.properties:
        key = req.name
        t_comp = time.time()

        cache_key = None
        if cache is not None:
            cache_key = make_property_cache_key(
                spec_id=spec_id, backend_cfg=cfg.backend, req=req,
            )

            cached = cache.load_json(cache_key) 
            if cached is not None:
                # reconstruct PropertyResult from cached payload
                results[key] = PropertyResult(
                    name=req.name,
                    value=float(cached["value"]),
                    error=cached.get("error"),
                    meta=dict(cached.get("meta") or {}),
                )
                continue

        try:
            # if we want to avoid duplicate compute in parallel, lock around compute+write
            if cache is not None and cache_key is not None:
                lock_path = cache.get_lock_path(cache_key)
                with cache_lock(lock_path, timeout_s=60.0):
                    # check again after waiting
                    cached = cache.load_json(cache_key)
                    if cached is not None:
                        results[key] = PropertyResult(
                            name=req.name,
                            value=float(cached["value"]),
                            error=cached.get("error"),
                            meta=dict(cached.get("meta") or {}),
                        )
                        continue

                    res = compute_property(state, req)
                    meta = dict(res.meta or {})
                    meta.setdefault("compute_time", time.time() - t_comp)

                    payload = {
                        "_schema": {"type": "property_result", "version": 1},
                        "value": res.value,
                        "error": res.error,
                        "meta": meta,
                    }
                    cache.save_json(cache_key, payload)
            else:
                res = compute_property(state, req)
                meta = dict(res.meta or {})
                meta.setdefault("compute_time", time.time() - t_comp)
                res = PropertyResult(
                    name=res.name,
                    value=res.value,
                    error=res.error,
                    meta=meta,
                )

            results[key] = res

        except Exception as e:
            if raise_property_error:
                raise
            res = PropertyResult(
                name=req.name,
                value=float("nan"),
                error=None,
                meta={
                    "method": req.method,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "compute_time": time.time() - t_comp,
                },
            )
        results[key] = res
    total_seconds = time.time() - t0
    metadata = dict(cfg.meta_data)
    metadata.setdefault("total_time", total_seconds)
    return ExperimentRun(
        spec_id=spec_id,
        backend=cfg.backend,
        results=results,
        state_info=state_info,
        meta_data=metadata,
    )
