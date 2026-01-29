
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from src.experiments.runner_config import ExperimentConfig, ExperimentRun
from src.properties.compute import compute_property
from src.properties.request import PropertyRequest
from src.properties.results import PropertyResult
from src.states.types import DenseState, MPSState


def _summarize_state(state: Any) -> dict[str, Any]:
    """
    Collect lightweight info about the produced state without depending on the backend.
    """
    info: dict[str, Any] = {"state_type": type(state).__name__}

    # Your wrappers:
    if isinstance(state, DenseState):
        info.update({"n": state.n, "d": state.d, "dtype": str(state.vec.dtype), "shape": tuple(state.vec.shape)})
    elif isinstance(state, MPSState):
        info.update({"n": state.n, "d": state.d})
        # Optional: try to record bonds if available
        mps = state.mps
        if hasattr(mps, "bond_sizes"):
            try:
                info["bond_sizes"] = list(mps.bond_sizes())
            except Exception:
                pass
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
) -> ExperimentRun:
    t0 = time.time()
    spec_id = cfg.spec.spec_id()

    if cfg.backend.name not in backend_registry:
        raise KeyError(
            f"Unknown backend '{cfg.backend.name}'. "
            f"Available: {list(backend_registry.keys())}"
        )
    backend = backend_registry[cfg.backend.name]

    sim_second = None
    if state is None:
        t_sim = time.time()
        state = backend.simulate(
            cfg.spec,
            representation=cfg.backend.representation,
            **cfg.backend.params,
        )
        sim_second = time.time() - t_sim

    state_info = _summarize_state(state)
    if sim_second is not None:
        state_info["simulate_seconds"] = sim_second

    results: dict[str, Any] = {}
    for req in cfg.properties:
        key = req.key() if hasattr(req, "key") else f"{req.name}:{req.method}"
        t_comp = time.time()

        try:
            res = compute_property(state, req)
            if res.meta is None:
                res_details = {"compute_time": time.time() - t_comp}
                res = type(res)(name=res.name, value=res.value, error=res.error, details=res_details)
            else:
                res.meta.setdefault("compute_time", time.time() - t_comp)
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

