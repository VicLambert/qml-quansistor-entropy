
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from itertools import product
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ExperimentConfig:
    circuit_family: str
    n_qubits: int
    n_layers: int
    d: int

    family_params: dict[str, Any]

    backend: str
    backend_params: dict[str, Any]

    properties: list[dict[str, Any]]

    base_seed: int
    replicate: int
    run_seed: int

    tags: dict[str, Any]

def _hash_job(obj: Any, bits: int = 32) -> int:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    x = int.from_bytes(h,"little")
    if bits == 32:
        return x & 0xFFFFFFFF
    return x

def apply_nested_sweep_keys(cfg: ExperimentConfig, cond: Mapping[str, Any]) -> ExperimentConfig:
    backend_params = dict(cfg.backend_params)
    family_params = dict(cfg.family_params)
    properties = [dict(p) for p in cfg.properties]

    for k, v in cond.items():
        if k.startswith("backend."):
            backend_params[k.split(".", 1)[1]] = v
        elif k.startswith("family."):
            family_params[k.split(".", 1)[1]] = v
        elif k.startswith("prop."):
            # Example: "prop.sre.chiP_max"
            _, prop_name, prop_key = k.split(".", 2)
            for p in properties:
                if p.get("name") == prop_name:
                    p.setdefault("params", {})[prop_key] = v

    return replace(cfg, backend_params=backend_params, family_params=family_params, properties=properties)

def derive_run_seed(base_seed: int, condition: Mapping[str, Any], replicate: int) -> int:
    payload = {"base_seed": base_seed, "condition": dict(condition), "replicate": replicate}
    return _hash_job(payload, bits=32)

def generate_cond(axes: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    keys = list(axes.keys())
    values = [axes[k] for k in keys]
    conds = []
    for prod in product(*values):
        conds.append({k: v for k, v in zip(keys, prod)})
    return conds

def generate_jobs(
    base_cfg: ExperimentConfig,
    axes: Mapping[str, Sequence[Any]],
    repeats: int,
    *,
    seed_key: str = "base_seed",
) -> list[ExperimentConfig]:
    conditions = generate_cond(axes)
    jobs : list[ExperimentConfig] = []

    for cond in conditions:
        cond_fingerprint = {
            "circuit_family": base_cfg.circuit_family,
            "d": base_cfg.d,
            **cond,
            "family_params": base_cfg.family_params,
            "backend_name": base_cfg.backend,
            "backend_params": base_cfg.backend_params,
            "properties": base_cfg.properties,
        }
        for r in range(repeats):
            run_seed = derive_run_seed(base_cfg.base_seed, cond_fingerprint, r)
            cfg = replace(
                base_cfg,
                replicate=r,
                run_seed=run_seed,
                tags={**(base_cfg.tags or {}), **cond, "replicate": r},
            )

            copy_dict = cfg.__dict__.copy()
            for k, v in cond.items():
                if k in copy_dict:
                    copy_dict[k] = v
                else:
                    pass

            cfg = ExperimentConfig(**copy_dict)
            cfg = apply_nested_sweep_keys(cfg, cond)

            jobs.append(cfg)

    return jobs


@dataclass(frozen=True)
class AggregateResults:
    mean: float
    std: float
    stderr: float
    n: int

def aggregate_by_cond(
    job_results: list[dict[str, Any]],
    *,
    group_keys: Sequence[str],
    value_path: tuple[str, ...] = ("results", "SRE", "value"),
) -> dict[tuple[Any, ...], AggregateResults]:
    groups = dict[tuple[Any, ...], list[float]] = {}

    for output in job_results:
        tags = output.get("tags", {})
        g_key = tuple(tags[k] for k in group_keys)

        x = output
        for p in value_path:
            x = x[p]
        val = float(x)

        groups.setdefault(g_key, []).append(val)
    stats: dict[tuple[Any, ...], AggregateResults] = {}
    for k, vals in groups.items():
        arr = np.asarray(vals, dtype=float)
        n = arr.size
        mean=float(arr.mean())
        std = float(arr.std(ddof=1)) if n > 1 else 0.0
        stderr = float(std / np.sqrt(n)) if n > 1 else 0.0
        stats[k] = AggregateResults(mean=mean, std=std, stderr=stderr, n=n)
    return stats
