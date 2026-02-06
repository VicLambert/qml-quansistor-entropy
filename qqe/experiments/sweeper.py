"""Module for generating experiment configurations and aggregating results.

This module provides utilities for sweeping over parameter spaces, generating experiment jobs,
and aggregating results for quantum circuit experiments.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging

from dataclasses import dataclass, replace
from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np

from qqe.circuit.patterns import TdopingRules
from qqe.experiments.core import ExperimentConfig
from qqe.properties.compute import PropertyRequest
from qqe.states.types import BackendConfig
from qqe.utils.reading import _to_jsonable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JobConfig:
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
    jsonable_obj = _to_jsonable(obj)
    s = json.dumps(jsonable_obj, sort_keys=True, separators=(",", ":"))
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    x = int.from_bytes(h, "little")
    if bits == 32:
        return x & 0xFFFFFFFF
    return x

def derive_run_seed(base_seed: int, condition: Mapping[str, Any], replicate: int) -> int:
    payload = {"base_seed": base_seed, "condition": dict(condition), "replicate": replicate}
    return _hash_job(payload, bits=32)


def generate_cond(axes: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    keys = list(axes.keys())
    values = [axes[k] for k in keys]
    return [dict(zip(keys, prod)) for prod in product(*values)]


def apply_nested_sweep_keys(
    job: JobConfig,
    cond: Mapping[str, Any],
) -> JobConfig:
    backend_params = dict(job.backend_params)
    family_params = dict(job.family_params)
    properties = [dict(p) for p in job.properties]

    for k, v in cond.items():
        if k.startswith("backend."):
            backend_params[k.split(".", 1)[1]] = v
        elif k.startswith("family."):
            if job.circuit_family != "clifford":
                continue
            family_params[k.split(".", 1)[1]] = v
        elif k.startswith("prop."):
            # Example: "prop.sre.chiP_max"
            _, prop_name, prop_key = k.split(".", 2)
            for p in properties:
                if p.get("name") == prop_name:
                    p.setdefault("params", {})[prop_key] = v

    return replace(
        job,
        backend_params=backend_params,
        family_params=family_params,
        properties=properties,
    )


def generate_jobs(
    base_job: JobConfig,
    axes: Mapping[str, Sequence[Any]],
    repeats: int,
    *,
    seed_key: str = "base_seed",
) -> list[JobConfig]:
    conditions = generate_cond(axes)
    jobs: list[JobConfig] = []
    fields = set(base_job.__dataclass_fields__.keys())


    for cond in conditions:
        cond_fingerprint = {
            "circuit_family": base_job.circuit_family,
            "d": base_job.d,
            **cond,
            "family_params": base_job.family_params,
            "backend_name": base_job.backend,
            "backend_params": base_job.backend_params,
            "properties": base_job.properties,
        }
        for r in range(repeats):
            run_seed = derive_run_seed(base_job.base_seed, cond_fingerprint, r)

            job = replace(
                base_job,
                replicate=r,
                run_seed=run_seed,
                tags={**(base_job.tags or {}), **cond, "replicate": r},
            )

            direct_updates = {k: v for k, v in cond.items() if k in fields}
            if direct_updates:
                job = replace(job, **direct_updates)
            core_tags = {
                "circuit_family": job.circuit_family,
                "backend": job.backend,
                "n_qubits": job.n_qubits,
                "n_layers": job.n_layers,
                "d": job.d,
                "replicate": job.replicate,
                "run_seed": job.run_seed,
            }

            job = apply_nested_sweep_keys(job, cond)

            tags = {**(base_job.tags or {}), **cond, **core_tags}
            if job.circuit_family != "clifford":
                tags.pop("family.tcount", None)

            job = replace(job, tags=tags)
            jobs.append(job)

    return jobs

def compile_job(
    job: JobConfig,
    *,
    family_registry: dict[str, Any],
) -> ExperimentConfig:
    if job.circuit_family not in family_registry:
        msg = (
            f"Unknown circuit family '{job.circuit_family}'. "
            f"Available: {list(family_registry.keys())}"
        )
    family_params = dict(job.family_params or {})

    if job.circuit_family == "clifford":
        tcount = int(family_params.pop("tcount", 0))  # remove it so it doesn't get forwarded blindly
        # keep any explicit tdoping if user already provided one
        if "tdoping" not in family_params:
            family_params["tdoping"] = TdopingRules(
                count=tcount,
                placement="center_pair",
                per_layer=2,
            )

    family_cls = family_registry[job.circuit_family]
    family = family_cls(**family_params) if callable(family_cls) else family_cls
    spec = family.make_spec(
        n_qubits=job.n_qubits,
        n_layers=job.n_layers,
        d=job.d,
        seed=job.run_seed,
        **(family_params or {}),
    )

    backend_cfg = BackendConfig(
        name=job.backend,
        representation=job.backend_params.get("representation", "dense"),
        params={k: v for k, v in (job.backend_params or {}).items() if k != "representation"},
    )

    props_reqs: list[PropertyRequest] = []
    for p in job.properties:
        params = dict(p.get("params", {}))

        method = (p.get("method", "default") or "default").lower()
        name = p["name"]

        # Auto-seed stochastic methods
        if method == "sampling":
            params.setdefault("seed", job.run_seed)
            params.setdefault("n_samples", 10000)   # choose your default

        props_reqs.append(PropertyRequest(name=name, method=method, params=params))

    meta= dict(job.tags or {})
    meta.update({"run_seed": job.run_seed, "replicate": job.replicate})

    return ExperimentConfig(
        spec=spec,
        backend=backend_cfg,
        properties=props_reqs,
        meta_data=meta,
    )




@dataclass(frozen=True)
class AggregateResults:
    mean: float
    std: float
    stderr: float
    repeat: int

def aggregate_by_cond(
    job_results: list[dict[str, Any]],
    *,
    group_keys: Sequence[str],
    value_path: tuple[str, ...] = ("results", "SRE", "value"),
) -> dict[tuple[Any, ...], AggregateResults]:
    groups: dict[tuple[Any, ...], list[float]] = {}

    for output in job_results:
        tags = output.get("meta_data") or {}
        tags = output.get("tags") or output.get("meta_data") or {}
        g_key_values = [tags.get(k) for k in group_keys]
        # Skip outputs missing any required grouping tag to avoid None in grouping keys
        if any(v is None for v in g_key_values):
            continue
        g_key = tuple(g_key_values)

        try:
            current = output
            for p in value_path:
                current = current[p]
            assert isinstance(current, float), "Expected numeric value at specified value_path"
            val = float(current)
        except Exception:
            # Skip outputs where the value_path cannot be fully resolved or cast to float
            continue

        groups.setdefault(g_key, []).append(val)

    stats: dict[tuple[Any, ...], AggregateResults] = {}
    for k, vals in groups.items():
        arr = np.asarray(vals, dtype=float)
        size = arr.size
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if size > 1 else 0.0
        stderr = float(std / np.sqrt(size)) if size > 1 else 0.0
        stats[k] = AggregateResults(mean=mean, std=std, stderr=stderr, repeat=size)
    return stats
