"""Main script for simulating quantum circuits and computing properties.

This module simulates various quantum circuit families (Haar, Clifford, Quansistor)
and computes their stabilizer Rényi entropy (SRE) using different methods.
"""

from __future__ import annotations

import logging

from pathlib import Path
from typing import Any

from dask.distributed import Client, as_completed
from tqdm import tqdm

from backend import PennylaneBackend, QuimbBackend
from circuit.families import (
    CliffordBrickwork,
    HaarBrickwork,
    QuansistorBrickwork,
)
from experiments.runner import run_experiment
from experiments.sweeper import (
    JobConfig,
    compile_job,
    generate_jobs,
    aggregate_by_cond,
)
from experiments.visualizer import plot_pennylane_circuit
from utils import FileCache, RunStore, configure_logger, make_run_id
from parallel import dask_client

logger = logging.getLogger(__name__)

OUT = Path("circuit_outputs")
OUT.mkdir(exist_ok=True)
PROJECT_ROOT = Path(__file__).resolve().parent  # directory containing main.py
RUNS_ROOT = PROJECT_ROOT / "outputs" / "runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
cache = FileCache(PROJECT_ROOT / "outputs" / "cache")


def run_jobs(
    job: JobConfig,
    *,
    family_registry: dict[str, Any],
    backend_registry: dict[str, Any],
    cache: Any | None = None,
) -> dict[str, Any]:
    """Run a single job and return the results."""
    cfg = compile_job(
        job,
        family_registry=family_registry,
    )
    backend_name = cfg.backend.name
    backend_factory = backend_registry[backend_name]
    backend_instance = backend_factory() if callable(backend_factory) else backend_factory

    out = run_experiment(
        cfg,
        backend_registry={backend_name: backend_instance},
        cache=cache,
    )

    tcount = (job.family_params or {}).get("tcount", None)

    return {
        "tags": {
            "circuit_family": job.circuit_family,
            **(job.tags or {}),
            "n_qubits": cfg.spec.n_qubits,
            "n_layers": cfg.spec.n_layers,
            "d": cfg.spec.d,
            "family.tcount": tcount,
        },
        "results": {k: {"value": v.value, "meta": v.meta} for k, v in out.results.items()},
    }


if __name__ == "__main__":
    configure_logger(logging.INFO, logging.INFO)
    family_name = "clifford"
    n_qubits = 10
    d = 2
    n_layers = 40
    seed = 33
    tcount = 38
    method = "fwht"
    repeat = 10

    logger.info(
        "Simulating %s circuit with %d qubits, %d layers, d=%d.",
        family_name,
        n_qubits,
        n_layers,
        d,
    )

    run_id = make_run_id(label=f"{family_name}__n{n_qubits}_l{n_layers}_{method}")
    run_store = RunStore(RUNS_ROOT, run_id)

    family_registry = {
        "haar": HaarBrickwork,
        "clifford": CliffordBrickwork,
        "quansistor": QuansistorBrickwork,
    }

    backend_registry = {
        "pennylane": lambda: PennylaneBackend(),
        "quimb": lambda: QuimbBackend(),
    }

    experiment = JobConfig(
        circuit_family=family_name,
        n_qubits=n_qubits,
        n_layers=n_layers,
        d=d,
        family_params={},
        backend="quimb",
        backend_params={},
        properties=[{"name": "SRE", "method": method}],
        base_seed=seed,
        replicate=repeat,
        run_seed=0,
        tags={},
    )

    # Define sweep axes
    # To vary tcount use only Clifford family
    axes = {
        "circuit_family": ["haar", "clifford", "quansistor"],
        "n_qubits": list(range(6, n_qubits + 1, 2)),
        "family.tcount": [tcount],
    }

    run_store.write_run_header(
        {
            "circuit_family": experiment.circuit_family,
            "backend": experiment.backend,
            "axes": axes,
            "repeats": repeat,
            "properties": experiment.properties,
            "base_seed": experiment.base_seed,
        },
    )

    outputs = []

    jobs = generate_jobs(experiment, axes, repeats=repeat)

    with dask_client(
            mode="local",
            n_workers=4,
            threads_per_worker=1,
            dashboard=True,
        ) as client:
        for job in jobs:
            run_store.log_job(job)
        futures = [
            client.submit(
                run_jobs,
                job,
                family_registry=family_registry,
                backend_registry=backend_registry,
                cache=cache,
                pure=False,
            )
            for job in jobs
        ]

        try:
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Running jobs", unit="job"):
                out = fut.result()
                run_store.log_result(out)
                outputs.append(out)
        except Exception as e:
            logger.exception("Job failed")
            run_store.log_error({"error": str(e)})


    stats = aggregate_by_cond(
        outputs,
        group_keys=("circuit_family", "n_qubits"),
        value_path=("results", "SRE", "value"),
    )
    summary = {
        "group_keys": ["circuit_family", "n_qubits"],
        "stats": {
            str(k): {"mean": s.mean, "std": s.std, "stderr": s.stderr, "n": s.n}
            for k, s in stats.items()
        },
    }
    run_store.write_summary(summary)
    logger.info("Saved run to: %s", run_store.dir)

