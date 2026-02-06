"""Main script for simulating quantum circuits and computing properties.

This module simulates various quantum circuit families (Haar, Clifford, Quansistor)
and computes their stabilizer Rényi entropy (SRE) using different methods.
"""

from __future__ import annotations

import json
import logging

from pathlib import Path
from typing import Any

import numpy as np
import typer

from dask.distributed import as_completed
from tqdm import tqdm

from qqe.backend import PennylaneBackend, QuimbBackend
from qqe.circuit.families import (
    CliffordBrickwork,
    HaarBrickwork,
    QuansistorBrickwork,
)
from qqe.experiments.core import run_experiment
from qqe.experiments.plotting import (
    plot_sre,
    plot_sredensity_v_tcount,
)
from qqe.experiments.sweeper import (
    JobConfig,
    aggregate_by_cond,
    compile_job,
    generate_jobs,
)
from qqe.parallel import dask_client
from qqe.utils import FileCache, RunStore, configure_logger, make_run_id

logger = logging.getLogger(__name__)


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
    logger.info("job tags=%s family_params=%s", job.tags, job.family_params)

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

def _plot(
    results_path: str | None,
    quantity: str,
    method: str,
    n_layers: int,
) -> None:
    if not results_path:
        logger.error("results_path must be provided for plotting.")
        return

    p = Path(results_path)
    if p.is_dir():
        file = p / "summary.json"
        if file.exists():
            p = file
        else:
            logger.error("No summary.json found in directory: %s", p)
            return
    with p.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    group_keys = summary.get("group_keys", [])
    if group_keys == ["circuit_family", "n_qubits"]:
        plot_sre(
            results=summary,
            quantity=quantity,
            save_path= f"outputs/figures/sre_vs_nqubits_{method}_{n_layers}l.png",
            show=True,
        )
    elif group_keys == ["n_qubits", "family.tcount"]:
        plot_sredensity_v_tcount(
            results=summary,
            save_path= f"outputs/figures/sredensity_vs_tcount_{method}_{n_layers}l.png",
            show=True,
        )
    else:
        logger.error("Unknown group_keys in summary: %s", group_keys)
        return
    logger.info("Plotting completed.")


def _run_once(
    experiment: JobConfig,
    axes: dict[str, list[Any]],
    run_label: str,
    family_registry: dict[str, Any],
    backend_registry: dict[str, Any],
    repeat: int,
) -> tuple[list[dict[str, Any]], RunStore]:
    run_id = make_run_id(label=run_label)
    run_store = RunStore(RUNS_ROOT, run_id)

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
    outputs: list[dict[str, Any]] = []
    jobs = generate_jobs(experiment, axes, repeats=repeat)

    with dask_client(mode="local", n_workers=4, threads_per_worker=1, dashboard=True) as client:
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
    return outputs, run_store

def _sweep_jobs(
    experiment: JobConfig,
    sweep_type: str,
    sweep_axes: dict[str, dict[str, list[Any]]],
    family_registry: dict[str, Any],
    backend_registry: dict[str, Any],
    repeat: int,
    quantity: str,
    method: str,
) -> tuple[list[dict[str, Any]], RunStore, list[str]]:
    if sweep_type not in sweep_axes:
        logger.error("Invalid or missing sweep_type: %s", sweep_type)
        raise ValueError(f"Invalid sweep_type: {sweep_type}")

    axes = sweep_axes[sweep_type]
    run_label = f"{quantity}_sweep_{sweep_type}_{method}"
    run_id = make_run_id(label=run_label)
    run_store = RunStore(RUNS_ROOT, run_id)

    run_store.write_run_header(
        {
            "backend": experiment.backend,
            "axes": axes,
            "repeats": repeat,
            "properties": experiment.properties,
            "base_seed": experiment.base_seed,
        },
    )

    outputs: list[dict[str, Any]] = []
    jobs = generate_jobs(experiment, axes, repeats=repeat)

    with dask_client(mode="local", n_workers=4, threads_per_worker=1, dashboard=True) as client:
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
    if sweep_type == "n_qubits" or sweep_type == "n_layers":
        group_keys = ["circuit_family", "n_qubits" if sweep_type == "n_qubits" else "n_layers"]
    else:
        group_keys = ["n_qubits", "family.tcount"]
    return outputs, run_store, group_keys

def main(
    run: str,                               #run_once, sweep, plot...
    n_qubits: int,
    n_layers: int,
    circuit_family: str = "clifford",
    *,
    quantity: str = "SRE",                  #SRE, entanglement_entropy...
    backend: str = "quimb",
    method: str = "fwht",
    repeat: int = 5,
    seed: int = 42,
    sweep_type: str | None = None,          #n_qubits, n_layers, tcount
    results_path: str | None = None,
) -> None:
    """Main function to run simulations and compute SRE properties.

    Args:
        run: Type of run to perform ('run_once', 'sweep', 'plot').
        n_qubits: Number of qubits in the circuit.
        n_layers: Number of layers in the circuit.
        circuit_family: Circuit family to simulate ('haar', 'clifford', 'quansistor').
        quantity: Quantity to compute ('SRE').
        backend: Backend to use for simulation ('pennylane' or 'quimb').
        method: Method to compute SRE ('fwht', 'exact', etc.).
        repeat: Number of repetitions for averaging results.
        seed: Base random seed for reproducibility.
        sweep_type: Type of sweep to perform (if any).
        results_path: Path to obtain results (if any).
    """
    run = (run or "").strip().lower()

    if run not in {"run_once", "sweep", "plot"}:
        logger.error("Invalid run type: %s. Must be 'run', 'sweep', or 'plot'.", run)
        return

    if run == "plot":
        _plot(results_path, quantity, method, n_layers)
        return

    family_registry: dict[str, Any] = {
        "haar": HaarBrickwork,
        "clifford": CliffordBrickwork,
        "quansistor": QuansistorBrickwork,
    }
    backend_registry: dict[str, Any] = {
        "pennylane": lambda: PennylaneBackend(),
        "quimb": lambda: QuimbBackend(),
    }

    logger.info(
        "Simulating %s circuit with %d qubits, %d layers, d=2.",
        circuit_family,
        n_qubits,
        n_layers,
    )
    rng = np.random.default_rng(seed)
    experiment = JobConfig(
        circuit_family=circuit_family,
        n_qubits=n_qubits,
        n_layers=n_layers,
        d=2,
        family_params={},
        backend=backend,
        backend_params={},
        properties=[{"name": quantity, "method": method}],
        base_seed=seed,
        replicate=repeat,
        run_seed=int(rng.integers(0, 2**32 - 1)),
        tags={},
    )
    tcount_default = n_layers - 1
    tcount_max = int(getattr(experiment, "family_params", {}).get("tcount", 0) or 0)
    if tcount_max == 0:
        tcount_max = min(2 * n_layers, 2 * n_qubits * n_layers)

    sweep_axes = {
        "n_qubits": {
            "circuit_family": ["haar", "clifford", "quansistor"],
            "n_qubits": list(range(4, n_qubits + 1, 2)),
            "family.tcount": [tcount_default],
        },
        "n_layers": {
            "circuit_family": ["haar", "quansistor"],
            "n_layers": list(range(1, n_layers + 1, 1)),
            "family.tcount": [tcount_default],
        },
        "tcount": {
            "circuit_family": ["clifford"],
            "n_qubits": list(range(4, n_qubits + 1, 2)),
            "family.tcount": list(range(0, tcount_default + 1, 2)),
        },
    }

    if run == "run_once":
        axes = {
            "circuit_family": [circuit_family],
            "n_qubits": [n_qubits],
            "n_layers": [n_layers],
            # use default tcount for clifford-like; None for others
            "family.tcount": [tcount_default] if circuit_family == "clifford" else [None],
        }
        run_label = f"{circuit_family}__n{n_qubits}_l{n_layers}_{method}_once"
        outputs, run_store = _run_once(
            experiment,
            axes,
            run_label,
            family_registry,
            backend_registry,
            repeat,
        )
        group_keys = ["circuit_family", "n_qubits", "n_layers"]

    else:
        if sweep_type is None or sweep_type not in sweep_axes:
            logger.error("Invalid or missing sweep_type: %s", sweep_type)
            return
        outputs, run_store, group_keys = _sweep_jobs(
            experiment,
            sweep_type,
            sweep_axes,
            family_registry,
            backend_registry,
            repeat,
            quantity,
            method,
        )

    stats = aggregate_by_cond(
        outputs,
        group_keys=tuple(group_keys),       #TODO implement automatic detection of grouping keys
        value_path=("results", quantity, "value"),
    )

    summary = {
        "group_keys": list(group_keys),
        "stats": {
            str(k): {"mean": s.mean, "std": s.std, "stderr": s.stderr, "n": s.repeat}
            for k, s in stats.items()
        },
    }
    fig_dir = PROJECT_ROOT / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if run == "sweep" and sweep_type == "tcount":
        plot_sredensity_v_tcount(
            results=summary,
            save_path= str(fig_dir / f"sredensity_vs_tcount_{method}_{n_layers}l.png"),
            show=False,
        )
    elif run == "sweep" and sweep_type == "n_qubits":
        plot_sre(
            results=summary,
            quantity=quantity,
            save_path= str(fig_dir / f"{quantity}_vs_nqubits_{method}_{n_layers}l.png"),
            show=False,
        )
    elif run == "sweep" and sweep_type == "n_layers":
        plot_sre(
            results=summary,
            quantity=quantity,
            save_path= str(fig_dir / f"{quantity}_vs_nlayers_{method}_{n_qubits}q.png"),
            show=False,
        )
    else:
        logger.info("No plotting for run type '%s' with sweep_type '%s'.", run, sweep_type)

    logger.info("Saved figures to: %s", fig_dir)

    run_store.write_summary(summary)
    logger.info("Saved run to: %s", run_store.dir)



if __name__ == "__main__":
    configure_logger(logging.INFO, logging.INFO)
    typer.run(main)
