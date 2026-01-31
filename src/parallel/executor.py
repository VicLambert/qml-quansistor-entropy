from __future__ import annotations

import logging
import os

from typing import TYPE_CHECKING, Any, Sequence

from dask.distributed import Client, as_completed

from experiments.runner import run_experiment

if TYPE_CHECKING:
    from experiments.sweeper import ExperimentConfig

logger = logging.getLogger(__name__)


def _set_worker_thread_env() -> None:
    """Prevent BLAS oversubscription: each worker should typically use 1 thread."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def run_experiment_job(
    cfg: ExperimentConfig,
    *,
    backend_registry: dict[str, Any],
) -> dict[str, Any]:
    """Run a single experiment job and return the results as a dictionary."""
    _set_worker_thread_env()
    out = run_experiment(cfg, backend_registry=backend_registry)
    return {
        "spec_id": out.spec_id,
        "state_info": dict(out.state_info or {}),
        "meta_data": dict(out.meta_data or {}),
        "tags": dict(cfg.meta_data or {}),  # handy: what you sweep over
        "results": {
            name: {"value": r.value, "error": r.error, "meta": dict(r.meta or {})}
            for name, r in out.results.items()
        },
    }


def run_dask_experiments(
    client: Client,
    jobs: Sequence[ExperimentConfig],
    *,
    backend_registry: dict[str, Any],
    retries: int = 0,
    priority: int = 0,
    gather_errors: bool = True,
) -> list[dict[str, Any]]:
    """Run a list of experiment jobs in parallel using Dask.

    Args:
        client: Dask distributed client.
        jobs: List of ExperimentConfig defining the jobs to run.
        backend_registry: Registry of available backends.
        retries: Number of times to retry failed jobs.
        priority: Priority level for the jobs.
        gather_errors: If True, collect errors instead of raising.

    Returns:
        List of results corresponding to each job.
    """
    if not jobs:
        return []

    futures = [
        client.submit(
            run_experiment_job,
            job,
            backend_registry=backend_registry,
            priority=priority,
            retries=retries,
            pure=False,
        )
        for job in jobs
    ]

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    try:
        for future in as_completed(futures):
            outcome = future.result()
            results.append(outcome)
    except Exception as e:
        if gather_errors:
            logger.exception(
                "Job failed after %d/%d completions (error_type=%s, error_message=%s)",
                len(results),
                len(jobs),
                type(e).__name__,
                str(e),
            )
            errors.append({"error_type": type(e).__name__, "error_message": str(e)})
        else:
            raise
    if errors:
        logger.warning("%d/%d jobs failed", len(errors), len(jobs))
    return results
