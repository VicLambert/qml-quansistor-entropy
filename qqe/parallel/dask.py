
from __future__ import annotations

import logging
import os

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator, Literal, Sequence

from dask.distributed import Client, LocalCluster, performance_report, as_completed
from dask_jobqueue import SLURMCluster

from qqe.experiments.core import run_experiment
from qqe.experiments.sweeper import ExperimentConfig

logger = logging.getLogger(__name__)
_global_client: Client | None = None

@dataclass
class DaskConfig:
    mode: Literal["local", "threaded", "distributed", "synchronous"] = "local"
    n_workers: int | None = None
    threads_per_worker: int = 1
    memory_per_worker: str | None = None
    scheduler_address: str | None = None
    dashboard: bool = True

@dataclass
class SLURMConfig:
    account: str
    n_workers: int
    processes: int
    cores_per_worker: int
    memory_per_worker: str | None = "4GB"


def get_client() -> Client | None:
    return _global_client

def set_global_client(client: Client | None) -> None:
    global _global_client
    _global_client = client

def create_local_cluster(
    n_workers: int | None,
    threads_per_worker: int = 1,
    memory_per_worker: str | None = None,
    scheduler: Literal["threads", "processes", "synchronous"] = "threads",
    processes: bool = True,
    dashboard: bool = True,
    dashboard_address: str = "127.0.0.1:8787",
    silence_logs: int = logging.WARNING,
) -> LocalCluster:
    """Create a local Dask cluster for parallel processing."""
    if n_workers is None:
        n_workers = os.cpu_count() or 1

    logger.info(
        "Creating local Dask cluster with %d workers, %d threads per worker.",
        n_workers,
        threads_per_worker,
    )

    if scheduler == "synchronous":
        logger.warning(
            "Using synchronous scheduler; parallelism will be disabled.",
        )

    return LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_per_worker,
        scheduler_port=0,
        silence_logs=silence_logs,
        processes=processes,
        dashboard_address=dashboard_address if dashboard else ":8787",
    )

def create_slurm_cluster(
    account: str,
    n_workers: int,
    processes: int,
    cores_per_worker: int,
    memory_per_worker: str | None = "4GB",
    walltime: str = "0-1:20:00",
    dashboard: bool = True,
    dashboard_address: str = "127.0.0.1:8787",
) -> SLURMCluster:
    """Create a Dask cluster using SLURM for job scheduling."""
    logger.info(
        "Creating SLURM Dask cluster with %d workers, %d cores per worker.",
        n_workers,
        cores_per_worker,
    )

    return SLURMCluster(
        account=account,
        n_workers=n_workers,
        walltime=walltime,
        cores=cores_per_worker,
        memory=memory_per_worker,
        processes=processes,
        log_directory="logs/workers/",
        job_script_prologue=[
                "module load python scipy-stack",
                "export OMP_NUM_THREADS=1",
                "export OPENBLAS_NUM_THREADS=1",
                "export MKL_NUM_THREADS=1",
                "source /project/6099921/NDOT/QML-gates-tests/qqe/.venv/bin/activate",
            ],
        scheduler_options={
            "host": "0.0.0.0",
            "dashboard_address":":8787",
        }
    )

def create_distributed_client(
    scheduler_address: str,
    timeout: int = 30,
    silence_logs: int = logging.WARNING,
) -> Client:
    """Create a Dask client connected to a distributed cluster."""
    logger.info(
        "Connecting to distributed Dask cluster at %s.",
        scheduler_address,
    )

    client = Client(
        address=scheduler_address,
        timeout=timeout,
        silence_logs=silence_logs,
    )

    logger.info(
        "Connected to Dask cluster with %d workers.",
        len(client.scheduler_info().get("workers", {})),
    )

    return client

def create_client(
    mode: Literal["local", "distributed", "threaded", "synchronous", "slurm"] = "local",
    *,
    n_workers: int,
    threads_per_worker: int = 1,
    memory_per_worker: str | None = None,
    scheduler_address: str | None = None,
    dashboard: bool = True,
    walltime: str = "0-1:00:00",
    silence_logs: int = logging.WARNING,
) -> Client:
    if mode == "synchronous":
        logger.info("Creating synchronous Dask client.")
        client = Client(
            processes=False, n_workers=1, threads_per_worker=1, silence_logs=silence_logs,
        )

    elif mode in ("local", "threaded"):
        processes = (mode == "local")
        scheduler_type = "processes" if mode == "local" else "threads"
        cluster = create_local_cluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_per_worker=memory_per_worker,
            scheduler=scheduler_type,
            dashboard=dashboard,
            silence_logs=silence_logs,
            processes=processes,
        )
        client = Client(cluster)

    elif mode == "distributed":
        if scheduler_address is None:
            msg = "scheduler_address must be provided for distributed mode."
            raise ValueError(msg)
        client = create_distributed_client(
            scheduler_address=scheduler_address,
            silence_logs=silence_logs,
        )
    elif mode == "slurm":
        cluster = create_slurm_cluster(
            account="def-boudrea1",
            walltime=walltime,
            n_workers=n_workers,
            processes=1,
            cores_per_worker=threads_per_worker,
            memory_per_worker=memory_per_worker,
            dashboard=dashboard,
        )
        client = Client(cluster)
    else:
        msg = f"Unknown mode: {mode}"
        raise ValueError(msg)

    logger.info("Dask client created successfully.")
    logger.info("Client dashboard available at: %s", client.dashboard_link)
    return client


@contextmanager
def dask_client(
    mode: Literal["local", "distributed", "threaded", "synchronous", "slurm"] = "local",
    *,
    n_workers: int | None = None,
    threads_per_worker: int = 1,
    memory_per_worker: str | None = None,
    scheduler_address: str | None = None,
    dashboard: bool = True,
    walltime: str = "0-1:00:00",
    performance_report_file: str | None = None,
    silence_logs: int = logging.WARNING,
    set_global: bool = False,
) -> Generator[Client, None, None]:
    """Context manager for Dask client."""
    client = create_client(
        mode=mode,
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_per_worker=memory_per_worker,
        scheduler_address=scheduler_address,
        dashboard=dashboard,
        walltime=walltime,
        silence_logs=silence_logs,
    )

    if set_global:
        set_global_client(client)

    try:
        if performance_report_file:
            with performance_report(filename=performance_report_file):
                yield client
        else:
            yield client
    finally:
        if set_global:
            set_global_client(None)

        cluster = client.cluster
        client.close()
        if cluster is not None:
            cluster.close()
        logger.info("Dask client and cluster closed.")


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
