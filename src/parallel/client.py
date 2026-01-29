"""Client module for parallel processing with dask."""

from __future__ import annotations

import logging
import os

from contextlib import contextmanager
from typing import Generator, Literal

from dask.distributed import Client, LocalCluster, performance_report

logger = logging.getLogger(__name__)

_global_client: Client | None = None


def get_client() -> Client:
    return _global_client


def set_global_client(client: Client) -> None:
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
        dashboard_address=dashboard_address if dashboard else None,
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
    mode: Literal["local", "distributed", "threaded", "synchronous"] = "local",
    *,
    n_workers: int | None = None,
    threads_per_worker: int = 1,
    memory_per_worker: str | None = None,
    scheduler_address: str | None = None,
    dashboard: bool = True,
    silence_logs: int = logging.WARNING,
) -> Client:
    if mode == "synchronous":
        logger.info("Creating synchronous Dask client.")
        client = Client(
            processes=False, n_workers=1, threads_per_worker=1, silence_logs=silence_logs
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
    else:
        msg = f"Unknown mode: {mode}"
        raise ValueError(msg)

    logger.info("Dask client created successfully.")
    logger.info("Client dashboard available at: %s", client.dashboard_link)
    return client


@contextmanager
def dask_client(
    mode: Literal["local", "distributed", "threaded", "synchronous"] = "local",
    *,
    n_workers: int | None = None,
    threads_per_worker: int = 1,
    memory_per_worker: str | None = None,
    scheduler_address: str | None = None,
    dashboard: bool = True,
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



