from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class DaskConfig:
    mode: Literal["local", "threaded", "distributed", "synchronous"] = "local"
    n_workers: int | None = None
    threads_per_worker: int = 1
    memory_per_worker: str | None = None
    scheduler_address: str | None = None
    dashboard: bool = True

# class DaskConfig:
#     """Configuration settings for Dask parallel processing."""
#     scheduler: Literal["threads", "processes", "distributed"] = "threads"
#     n_workers: int | None = None
#     threads_per_worker: int = 1
#     memory_per_worker: str | None = None
#     memory_limit: str = "None"
#     scheduler_address: str | None = None
#     silence_logs: bool = True
#     dashboard_address: str | None = None

#     def __post_init__(self):
#         """Validate the DaskConfig after initialization."""
#         if self.scheduler == "distributed" and self.scheduler_address is None:
#             msg = "scheduler_address must be provided for distributed scheduler."
#             raise ValueError(msg)
#         if self.scheduler != "distributed" and self.scheduler_address is not None:
#             msg = "scheduler_address should only be provided for distributed scheduler."
#             raise ValueError(msg)
