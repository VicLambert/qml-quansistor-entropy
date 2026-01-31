from __future__ import annotations

import json
import os
import socket

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _to_jsonable(x: Any) -> Any:
    # small, practical converter; you can replace with your serialize.py later
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, Path):
        return str(x)
    # numpy scalars
    try:
        import numpy as np
        if isinstance(x, (np.integer, np.floating)):
            return x.item()
    except Exception:
        pass
    return x


def make_run_id(*, label: str = "run") -> str:
    # Example: 2026-01-30_19-12-05__run
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    return f"{ts}__{label}"


class RunStore:
    """Creates outputs/runs/<run_id>/ and provides helpers to write.

    - run.json (metadata)
    - jobs.jsonl
    - results.jsonl
    - errors.jsonl
    - summary.json
    """

    def __init__(self, root: Path, run_id: str):
        self.root = root
        self.dir = root / run_id
        self.dir.mkdir(parents=True, exist_ok=True)

        self.run_path = self.dir / "run.json"
        self.jobs_path = self.dir / "jobs.jsonl"
        self.results_path = self.dir / "results.jsonl"
        self.errors_path = self.dir / "errors.jsonl"
        self.summary_path = self.dir / "summary.json"

    def write_run_header(self, payload: dict[str, Any]) -> None:
        payload = {
            **payload,
            "created_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "pid": os.getpid(),
            "host": socket.gethostname(),
        }
        with open(self.run_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=_to_jsonable)

    def append_jsonl(self, path: Path, obj: Any) -> None:
        line = json.dumps(obj, default=_to_jsonable, separators=(",", ":"))
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    def log_job(self, job_cfg: Any) -> None:
        self.append_jsonl(self.jobs_path, job_cfg)

    def log_result(self, result: Any) -> None:
        self.append_jsonl(self.results_path, result)

    def log_error(self, err: dict[str, Any]) -> None:
        self.append_jsonl(self.errors_path, err)

    def write_summary(self, summary: dict[str, Any]) -> None:
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=_to_jsonable)
