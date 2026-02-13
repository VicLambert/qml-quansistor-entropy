from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import socket
import time

from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from qqe.properties.results import PropertyRequest
from qqe.states.types import BackendConfig

logger = logging.getLogger(__name__)


def make_cache_key(
    *, spec_id: str, backend_cfg: dict[str, Any], prop_req: dict[str, Any],
) -> str:
    """Create a unique cache key for (circuit, backend, property)."""
    payload = {
        "spec_id": spec_id,
        "backend_cfg": backend_cfg,
        "property": prop_req,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


class FileCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _get_path(self, key: str) -> Path:
        return self.root / "results" / key[:2] / f"{key}.json"

    def get_lock_path(self, key: str) -> Path:
        return self.root / "locks" / key[:2] / f"{key}.lock"

    def load_json(self, key: str) -> dict[str, Any] | None:
        path = self._get_path(key)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def save_json(self, key: str, data: dict[str, Any]) -> None:
        path = self._get_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except OSError as e:
            with suppress(OSError):
                tmp_path.unlink(missing_ok=True)
            msg = f"Failed to write cache file {path}: {e}"
            raise RuntimeError(msg) from e


def store_results(cache: FileCache, key: str, result: dict[str, Any]) -> None:
    cache.save_json(key, result)


def maybe_load_results(cache: FileCache, key: str) -> dict | None:
    return cache.load_json(key)


def hash_key(payload: Any, *, n_hex: int = 32) -> str:
    h = hashlib.blake2b(_stable_json(payload).encode("utf-8"), digest_size=16).hexdigest()
    return h[:n_hex]


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


@contextmanager
def cache_lock(
    lock_path: Path, *, timeout_s: float = 60.0, stale_s: float = 6 * 3600,
) -> Iterator[None]:
    """Context manager to acquire a file lock for a cache entry."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    delay = 0.02
    fd: int | None = None

    try:
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                # stale cleanup
                try:
                    age = time.time() - lock_path.stat().st_mtime
                    if age > stale_s:
                        lock_path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue

                if time.time() - start > timeout_s:
                    msg = f"Failed to acquire lock {lock_path} in {timeout_s:.1f}s"
                    raise TimeoutError(msg) from None

                time.sleep(delay + np.random.random() * delay)
                delay = min(0.5, delay * 1.5)

        yield

    finally:
        if fd is not None:
            with suppress(OSError):
                os.close(fd)
        with suppress(OSError):
            lock_path.unlink(missing_ok=True)


@dataclass
class CachePolicy:
    enabled: bool = True
    write: bool = True
    read: bool = True
    namespace: str = "results"


def make_property_cache_key(
    *,
    spec_id: str,
    backend_cfg: BackendConfig,
    req: PropertyRequest,
) -> str:
    """Create a cache key for a property computation, or None if caching is disabled."""
    payload = {
        "kind": "property_result",
        "version": 1,
        "spec_id": spec_id,
        "backend": {
            "name": backend_cfg.name,
            "representation": backend_cfg.representation,
            "params": backend_cfg.params,
        },
        "property": {
            "name": req.normalized_name(),
            "method": req.normalized_method(),
            "params": req.params,
        },
    }
    return hash_key(payload)

def _to_jsonable(x: Any) -> Any:
    """Convert an object to a JSON-serializable format."""
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(item) for item in x]
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if is_dataclass(x) and not isinstance(x, type):
        return asdict(x)  # Convert dataclass to dict
    if hasattr(x, "__dict__"):
        return _to_jsonable(x.__dict__)
    # Fallback: convert to string
    return str(x)


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



def write_json(path: str | Path, obj: Any, *, indent: int = 2, compress: bool = False) -> None:
    """Write an object as a JSON file with atomic rename to prevent corruption."""
    jsonable_obj = _to_jsonable(obj)
    temp_path = Path(str(path) + ".tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            if compress:
                json.dump(jsonable_obj, f, separators=(",", ":"))
            else:
                json.dump(jsonable_obj, f, indent=indent)
        os.replace(temp_path, Path(path))
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            logger.warning("Failed to remove temporary file %s during cleanup", temp_path)


def read_json(path: str | Path) -> Any:
    """Read an object from a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
