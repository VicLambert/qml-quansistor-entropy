
from __future__ import annotations

import hashlib
import json
import os
import time

from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator
from properties.request import PropertyRequest
from experiments.runner_config import BackendConfig

import numpy as np
import filelock as fl


def make_cache_key(*, spec_id: str, backend_cfg: dict[str, Any], prop_req: dict[str, Any]) -> str:
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
            with open(path, "r", encoding="utf-8") as f:
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
def cache_lock(lock_path: Path, *, timeout_s: float = 60.0, stale_s: float = 6*3600) -> Iterator[None]:
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
