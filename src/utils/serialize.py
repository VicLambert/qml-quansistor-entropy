
from __future__ import annotations

from pathlib import Path
import json
import logging
import os
import dataclasses
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

def _to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj

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
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

