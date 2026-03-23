
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pred_config import PredConfig

def run_inference(cfg: PredConfig):
    ...