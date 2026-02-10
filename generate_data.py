from __future__ import annotations

import json
import logging
import typer

from pathlib import Path
from typing import Any

import numpy as np
import itertools

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


data = [
    ["cid", "family", "n_qubits", "n_layers", "seed", "state_vector", "SRE", "SRE_std"], # state_vector
]

n_seeds = 50

qubits_range = np.arange(4, 11, 2)
layers_range = np.arange(1, 51, 2)
tcount_range = np.arange(0, 100, 2)

def make_seed(n_qubits, n_layers, rep) -> int:
    return hash((n_qubits, n_layers, rep)) % (2**32)

def calculate_tcount(n_layers: int, per_layer: int = 2) -> int:
    """Calculate the number of t gates for a given number of layers.

    Args:
        n_layers: Number of layers in the circuit.
        per_layer: Number of t gates per layer (default: 2).

    Returns:
        Total t gate count: per_layer × (n_layers - 1)
        (Last layer is excluded from t-gate placement)
    """
    return per_layer * max(0, n_layers - 1)

def is_valid_config(n_qubits, n_layers, tcount) -> bool:
    max_tcount = calculate_tcount(n_layers, per_layer=2)
    return tcount <= max_tcount

def make_cid(family, n_qubits, n_layers, seed) -> str:
    return f"{family}_Q{n_qubits}_L{n_layers}_S{seed}"

def generate_data_params(qubits_range, layers_range, n_seeds):
    params = list(itertools.product(qubits_range, layers_range))
    data = []
    for rep in range(n_seeds):
        for n_qubits, n_layers in params:
            seed = make_seed(n_qubits, n_layers, rep)
            cid = make_cid("haar", n_qubits, n_layers, seed)
            data.append([cid, "haar", n_qubits, n_layers, seed, None, None, None])
    return data

def main():
    data = generate_data_params(qubits_range, layers_range, n_seeds)
    for row in data:
        print(", ".join(str(x) for x in row))
    print(len(data))

if __name__ == "__main__":
    configure_logger(logging.INFO, logging.INFO)
    typer.run(main)
