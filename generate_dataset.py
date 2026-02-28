from __future__ import annotations

import itertools
import json
import logging

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer

from dask.distributed import as_completed
from tqdm import tqdm

from qqe.backend import PennylaneBackend, QuimbBackend
from qqe.circuit.families import (
    CliffordBrickwork,
    HaarBrickwork,
    QuansistorBrickwork,
    RandomCircuit,
)
from qqe.experiments.core import run_experiment
#from qqe.parallel import dask_client
from qqe.utils import FileCache, configure_logger

from qqe.circuit.patterns import TdopingRules, to_qasm
from qqe.GNN.encoder import qasm_to_pyg_graph

logger = logging.getLogger(__name__)


family_registry = {
    "haar": HaarBrickwork,
    "clifford": CliffordBrickwork,
    "quansistor": QuansistorBrickwork,
    "random": RandomCircuit,
}
PROJECT_ROOT = Path(__file__).resolve().parent
cache = FileCache(PROJECT_ROOT / "outputs" / "cache")

def main():
    n_qubits = 8
    n_layers = 30
    seed = 42
    n_bins = 50
    family = "random"
    output_path = PROJECT_ROOT / "qqe/data/" / f"encoding_data_{family}"
    # family = "random"
    tdoping=TdopingRules(count=2*n_layers, per_layer=2)

    circuit = family_registry[family]()

    circuit_spec = circuit.make_spec(
        n_qubits=n_qubits,
        n_layers=n_layers,
        d=2,
        seed=seed,
        tdoping=tdoping if family == "clifford" else None,
    )
    # enc = QasmCountEncoder(n_bins=n_bins)

    gates = circuit_spec.gates
    qasm = to_qasm(circuit_spec, gates)
    # features, meta = encode_qasm_to_feature_dict(qasm, n_bins=n_bins)


    graph_data, gate_counts = qasm_to_pyg_graph(
        qasm_str=qasm,
        n_bins=n_bins,
        family=family,
        global_feature_variant="binned",
    )

    d_q = ["IN", "OUT", "RX", "RY", "RZ", "CX", "I", "H", "S", "T", "HAAR", "QX", "QY"]
    x_np = graph_data.x.detach().cpu().numpy()
    A_np = graph_data.edge_index.detach().cpu().numpy()
    
    results = pd.DataFrame(
        x_np,
        columns=d_q + [f"q_{i}" for i in range(n_qubits)],
    )
    results.to_json(output_path.with_suffix(".json"), index=False)
    results = pd.DataFrame(
        A_np,
    )
    results.to_json(output_path.with_suffix(".json"), index=False)
    # print(gate_counts)

if __name__ == "__main__":
    main()
