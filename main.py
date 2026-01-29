"""Main script for simulating quantum circuits and computing properties.

This module simulates various quantum circuit families (Haar, Clifford, Quansistor)
and computes their stabilizer Rényi entropy (SRE) using different methods.
"""

import time

from pathlib import Path

import numpy as np

from backend import QuimbBackend
from circuit.families import (
    CliffordBrickwork,
    HaarBrickwork,
    QuansistorBrickwork,
    TdopingRules,
)
from experiments.sweeper import (
    ExperimentConfig,
)
from experiments.visualizer import plot_pennylane_circuit
from properties.SRE.fwht_sre import compute as fwht_sre
from properties.SRE.monte_carlo_sre import compute as MCMC_sre
from states.types import DenseState

OUT = Path("circuit_outputs")
OUT.mkdir(exist_ok=True)

if __name__ == "__main__":
    family_name = "haar"
    n_qubits = 14
    d = 2
    n_layers = 40
    seed = 33
    tcount = 38

    families_dict = {
        "clifford": CliffordBrickwork(
            tdoping=TdopingRules(
                count=tcount,
                placement="center_pair",
                per_layer=2,
            )
        ),
        "haar": HaarBrickwork(),
        "quansistor": QuansistorBrickwork(),
    }
    t_0 = time.time()
    spec = families_dict[family_name].make_spec(
        n_qubits=n_qubits,
        n_layers=n_layers,
        d=d,
        seed=seed,
    )
    backend = QuimbBackend()
    raw_state = backend.simulate(spec, state_type="dense")

    if hasattr(raw_state, "vector"):  # if backend returns DenseState
        vec = np.asarray(raw_state.vector, dtype=complex).reshape(-1)

    state = DenseState(vector=vec, n_qubits=n_qubits, d=d, backend="pennylane")

    t_1 = time.time()
    print(f"Simulation done after: {t_1-t_0:.3f} s.")

    if family_name == "haar":
        sre_haar = -np.log2(4 / ((d**n_qubits) + 3))
        print(f"SRE expected value for a Haar-Random circuit: {sre_haar:.6f}")
    if family_name == "clifford" and tcount == 0:
        print(f"SRE expected value for a fully Clifford circuit: {0.0}")

    # res_exact = sre_exact_dense(state, d=2)

    # print(f"SRE exact dense: {res_exact.value:.6f}")
    # t_2 = time.time()
    # print(f"Computed in: {t_2-t_1:.3f} s.")

    res_fwht = fwht_sre(state)

    print(f"SRE using FWHT: {res_fwht.value:.6f}")
    t_3 = time.time()
    print(f"Computed in: {t_3-t_1:.3f} s.")

    res_sampled = MCMC_sre(state, seed=spec.global_seed)
    sampling_bias = res_sampled.meta["se_sre_approx"]

    print(sampling_bias)
    print(rf"SRE using sampling: {res_sampled.value - sampling_bias:.6f}.")
    t_4 = time.time()
    print(f"Computed in: {t_4-t_3:.3f} s.")

    # res_entanglement_entropy = compute_entanglement_entropy(state)
    # print(f"SRE exact dense: {res_entanglement_entropy.value:.6f}")
    # t_2 = time.time()
    # print(f"Computed in: {t_2-t_1:.3f} s.")

    # res_renyi_entropy = compute_renyi_entanglement_entropy(state)
    # print(f"SRE exact dense: {res_renyi_entropy.value:.6f}")
    # t_3 = time.time()
    # print(f"Computed in: {t_3-t_2:.3f} s.")

    experiment = ExperimentConfig(
        circuit_family="clifford",
        n_qubits=8,
        n_layers=30,
        d=2,
        family_params={},
        backend="pennylane",
        backend_params={},
        properties=[],
        base_seed=seed,
        replicate=15,
        run_seed=0,
        tags={},
    )

    path_name = str(OUT / f"pennylane_{family_name}_circuit_for_SRE.png")
    plot_pennylane_circuit(
        spec,
        save_path=path_name,
    )
    print(f"Total computation time: {t_3 - t_0:.3f} s.")
