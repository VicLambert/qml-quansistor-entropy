"""Main script for simulating quantum circuits and computing properties.

This module simulates various quantum circuit families (Haar, Clifford, Quansistor)
and computes their stabilizer Rényi entropy (SRE) using different methods.
"""

import logging
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
from experiments.runner import run_experiment
from experiments.sweeper import (
    JobConfig,
    aggregate_by_cond,
    compile_job,
    generate_jobs,
)
from experiments.visualizer import plot_pennylane_circuit
from parallel.client import dask_client
from parallel.executor import run_dask_experiments
from properties.SRE.fwht_sre import compute as fwht_sre
from properties.SRE.monte_carlo_sre import compute as MCMC_sre
from states.types import DenseState
from utils import FileCache, configure_logger, make_run_id, RunStore

logger = logging.getLogger(__name__)

OUT = Path("circuit_outputs")
OUT.mkdir(exist_ok=True)
PROJECT_ROOT = Path(__file__).resolve().parent  # directory containing main.py
RUNS_ROOT = PROJECT_ROOT / "outputs" / "runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
cache = FileCache(PROJECT_ROOT / "outputs" / "cache")

def make_family_instance(name: str):
    # family_registry entries are factories: params -> family instance
    if name == "haar":
        return lambda params: HaarBrickwork(**params)
    if name == "clifford":
        return lambda params: CliffordBrickwork(**params)
    if name == "quansistor":
        return lambda params: QuansistorBrickwork(**params)
    raise KeyError(name)

if __name__ == "__main__":
    configure_logger(logging.INFO, logging.INFO)
    family_name = "quansistor"
    n_qubits = 10
    d = 2
    n_layers = 40
    seed = 33
    tcount = 38
    method="fwht"
    repeat = 5

    logger.info("Simulating %s circuit with %d qubits, %d layers, d=%d.", family_name, n_qubits, n_layers, d)

    run_id = make_run_id(label=f"{family_name}__n{n_qubits}_l{n_layers}_{method}")
    run_store = RunStore(RUNS_ROOT, run_id)

    families_dict = {
        "clifford": CliffordBrickwork(
            tdoping=TdopingRules(
                count=tcount,
                placement="center_pair",
                per_layer=2,
            ),
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
    logger.info("Simulation done after: %.3f s.", t_1-t_0)

    if family_name == "haar":
        for nq in [6, 8, 10, 12, 14, 16]:
            sre_haar = -np.log2(4 / ((d**nq) + 3))
            logger.info("SRE expected value for a Haar-Random circuit with %d qubits: %.6f", nq, sre_haar)
    if family_name == "clifford" and tcount == 0:
        logger.info("SRE expected value for a fully Clifford circuit: %.6f", 0.0)

    # res_exact = sre_exact_dense(state, d=2)

    # print(f"SRE exact dense: {res_exact.value:.6f}")
    # t_2 = time.time()
    # print(f"Computed in: {t_2-t_1:.3f} s.")

    # res_fwht = fwht_sre(state)

    # logger.info("SRE using FWHT: %.6f", res_fwht.value)
    # t_3 = time.time()
    # logger.info("Computed in: %.3f s.", t_3-t_1)

    # res_sampled = MCMC_sre(state, seed=spec.global_seed)
    # sampling_bias = res_sampled.meta["se_sre_approx"]

    # logger.info("Sampling bias: %s", sampling_bias)
    # logger.info("SRE using sampling: %.6f.", res_sampled.value - sampling_bias)
    # t_4 = time.time()
    # logger.info("Computed in: %.3f s.", t_4-t_3)

    # res_entanglement_entropy = compute_entanglement_entropy(state)
    # logger.info(f" Von Neumann entanglement entropy: {res_entanglement_entropy.value:.6f}")
    # t_2 = time.time()
    # logger.info(f"Computed in: {t_2-t_1:.3f} s.")

    # res_renyi_entropy = compute_renyi_entanglement_entropy(state)
    # logger.info(f"Rényi entanglement entropy: {res_renyi_entropy.value:.6f}")
    # t_3 = time.time()
    # logger.info(f"Computed in: {t_3-t_2:.3f} s.")


    family_registry = {
        "haar": make_family_instance("haar"),
        "clifford": make_family_instance("clifford"),
        "quansistor": make_family_instance("quansistor"),
    }

    backend_registry = {
        # "pennylane": PennylaneBackend(),
        "quimb": QuimbBackend(),
    }


    experiment = JobConfig(
        circuit_family=family_name,
        n_qubits=8,
        n_layers=30,
        d=2,
        family_params={
            "tdoping": TdopingRules(
            count=tcount,
            placement="center_pair",
            per_layer=2,
            ),
        } if family_name == "clifford" else {},
        backend="quimb",
        backend_params={},
        properties=[{"name": "SRE", "method": "fwht"}],
        base_seed=seed,
        replicate=0,
        run_seed=0,
        tags={},
    )

    axes = {
        "n_qubits": list(range(6, n_qubits+1, 2)),
    }

    run_store.write_run_header({
        "circuit_family": experiment.circuit_family,
        "backend": experiment.backend,
        "axes": axes,
        "repeats": repeat,
        "properties": experiment.properties,
        "base_seed": experiment.base_seed,
        },
    )

    outputs = []

    jobs = generate_jobs(experiment, axes, repeats=repeat)
    for job in jobs:
        t_start = time.time()
        cfg = compile_job(job, family_registry=family_registry)
        out = run_experiment(cfg, backend_registry=backend_registry, cache=cache)
        # convert ExperimentRun (dataclass) to dict for aggregator if needed
        outputs.append({
            "tags": cfg.meta_data,
            "results": {k: {"value": v.value, "meta": v.meta} for k, v in out.results.items()},
        })
        t_end = time.time()
        logger.info("Job done in %.3f s.", t_end - t_start)

    stats = aggregate_by_cond(outputs, group_keys=("n_qubits", "n_layers"), value_path=("results", "SRE", "value"))
    # logger.info(stats)
    logger.info("Aggregated results:")
    for k, s in stats.items():
        logger.info("%s: %s", k, s.mean)

    # jobs = generate_jobs(experiment, axes, repeats=5)

    # def make_quimb_backend():
    #     return QuimbBackend()


    # backend_registry = {"quimb": make_quimb_backend}
    # dask_jobs = [
    #     compile_job(job, family_registry=family_registry)
    #     for job in jobs
    # ]

    # with dask_client(mode="local", n_workers=4, threads_per_worker=1, dashboard=True) as client:
    #     results = run_dask_experiments(
    #         client,
    #         dask_jobs,
    #         backend_registry=backend_registry,
    #         gather_errors=True,
    #     )

    # stats = aggregate_by_cond(results, group_keys=["n_qubits"], value_path=("results", "SRE", "value"))

    # for k, s in stats.items():
    #     logger.info("%s: %s", k, s.mean)

    path_name = str(OUT / f"pennylane_{family_name}_circuit_for_SRE.png")
    plot_pennylane_circuit(
        spec,
        save_path=path_name,
    )
    # logger.info("Total computation time: %.3f s.", t_3 - t_0)
