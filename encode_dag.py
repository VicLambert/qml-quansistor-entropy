
from qqe.circuit.families import (
    CliffordBrickwork,
    HaarBrickwork,
    QuansistorBrickwork,
    RandomCircuit,
)
from qqe.circuit.patterns import TdopingRules, to_qasm
from qqe.GNN.encoder import qasm_to_pyg_graph

family_registry = {
    "haar": HaarBrickwork,
    "clifford": CliffordBrickwork,
    "quansistor": QuansistorBrickwork,
    "random": RandomCircuit,
}

def main():
    n_qubits = 8
    n_layers = 30
    seed = 42
    n_bins = 50

    # family = "random"
    tdoping=TdopingRules(count=2*n_layers, per_layer=2)

    for family in family_registry:
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
        # print(meta["family"], meta["n_qubits"])
        # print(dict(features.items()))
        print(f"------ Family: {family} -----")
        print(graph_data)
        print(gate_counts)

if __name__ == "__main__":
    main()
