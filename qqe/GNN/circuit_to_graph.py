
from __future__ import annotations

from dataclasses import dataclass
from qqe.circuit.spec import CircuitSpec

@dataclass
class DAG:
    parents: list[set[int]]
    children: list[set[int]]


def circuit_spec_to_dag(spec: CircuitSpec) -> DAG:
    n_nodes = len(spec.gates)
    parents = [set() for _ in range(n_nodes)]
    children = [set() for _ in range(n_nodes)]

    last_node_on_qubit: dict[int, int] = {}

    for i, gate in enumerate(spec.gates):
        wires = tuple(int(wire) for wire in gate.wires)

        ps = {last_node_on_qubit[wire] for wire in wires if wire in last_node_on_qubit}
        for p in ps:
            parents[i].add(p)
            children[p].add(i)

        for wire in wires:
            last_node_on_qubit[wire] = i

    return DAG(parents=parents, children=children)

def layer_from_tags(tags) -> int | None:
    # tags is a tuple like ("layer", "L3", "clifford") or ("layer","L3","wire","W2","T-gate")
    if not tags:
        return None
    for i in range(len(tags) - 1):
        if tags[i] == "layer" and isinstance(tags[i+1], str) and tags[i+1].startswith("L"):
            try:
                return int(tags[i+1][1:])
            except ValueError:
                return None
    return None

def circuit_spec_to_nx_dag(spec: CircuitSpec):
    import networkx as nx
    dag = circuit_spec_to_dag(spec)
    G = nx.MultiDiGraph()

    for i, g in enumerate(spec.gates):
        subset = layer_from_tags(g.tags)
        if subset is None:
            subset=0
        G.add_node(i, kind=g.kind, wires=tuple(g.wires), tags=g.tags, subset=subset)

    for u, vs in enumerate(dag.children):
        for v in vs:
            G.add_edge(u, v)

    return G
