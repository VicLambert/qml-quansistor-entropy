from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import networkx as nx

from qqe.circuit.spec import CircuitSpec, GateSpec

Wires = tuple[int, ...]

@dataclass(frozen=True)
class GateNode:
    idx: int
    gate: GateSpec

@dataclass
class CircuitDag:
    nodes : list[GateNode]
    parents: list[set[int]]
    children: list[set[int]]

    def topological_order(self) -> list[int]:
        in_deg = [len(p) for p in self.parents]
        q = [i for i, deg in enumerate(in_deg) if deg == 0]
        order: list[int] = []
        head = 0

        while head < len(q):
            u = q[head]
            head += 1
            order.append(u)
            for child in self.children[u]:
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    q.append(child)

        if len(order) != len(self.nodes):
            msg = "Circuit is not a DAG (contains cycles)"
            raise ValueError(msg)

        return order

    def asap_layer(self) -> list[list[int]]:
        order = self.topological_order()
        layer = [0] * len(self.nodes)

        for node_idx in order:
            if self.parents[node_idx]:
                layer[node_idx] = 1 + max(layer[p] for p in self.parents[node_idx])

        L = max(layer, default=1)
        layers: list[list[int]] = [[] for _ in range(L + 1)]
        for u, l in enumerate(layer):
            layers[l].append(u)
        return layers


def circuit_spec_to_dag(
    spec: CircuitSpec,
) -> CircuitDag:
    nodes = [GateNode(idx=i, gate=g) for i, g in enumerate(spec.gates)]
    n = len(nodes)

    parents = [set() for _ in range(n)]
    children = [set() for _ in range(n)]

    last_on_wire: dict[int, int] = {}

    for i, g in enumerate(spec.gates):
        wires: Iterable[int] = tuple(g.wires)

        direct_parents = set()
        for w in wires:
            if w in last_on_wire:
                direct_parents.add(last_on_wire[w])

        for p in direct_parents:
            parents[i].add(p)
            children[p].add(i)

        for w in wires:
            last_on_wire[w] = i
    return CircuitDag(
        nodes=nodes,
        parents=parents,
        children=children,
    )

def dag_to_graph(dag: CircuitDag) -> nx.DiGraph:
    """Convert CircuitDag to PennyLane CircuitGraph."""
    G = nx.DiGraph()
    for node in dag.nodes:
        g = node.gate
        G.add_node(
            node.idx,
            kind=g.kind,
            wires=tuple(g.wires),
            tags=tuple(g.tags),
            params=tuple(g.params),
        )
    for i, parent in enumerate(dag.parents):
        for p in parent:
            G.add_edge(i , p)
    return G
