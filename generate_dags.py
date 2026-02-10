from __future__ import annotations

import logging

from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from qqe.backend import PennylaneBackend, QuimbBackend
from qqe.circuit.families import (
    CliffordBrickwork,
    HaarBrickwork,
    QuansistorBrickwork,
)
from qqe.circuit.patterns import TdopingRules
from qqe.circuit.gates import clifford_recipe_unitary
from qqe.GNN import circuit_spec_to_nx_dag

logger = logging.getLogger(__name__)


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


quantity = "SRE"
backend: str = "quimb"
method: str = "fwht"
seed = 42
repeat = 1

rng = np.random.default_rng(seed)
n_qubits = 6
n_layers = 10

family_registry: dict[str, Any] = {
    "haar": HaarBrickwork,
    "clifford": CliffordBrickwork,
    "quansistor": QuansistorBrickwork,
}
backend_registry: dict[str, Any] = {
    "pennylane": lambda: PennylaneBackend(),
    "quimb": lambda: QuimbBackend(),
}

circuit = CliffordBrickwork(
    name="clifford",
    tdoping=TdopingRules(
        count=calculate_tcount(n_layers, per_layer=2),
        placement="center_pair",
        per_layer=2,
    ),
)

# circuit = QuansistorBrickwork(
#     name="quansistor",
# )

# circuit = HaarBrickwork(
#     name="haar",
# )
spec = circuit.make_spec(
    n_qubits=n_qubits,
    n_layers=n_layers,
    d=2,
    seed=seed,
)
dag = circuit_spec_to_nx_dag(spec)

# Create a layout based on layers and qubits
pos = {}
layer_node_counts = {}

for node, attrs in dag.nodes(data=True):
    layer = attrs["subset"]
    wires = attrs["wires"]

    # Count nodes at each layer for vertical spacing
    if layer not in layer_node_counts:
        layer_node_counts[layer] = 0

    # Position x by layer, y by qubit(s) involved
    x = layer * 2  # Horizontal spacing by layer
    y = -sum(wires) / len(wires)  # Average qubit index (negative for better layout)

    pos[node] = (x, y)

# Create custom labels based on node attributes
labels = {}
for node, attrs in dag.nodes(data=True):
    kind = attrs["kind"]
    wires = attrs["wires"]
    tags = attrs["tags"]

    # Format wire display
    wire_str = f"q{wires[0]}" if len(wires) == 1 else f"q{wires[0]}-q{wires[1]}"

    # Check for special tag info
    special_tag = None
    if "T-gate" in tags:
        special_tag = "T"
    elif kind == "clifford":
        # First check if decomposition is already in tags
        decomp_tag = None
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("decomp_"):
                decomp_tag = tag.replace("decomp_", "")
                break

        if decomp_tag:
            special_tag = decomp_tag
        else:
            # Fallback: compute from seed
            gate_spec = spec.gates[node]
            if gate_spec.seed is not None:
                u_a, u_b, _ = clifford_recipe_unitary(gate_spec.seed)
                special_tag = f"{u_a}⊗{u_b}+CX"
            else:
                special_tag = "Cliff"
    elif kind == "quansistor":
        # Extract axis if present
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("axis_"):
                axis = tag.split("_")[1]
                special_tag = f"Q-{axis}"
                break
        if special_tag is None:
            special_tag = "Q"

    # Create concise label
    if special_tag:
        labels[node] = f"{special_tag}\n{wire_str}"
    else:
        labels[node] = f"{kind}\n{wire_str}"

# Color nodes by gate type
gate_colors = {
    "X": "#FF6B6B",  # Red
    "Y": "#4ECDC4",  # Teal
    "Z": "#45B7D1",  # Blue
    "H": "#FFA07A",  # Light Salmon
    "S": "#98D8C8",  # Mint
    "T": "#F7DC6F",  # Yellow
    "CNOT": "#BB8FCE",  # Purple
    "CZ": "#85C1E2",  # Sky Blue
    "clifford": "#D2B4DE",  # Lavender
    "quansistor_X": "#A9DFBF",  # Light Green for X-axis
    "quansistor_Y": "#F8B88B",  # Light Orange for Y-axis
    "quansistor": "#D5D8DC",  # Gray (fallback)
    "haar": "#FAD7A0",  # Peach
}

# Assign colors based on gate type and axis
node_colors = []
for node in dag.nodes():
    kind = dag.nodes[node]["kind"]
    tags = dag.nodes[node]["tags"]

    if kind == "quansistor":
        # Extract axis from tags
        axis = None
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("axis_"):
                axis = tag.split("_")[1]
                break

        if axis:
            node_colors.append(
                gate_colors.get(f"quansistor_{axis}", gate_colors["quansistor"]),
            )
        else:
            node_colors.append(gate_colors["quansistor"])
    else:
        node_colors.append(gate_colors.get(kind, "#D3D3D3"))

# Draw the circuit DAG
plt.figure(figsize=(16, 10))
nx.draw(
    dag,
    pos=pos,
    labels=labels,
    with_labels=True,
    node_size=1800,
    node_color=node_colors,
    arrowsize=15,
    arrowstyle="->",
    edge_color="gray",
    width=2,
    font_size=6,
    font_color="black",
    font_weight="bold",
)

# Add layer labels
layer_positions = {}
for layer in layer_node_counts:
    layer_positions[layer] = layer * 2

ax = plt.gca()
for layer, x in layer_positions.items():
    ax.text(
        x,
        max([p[1] for p in pos.values()]) + 1.5,
        f"Layer {layer}",
        ha="center",
        fontsize=10,
        fontweight="bold",
    )

plt.title("Quantum Circuit DAG (Layers × Qubits Layout)", fontsize=14, fontweight="bold")
# plt.tight_layout()
plt.savefig(f"DAG_{circuit.name}.png", dpi=150, bbox_inches="tight")
plt.show()


# jobs = generate_jobs(
#     base_job=spec_clifford,
#     axes={
#         "circuit_family": ["clifford"],
#         "n_qubits": [8],
#         "n_layers": [10],
#         "family.tcount": [calculate_tcount(10, per_layer=2)],
#     },
#     repeats=repeat,
# )
# for job in jobs:
#     cfg = compile_job(
#         job,
#         family_registry=family_registry,
#     )
