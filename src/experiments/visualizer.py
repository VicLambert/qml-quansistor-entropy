from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml

from src.circuit.matrix_factory import gate_unitary
from src.circuit.spec import CircuitSpec
from src.states.types import DenseState


def _layer_from_tags(tags: tuple[str, ...]) -> Optional[int]:
    # Expected tags include "L0", "L1", ... OR ("layer","L0",...)
    for t in tags:
        if isinstance(t, str) and t.startswith("L") and t[1:].isdigit():
            return int(t[1:])
    # fallback: find "layer" then next token "Lk"
    for i, t in enumerate(tags[:-1]):
        if t == "layer":
            nxt = tags[i + 1]
            if isinstance(nxt, str) and nxt.startswith("L") and nxt[1:].isdigit():
                return int(nxt[1:])
    return None

def plot_pennylane_circuit(
    spec: CircuitSpec,
    *,
    device_name: str = "lightning.qubit",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plot a quantum circuit using PennyLane's native matplotlib drawer.
    Requires spec.gates to be materialized.
    """
    if not spec.gates:
        raise ValueError("CircuitSpec.gates is empty. Cannot visualize PennyLane circuit.")

    dev = qml.device(device_name, wires=spec.n_qubits)

    @qml.qnode(dev)
    def circuit():
        for gate in spec.gates:
            U = gate_unitary(gate)
            # Extract gate type from tags for better visualization
            gate_label = None
            if "T-gate" in gate.tags:
                gate_label = "T"
            elif "clifford" in gate.tags:
                gate_label = "C"
            qml.QubitUnitary(U, wires=gate.wires, id=gate_label)
        return qml.state()

    drawer = qml.draw_mpl(circuit)
    fig, ax = drawer()

    if title is None:
        title = f"PennyLane circuit ({spec.family}), n={spec.n_qubits}, L={spec.n_layers}"
    ax.set_title(title, fontsize=12, fontweight="bold")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        fig.show()

    return fig

@dataclass(frozen=True)
class GateDraw:
    layer: int
    kind: str
    wires: tuple[int, ...]
    tags: tuple[str, ...] = ()

def _get_gate_label(gate: GateDraw) -> str:
    """Generate a label for a gate that includes type information from tags."""
    # Prefer explicit labels from tags
    if "T-gate" in gate.tags:
        return "T"
    if "clifford" in gate.tags:
        return "C"
    # Fallback to gate kind
    return gate.kind[:1].upper() if gate.kind else "U"


def _collect_gates_by_layer(spec: CircuitSpec) -> dict[int, list[GateDraw]]:
    if not spec.gates:
        raise ValueError(
            "CircuitSpec.gates is empty. Visualizer requires a *materialized* circuit. "
            "Call your Family.gates(spec) and set spec.gates before plotting.",
        )

    by_layer: dict[int, list[GateDraw]] = {}
    for g in spec.gates:
        L = _layer_from_tags(g.tags)
        if L is None:
            raise ValueError(
                f"Gate {g.kind} on wires={g.wires} is missing a layer tag (expected 'Lk'). "
                f"Tags were: {g.tags}",
            )
        by_layer.setdefault(L, []).append(GateDraw(layer=L, kind=g.kind, wires=g.wires, tags=g.tags))

    return by_layer


def plot_circuit_diagram(
    spec: CircuitSpec,
    *,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize_per_layer: float = 1.2,
) -> None:
    """Draw a circuit diagram: qubit wires vs layers, with gate symbols placed per layer.

    - 1-qubit gates: box with label (e.g. T)
    - 2-qubit gates: vertical connector + two boxes (or a single label at midpoint)
    """
    by_layer = _collect_gates_by_layer(spec)

    n = spec.n_qubits
    L = spec.n_layers

    # layout: x = layer index (centered at integer positions), y = wire index (top to bottom)
    fig_w = max(6.0, figsize_per_layer * L)
    fig_h = max(3.0, 0.6 * n)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Draw horizontal wires
    for q in range(n):
        y = (n - 1) - q  # put wire 0 at top
        ax.plot([-0.5, L - 0.5], [y, y], linewidth=1)
        ax.text(-0.7, y, f"q{q}", va="center", ha="right", fontsize=10)

    # helper to draw a 1q box
    def draw_1q_gate(x: float, q: int, label: str):
        y = (n - 1) - q
        w, h = 0.55, 0.42
        rect = plt.Rectangle((x - w / 2, y - h / 2), w, h, fill=False, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=10)

    # helper to draw a 2q gate connector
    def draw_2q_gate(x: float, a: int, b: int, label: str):
        ya = (n - 1) - a
        yb = (n - 1) - b
        y_top, y_bot = (ya, yb) if ya > yb else (yb, ya)

        # vertical line between qubits
        ax.plot([x, x], [y_bot, y_top], linewidth=1.5)

        # small boxes on each wire
        draw_1q_gate(x, a, label)
        draw_1q_gate(x, b, label)

    # Place gates per layer
    for layer in range(L):
        gates = by_layer.get(layer, [])

        # If multiple gates act in same layer, we still place them at same x=layer.
        # To avoid overlap on same qubit, we add small x-offsets ("slots") within the layer.
        # Simple slotting: sort by min wire index, then assign offsets.
        gates_sorted = sorted(gates, key=lambda g: (min(g.wires), len(g.wires), g.kind))

        # Track occupied wires at each slot
        slots: list[set[int]] = []

        def allocate_slot(wires: tuple[int, ...]) -> int:
            ws = set(wires)
            for si, occ in enumerate(slots):
                if occ.isdisjoint(ws):
                    occ.update(ws)
                    return si
            slots.append(set(ws))
            return len(slots) - 1

        for g in gates_sorted:
            si = allocate_slot(g.wires)
            x = layer + (si - 0.5 * (len(slots) - 1)) * 0.18  # spread within layer

            label = _get_gate_label(g)

            if len(g.wires) == 1:
                draw_1q_gate(x, g.wires[0], label)
            elif len(g.wires) == 2:
                a, b = g.wires
                draw_2q_gate(x, a, b, label)
            else:
                # if you ever add k-qubit gates, you can generalize later
                raise NotImplementedError(f"Cannot draw gate with wires={g.wires}")

        # layer label at bottom
        ax.text(layer, -0.7, f"L{layer}", ha="center", va="top", fontsize=10)

    ax.set_xlim(-1.0, L - 0.2)
    ax.set_ylim(-1.2, n - 0.0)
    ax.axis("off")

    if title is None:
        title = f"Circuit diagram ({spec.family}), n={spec.n_qubits}, L={spec.n_layers}"
    ax.set_title(title, fontsize=12, fontweight="bold")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def plot_state_probabilities_dense(
    state: DenseState,
    *,
    top_k: int = 10,
    save_path: Optional[str] = None,
    show: bool = True,
    title: Optional[str] = None,
) -> None:
    vec = np.asarray(state.vector).reshape(-1)
    probs = np.abs(vec) ** 2

    top_idx = np.argsort(probs)[-top_k:][::-1]
    top_probs = probs[top_idx]
    basis = [format(i, f"0{state.n_qubits}b") for i in top_idx]

    fig, ax = plt.subplots(figsize=(max(8, 0.7 * top_k), 4.5))
    bars = ax.bar(range(top_k), top_probs)

    for bar, p in zip(bars, top_probs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{p:.4f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(top_k))
    ax.set_xticklabels(basis, rotation=45, ha="right")
    ax.set_ylabel("Probability")
    ax.grid(axis="y", alpha=0.3)

    if title is None:
        title = f"Top-{top_k} basis probabilities ({state.backend})"
    ax.set_title(title, fontweight="bold")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)
