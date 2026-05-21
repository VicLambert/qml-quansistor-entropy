from __future__ import annotations

import os
import ast
import logging

import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml

from matplotlib.patches import Rectangle

from qqe.src.circuit.gates import gate_unitary
from qqe.src.circuit.spec import CircuitSpec
from qqe.src.states.types import DenseState
from qqe.src.GNN.training.train import TrainHistory

logger = logging.getLogger(__name__)

def _layer_from_tags(tags: tuple[str, ...]) -> int | None:
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
    title: str | None = None,
    save_path: str | None = None,
    show: bool = False,
):
    """Plot a quantum circuit using PennyLane's native matplotlib drawer.

    Requires spec.gates to be materialized.
    """
    if not spec.gates:
        msg = "CircuitSpec.gates is empty. Cannot visualize PennyLane circuit."
        raise ValueError(msg)

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
    plt.title(title, fontsize=12, fontweight="bold")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

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
        msg = (
            "CircuitSpec.gates is empty. Visualizer requires a *materialized* circuit. "
            "Call your Family.gates(spec) and set spec.gates before plotting."
        )
        raise ValueError(msg)

    by_layer: dict[int, list[GateDraw]] = {}
    for g in spec.gates:
        L = _layer_from_tags(g.tags)
        if L is None:
            msg = (
                f"Gate {g.kind} on wires={g.wires} is missing a layer tag (expected 'Lk'). "
                f"Tags were: {g.tags}"
            )
            raise ValueError(msg)
        by_layer.setdefault(L, []).append(
            GateDraw(layer=L, kind=g.kind, wires=g.wires, tags=g.tags),
        )

    return by_layer


def plot_circuit_diagram(
    spec: CircuitSpec,
    *,
    title: str | None = None,
    save_path: str | None = None,
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
        rect = Rectangle((x - w / 2, y - h / 2), w, h, fill=False, linewidth=1.5)
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
                msg = f"Cannot draw gate with wires={g.wires}"
                raise NotImplementedError(msg)

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
    save_path: str | None = None,
    show: bool = True,
    title: str | None = None,
) -> None:
    vec = np.asarray(state.vector).reshape(-1)
    probs = np.abs(vec) ** 2

    top_idx = np.argsort(probs)[-top_k:][::-1]
    top_probs = probs[top_idx]
    basis = [format(i, f"0{state.n_qubits}b") for i in top_idx]

    fig, ax = plt.subplots(figsize=(max(8, 0.7 * top_k), 4.5))
    bars = ax.bar(range(top_k), top_probs)

    for bar, p in zip(bars, top_probs, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{p:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

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


def plot_sre(
    results: dict[str, Any],
    *,
    quantity: str = "SRE",
    save_path: str | None = None,
    show: bool = True,
    title: str | None = None,
) -> None:
    """Plot SRE (Structural Rényi Entropy) versus number of qubits.

    Parameters
    ----------
    results : dict[str, Any]
        Dictionary containing statistics with keys as (family, n_qubits) tuples.
    save_path : str | None, optional
        Path to save the plot. If None, plot is not saved.
    show : bool, optional
        Whether to display the plot. Default is True.
    title : str | None, optional
        Title for the plot. If None, a default title is used.
    """
    stats = results.get("stats", results)
    label = results.get("group_keys", ["n_layers", "n_qubits"])[1]
    print("label", label)

    parsed: list[tuple[str, int, dict[str, Any]]] = []
    for key, value in stats.items():
        family = None
        n_qubits = None

        if isinstance(key, tuple) and len(key) == 2:
            family, n_qubits = key

        elif isinstance(key, str):
            # Handles keys like "('haar', 6)"
            try:
                k = ast.literal_eval(key)
            except (ValueError, SyntaxError):
                continue
            if isinstance(k, tuple) and len(k) == 2:
                family, n_qubits = k

        if family is None or n_qubits is None:
            continue

        parsed.append((str(family), int(n_qubits), value))

    if not parsed:
        msg = (
            "No valid (family, n_qubits) keys found in stats. "
            "Keys must be tuples like ('haar', 6) or strings like \"('haar', 6)\","
        )
        raise ValueError(msg)

    families: dict[str, list[tuple[int, float, float]]] = {}
    for family, n_qubits, value in parsed:
        sre = float(value["mean"])
        err = float(value["stderr"])
        families.setdefault(family, []).append((n_qubits, sre, err))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for family, rows in sorted(families.items(), key=lambda x: x[0]):
        rows.sort(key=lambda x: x[0])
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        es = [r[2] for r in rows]
        ax.errorbar(xs, ys, yerr=es, label=family, marker="o", capsize=3)

    xlabel = "Number of qubits" if label == "n_qubits" else "Number of layers"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(quantity)
    ax.grid(alpha=0.3)
    ax.legend(title="Circuit Family")
    ax.set_title(title or f"{quantity} vs {xlabel}", fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def _gse(d: int, n_qubits: Any, NT: Any) -> float:
    phys_dim = d ** n_qubits
    f = (-4 + 3 * (2*phys_dim - phys_dim)) / (4 * (2*phys_dim - 1))

    num = np.asarray( 4 +(phys_dim - 1) * f ** (NT), dtype=float)
    return -np.log2(num / (3 + phys_dim))

    # term1 = 3 / (d**n_qubits + 2)
    # term2 = (d**n_qubits - 1) / (d**n_qubits + 2)
    # inner = (2/d * d**(2*n_qubits) - (d-2) * 2/d * d**n_qubits - 1) / (d**(2*n_qubits) - 1)
    # return -np.log2(term1 + term2 * inner**(q*n_qubits))

def plot_sredensity_v_tcount(
    results: dict[str, Any],
    n_layers: int,
    *,
    save_path: str | None = None,
    show: bool = True,
    title: str | None = None,
) -> None:
    """Plot SRE (Structural Rényi Entropy) versus number of qubits.

    Parameters
    ----------
    results : dict[str, Any]
        Dictionary containing statistics with keys as (n_qubits, tcount) tuples.
    save_path : str | None, optional
        Path to save the plot. If None, plot is not saved.
    show : bool, optional
        Whether to display the plot. Default is True.
    title : str | None, optional
        Title for the plot. If None, a default title is used.
    """
    stats = results.get("stats", results)

    parsed: list[tuple[int, int, float, float]] = []  # (n_qubits, tcount, mean, stderr)

    for key, value in stats.items():
        # Parse key: either tuple (n_qubits, tcount) or string "(n_qubits, tcount)"
        if isinstance(key, tuple) and len(key) == 2:
            n_qubits, tcount = key
        elif isinstance(key, str):
            try:
                k = ast.literal_eval(key)
            except (ValueError, SyntaxError):
                continue
            if not (isinstance(k, tuple) and len(k) == 2):
                continue
            n_qubits, tcount = k
        else:
            continue

        try:
            n_qubits_i = int(n_qubits)
            tcount_i = int(tcount)
            mean_f = float(value["mean"])
            stderr_f = float(value.get("stderr", 0.0))
        except (TypeError, ValueError, KeyError):
            continue

        parsed.append((n_qubits_i, tcount_i, mean_f, stderr_f))

    if not parsed:
        msg = (
            "No valid (n_qubits, tcount) keys found in stats. "
            "Expected keys like (6, 38) or '(6, 38)'."
        )
        raise ValueError(msg)

    # Group by n_qubits (one curve per n_qubits)
    curves: dict[int, list[tuple[int, float, float]]] = {}
    for n_qubits, tcount, mean, stderr in parsed:
        curves.setdefault(n_qubits, []).append((tcount, mean, stderr))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ns = []
    for n_qubits, rows in sorted(curves.items(), key=lambda x: x[0]):
        ns.append(int(n_qubits))
        rows.sort(key=lambda r: r[0])  # sort by tcount
        xs = np.array([r[0] for r in rows])
        ys = np.array([r[1] for r in rows])
        es = np.array([r[2] for r in rows])
        ax.errorbar(xs/(n_qubits), ys/n_qubits, yerr=es/n_qubits, label=f"n={n_qubits}", marker="o", capsize=3)

    NT = np.linspace(0, 2*n_layers, 1000, dtype=float)

    for j, n_qubits in enumerate(ns):
        alpha = 0.2 + 0.8 * j / (len(ns) - 1)
        alpha = min(1, max(0, alpha))
        q = NT
        gsre = _gse(2, n_qubits, NT)
        ax.plot(NT/n_qubits, gsre/n_qubits, linestyle="-", color="black", alpha=alpha, label=f"GSE n={n_qubits}")
    M_max = np.log2(2**ns[-1])
    plt.axhline(y=(M_max)/ns[-1], linestyle="--", alpha=0.7)
    q_c = np.log2(2) / np.log2(4/3)
    plt.axvline(x=q_c,  linestyle=":", alpha=0.7)

    ax.set_xlabel("q (T-count per qubit)")
    ax.set_ylabel(r"$m_2$ (SRE per qubit)")
    ax.grid(alpha=0.3)
    ax.legend(title="Number of qubits")
    ax.set_title(title or r"$m_2$ vs q", fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)

def _unique_path(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return str(p)

    stem = p.stem
    suffix = p.suffix
    parent = p.parent

    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return str(candidate)
        i += 1

def plot_training_curves(
    hist: TrainHistory,
    title: str = "Training curves",
    save_fig: bool = False,
    fig_path: str | None = None,
):
    if fig_path is None:
        raise ValueError("fig_path must be provided if save_fig is True")
    os.makedirs(fig_path, exist_ok=True)
    epochs = list(range(1, len(hist.train_loss) + 1))

    plt.figure()
    plt.plot(epochs, hist.train_loss, label="train")
    plt.plot(epochs, hist.val_loss, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_fig and fig_path is not None:
        safe_path = _unique_path(fig_path)
        plt.savefig(safe_path)

    plt.show()


# =========================================================
# Plotting predictions
# =========================================================

def plot_fixed_layers_vary_qubits(
    rows: list[dict[str, Any]],
    *,
    n_layers: int,
    output_path: str | Path | None = None,
    split_by_family: bool = True,
):
    filtered = [r for r in rows if int(r["n_layers"]) == int(n_layers)]
    if not filtered:
        logger.info("No predictions found for n_layers=%s", n_layers)
        return

    plt.figure(figsize=(8, 5))

    if split_by_family:
        families = sorted({str(r.get("family", "unknown")) for r in filtered})
        for family in families:
            family_rows = [r for r in filtered if str(r.get("family", "unknown")) == family]

            groups: dict[int, list[float]] = {}
            for r in family_rows:
                q = int(r["n_qubits"])
                groups.setdefault(q, []).append(float(r["prediction"]))

            x = sorted(groups.keys())
            y = [np.mean(groups[q]) for q in x]
            yerr = [np.std(groups[q], ddof=0) for q in x]

            plt.errorbar(x, y, yerr=yerr, marker="o", linestyle="-", capsize=3, label=family)

        plt.legend(title="family")
        plt.title(f"Predicted SRE vs qubits (n_layers={n_layers})")
    else:
        groups: dict[int, list[float]] = {}
        for r in filtered:
            q = int(r["n_qubits"])
            groups.setdefault(q, []).append(float(r["prediction"]))

        x = sorted(groups.keys())
        y = [np.mean(groups[q]) for q in x]
        yerr = [np.std(groups[q], ddof=0) for q in x]

        plt.errorbar(x, y, yerr=yerr, marker="o", linestyle="-", capsize=3)
        plt.title(f"Predicted SRE vs qubits (n_layers={n_layers})")

    plt.xlabel("Number of qubits")
    plt.ylabel("Predicted SRE")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=180)
        plt.close()
    else:
        plt.show()

def plot_fixed_qubits_vary_layers(
    rows: list[dict[str, Any]],
    *,
    n_qubits: int,
    output_path: str | Path | None = None,
    split_by_family: bool = True,
):
    filtered = [r for r in rows if int(r["n_qubits"]) == int(n_qubits)]
    if not filtered:
        logger.info("No predictions found for n_qubits=%s", n_qubits)
        return

    plt.figure(figsize=(8, 5))

    if split_by_family:
        families = sorted({str(r.get("family", "unknown")) for r in filtered})
        for family in families:
            family_rows = [r for r in filtered if str(r.get("family", "unknown")) == family]

            groups: dict[int, list[float]] = {}
            for r in family_rows:
                L = int(r["n_layers"])
                groups.setdefault(L, []).append(float(r["prediction"]))

            x = sorted(groups.keys())
            y = [np.mean(groups[L]) for L in x]
            yerr = [np.std(groups[L], ddof=0) for L in x]

            plt.errorbar(x, y, yerr=yerr, marker="o", linestyle="-", capsize=3, label=family)

        plt.legend(title="family")
        plt.title(f"Predicted SRE vs layers (n_qubits={n_qubits})")
    else:
        groups: dict[int, list[float]] = {}
        for r in filtered:
            L = int(r["n_layers"])
            groups.setdefault(L, []).append(float(r["prediction"]))

        x = sorted(groups.keys())
        y = [np.mean(groups[L]) for L in x]
        yerr = [np.std(groups[L], ddof=0) for L in x]

        plt.errorbar(x, y, yerr=yerr, marker="o", linestyle="-", capsize=3)
        plt.title(f"Predicted SRE vs layers (n_qubits={n_qubits})")

    plt.xlabel("Number of layers")
    plt.ylabel("Predicted SRE")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=180)
        plt.close()
    else:
        plt.show()


def pearson_corr_fast(df: pd.DataFrame, col_x: str, col_y: str) -> float:
    x = df[col_x].to_numpy(dtype=float)
    y = df[col_y].to_numpy(dtype=float)
    return float(np.corrcoef(x, y)[0, 1])

def view_correlation(
    df: pd.DataFrame,
    nq: int = 10,
    nl: int = 100,
    col_x: str = "target",
    col_y: str = "prediction",
):


    subset = df[(df["n_qubits"] == nq) & (df["n_layers"] == nl)].copy()

    # Reset index so "sample index" is clean
    subset = subset.reset_index(drop=True)

    plt.figure(figsize=(8, 4))

    x = subset.index  # sample index (what you want)

    plt.plot(x, subset[col_x], label="target", marker="o")
    plt.plot(x, subset[col_y], label="prediction", marker="x")

    plt.xlabel("Sample index")
    plt.ylabel("Value")
    plt.title(f"Target vs Prediction (n_qubits={nq}, n_layers={nl})")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.show()
    r = pearson_corr_fast(subset, col_x, col_y)
    print(f"Pearson r = {r:.4f}")

def _plot_SRE_distribution(
    dataset,
    bins=30,
    value_range=None,
    density=True,
    show_mean=True,
    min_count=1,
    title="SRE frequency by number of qubits",
):
    grouped_sre = {}

    # -----------------------------
    # Collect SRE values by qubit count
    # -----------------------------
    for data in dataset:
        if not hasattr(data, "num_qubits"):
            continue
        if not hasattr(data, "y") or data.y is None:
            continue

        q = int(data.num_qubits)

        # robust extraction of scalar y
        y = data.y
        if torch.is_tensor(y):
            if y.numel() == 0:
                continue
            y_val = float(y.view(-1)[0].item())
        else:
            y_val = float(np.array(y).reshape(-1)[0])

        if not np.isfinite(y_val):
            continue

        grouped_sre.setdefault(q, []).append(y_val)

    # remove groups with too few points
    grouped_sre = {q: vals for q, vals in grouped_sre.items() if len(vals) >= min_count}

    if not grouped_sre:
        raise ValueError("No valid finite SRE values found in the dataset.")

    # -----------------------------
    # Determine histogram range
    # -----------------------------
    all_vals = np.concatenate([np.asarray(v, dtype=float) for v in grouped_sre.values()])
    if value_range is None:
        xmin = float(np.nanmin(all_vals))
        xmax = float(np.nanmax(all_vals))
        if xmin == xmax:
            xmin -= 0.5
            xmax += 0.5
        value_range = (xmin, xmax)

    bin_edges = np.linspace(value_range[0], value_range[1], bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(9, 6))

    for q in sorted(grouped_sre):
        vals = np.array(grouped_sre[q])

        counts, _ = np.histogram(vals, bins=bin_edges)

        if density:
            freq = 100 * counts / counts.sum()
        else:
            freq = counts

        # plot curve and capture color
        line, = plt.plot(bin_centers, freq, label=f"{q} qubits")
        color = line.get_color()

        # plot mean with SAME color
        if show_mean:
            mean_val = vals.mean()
            plt.axvline(
                mean_val,
                linestyle="--",
                linewidth=1.5,
                color=color,
                alpha=0.9,
            )

            ymax = freq.max()
            plt.text(
                mean_val - 0.5,
                ymax + 0.5,
                f"{mean_val:.2f}",
                color=color,
                ha="center",
                fontsize=8,
            )

    plt.xlabel("Stabilizer Rényi Entropy")
    plt.ylabel("Frequency (%)")
    plt.title(title, fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

