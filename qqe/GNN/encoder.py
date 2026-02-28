from __future__ import annotations

import re

import numpy as np
import torch

from scipy.stats import norm
from torch_geometric.data import Data

# ---------------- helpers ----------------

def bin_theta(theta: float, n_bins: int) -> int:
    twopi = 2.0 * np.pi
    t = float(theta) % twopi
    idx = int(np.floor(n_bins * t / twopi))
    return max(0, min(idx, n_bins - 1))

def bin_gaussian_quantile(x: float, n_bins: int) -> int:
    u = float(norm.cdf(float(x)))
    idx = int(np.floor(n_bins * u))
    return max(0, min(idx, n_bins - 1))


# ---------------- QASM parsing ----------------

_QREG_RE = re.compile(r"qreg\s+q\[(\d+)\]\s*;")

_HEADER_PREFIXES = ("openqasm", "include", "opaque", "gate", "qreg")

_GATE_LINE_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(\s*([^\)]*)\s*\))?\s+"
    r"q\[(\d+)\]\s*(?:,\s*q\[(\d+)\]\s*)?;\s*$",
)

def _split_params(params_str: str | None) -> list[float]:
    if not params_str:
        return []
    parts = [p.strip() for p in params_str.split(",") if p.strip() != ""]
    return [float(p) for p in parts]

def _parse_n_qubits(qasm: str) -> int:
    m = _QREG_RE.search(qasm)
    if not m:
        raise ValueError("Could not find qreg q[N]; in QASM.")
    return int(m.group(1))

def _parse_ops(qasm: str) -> list[tuple[str, list[float], int, int | None]]:
    """Returns ops as tuples: (name, params, w0, w1)
    Skips header/declaration lines.
    """
    ops = []
    for raw in qasm.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if any(low.startswith(p) for p in _HEADER_PREFIXES):
            continue

        m = _GATE_LINE_RE.match(line)
        if not m:
            continue

        name = m.group(1)
        params = _split_params(m.group(2))
        w0 = int(m.group(3))
        w1 = int(m.group(4)) if m.group(4) is not None else None
        ops.append((name.lower(), params, w0, w1))
    return ops

def infer_family_from_qasm(qasm: str) -> str:
    ops = _parse_ops(qasm)
    names = {name for (name, _, _, _) in ops}
    if not names:
        return "unknown"
    if "qx" in names or "qy" in names:
        return "quansistor"
    if "haar" in names or "haar2" in names:
        return "haar"
    if any(n in ("rx", "ry", "rz") for n in names):
        return "random"
    if any(n in ("h", "s", "t", "id", "cx") for n in names):
        return "clifford"
    return "unknown"


# ---------------- feature templates ----------------

def _init_rotation_features(n_bins: int) -> dict[str, int]:
    feats: dict[str, int] = {}
    for g in ("rx", "ry", "rz"):
        for b in range(n_bins):
            feats[f"{g}_bin_{b}"] = 0
    feats["CNOT_count"] = 0
    return feats

def _init_quansistor_features(n_bins: int) -> dict[str, int]:
    feats: dict[str, int] = {}
    for axis in ("qx", "qy"):
        for p in ("a", "b", "g"):
            for b in range(n_bins):
                feats[f"{axis}_{p}_bin_{b}"] = 0
    return feats

def _init_haar_features() -> dict[str, int]:
    return {"haar_count": 0}

def _init_clifford_features() -> dict[str, int]:
    return {
        "I_count": 0,
        "H_count": 0,
        "S_count": 0,
        "T_count": 0,
        "CNOT_count": 0,
    }


# ---------------- main encoder ----------------

def encode_qasm_to_feature_dict(
    qasm: str,
    *,
    n_bins: int = 50,
    family: str | None = None,
    require_family: bool = True,
) -> tuple[dict[str, int], dict[str, object]]:
    n_qubits = _parse_n_qubits(qasm)
    fam = family or infer_family_from_qasm(qasm)

    if require_family and fam == "unknown":
        raise ValueError("Could not infer family from QASM. Pass family=... explicitly.")

    ops = _parse_ops(qasm)

    if fam in ("random", "rotations", "rotations+cx"):
        feats = _init_rotation_features(n_bins)
        for name, params, w0, w1 in ops:
            if name in ("rx", "ry", "rz") and len(params) == 1:
                b = bin_theta(params[0], n_bins)
                feats[f"{name}_bin_{b}"] += 1
            elif name in ("cx", "cnot"):
                feats["CNOT_count"] += 1
        return feats, {"family": fam, "n_qubits": n_qubits, "n_bins": n_bins}

    if fam == "quansistor":
        feats = _init_quansistor_features(n_bins)
        for name, params, w0, w1 in ops:
            if name not in ("qx", "qy"):
                continue
            if len(params) < 3:
                continue
            a, b, g = float(params[0]), float(params[1]), float(params[2])
            feats[f"{name}_a_bin_{bin_gaussian_quantile(a, n_bins)}"] += 1
            feats[f"{name}_b_bin_{bin_gaussian_quantile(b, n_bins)}"] += 1
            feats[f"{name}_g_bin_{bin_gaussian_quantile(g, n_bins)}"] += 1
        return feats, {"family": fam, "n_qubits": n_qubits, "n_bins": n_bins}

    if fam == "haar":
        feats = _init_haar_features()
        for name, params, w0, w1 in ops:
            if name in ("haar", "haar2") and w1 is not None:
                feats["haar_count"] += 1
        return feats, {"family": fam, "n_qubits": n_qubits, "n_bins": n_bins}

    if fam == "clifford":
        feats = _init_clifford_features()
        for name, params, w0, w1 in ops:
            if name == "id":
                feats["I_count"] += 1
            elif name == "h":
                feats["H_count"] += 1
            elif name == "s":
                feats["S_count"] += 1
            elif name == "t":
                feats["T_count"] += 1
            elif name in ("cx", "cnot"):
                feats["CNOT_count"] += 1
        return feats, {"family": fam, "n_qubits": n_qubits, "n_bins": n_bins}

    # fallback: count gate names
    feats: dict[str, int] = {}
    for name, params, w0, w1 in ops:
        feats[f"{name}_count"] = feats.get(f"{name}_count", 0) + 1

    return feats, {"family": fam, "n_qubits": n_qubits, "n_bins": n_bins}


def qasm_to_pyg_graph(
    qasm_str: str,
    num_qubits_hint: int | None = None,
    global_feature_variant: str = "baseline",
    n_bins: int = 50,
    family: str | None = None,
) -> tuple[Data, dict[str, int]]:
    """Convert a QASM string to a PyG Data graph using existing qqe architecture.

    Nodes: input per qubit, gate nodes, and output per qubit.
    Directed edges follow temporal flow on each wire.

    Args:
        qasm_str: OpenQASM 2.0 string
        num_qubits_hint: Fallback if parsing fails
        global_feature_variant: "baseline" or "binned"
        n_bins: Number of bins for parameter discretization
        family: Circuit family (haar, clifford, quansistor, random)

    Returns:
        (data, gate_counts) where data contains:
            - x: [num_nodes, node_feature_dim]
            - edge_index: [2, num_edges]
            - num_qubits: stored in data.num_qubits
            - global_features: Tensor with global circuit statistics
    """
    # Parse number of qubits
    try:
        num_qubits = _parse_n_qubits(qasm_str)
    except ValueError:
        if num_qubits_hint is not None:
            num_qubits = num_qubits_hint
        else:
            raise ValueError("Could not determine number of qubits")

    # Parse operations
    ops = _parse_ops(qasm_str)

    # Node and edge lists
    x_features: list[torch.Tensor] = []
    edge_src: list[int] = []
    edge_dst: list[int] = []
    last_node_for_qubit: list[int] = []

    # Create input nodes (one per qubit)
    for q in range(num_qubits):
        idx = len(x_features)
        x_features.append(_encode_node_feature("input", q, num_qubits))
        last_node_for_qubit.append(idx)

    # Process gates
    for gate_name, params, w0, w1 in ops:
        if w1 is None:
            # Single-qubit gate
            node_idx = len(x_features)
            x_features.append(_encode_node_feature(gate_name, w0, num_qubits, params))
            edge_src.append(last_node_for_qubit[w0])
            edge_dst.append(node_idx)
            last_node_for_qubit[w0] = node_idx
        else:
            # Two-qubit gate
            node_idx = len(x_features)
            x_features.append(_encode_node_feature(gate_name, [w0, w1], num_qubits, params))
            edge_src.extend([last_node_for_qubit[w0], last_node_for_qubit[w1]])
            edge_dst.extend([node_idx, node_idx])
            last_node_for_qubit[w0] = node_idx
            last_node_for_qubit[w1] = node_idx

    # Create output nodes (one per qubit)
    for q in range(num_qubits):
        node_idx = len(x_features)
        x_features.append(_encode_node_feature("measurement", q, num_qubits))
        edge_src.append(last_node_for_qubit[q])
        edge_dst.append(node_idx)

    # Stack node features
    if x_features:
        x = torch.stack(x_features, dim=0)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    else:
        x = torch.zeros((0, _get_node_feature_dim()), dtype=torch.float)
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Extract global features using existing encoder
    gate_features, meta = encode_qasm_to_feature_dict(
        qasm_str, n_bins=n_bins, family=family, require_family=False,
    )

    # Build global feature vector
    if global_feature_variant == "baseline":
        global_feat = _global_features_baseline(ops, num_qubits)
    elif global_feature_variant == "binned":
        global_feat = _global_features_binned(gate_features, meta, n_bins)
    else:
        raise ValueError(f"Unknown global_feature_variant: {global_feature_variant}")

    global_feat = torch.nan_to_num(global_feat, nan=0.0, posinf=0.0, neginf=0.0)

    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index)
    data.num_qubits = num_qubits
    data.global_features = global_feat

    return data, gate_features


def _encode_node_feature(
    gate_type: str,
    qubits: int | list[int],
    num_qubits: int,
    params: list[float] | None = None,
) -> torch.Tensor:
    """Encode a node feature vector."""
    # Gate type one-hot encoding
    gate_types = [
        "input", "measurement",
        "h", "s", "t", "id",
        "rx", "ry", "rz",
        "cx",
        "qx", "qy", "haar",
    ]
    gate_idx = gate_types.index(gate_type) if gate_type in gate_types else len(gate_types)
    gate_onehot = torch.zeros(len(gate_types))
    gate_onehot[gate_idx] = 1.0

    # Qubit mask (which qubits this gate acts on)
    qubit_mask = torch.zeros(num_qubits)
    if isinstance(qubits, int):
        qubit_mask[qubits] = 1.0
    else:
        for q in qubits:
            qubit_mask[q] = 1.0

    return torch.cat([gate_onehot, qubit_mask])


def _get_node_feature_dim() -> int:
    """Calculate node feature dimension."""
    return 14 + 1 + 4  # gate_types + unknown + 4 params (will be + num_qubits dynamically)


def _global_features_baseline(
    ops: list[tuple[str, list[float], int, int | None]],
    num_qubits: int,
) -> torch.Tensor:
    """Compute baseline global features: [depth, num_param_gates, num_qubits, total_gates, rx, ry, rz, cx]"""
    gate_counts = {"rx": 0, "ry": 0, "rz": 0, "cx": 0, "cnot": 0}
    depth = 0
    num_param_gates = 0

    for gate_name, params, w0, w1 in ops:
        depth += 1
        if gate_name in gate_counts:
            gate_counts[gate_name] += 1
        if gate_name in ("rx", "ry", "rz", "qx", "qy"):
            num_param_gates += 1

    total_gates = len(ops)
    cx_count = gate_counts["cx"] + gate_counts["cnot"]

    return torch.tensor(
        [
            depth,
            num_param_gates,
            num_qubits,
            total_gates,
            gate_counts["rx"],
            gate_counts["ry"],
            gate_counts["rz"],
            cx_count,
        ],
        dtype=torch.float,
    )


def _global_features_binned(
    gate_features: dict[str, int],
    meta: dict[str, object],
    n_bins: int,
) -> torch.Tensor:
    """Convert binned gate features to a fixed-size global vector."""
    # Flatten the binned features dictionary into a vector
    feature_list = []

    # Add basic metadata
    feature_list.append(float(meta.get("n_qubits", 0)))
    feature_list.append(float(n_bins))

    # Sort keys for consistent ordering
    for key in sorted(gate_features.keys()):
        feature_list.append(float(gate_features[key]))

    return torch.tensor(feature_list, dtype=torch.float)


