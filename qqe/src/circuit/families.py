"""Base protocol for circuit family implementations.

This module defines the Family protocol that specifies the interface for
circuit family implementations, including methods for generating circuit
specifications and retrieving gate specifications.
"""

from __future__ import annotations

import logging

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from circuit.gates import clifford_recipe_unitary
from circuit.patterns import TdopingRules, brickwork_pattern
from circuit.spec import CircuitSpec, GateSpec
from rng.seeds import gate_seed

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# General family interface
# ---------------------------------------------------------------------
class Family(Protocol):
    name: str

    def make_spec(
        self,
        n_qubits: int,
        n_layers: int,
        d: int,
        seed: int,
        *,
        connectivity: str = "loop",
        pattern: str = "random",
        **kwargs: Any,
    ) -> CircuitSpec:
        """Returns the circuit specification."""
        ...

    def gates(self, spec: CircuitSpec) -> Iterable[GateSpec]:
        """Returns the GatesSpec."""
        ...

# ---------------------------------------------------------------------
# Spec helpers
# ---------------------------------------------------------------------
def _make_spec(
    family: Family,
    n_qubits: int,
    n_layers: int,
    d: int,
    seed: int,
    connectivity: str,
    pattern: str,
    params: dict[str, Any],
) -> CircuitSpec:
    spec = CircuitSpec(
        n_qubits=int(n_qubits),
        n_layers=int(n_layers),
        d=int(d),
        global_seed=int(seed),
        family=family.name,
        connectivity=connectivity,
        pattern=pattern,
        params=params,
    )
    return replace(spec, gates=tuple(family.gates(spec)))

def _sample_quansistor_params(
    rng: np.random.Generator,
    param_regime: str,
    param_scale: float | None = None,
) -> tuple[float, float, float]:
    if param_regime == "identity_like":
        sigma = 0.02 if param_scale is None else float(param_scale)
        a, b, g = rng.normal(0.0, sigma, 3)

    elif param_regime == "weak":
        sigma = 0.20 if param_scale is None else float(param_scale)
        a, b, g = rng.normal(0.0, sigma, 3)

    elif param_regime == "moderate":
        sigma = 0.75 if param_scale is None else float(param_scale)
        a, b, g = rng.normal(0.0, sigma, 3)

    elif param_regime == "structured_equal_ab":
        scale = np.pi if param_scale is None else float(param_scale) * np.pi
        base = rng.uniform(-scale, scale)
        g = rng.uniform(-0.25, 0.25)
        a, b = base, base

    elif param_regime == "structured_opposite_ab":
        scale = np.pi if param_scale is None else float(param_scale) * np.pi
        base = rng.uniform(-scale, scale)
        g = rng.uniform(-0.25, 0.25)
        a, b = base, -base

    elif param_regime == "generic_uniform":
        a, b, g = rng.uniform(-np.pi, np.pi, 3)

    else:
        raise ValueError(f"Unknown param_regime={param_regime}")

    return float(a), float(b), float(g)

# ---------------------------------------------------------------------
# Haar random family
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class HaarBrickwork:
    """Haar random circuit family with brickwork pattern."""
    name: str = "haar"

    def make_spec(
        self,
        n_qubits: int,
        n_layers: int,
        d: int,
        seed: int,
        *,
        connectivity: str = "line",
        pattern: str = "brickwork",
        **kwargs: Any,
    ) -> CircuitSpec:
        params = dict(kwargs)

        return _make_spec(
            self,
            n_qubits,
            n_layers,
            d,
            seed,
            connectivity,
            pattern,
            params,
        )

    def gates(self, spec: CircuitSpec) -> Generator[GateSpec]:
        gate_probability = float(spec.params.get("gate_probability", 1.0))
        haar_strength = float(spec.params.get("haar_strength", 1.0))
        haar_mode = str(spec.params.get("haar_mode", "full_haar"))

        for layer in range(spec.n_layers):
            pairs = brickwork_pattern(spec.n_qubits, layer, connectivity=spec.connectivity)

            for slot, (a, b) in enumerate(pairs):
                s = gate_seed(
                    spec.global_seed,
                    layer=layer,
                    slot=slot,
                    wires=(a, b),
                    kind="haar",
                )
                rng = np.random.default_rng(s)
                if rng.random() > gate_probability:
                    yield GateSpec(
                        kind="I",
                        wires=(int(a),),
                        d=spec.d,
                        seed=s,
                        tags=("layer", f"L{layer}", "haar_skipped_identity", f"wire_{a}"),
                        params=(),
                    )
                    yield GateSpec(
                        kind="I",
                        wires=(int(b),),
                        d=spec.d,
                        seed=s,
                        tags=("layer", f"L{layer}", "haar_skipped_identity", f"wire_{b}"),
                        params=(),
                    )
                    continue

                yield GateSpec(
                    kind="haar",
                    wires=(a, b),
                    d=spec.d,
                    seed=s,
                    tags=("layer", f"L{layer}", "haar"),
                    params=(s, float(haar_strength), haar_mode),
                )

# ---------------------------------------------------------------------
# Clifford family
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class CliffordBrickwork:
    """Clifford circuit family with brickwork pattern and optional T-gate doping."""

    name: str = "clifford"
    tdoping: TdopingRules | None = None

    def make_spec(
        self,
        n_qubits: int,
        n_layers: int,
        d: int,
        seed: int,
        *,
        connectivity: str = "line",
        pattern: str = "brickwork",
        **kwargs: Any,
    ) -> CircuitSpec:
        """Create a circuit specification for a Clifford brickwork circuit.

        Parameters
        ----------
        n_qubits : int
            Number of qubits in the circuit.
        n_layers : int
            Number of layers in the circuit.
        d : int
            Depolarizing error parameter.
        seed : int
            Global random seed for reproducibility.
        connectivity : str, optional
            Qubit connectivity pattern, by default "line".
        pattern : str, optional
            Gate pattern type, by default "brickwork".
        **kwargs : Any
            Additional parameters passed to the circuit specification.

        Returns:
        -------
        CircuitSpec
            A circuit specification object with the given parameters.
        """
        params = dict(kwargs)
        params["tdoping"] = params.get("tdoping", self.tdoping)

        return _make_spec(
            self,
            n_qubits,
            n_layers,
            d,
            seed,
            connectivity,
            pattern,
            params,
        )

    def gates(self, spec: CircuitSpec) -> Iterable[GateSpec]:
        tdoping = spec.params.get("tdoping", None)
        logger.info(f"Generating gates for CliffordBrickwork with tdoping={tdoping}")

        # Prepare T-gate placement for all layers upfront
        t_wires_per_layer = {}
        if tdoping is not None and tdoping.count > 0:
            # Determine center qubits based on circuit size
            if spec.n_qubits % 2 == 0:
                center_wires = (spec.n_qubits // 2 - 1, spec.n_qubits // 2)
            else:
                center_wires = (spec.n_qubits // 2,)

            # Distribute tdoping.count T-gates across layers (excluding last layer)
            total_t_gates = tdoping.count
            n_available_layers = spec.n_layers - 1  # exclude last layer

            if n_available_layers > 0:
                # Calculate how many T-gates per layer
                t_gates_per_layer = total_t_gates // n_available_layers
                remainder = total_t_gates % n_available_layers

                for layer in range(n_available_layers):
                    # Distribute the remainder across first few layers
                    n_t_this_layer = t_gates_per_layer + (1 if layer < remainder else 0)

                    # Limit by available center wires and per_layer setting
                    n_t_this_layer = min(n_t_this_layer, tdoping.per_layer, len(center_wires))

                    if n_t_this_layer > 0:
                        t_wires_per_layer[layer] = list(center_wires[:n_t_this_layer])

        # Yield gates layer by layer: Clifford gates, then T-gates
        for layer in range(spec.n_layers):
            pairs = brickwork_pattern(spec.n_qubits, layer, connectivity=spec.connectivity)

            # Yield Clifford gates for this layer
            for slot, (a, b) in enumerate(pairs):
                s = gate_seed(
                    spec.global_seed,
                    layer=layer,
                    slot=slot,
                    wires=(a, b),
                    kind="clifford",
                    ordered_wires=True,
                )

                # Get the actual Clifford decomposition
                u_a, u_b, _ = clifford_recipe_unitary(s)

                yield GateSpec(kind=u_a, wires=(a,), d=spec.d, seed=s,
                                tags=("layer", f"L{layer}", "clifford_1q", f"wire_{a}", f"ua_{u_a}"))

                # 1q gate on wire b
                yield GateSpec(kind=u_b, wires=(b,), d=spec.d, seed=s,
                                tags=("layer", f"L{layer}", "clifford_1q", f"wire_{b}", f"ub_{u_b}"))

                # entangler
                yield GateSpec(kind="CNOT", wires=(a, b), d=spec.d, seed=s,
                            tags=("layer", f"L{layer}", "CNOT", f"wire_{a}_{b}"))

            # Yield T-gates for this layer (only if not the last layer)
            if layer in t_wires_per_layer:
                for wire in t_wires_per_layer[layer]:
                    yield GateSpec(
                        kind="T",
                        wires=(wire,),
                        d=spec.d,
                        seed=None,
                        tags=("layer", f"L{layer}", "T-gate", f"wire_{wire}"),
                    )

# ---------------------------------------------------------------------
# Quansistor family
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class QuansistorBrickwork:
    name: str = "quansistor"

    def make_spec(
        self,
        n_qubits: int,
        n_layers: int,
        d: int,
        seed: int,
        *,
        connectivity: str = "line",
        pattern: str = "brickwork",
        **kwargs: Any,
    ) -> CircuitSpec:
        params = dict(kwargs)
        return _make_spec(
            self,
            n_qubits,
            n_layers,
            d,
            seed,
            connectivity,
            pattern,
            params,
        )

    def gates(self, spec: CircuitSpec) -> Iterable[GateSpec]:
        for layer in range(spec.n_layers):
            pairs = brickwork_pattern(spec.n_qubits, layer, connectivity=spec.connectivity)

            for slot, (q0, q1) in enumerate(pairs):
                s = gate_seed(
                    spec.global_seed,
                    layer=layer,
                    slot=slot,
                    wires=(q0, q1),
                    kind="quansistor",
                )
                rng = np.random.default_rng(s)

                gate_probability = float(spec.params.get("gate_probability", 1.0))
                param_regime = str(spec.params.get("param_regime", "generic_uniform"))
                param_scale = spec.params.get("param_scale", None)

                alpha, beta, gamma = _sample_quansistor_params(
                    rng=rng,
                    param_regime=param_regime,
                    param_scale=param_scale,
                )

                if rng.random() > gate_probability:
                    yield GateSpec(
                        kind="I",
                        wires=(int(q0),),
                        d=spec.d,
                        seed=s,
                        tags=("layer", f"L{layer}", "quansistor_skipped_identity", f"wire_{q0}"),
                        params=(),
                    )
                    yield GateSpec(
                        kind="I",
                        wires=(int(q1),),
                        d=spec.d,
                        seed=s,
                        tags=("layer", f"L{layer}", "quansistor_skipped_identity", f"wire_{q1}"),
                        params=(),
                    )
                    continue


                axis = str(rng.choice(["X", "Y"]))
                params = (float(alpha), float(beta), float(gamma), axis)

                yield GateSpec(
                    kind="quansistor",
                    wires=(int(q0), int(q1)),
                    d=spec.d,
                    seed=s,
                    tags=(
                        "layer",
                        f"L{layer}",
                        "quansistor",
                        f"axis_{axis}",
                        f"wire_{q0}_{q1}",
                        f"regime_{param_regime}",
                    ),
                    params=params,
                )

# ---------------------------------------------------------------------
# Random rotations family
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class RandomCircuit:
    name: str = "random"
    p_cnot: float = 0.8
    rot_set: tuple[str, ...] = ("RX", "RY", "RZ")

    def make_spec(
        self,
        n_qubits: int,
        n_layers: int,
        d: int,
        seed: int,
        *,
        connectivity: str = "line",
        pattern: str = "random",
        **kwargs: Any,
    ) -> CircuitSpec:
        params = dict(kwargs)

        return _make_spec(
            self,
            n_qubits,
            n_layers,
            d,
            seed,
            connectivity,
            pattern,
            params,
        )

    def _allowed_edges(self, n: int, connectivity: str) -> list[tuple[int, int]]:
        if connectivity == "line":
            return [(i, i + 1) for i in range(n - 1)]
        if connectivity == "ring":
            return [(i, (i + 1) % n) for i in range(n)]
        if connectivity == "all":
            return [(i, j) for i in range(n) for j in range(i + 1, n)]
        raise ValueError(f"Unknown connectivity={connectivity}")

    def _random_disjoint_pairs(
        self, n: int, rng: np.random.Generator,
    ) -> list[tuple[int, int]]:
        """Random perfect-ish matching on vertices (ignores hardware edges)."""
        perm = rng.permutation(n).tolist()
        return [(perm[i], perm[i + 1]) for i in range(0, n - 1, 2)]

    def _random_disjoint_pairs_on_edges(
        self, n: int, edges: list[tuple[int, int]], rng: np.random.Generator,
    ) -> list[tuple[int, int]]:
        """Random maximal matching restricted to a given edge set.
        Greedy: shuffle edges, take if both endpoints unused.
        """
        edges_shuffled = edges.copy()
        rng.shuffle(edges_shuffled)

        used = set()
        pairs: list[tuple[int, int]] = []
        for a, b in edges_shuffled:
            if a in used or b in used:
                continue
            used.add(a); used.add(b)
            pairs.append((a, b))
        return pairs

    def _sample_rotation_angle(
        self,
        rng: np.random.Generator,
        angle_regime: str,
        angle_scale: float | None,
    ) -> float:
        if angle_regime == "identity_like":
            scale = 0.02 if angle_scale is None else float(angle_scale)
            return float(rng.normal(0.0, scale))

        if angle_regime == "clifford_like":
            scale = 0.02 if angle_scale is None else float(angle_scale)
            base = float(rng.choice([0.0, np.pi / 2, np.pi, 3 * np.pi / 2]))
            return float(base + rng.normal(0.0, scale))

        if angle_regime == "small_angles":
            scale = 0.25 if angle_scale is None else float(angle_scale)
            return float(rng.normal(0.0, scale))

        if angle_regime == "generic":
            return float(rng.uniform(0.0, 2 * np.pi))

        raise ValueError(f"Unknown angle_regime={angle_regime}")

    def gates(self, spec: CircuitSpec) -> Iterable[GateSpec]:
        p_cnot = float(spec.params.get("p_cnot", self.p_cnot))
        rot_set = tuple(spec.params.get("rot_set", self.rot_set))
        n = spec.n_qubits
        gate_probability = float(spec.params.get("gate_probability", 1.0))
        angle_regime = str(spec.params.get("angle_regime", "generic"))
        angle_scale = spec.params.get("angle_scale", None)

        allowed_edges = self._allowed_edges(n, spec.connectivity)


        for step in range(spec.n_layers):
            # one seed per "step" so the sampling matches "one random gate per step"
            for wire in range(n):
                s1 = gate_seed(
                    spec.global_seed,
                    layer=step,
                    slot=wire,
                    wires=(wire,),
                    kind="random",
                )
                rng1 = np.random.default_rng(s1)
                if rng1.random() > gate_probability:
                    yield GateSpec(
                        kind="I",
                        wires=(int(wire),),
                        d=spec.d,
                        seed=s1,
                        tags=("layer", f"L{step}", "random_skipped_identity", f"wire_{wire}"),
                        params=(),
                    )
                else:
                    kind = str(rng1.choice(rot_set))
                    theta = self._sample_rotation_angle(rng1, angle_regime, angle_scale)

                    yield GateSpec(
                        kind=kind,
                        wires=(int(wire),),
                        d=spec.d,
                        seed=s1,
                        tags=("layer", f"L{step}", "random_rotation", f"wire_{wire}", f"regime_{angle_regime}"),
                        params=(theta,),
                    )
            s2 = gate_seed(
                spec.global_seed,
                layer=step,
                slot=0,
                wires=(),
                kind="pairing",
                ordered_wires=True,
            )
            rng2 = np.random.default_rng(s2)
            pairs = self._random_disjoint_pairs_on_edges(n, allowed_edges, rng2)
            for slot, (a, b) in enumerate(pairs):
                s3 = gate_seed(
                    spec.global_seed,
                    layer=step,
                    slot=slot,
                    wires=(a, b),
                    kind="cnot",
                    ordered_wires=True,
                )
                rng3 = np.random.default_rng(s3)

                if rng3.random() < p_cnot * gate_probability:
                    ctrl, tgt = (a, b) if rng3.random() < 0.5 else (b, a)
                    yield GateSpec(
                        kind="CNOT",
                        wires=(ctrl, tgt),
                        d=spec.d,
                        seed=s3,
                        params=(),
                        tags=("layer", f"L{step}", "2q", "CNOT"),
                    )
