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

from qqe.circuit.gates import clifford_recipe_unitary
from qqe.circuit.patterns import TdopingRules, brickwork_pattern
from qqe.circuit.spec import CircuitSpec, GateSpec
from qqe.rng.seeds import gate_seed

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)


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


def quansistor_blocks(n_qubits: int, n_layer: int) -> list[tuple[int, int, int, int]]:
    start = (n_layer % 2) * 2  # shift by 2 wires each odd layer
    blocks: list[tuple[int, int, int, int]] = []
    for i in range(start, n_qubits - 3, 4):
        blocks.append((i, i + 1, i + 2, i + 3))
    return blocks


def leftover_pairs(n_qubits: int, used: set[int], connectivity: str) -> list[tuple[int, int]]:
    left = [i for i in range(n_qubits) if i not in used]
    leftover = set(left)

    pairs: list[tuple[int, int]] = []
    used = set()

    for i in range(n_qubits - 1):
        a, b = i, i + 1
        if a in leftover and b in leftover and a not in used and b not in used:
            pairs.append((a, b))
            used.update((a, b))
    if connectivity in ("loop", "ring"):
        a, b = n_qubits - 1, 0
        if a in leftover and b in leftover and a not in used and b not in used:
            pairs.append((a, b))
            used.update((a, b))
    return pairs


def kv(**kwargs) -> tuple[str, ...]:
    # deterministic ordering
    return tuple(f"{k}={v}" for k, v in sorted(kwargs.items()))


_BLOCK_STEPS = (
    (0, 1),  # q0 q1
    (2, 3),  # q2 q3
    (1, 2),  # q1 q2
    (0, 1),  # q0 q1
    (2, 3),  # q2 q3
)


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
        tdoping = params.get("tdoping", self.tdoping)

        params["tdoping"] = tdoping
        spec = CircuitSpec(
            n_qubits=n_qubits,
            n_layers=n_layers,
            d=d,
            global_seed=seed,
            family=self.name,
            connectivity=connectivity,
            pattern=pattern,
            params=params,
        )
        return replace(spec, gates=tuple(self.gates(spec)))

    # 
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
                decomp = f"{u_a}⊗{u_b}+CNOT"

                yield GateSpec(
                    kind="clifford",
                    wires=(a, b),
                    d=spec.d,
                    seed=s,
                    tags=(
                        "layer",
                        f"L{layer}",
                        "clifford",
                        f"wire_{a}_{b}",
                        f"decomp_{decomp}",
                    ),
                )

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


@dataclass(frozen=True)
class HaarBrickwork:
    """Haar random circuit family with brickwork pattern.

    Attributes:
        name: The name of the circuit family.
    """

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
        spec = CircuitSpec(
            n_qubits=n_qubits,
            n_layers=n_layers,
            d=d,
            global_seed=seed,
            family=self.name,
            connectivity=connectivity,
            pattern=pattern,
            params={},
        )
        return replace(spec, gates=tuple(self.gates(spec)))

    def gates(self, spec: CircuitSpec) -> Generator[GateSpec]:
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
                yield GateSpec(
                    kind="haar",
                    wires=(a, b),
                    d=spec.d,
                    seed=s,
                    tags=("layer", f"L{layer}", "haar"),
                )


def _axis_from_seed(seed: int) -> str:
    # deterministic, reproducible across runs
    rng = np.random.default_rng(seed)
    return "X" if rng.integers(0, 2) == 0 else "Y"


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
        spec = CircuitSpec(
            n_qubits=n_qubits,
            n_layers=n_layers,
            d=d,
            global_seed=seed,
            family=self.name,
            connectivity=connectivity,
            pattern=pattern,
            params={},
        )
        return replace(spec, gates=tuple(self.gates(spec)))

    def gates(self, spec: CircuitSpec) -> Iterable[GateSpec]:
        if spec.pattern == "custom":
            for layer in range(spec.n_layers):
                blocks = quansistor_blocks(spec.n_qubits, layer)
                used: set[int] = set()

                # ---- 4-qubit blocks -> 5 two-qubit gates each
                for block_idx, (q0, q1, q2, q3) in enumerate(blocks):
                    used.update((q0, q1, q2, q3))

                    # One seed namespace per block (so 5 gates share a single RNG stream conceptually)
                    # We derive per-step seeds deterministically from that block identity.
                    block_seed = gate_seed(
                        spec.global_seed,
                        kind="quansistor_block",
                        layer=layer,
                        slot=block_idx,
                        wires=(q0, q1, q2, q3),
                        ordered_wires=True,
                    )

                    # Emit the 5 sequential 2-qubit quansistor gates
                    wires4 = (q0, q1, q2, q3)
                    for step_idx, (i, j) in enumerate(_BLOCK_STEPS):
                        a, b = wires4[i], wires4[j]

                        # step seed derived from block_seed (NOT spec.global_seed directly)
                        s = gate_seed(
                            block_seed,
                            kind="quansistor",
                            layer=step_idx,  # local to block
                            slot=0,
                            wires=(a, b),
                            ordered_wires=True,
                            extra=f"L{layer}_B{block_idx}_S{step_idx}",
                        )

                        yield GateSpec(
                            kind="quansistor",
                            wires=(a, b),
                            d=spec.d,
                            seed=s,
                            tags=(
                                "layer",
                                f"L{layer}",
                                "block",
                                f"B{block_idx}",
                                "step",
                                f"S{step_idx}",
                            ),
                            params=(("block", (q0, q1, q2, q3)), ("step", step_idx)),
                        )
        elif spec.pattern == "brickwork":
            for layer in range(spec.n_layers):
                pairs = brickwork_pattern(spec.n_qubits, layer, connectivity=spec.connectivity)

                for slot, (a, b) in enumerate(pairs):
                    s = gate_seed(
                        spec.global_seed,
                        layer=layer,
                        slot=slot,
                        wires=(a, b),
                        kind="quansistor",
                    )
                    axis = _axis_from_seed(s)

                    yield GateSpec(
                        kind="quansistor",
                        wires=(a, b),
                        d=spec.d,
                        seed=s,
                        tags=(
                            "layer",
                            f"L{layer}",
                            "quansistor",
                            f"axis_{axis}",
                            f"wire_{a}_{b}",
                        ),
                    )
@dataclass(frozen=True)
class RandomCircuit:
    name: str = "random"
    p_cnot: float = 0.1
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
        spec = CircuitSpec(
            n_qubits=n_qubits,
            n_layers=n_layers,
            d=d,
            global_seed=seed,
            family=self.name,
            connectivity=connectivity,
            pattern=pattern,
            params={},
        )
        return replace(spec, gates=tuple(self.gates(spec)))
    
    def _allowed_pairs(self, n: int, connectivity: str) -> list[tuple[int,int]]:
        if connectivity == "all":
            return [(i, j) for i in range(n) for j in range(i+1, n)]
        if connectivity == "line":
            return [(i, i+1) for i in range(n-1)]
        if connectivity == "ring":
            return [(i, (i+1) % n) for i in range(n)]
        raise ValueError(f"Unknown connectivity={connectivity}")

    def gates(self, spec: CircuitSpec) -> Iterable[GateSpec]:
        p_cnot = float(spec.params.get("p_cnot", self.p_cnot))
        rot_set = tuple(spec.params.get("rot_set", self.rot_set))
        pairs = self._allowed_pairs(spec.n_qubits, spec.connectivity)

        for step in range(spec.n_layers):
            # one seed per "step" so the sampling matches "one random gate per step"
            s = gate_seed(
                spec.global_seed,
                layer=step,
                slot=0,
                wires=(),
                kind="step",
                ordered_wires=True,
            )
            rng = np.random.default_rng(s)
            if rng.random() < p_cnot:
                a, b = pairs[int(rng.integers(0, len(pairs)))]
                ctrl, tgt = (a, b) if rng.random() < 0.5 else (b, a)
                yield GateSpec(
                    kind="CNOT",
                    wires=(ctrl, tgt),
                    d=spec.d,
                    seed=s,
                    params=(),
                    tags=("step", f"S{step}", "2q", "CNOT"),
                )
            else:
                wire = int(rng.integers(0, spec.n_qubits))
                kind = str(rng.choice(rot_set))
                theta = float(rng.uniform(0, 2*np.pi))
                yield GateSpec(
                    kind=kind,                 # "RX"/"RY"/"RZ"
                    wires=(wire,),
                    d=spec.d,
                    seed=s,
                    params=(theta,),
                    tags=("step", f"S{step}", "1q", kind),
                )