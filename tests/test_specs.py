"""Tests for circuit specification classes."""

import pytest

from circuit.spec import CircuitSpec, GateSpec


class TestGateSpec:
    """Tests for GateSpec dataclass."""

    def test_gate_spec_creation_minimal(self):
        """Test creating a GateSpec with minimal arguments."""
        gate = GateSpec(kind="X", wires=(0, 1), d=2)
        assert gate.kind == "X"
        assert gate.wires == (0, 1)
        assert gate.d == 2
        assert gate.seed is None
        assert gate.tags == ()
        assert gate.meta == {}

    def test_gate_spec_creation_full(self):
        """Test creating a GateSpec with all arguments."""
        meta = {"param1": 0.5}
        gate = GateSpec(
            kind="haar",
            wires=(0, 1, 2),
            d=3,
            seed=123,
            tags=("layer", "L0", "haar"),
            meta=meta,
        )
        assert gate.kind == "haar"
        assert gate.wires == (0, 1, 2)
        assert gate.d == 3
        assert gate.seed == 123
        assert gate.tags == ("layer", "L0", "haar")
        assert gate.meta == meta

    def test_gate_spec_frozen(self):
        """Test that GateSpec is immutable."""
        gate = GateSpec(kind="X", wires=(0, 1), d=2)
        with pytest.raises(AttributeError):
            gate.kind = "Y"

    def test_gate_spec_single_wire(self):
        """Test GateSpec with single-qubit gate."""
        gate = GateSpec(kind="T", wires=(0,), d=1)
        assert len(gate.wires) == 1
        assert gate.wires[0] == 0

    def test_gate_spec_many_wires(self):
        """Test GateSpec with multi-qubit gate (quansistor block)."""
        gate = GateSpec(kind="quansistor_block", wires=(0, 1, 2, 3), d=4)
        assert len(gate.wires) == 4


class TestCircuitSpec:
    """Tests for CircuitSpec dataclass."""

    def test_circuit_spec_creation_minimal(self):
        """Test creating a CircuitSpec with minimal arguments."""
        spec = CircuitSpec(
            n_qubits=8,
            n_layers=3,
            d=2,
            family="haar",
            topology="line",
            global_seed=42,
        )
        assert spec.n_qubits == 8
        assert spec.n_layers == 3
        assert spec.d == 2
        assert spec.family == "haar"
        assert spec.topology == "line"
        assert spec.global_seed == 42
        assert spec.gates == ()
        assert spec.params == {}

    def test_circuit_spec_creation_full(self):
        """Test creating a CircuitSpec with all arguments."""
        gates = (
            GateSpec(kind="X", wires=(0, 1), d=2),
            GateSpec(kind="Y", wires=(2, 3), d=2),
        )
        params = {"tdoping": None, "custom": "value"}
        spec = CircuitSpec(
            n_qubits=4,
            n_layers=2,
            d=2,
            family="clifford",
            topology="loop",
            global_seed=123,
            gates=gates,
            params=params,
        )
        assert spec.n_qubits == 4
        assert spec.n_layers == 2
        assert len(spec.gates) == 2
        assert spec.params == params

    def test_circuit_spec_frozen(self):
        """Test that CircuitSpec is immutable."""
        spec = CircuitSpec(
            n_qubits=4,
            n_layers=2,
            d=2,
            family="haar",
            topology="line",
            global_seed=42,
        )
        with pytest.raises(AttributeError):
            spec.n_qubits = 8

    def test_circuit_spec_different_families(self):
        """Test CircuitSpec with different family names."""
        for family in ["haar", "clifford", "quansistor"]:
            spec = CircuitSpec(
                n_qubits=6,
                n_layers=4,
                d=2,
                family=family,
                topology="line",
                global_seed=999,
            )
            assert spec.family == family

    def test_circuit_spec_different_topologies(self):
        """Test CircuitSpec with different topologies."""
        for topology in ["line", "loop"]:
            spec = CircuitSpec(
                n_qubits=6,
                n_layers=4,
                d=2,
                family="haar",
                topology=topology,
                global_seed=999,
            )
            assert spec.topology == topology
