"""Tests for circuit family implementations."""

from circuit.families import FAMILY_REGISTRY
from circuit.families.clifford import CliffordBrickwork
from circuit.families.haar import HaarBrickwork
from circuit.families.quansistor import (QuansistorBrickwork, leftover_pairs,
                                         quansistor_block)
from circuit.spec import CircuitSpec, GateSpec


class TestFamilyRegistry:
    """Tests for the FAMILY_REGISTRY."""

    def test_family_registry_contains_families(self):
        """Test that registry contains expected families."""
        expected = {"haar", "clifford", "quansistor"}
        assert set(FAMILY_REGISTRY.keys()) == expected

    def test_family_registry_families_are_instances(self):
        """Test that registry values are family instances."""
        assert isinstance(FAMILY_REGISTRY["haar"], HaarBrickwork)
        assert isinstance(FAMILY_REGISTRY["clifford"], CliffordBrickwork)
        assert isinstance(FAMILY_REGISTRY["quansistor"], QuansistorBrickwork)


class TestHaarBrickwork:
    """Tests for HaarBrickwork family."""

    def test_haar_name(self):
        """Test HaarBrickwork name."""
        family = HaarBrickwork()
        assert family.name == "haar"

    def test_haar_make_spec(self):
        """Test HaarBrickwork.make_spec."""
        family = HaarBrickwork()
        spec = family.make_spec(n_qubits=6, n_layers=3, d=2, seed=123, topology="line")

        assert isinstance(spec, CircuitSpec)
        assert spec.n_qubits == 6
        assert spec.n_layers == 3
        assert spec.d == 2
        assert spec.global_seed == 123
        assert spec.family == "haar"
        assert spec.topology == "line"

    def test_haar_gates(self):
        """Test HaarBrickwork.gates generator."""
        family = HaarBrickwork()
        spec = family.make_spec(n_qubits=4, n_layers=2, d=2, seed=123, topology="line")
        gates = list(family.gates(spec))

        assert len(gates) > 0
        assert all(isinstance(g, GateSpec) for g in gates)
        assert all(g.kind == "haar" for g in gates)
        assert all(len(g.wires) == 2 for g in gates)  # 2-qubit gates

    def test_haar_gates_structure(self):
        """Test that Haar gates have expected structure."""
        family = HaarBrickwork()
        spec = family.make_spec(n_qubits=4, n_layers=2, d=2, seed=123, topology="line")
        gates = list(family.gates(spec))

        # Each layer should have specific number of gates
        layer_0_gates = [g for g in gates if "L0" in g.tags]
        layer_1_gates = [g for g in gates if "L1" in g.tags]
        assert len(layer_0_gates) > 0
        assert len(layer_1_gates) > 0

    def test_haar_deterministic(self):
        """Test that Haar gate generation is deterministic."""
        family = HaarBrickwork()
        spec1 = family.make_spec(n_qubits=6, n_layers=3, d=2, seed=999, topology="line")
        spec2 = family.make_spec(n_qubits=6, n_layers=3, d=2, seed=999, topology="line")

        gates1 = list(family.gates(spec1))
        gates2 = list(family.gates(spec2))

        assert gates1 == gates2

    def test_haar_different_seeds_produce_different_gates(self):
        """Test that different seeds produce different gate seeds."""
        family = HaarBrickwork()
        spec1 = family.make_spec(n_qubits=6, n_layers=3, d=2, seed=999, topology="line")
        spec2 = family.make_spec(n_qubits=6, n_layers=3, d=2, seed=1000, topology="line")

        gates1 = list(family.gates(spec1))
        gates2 = list(family.gates(spec2))

        # Gate seeds should be different
        seeds1 = [g.seed for g in gates1]
        seeds2 = [g.seed for g in gates2]
        assert seeds1 != seeds2


class TestCliffordBrickwork:
    """Tests for CliffordBrickwork family."""

    def test_clifford_name(self):
        """Test CliffordBrickwork name."""
        family = CliffordBrickwork()
        assert family.name == "clifford"

    def test_clifford_make_spec(self):
        """Test CliffordBrickwork.make_spec."""
        family = CliffordBrickwork()
        spec = family.make_spec(n_qubits=6, n_layers=3, d=2, seed=123, topology="line")

        assert isinstance(spec, CircuitSpec)
        assert spec.n_qubits == 6
        assert spec.n_layers == 3
        assert spec.family == "clifford"

    def test_clifford_gates_contain_clifford_gates(self):
        """Test that Clifford gates are generated."""
        family = CliffordBrickwork()
        spec = family.make_spec(n_qubits=6, n_layers=3, d=2, seed=123, topology="line")
        gates = list(family.gates(spec))

        clifford_gates = [g for g in gates if g.kind == "clifford"]
        assert len(clifford_gates) > 0
        assert all(len(g.wires) == 2 for g in clifford_gates)

    def test_clifford_with_tdoping(self):
        """Test CliffordBrickwork with T-doping rules."""
        from circuit.families.pattern.tdoping import TdopingRules

        tdoping = TdopingRules(count=2, placement="center_pair")
        family = CliffordBrickwork(tdoping=tdoping)
        spec = family.make_spec(n_qubits=6, n_layers=3, d=2, seed=123, topology="line")

        assert spec.params["tdoping"] == tdoping


class TestQuansistorBrickwork:
    """Tests for QuansistorBrickwork family."""

    def test_quansistor_name(self):
        """Test QuansistorBrickwork name."""
        family = QuansistorBrickwork()
        assert family.name == "quansistor"

    def test_quansistor_make_spec(self):
        """Test QuansistorBrickwork.make_spec."""
        family = QuansistorBrickwork()
        spec = family.make_spec(n_qubits=8, n_layers=4, d=2, seed=123, topology="line")

        assert isinstance(spec, CircuitSpec)
        assert spec.n_qubits == 8
        assert spec.n_layers == 4
        assert spec.family == "quansistor"

    def test_quansistor_gates_contain_blocks(self):
        """Test that Quansistor gates contain 4-qubit blocks."""
        family = QuansistorBrickwork()
        spec = family.make_spec(n_qubits=8, n_layers=2, d=2, seed=123, topology="line")
        gates = list(family.gates(spec))

        block_gates = [g for g in gates if g.kind == "quansistor_block"]
        assert len(block_gates) > 0
        assert all(len(g.wires) == 4 for g in block_gates)


class TestQuansistorBlock:
    """Tests for quansistor_block function."""

    def test_quansistor_block_layer_0(self):
        """Test quansistor_block for layer 0."""
        blocks = quansistor_block(n_qubits=8, n_layer=0)
        assert blocks == [(0, 1, 2, 3), (4, 5, 6, 7)]

    def test_quansistor_block_layer_1(self):
        """Test quansistor_block for layer 1."""
        blocks = quansistor_block(n_qubits=8, n_layer=1)
        # Layer 1 (odd) starts at 1, range(1, 8-3, 4) = range(1, 5, 4) gives only i=1
        assert blocks == [(1, 2, 3, 4)]

    def test_quansistor_block_small_n(self):
        """Test quansistor_block with small n."""
        blocks = quansistor_block(n_qubits=4, n_layer=0)
        assert blocks == [(0, 1, 2, 3)]

    def test_quansistor_block_very_small_n(self):
        """Test quansistor_block with n < 4."""
        assert quansistor_block(n_qubits=3, n_layer=0) == []
        assert quansistor_block(n_qubits=2, n_layer=0) == []

    def test_quansistor_block_alternates_by_parity(self):
        """Test that layers alternate parity."""
        n = 10
        blocks_0 = quansistor_block(n, n_layer=0)
        blocks_1 = quansistor_block(n, n_layer=1)
        assert blocks_0 != blocks_1
        assert blocks_0[0][0] == 0
        assert blocks_1[0][0] == 1


class TestLeftoverPairs:
    """Tests for leftover_pairs function."""

    def test_leftover_pairs_returns_list(self):
        """Test that leftover_pairs returns a list."""
        result = leftover_pairs(n_qubits=8, used=set(), topology="line")
        assert isinstance(result, list)

    def test_leftover_pairs_not_implemented_yet(self):
        """Test that leftover_pairs returns empty list (not yet implemented)."""
        result = leftover_pairs(n_qubits=8, used={0, 1, 2, 3}, topology="line")
        # Currently not implemented, should return empty
        assert result == []


class TestFamilyRegistrySmokeTest:
    """Smoke tests for all families in registry."""

    def test_registry_smoke_test(self):
        """Test that all families can generate specs and gates."""
        for name, family in FAMILY_REGISTRY.items():
            # Create spec
            spec = family.make_spec(
                n_qubits=6,
                n_layers=3,
                d=2,
                seed=123,
                topology="line",
            )
            assert spec is not None

            # Generate gates
            gates = list(family.gates(spec))
            assert len(gates) > 0
            assert all(isinstance(g, GateSpec) for g in gates)

    def test_registry_all_gates_have_valid_wires(self):
        """Test that all generated gates have valid wire indices."""
        for name, family in FAMILY_REGISTRY.items():
            spec = family.make_spec(
                n_qubits=6,
                n_layers=3,
                d=2,
                seed=123,
                topology="line",
            )
            gates = list(family.gates(spec))

            for gate in gates:
                # All wires should be within range
                assert all(0 <= w < spec.n_qubits for w in gate.wires)
                # No duplicate wires in same gate
                assert len(set(gate.wires)) == len(gate.wires)
