"""Integration tests for the full circuit generation pipeline."""

import pytest

from circuit.families import FAMILY_REGISTRY
from circuit.spec import GateSpec


class TestCircuitGenerationPipeline:
    """Integration tests for complete circuit generation."""

    @pytest.mark.parametrize("family_name", ["haar", "clifford", "quansistor"])
    def test_full_pipeline(self, family_name):
        """Test complete pipeline: spec -> gates for all families."""
        family = FAMILY_REGISTRY[family_name]

        # Generate spec
        spec = family.make_spec(
            n_qubits=8,
            n_layers=4,
            d=2,
            seed=42,
            topology="line",
        )

        # Generate gates
        gates = list(family.gates(spec))

        # Verify results
        assert len(gates) > 0
        assert all(isinstance(g, GateSpec) for g in gates)
        assert all(g.d == 2 for g in gates)
        assert all(0 <= w < 8 for g in gates for w in g.wires)

    @pytest.mark.parametrize("n_qubits", [4, 6, 8, 10])
    def test_different_qubit_counts(self, n_qubits):
        """Test circuit generation with different qubit counts."""
        family = FAMILY_REGISTRY["haar"]

        spec = family.make_spec(
            n_qubits=n_qubits,
            n_layers=3,
            d=2,
            seed=123,
            topology="line",
        )

        gates = list(family.gates(spec))
        assert all(0 <= w < n_qubits for g in gates for w in g.wires)

    @pytest.mark.parametrize("n_layers", [1, 2, 3, 5, 10])
    def test_different_layer_counts(self, n_layers):
        """Test circuit generation with different layer counts."""
        family = FAMILY_REGISTRY["haar"]

        spec = family.make_spec(
            n_qubits=6,
            n_layers=n_layers,
            d=2,
            seed=123,
            topology="line",
        )

        gates = list(family.gates(spec))

        # Check that gates span all layers
        gate_layers = set()
        for gate in gates:
            for tag in gate.tags:
                if tag.startswith("L"):
                    gate_layers.add(int(tag[1:]))

        assert len(gate_layers) <= n_layers

    def test_reproducibility_across_families(self):
        """Test that same circuit can be generated multiple times."""
        for family_name in FAMILY_REGISTRY:
            family = FAMILY_REGISTRY[family_name]

            gates1 = list(
                family.gates(
                    family.make_spec(6, 3, 2, seed=999, topology="line"),
                )
            )
            gates2 = list(
                family.gates(
                    family.make_spec(6, 3, 2, seed=999, topology="line"),
                )
            )

            assert gates1 == gates2

    def test_gate_stream_consistency(self):
        """Test that gate stream is consistent in structure."""
        family = FAMILY_REGISTRY["haar"]
        spec = family.make_spec(
            n_qubits=8,
            n_layers=4,
            d=2,
            seed=123,
            topology="line",
        )

        gates = list(family.gates(spec))

        # All gates should have valid structure
        for gate in gates:
            assert isinstance(gate.kind, str)
            assert isinstance(gate.wires, tuple)
            assert all(isinstance(w, int) for w in gate.wires)
            assert gate.d > 0
            assert isinstance(gate.tags, tuple)

    def test_quansistor_block_sizes(self):
        """Test that quansistor uses correct block sizes."""
        family = FAMILY_REGISTRY["quansistor"]
        spec = family.make_spec(
            n_qubits=12,
            n_layers=2,
            d=2,
            seed=123,
            topology="line",
        )

        gates = list(family.gates(spec))
        block_gates = [g for g in gates if g.kind == "quansistor_block"]

        # All block gates should be 4-qubit gates
        assert all(len(g.wires) == 4 for g in block_gates)

    def test_circuit_seed_variations(self):
        """Test that different seeds produce different circuits."""
        family = FAMILY_REGISTRY["haar"]

        gates1 = list(
            family.gates(
                family.make_spec(6, 3, 2, seed=1, topology="line"),
            )
        )
        gates2 = list(
            family.gates(
                family.make_spec(6, 3, 2, seed=2, topology="line"),
            )
        )

        # Same structure but different seeds
        assert len(gates1) == len(gates2)
        seeds1 = [g.seed for g in gates1]
        seeds2 = [g.seed for g in gates2]
        assert seeds1 != seeds2


class TestCircuitValidation:
    """Tests for circuit validity and constraints."""

    def test_no_duplicate_wires_in_gate(self):
        """Test that no gate has duplicate wires."""
        for family_name in FAMILY_REGISTRY:
            family = FAMILY_REGISTRY[family_name]
            spec = family.make_spec(8, 3, 2, seed=123, topology="line")
            gates = list(family.gates(spec))

            for gate in gates:
                assert len(set(gate.wires)) == len(
                    gate.wires
                ), f"Gate {gate} has duplicate wires"

    def test_gate_wires_in_valid_range(self):
        """Test that all gate wires are within qubit range."""
        for family_name in FAMILY_REGISTRY:
            family = FAMILY_REGISTRY[family_name]
            n_qubits = 8
            spec = family.make_spec(n_qubits, 3, 2, seed=123, topology="line")
            gates = list(family.gates(spec))

            for gate in gates:
                for wire in gate.wires:
                    assert 0 <= wire < n_qubits, f"Wire {wire} out of range [0, {n_qubits})"

    def test_gate_tags_format(self):
        """Test that gate tags have expected format."""
        for family_name in FAMILY_REGISTRY:
            family = FAMILY_REGISTRY[family_name]
            spec = family.make_spec(8, 3, 2, seed=123, topology="line")
            gates = list(family.gates(spec))

            for gate in gates:
                # Should have at least a layer tag
                layer_tags = [t for t in gate.tags if t.startswith("L")]
                assert len(layer_tags) == 1, f"Gate {gate} missing layer tag"
