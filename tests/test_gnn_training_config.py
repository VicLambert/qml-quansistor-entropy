"""Tests for qqe.GNN.training.train_config module."""

from __future__ import annotations

import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qqe.GNN.training.train_config import (
    FAMILY_GATE_TYPES,
    FAMILY_REGISTRY,
    MASTER_GATE_TYPES,
    TrainConfig,
)


class TestTrainConfig:
    """Tests for TrainConfig dataclass."""

    def test_default_config(self):
        """Test creating TrainConfig with defaults."""
        config = TrainConfig()

        assert config.epochs == 30
        assert config.lr == 1e-3
        assert config.loss_type == "mse"
        assert config.batch_size == 32
        assert config.training_mode == "global"
        assert config.family is None
        assert config.target == "sre"
        assert config.global_feature_variant == "binned"
        assert config.node_feature_backend_variant is None
        assert config.seed == 42
        assert config.train_split == 0.8
        assert config.val_split == 0.1
        assert config.show_progress is True
        assert config.show_val_progress is False
        assert config.log_batch_loss_every == 10
        assert config.heartbeat == 60.0
        assert config.epoch_warning == 300.0

    def test_custom_config(self):
        """Test creating TrainConfig with custom values."""
        config = TrainConfig(
            epochs=50,
            lr=5e-4,
            loss_type="huber",
            batch_size=64,
            training_mode="per_family",
            family="haar",
            target="ee",
        )

        assert config.epochs == 50
        assert config.lr == 5e-4
        assert config.loss_type == "huber"
        assert config.batch_size == 64
        assert config.training_mode == "per_family"
        assert config.family == "haar"
        assert config.target == "ee"

    def test_config_is_dataclass(self):
        """Test that TrainConfig is a dataclass."""
        config = TrainConfig()
        # Dataclasses support __dataclass_fields__
        assert hasattr(config, "__dataclass_fields__")

    def test_config_supports_comparison(self):
        """Test that TrainConfig instances can be compared."""
        config1 = TrainConfig(epochs=30)
        config2 = TrainConfig(epochs=30)
        config3 = TrainConfig(epochs=50)

        assert config1 == config2
        assert config1 != config3

    def test_config_partial_override(self):
        """Test overriding specific config fields."""
        config = TrainConfig(
            epochs=100,
            loss_type="l1",
        )

        assert config.epochs == 100
        assert config.loss_type == "l1"
        # Other fields should have defaults
        assert config.lr == 1e-3
        assert config.batch_size == 32


class TestFamilyRegistry:
    """Tests for FAMILY_REGISTRY."""

    def test_family_registry_structure(self):
        """Test FAMILY_REGISTRY contains expected families."""
        expected_families = {"haar", "random", "clifford", "quansistor"}
        assert set(FAMILY_REGISTRY.keys()) == expected_families

    def test_family_registry_values_are_true(self):
        """Test all FAMILY_REGISTRY values are True."""
        for family, value in FAMILY_REGISTRY.items():
            assert value is True

    def test_family_registry_is_dict(self):
        """Test FAMILY_REGISTRY is a dictionary."""
        assert isinstance(FAMILY_REGISTRY, dict)


class TestMasterGateTypes:
    """Tests for MASTER_GATE_TYPES."""

    def test_master_gate_types_list(self):
        """Test MASTER_GATE_TYPES is a list."""
        assert isinstance(MASTER_GATE_TYPES, list)

    def test_master_gate_types_not_empty(self):
        """Test MASTER_GATE_TYPES contains gates."""
        assert len(MASTER_GATE_TYPES) > 0

    def test_master_gate_types_contains_basic_gates(self):
        """Test MASTER_GATE_TYPES contains expected basic gates."""
        expected = {"input", "measurement", "h", "cx"}
        assert expected.issubset(set(MASTER_GATE_TYPES))

    def test_master_gate_types_all_strings(self):
        """Test all master gate types are strings."""
        for gate_type in MASTER_GATE_TYPES:
            assert isinstance(gate_type, str)

    def test_master_gate_types_no_duplicates(self):
        """Test MASTER_GATE_TYPES has no duplicates."""
        assert len(MASTER_GATE_TYPES) == len(set(MASTER_GATE_TYPES))


class TestFamilyGateTypes:
    """Tests for FAMILY_GATE_TYPES."""

    def test_family_gate_types_structure(self):
        """Test FAMILY_GATE_TYPES has correct structure."""
        assert isinstance(FAMILY_GATE_TYPES, dict)
        expected_families = {"haar", "random", "clifford", "quansistor"}
        assert set(FAMILY_GATE_TYPES.keys()) == expected_families

    def test_family_gate_types_are_lists(self):
        """Test all FAMILY_GATE_TYPES values are lists."""
        for family, gates in FAMILY_GATE_TYPES.items():
            assert isinstance(gates, list), f"Family {family} gates not a list"

    def test_family_gate_types_not_empty(self):
        """Test all families have at least one gate type."""
        for family, gates in FAMILY_GATE_TYPES.items():
            assert len(gates) > 0, f"Family {family} has no gates"

    def test_family_gate_types_are_valid(self):
        """Test all family gate types are in MASTER_GATE_TYPES."""
        master_set = set(MASTER_GATE_TYPES)
        for family, gates in FAMILY_GATE_TYPES.items():
            for gate in gates:
                assert gate in master_set, f"Gate {gate} in {family} not in MASTER_GATE_TYPES"

    def test_family_specific_gates(self):
        """Test each family has expected gate types."""
        # Random should have rotation gates
        assert "rx" in FAMILY_GATE_TYPES["random"]
        assert "ry" in FAMILY_GATE_TYPES["random"]
        assert "rz" in FAMILY_GATE_TYPES["random"]

        # Clifford should have clifford gates
        assert "h" in FAMILY_GATE_TYPES["clifford"]
        assert "s" in FAMILY_GATE_TYPES["clifford"]
        assert "t" in FAMILY_GATE_TYPES["clifford"]

        # Haar should have haar
        assert "haar" in FAMILY_GATE_TYPES["haar"]

        # Quansistor should have qx and qy
        assert "qx" in FAMILY_GATE_TYPES["quansistor"]
        assert "qy" in FAMILY_GATE_TYPES["quansistor"]

    def test_all_families_have_input_measurement(self):
        """Test all families include input and measurement gates."""
        required = {"input", "measurement"}
        for family, gates in FAMILY_GATE_TYPES.items():
            assert required.issubset(
                set(gates)
            ), f"Family {family} missing input/measurement gates"

    def test_family_gate_types_no_duplicates_per_family(self):
        """Test no duplicate gates within a family."""
        for family, gates in FAMILY_GATE_TYPES.items():
            assert len(gates) == len(set(gates)), f"Family {family} has duplicate gates"

    def test_cnot_consistency(self):
        """Test CNOT gate appears in appropriate families."""
        # CNOT should be in random and clifford (they use it)
        assert "cx" in FAMILY_GATE_TYPES["random"]
        assert "cx" in FAMILY_GATE_TYPES["clifford"]


class TestConfigIntegration:
    """Integration tests for train_config module."""

    def test_config_with_all_families(self):
        """Test config works with all registered families."""
        for family in FAMILY_REGISTRY.keys():
            config = TrainConfig(
                training_mode="per_family",
                family=family,
            )
            assert config.family == family

    def test_registry_and_gate_types_consistency(self):
        """Test FAMILY_REGISTRY and FAMILY_GATE_TYPES are consistent."""
        # All families in FAMILY_GATE_TYPES should be in FAMILY_REGISTRY
        for family in FAMILY_GATE_TYPES.keys():
            assert (
                family in FAMILY_REGISTRY
            ), f"Family {family} in FAMILY_GATE_TYPES but not FAMILY_REGISTRY"

        # All families in FAMILY_REGISTRY should be in FAMILY_GATE_TYPES
        for family in FAMILY_REGISTRY.keys():
            assert (
                family in FAMILY_GATE_TYPES
            ), f"Family {family} in FAMILY_REGISTRY but not FAMILY_GATE_TYPES"
