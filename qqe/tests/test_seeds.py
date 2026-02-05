"""Tests for random number generation and seeding."""

import numpy as np
import pytest

from rng.seeds import (SeedSchedule, _encode_str, _normalize_wires,
                       _u64_to_bytes, gate_seed, rng_from_gate, spawn_seed)


class TestU64ToBytes:
    """Tests for _u64_to_bytes utility function."""

    def test_u64_to_bytes_zero(self):
        """Test conversion of 0."""
        assert _u64_to_bytes(0) == b"\x00\x00\x00\x00\x00\x00\x00\x00"

    def test_u64_to_bytes_small_number(self):
        """Test conversion of small number."""
        result = _u64_to_bytes(1)
        assert len(result) == 8
        assert result[0] == 1

    def test_u64_to_bytes_large_number(self):
        """Test conversion of large number."""
        result = _u64_to_bytes(2**32)
        assert len(result) == 8

    def test_u64_to_bytes_negative_raises(self):
        """Test that negative numbers raise ValueError."""
        with pytest.raises(ValueError):
            _u64_to_bytes(-1)


class TestEncodeStr:
    """Tests for _encode_str utility function."""

    def test_encode_str_simple(self):
        """Test encoding simple string."""
        assert _encode_str("hello") == b"hello"

    def test_encode_str_empty(self):
        """Test encoding empty string."""
        assert _encode_str("") == b""

    def test_encode_str_unicode(self):
        """Test encoding unicode string."""
        result = _encode_str("café")
        assert isinstance(result, bytes)


class TestNormalizeWires:
    """Tests for _normalize_wires utility function."""

    def test_normalize_wires_none(self):
        """Test normalizing None."""
        assert _normalize_wires(None, ordered=False) == ()

    def test_normalize_wires_single_int(self):
        """Test normalizing single integer."""
        assert _normalize_wires(0, ordered=False) == (0,)
        assert _normalize_wires(5, ordered=False) == (5,)

    def test_normalize_wires_tuple(self):
        """Test normalizing tuple of wires."""
        assert _normalize_wires((0, 1, 2), ordered=False) == (0, 1, 2)

    def test_normalize_wires_list(self):
        """Test normalizing list of wires."""
        result = _normalize_wires([0, 1, 2], ordered=False)
        assert result == (0, 1, 2)

    def test_normalize_wires_sorting(self):
        """Test that wires are sorted when ordered=False."""
        result = _normalize_wires((2, 0, 1), ordered=False)
        assert result == (0, 1, 2)

    def test_normalize_wires_ordered_preserves_order(self):
        """Test that wires preserve order when ordered=True."""
        result = _normalize_wires((2, 0, 1), ordered=True)
        assert result == (2, 0, 1)


class TestGateSeed:
    """Tests for gate_seed function."""

    def test_gate_seed_basic(self):
        """Test basic gate seed generation."""
        seed = gate_seed(
            global_seed=42,
            kind="haar",
            layer=0,
            slot=0,
            wires=(0, 1),
        )
        assert isinstance(seed, int)
        assert seed >= 0

    def test_gate_seed_deterministic(self):
        """Test that gate_seed is deterministic."""
        seed1 = gate_seed(
            global_seed=42,
            kind="haar",
            layer=0,
            slot=0,
            wires=(0, 1),
        )
        seed2 = gate_seed(
            global_seed=42,
            kind="haar",
            layer=0,
            slot=0,
            wires=(0, 1),
        )
        assert seed1 == seed2

    def test_gate_seed_different_for_different_params(self):
        """Test that different parameters produce different seeds."""
        seed1 = gate_seed(
            global_seed=42,
            kind="haar",
            layer=0,
            slot=0,
            wires=(0, 1),
        )
        seed2 = gate_seed(
            global_seed=42,
            kind="haar",
            layer=1,  # different layer
            slot=0,
            wires=(0, 1),
        )
        assert seed1 != seed2

    def test_gate_seed_different_global_seed(self):
        """Test that different global seeds produce different seeds."""
        seed1 = gate_seed(
            global_seed=42,
            kind="haar",
            layer=0,
            slot=0,
            wires=(0, 1),
        )
        seed2 = gate_seed(
            global_seed=43,
            kind="haar",
            layer=0,
            slot=0,
            wires=(0, 1),
        )
        assert seed1 != seed2

    def test_gate_seed_different_kind(self):
        """Test that different gate kinds produce different seeds."""
        seed1 = gate_seed(
            global_seed=42,
            kind="haar",
            layer=0,
            slot=0,
            wires=(0, 1),
        )
        seed2 = gate_seed(
            global_seed=42,
            kind="clifford",
            layer=0,
            slot=0,
            wires=(0, 1),
        )
        assert seed1 != seed2

    def test_gate_seed_with_extra(self):
        """Test gate seed with extra parameter."""
        seed = gate_seed(
            global_seed=42,
            kind="haar",
            layer=0,
            slot=0,
            wires=(0, 1),
            extra="custom_extra",
        )
        assert isinstance(seed, int)

    def test_gate_seed_none_wires(self):
        """Test gate seed with None wires."""
        seed = gate_seed(
            global_seed=42,
            kind="spawn",
            layer=0,
            wires=None,
        )
        assert isinstance(seed, int)

    def test_gate_seed_wire_order_unordered(self):
        """Test that wire order doesn't matter when ordered=False."""
        seed1 = gate_seed(
            global_seed=42,
            kind="haar",
            layer=0,
            slot=0,
            wires=(0, 1),
            ordered_wires=False,
        )
        seed2 = gate_seed(
            global_seed=42,
            kind="haar",
            layer=0,
            slot=0,
            wires=(1, 0),
            ordered_wires=False,
        )
        assert seed1 == seed2

    def test_gate_seed_wire_order_ordered(self):
        """Test that wire order matters when ordered=True."""
        seed1 = gate_seed(
            global_seed=42,
            kind="haar",
            layer=0,
            slot=0,
            wires=(0, 1),
            ordered_wires=True,
        )
        seed2 = gate_seed(
            global_seed=42,
            kind="haar",
            layer=0,
            slot=0,
            wires=(1, 0),
            ordered_wires=True,
        )
        assert seed1 != seed2


class TestRngFromGate:
    """Tests for rng_from_gate function."""

    def test_rng_from_gate_creates_generator(self):
        """Test that rng_from_gate creates a numpy Generator."""
        rng = rng_from_gate(
            global_seed=42,
            kind="haar",
            layer=0,
            slot=0,
            wires=(0, 1),
        )
        assert isinstance(rng, np.random.Generator)

    def test_rng_from_gate_deterministic(self):
        """Test that rng_from_gate produces deterministic results."""
        rng1 = rng_from_gate(
            global_seed=42,
            kind="haar",
            layer=0,
            slot=0,
            wires=(0, 1),
        )
        rng2 = rng_from_gate(
            global_seed=42,
            kind="haar",
            layer=0,
            slot=0,
            wires=(0, 1),
        )
        # Both rngs should produce same random numbers
        assert rng1.uniform() == rng2.uniform()


class TestSpawnSeed:
    """Tests for spawn_seed function."""

    def test_spawn_seed_creates_seed(self):
        """Test that spawn_seed creates a valid seed."""
        seed = spawn_seed(global_seed=42, name="test")
        assert isinstance(seed, int)
        assert seed >= 0

    def test_spawn_seed_deterministic(self):
        """Test that spawn_seed is deterministic."""
        seed1 = spawn_seed(global_seed=42, name="test")
        seed2 = spawn_seed(global_seed=42, name="test")
        assert seed1 == seed2

    def test_spawn_seed_different_names(self):
        """Test that different names produce different seeds."""
        seed1 = spawn_seed(global_seed=42, name="test1")
        seed2 = spawn_seed(global_seed=42, name="test2")
        assert seed1 != seed2


class TestSeedSchedule:
    """Tests for SeedSchedule dataclass."""

    def test_seed_schedule_creation(self):
        """Test creating a SeedSchedule."""
        schedule = SeedSchedule(global_seed=42)
        assert schedule.global_seed == 42
        assert schedule.ordered_wires is False

    def test_seed_schedule_seed_method(self):
        """Test SeedSchedule.seed method."""
        schedule = SeedSchedule(global_seed=42)
        seed = schedule.seed(kind="haar", layer=0, slot=0, wires=(0, 1))
        assert isinstance(seed, int)
        assert seed >= 0

    def test_seed_schedule_deterministic(self):
        """Test that SeedSchedule.seed is deterministic."""
        schedule = SeedSchedule(global_seed=42)
        seed1 = schedule.seed(kind="haar", layer=0, slot=0, wires=(0, 1))
        seed2 = schedule.seed(kind="haar", layer=0, slot=0, wires=(0, 1))
        assert seed1 == seed2

    def test_seed_schedule_respects_ordered_wires_default(self):
        """Test that SeedSchedule respects ordered_wires default."""
        schedule1 = SeedSchedule(global_seed=42, ordered_wires=False)
        schedule2 = SeedSchedule(global_seed=42, ordered_wires=True)

        seed1 = schedule1.seed(kind="haar", layer=0, slot=0, wires=(0, 1))
        seed2 = schedule2.seed(kind="haar", layer=0, slot=0, wires=(1, 0))

        # With ordered_wires=False, (0,1) and (1,0) should give same seed
        # With ordered_wires=True, they should give different seeds
        seed3 = schedule1.seed(kind="haar", layer=0, slot=0, wires=(1, 0))
        assert seed1 == seed3

    def test_seed_schedule_rng_method(self):
        """Test SeedSchedule.rng method."""
        schedule = SeedSchedule(global_seed=42)
        rng = schedule.rng(kind="haar", layer=0, slot=0, wires=(0, 1))
        assert isinstance(rng, np.random.Generator)

    def test_seed_schedule_frozen(self):
        """Test that SeedSchedule is immutable."""
        schedule = SeedSchedule(global_seed=42)
        with pytest.raises(AttributeError):
            schedule.global_seed = 100
