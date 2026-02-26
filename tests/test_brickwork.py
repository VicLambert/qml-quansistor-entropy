"""Tests for brickwork pattern generation."""

import pytest

from circuit.families.pattern.brickwork import brickwork_pattern


class TestBrickworkPattern:
    """Tests for brickwork pattern generation."""

    def test_brickwork_pattern_line_topology_even_n_layer_0(self):
        """Test brickwork pattern for even number of qubits, layer 0, line topology."""
        pattern = brickwork_pattern(n_qubits=8, layer=0, topology="line")
        assert pattern == [(0, 1), (2, 3), (4, 5), (6, 7)]

    def test_brickwork_pattern_line_topology_even_n_layer_1(self):
        """Test brickwork pattern for even number of qubits, layer 1, line topology."""
        pattern = brickwork_pattern(n_qubits=8, layer=1, topology="line")
        assert pattern == [(1, 2), (3, 4), (5, 6)]

    def test_brickwork_pattern_line_topology_odd_n_layer_0(self):
        """Test brickwork pattern for odd number of qubits, layer 0, line topology."""
        pattern = brickwork_pattern(n_qubits=7, layer=0, topology="line")
        assert pattern == [(0, 1), (2, 3), (4, 5)]

    def test_brickwork_pattern_line_topology_odd_n_layer_1(self):
        """Test brickwork pattern for odd number of qubits, layer 1, line topology."""
        pattern = brickwork_pattern(n_qubits=7, layer=1, topology="line")
        assert pattern == [(1, 2), (3, 4), (5, 6)]

    def test_brickwork_pattern_alternates_by_parity(self):
        """Test that even and odd layers alternate."""
        n = 6
        pattern_even = brickwork_pattern(n, layer=0, topology="line")
        pattern_odd = brickwork_pattern(n, layer=1, topology="line")
        assert pattern_even != pattern_odd
        # Even layer starts at 0
        assert pattern_even[0][0] == 0
        # Odd layer starts at 1
        assert pattern_odd[0][0] == 1

    def test_brickwork_pattern_no_wire_overlap(self):
        """Test that pairs in a layer don't share wires."""
        for n in [4, 6, 8, 10]:
            pattern = brickwork_pattern(n, layer=0, topology="line")
            wires = set()
            for a, b in pattern:
                assert a not in wires and b not in wires
                wires.add(a)
                wires.add(b)

    def test_brickwork_pattern_wire_range_valid(self):
        """Test that all wires are within valid range."""
        n = 10
        for layer in range(5):
            pattern = brickwork_pattern(n, layer=layer, topology="line")
            for a, b in pattern:
                assert 0 <= a < n
                assert 0 <= b < n

    def test_brickwork_pattern_nearest_neighbor(self):
        """Test that all pairs are nearest neighbors."""
        n = 8
        for layer in range(3):
            pattern = brickwork_pattern(n, layer=layer, topology="line")
            for a, b in pattern:
                assert abs(a - b) == 1

    def test_brickwork_pattern_loop_topology_not_implemented(self):
        """Test that loop topology raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            brickwork_pattern(n_qubits=8, layer=0, topology="loop")

    def test_brickwork_pattern_small_n(self):
        """Test brickwork pattern with small n."""
        # n=2
        assert brickwork_pattern(2, layer=0, topology="line") == [(0, 1)]
        assert brickwork_pattern(2, layer=1, topology="line") == []

    def test_brickwork_pattern_very_large_n(self):
        """Test brickwork pattern with large n."""
        n = 100
        pattern = brickwork_pattern(n, layer=0, topology="line")
        assert all(0 <= a < n and 0 <= b < n for a, b in pattern)
        assert len(pattern) == n // 2

    def test_brickwork_pattern_layer_periodicity(self):
        """Test that layers repeat with period 2."""
        n = 8
        layer_0 = brickwork_pattern(n, layer=0, topology="line")
        layer_2 = brickwork_pattern(n, layer=2, topology="line")
        assert layer_0 == layer_2
