"""Tests for gate matrix implementations."""

import numpy as np

from circuit.families.gates import circ_X, circ_Y


class TestCircX:
    """Tests for circ_X gate matrix function."""

    def test_circ_x_creates_4x4_matrix(self):
        """Test that circ_X returns a 4x4 matrix."""
        matrix = circ_X(a=0.5, b=0.3, g=0.2)
        assert matrix.shape == (4, 4)

    def test_circ_x_is_complex(self):
        """Test that circ_X matrix has complex dtype."""
        matrix = circ_X(a=0.5, b=0.3, g=0.2)
        assert matrix.dtype == np.complex128

    def test_circ_x_at_zero(self):
        """Test circ_X with zero parameters."""
        matrix = circ_X(a=0, b=0, g=0)
        assert matrix.shape == (4, 4)
        assert not np.isnan(matrix).any()

    def test_circ_x_deterministic(self):
        """Test that circ_X is deterministic."""
        matrix1 = circ_X(a=0.5, b=0.3, g=0.2)
        matrix2 = circ_X(a=0.5, b=0.3, g=0.2)
        np.testing.assert_array_equal(matrix1, matrix2)

    def test_circ_x_different_params(self):
        """Test that different parameters produce different matrices."""
        matrix1 = circ_X(a=0.5, b=0.3, g=0.2)
        matrix2 = circ_X(a=0.6, b=0.3, g=0.2)
        assert not np.allclose(matrix1, matrix2)

    def test_circ_x_no_nan(self):
        """Test that circ_X doesn't produce NaN values."""
        for a in np.linspace(-1, 1, 5):
            for b in np.linspace(-1, 1, 5):
                for g in np.linspace(-1, 1, 5):
                    matrix = circ_X(a, b, g)
                    assert not np.isnan(matrix).any()

    def test_circ_x_no_inf(self):
        """Test that circ_X doesn't produce infinite values."""
        for a in np.linspace(-1, 1, 5):
            for b in np.linspace(-1, 1, 5):
                for g in np.linspace(-1, 1, 5):
                    matrix = circ_X(a, b, g)
                    assert not np.isinf(matrix).any()


class TestCircY:
    """Tests for circ_Y gate matrix function."""

    def test_circ_y_creates_4x4_matrix(self):
        """Test that circ_Y returns a 4x4 matrix."""
        matrix = circ_Y(a=0.5, b=0.3, g=0.2)
        assert matrix.shape == (4, 4)

    def test_circ_y_is_complex(self):
        """Test that circ_Y matrix has complex dtype."""
        matrix = circ_Y(a=0.5, b=0.3, g=0.2)
        assert matrix.dtype == np.complex128

    def test_circ_y_at_zero(self):
        """Test circ_Y with zero parameters."""
        matrix = circ_Y(a=0, b=0, g=0)
        assert matrix.shape == (4, 4)
        assert not np.isnan(matrix).any()

    def test_circ_y_deterministic(self):
        """Test that circ_Y is deterministic."""
        matrix1 = circ_Y(a=0.5, b=0.3, g=0.2)
        matrix2 = circ_Y(a=0.5, b=0.3, g=0.2)
        np.testing.assert_array_equal(matrix1, matrix2)

    def test_circ_y_different_params(self):
        """Test that different parameters produce different matrices."""
        matrix1 = circ_Y(a=0.5, b=0.3, g=0.2)
        matrix2 = circ_Y(a=0.6, b=0.3, g=0.2)
        assert not np.allclose(matrix1, matrix2)

    def test_circ_y_no_nan(self):
        """Test that circ_Y doesn't produce NaN values."""
        for a in np.linspace(-1, 1, 5):
            for b in np.linspace(-1, 1, 5):
                for g in np.linspace(-1, 1, 5):
                    matrix = circ_Y(a, b, g)
                    assert not np.isnan(matrix).any()

    def test_circ_y_no_inf(self):
        """Test that circ_Y doesn't produce infinite values."""
        for a in np.linspace(-1, 1, 5):
            for b in np.linspace(-1, 1, 5):
                for g in np.linspace(-1, 1, 5):
                    matrix = circ_Y(a, b, g)
                    assert not np.isinf(matrix).any()
