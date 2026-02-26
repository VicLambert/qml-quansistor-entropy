"""Tests for the sre_exact_dense compute function."""

import numpy as np

from properties.SRE.sre_exact_dense import (compute, fast_kron, make_qudit_ops,
                                            pauli_expval_fast_kron)
from states.types import DenseState


class TestMakeQuditOps:
    """Tests for make_qudit_ops function."""

    def test_make_qudit_ops_dimension_qubits(self):
        """Test that make_qudit_ops generates correct number of ops for qubits."""
        ops = make_qudit_ops(dim=2)
        assert len(ops) == 4  # 2^2 Pauli matrices
        assert all(op.shape == (2, 2) for op in ops.values())

    def test_make_qudit_ops_dimension_qudits(self):
        """Test that make_qudit_ops generates correct number of ops for qudits."""
        ops = make_qudit_ops(dim=3)
        assert len(ops) == 9  # 3^2 generalized Paulis
        assert all(op.shape == (3, 3) for op in ops.values())

    def test_make_qudit_ops_unitary(self):
        """Test that generated operators are unitary."""
        ops = make_qudit_ops(dim=2)
        for op in ops.values():
            # Check U†U = I (unitary condition)
            product = op.conj().T @ op
            assert np.allclose(product, np.eye(2))

    def test_make_qudit_ops_traceless(self):
        """Test that non-identity Pauli operators are traceless."""
        ops = make_qudit_ops(dim=2)
        # All Pauli operators except identity should be traceless
        for idx in range(1, 4):
            trace = np.trace(ops[idx])
            assert np.abs(trace) < 1e-10


class TestFastKron:
    """Tests for fast_kron function."""

    def test_fast_kron_two_pauli_x(self):
        """Test fast_kron with two X Pauli operators."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        state = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
        result = fast_kron([X, X], state)

        # X ⊗ X |00⟩ = |11⟩
        expected = np.array([0, 0, 0, 1], dtype=complex)
        assert np.allclose(result, expected)

    def test_fast_kron_identity(self):
        """Test fast_kron with identity operators."""
        I = np.eye(2, dtype=complex)
        state = np.array([1, 0, 0, 0], dtype=complex)
        result = fast_kron([I, I], state)

        assert np.allclose(result, state)

    def test_fast_kron_dimension(self):
        """Test that fast_kron output has correct dimension."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        state = np.random.randn(8) + 1j * np.random.randn(8)
        result = fast_kron([X, X, X], state)

        assert len(result) == 8


class TestPauliExpvalFastKron:
    """Tests for pauli_expval_fast_kron function."""

    def test_pauli_expval_identity_state(self):
        """Test expectation value for identity operator on |0⟩."""
        state = np.array([1, 0], dtype=complex)  # |0⟩
        label = [0]  # Identity on single qubit
        expval = pauli_expval_fast_kron(state, label, dim=2)

        assert np.isclose(expval, 1.0)

    def test_pauli_expval_two_qubit_identity(self):
        """Test expectation value for identity on two-qubit state."""
        state = np.array([1, 0, 0, 0], dtype=complex) / np.sqrt(1)  # |00⟩
        label = [0, 0]  # Identity ⊗ Identity
        expval = pauli_expval_fast_kron(state, label, dim=2)

        assert np.isclose(expval, 1.0)

    def test_pauli_expval_real_expectation(self):
        """Test that expectation values are real for Hermitian operators."""
        # Create a random quantum state
        state = np.random.randn(4) + 1j * np.random.randn(4)
        state = state / np.linalg.norm(state)

        label = [1, 2]  # Some arbitrary Pauli operators
        expval = pauli_expval_fast_kron(state, label, dim=2)

        # Expectation value of Hermitian operator should be real
        assert np.abs(expval.imag) < 1e-10

    def test_pauli_expval_bounds(self):
        """Test that expectation values are bounded by 1."""
        state = np.random.randn(4) + 1j * np.random.randn(4)
        state = state / np.linalg.norm(state)

        label = [1, 2]
        expval = pauli_expval_fast_kron(state, label, dim=2)

        assert np.abs(expval) <= 1.0 + 1e-10  # Small tolerance for numerical errors


class TestComputeFunction:
    """Tests for the compute function."""

    def test_compute_returns_property_result(self):
        """Test that compute returns a PropertyResult object."""
        state_vector = np.array([1, 0], dtype=complex)  # |0⟩
        state = DenseState(vector=state_vector, n_qubits=1, d=2, backend="numpy")

        result = compute(state)

        assert result.name == "SRE"
        assert result.value is not None
        assert isinstance(result.value, (float, np.floating))

    def test_compute_meta_information(self):
        """Test that compute includes proper meta information."""
        state_vector = np.array([1, 0], dtype=complex)  # |0⟩
        state = DenseState(vector=state_vector, n_qubits=1, d=2, backend="numpy")

        result = compute(state)

        assert "method" in result.meta
        assert result.meta["method"] == "exact_dense"
        assert "n_qubits" in result.meta
        assert result.meta["n_qubits"] == 1

    def test_compute_product_state(self):
        """Test compute on a simple product state |00⟩."""
        # |00⟩ is a pure product state
        state_vector = np.array([1, 0, 0, 0], dtype=complex)
        state = DenseState(vector=state_vector, n_qubits=2, d=2, backend="numpy")

        result = compute(state)

        # SRE should be well-defined and finite
        assert np.isfinite(result.value)
        assert result.value >= 0  # SRE is non-negative

    def test_compute_bell_state(self):
        """Test compute on a maximally entangled Bell state |Φ+⟩."""
        # |Φ+⟩ = (|00⟩ + |11⟩) / sqrt(2)
        state_vector = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        state = DenseState(vector=state_vector, n_qubits=2, d=2, backend="numpy")

        result = compute(state)

        assert np.isfinite(result.value)
        assert result.value >= 0

    def test_compute_maximally_mixed_state(self):
        """Test compute on a maximally mixed state (uniform superposition)."""
        # |++++...⟩ is a uniform superposition
        state_vector = np.ones(4, dtype=complex) / 2  # For 2 qubits
        state = DenseState(vector=state_vector, n_qubits=2, d=2, backend="numpy")

        result = compute(state)

        assert np.isfinite(result.value)
        assert result.value >= 0

    def test_compute_different_qubit_numbers(self):
        """Test compute on different numbers of qubits."""
        for n_qubits in [1, 2, 3]:
            dim = 2**n_qubits
            state_vector = np.zeros(dim, dtype=complex)
            state_vector[0] = 1  # |0...0⟩
            state = DenseState(vector=state_vector, n_qubits=n_qubits, d=2, backend="numpy")

            result = compute(state)

            assert np.isfinite(result.value)
            assert result.meta["n_qubits"] == n_qubits

    def test_compute_qutrit_system(self):
        """Test compute on a qutrit (3-level) system."""
        state_vector = np.zeros(3, dtype=complex)
        state_vector[0] = 1  # |0⟩
        state = DenseState(vector=state_vector, n_qubits=1, d=3, backend="numpy")

        result = compute(state)

        assert np.isfinite(result.value)
        assert result.value >= 0

    def test_compute_normalized_state(self):
        """Test compute requires normalized input states."""
        # Create an unnormalized state
        state_vector = np.array([1, 1], dtype=complex)
        state = DenseState(vector=state_vector, n_qubits=1, d=2, backend="quimb")

        result = compute(state)

        # Should still work, but result depends on normalization
        assert np.isfinite(result.value)

    def test_compute_deterministic(self):
        """Test that compute is deterministic (same input gives same output)."""
        state_vector = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        state = DenseState(vector=state_vector, n_qubits=2, d=2, backend="quimb")

        result1 = compute(state)
        result2 = compute(state)

        assert np.isclose(result1.value, result2.value)

    def test_compute_no_side_effects(self):
        """Test that compute doesn't modify input state."""
        state_vector = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        state_vector_original = state_vector.copy()
        state = DenseState(vector=state_vector, n_qubits=2, d=2, backend="quimb")

        compute(state)

        assert np.allclose(state.vector, state_vector_original)

    def test_compute_single_qubit_pure_state(self):
        """Test compute on single qubit pure states."""
        # Test |0⟩
        state_0 = DenseState(
            vector=np.array([1, 0], dtype=complex),
            n_qubits=1,
            d=2,
            backend="quimb",
        )
        result_0 = compute(state_0)
        assert np.isfinite(result_0.value)

        # Test |+⟩ = (|0⟩ + |1⟩) / sqrt(2)
        state_plus = DenseState(
            vector=np.array([1, 1], dtype=complex) / np.sqrt(2),
            n_qubits=1,
            d=2,
            backend="quimb",
        )
        result_plus = compute(state_plus)
        assert np.isfinite(result_plus.value)

        # Both should give valid results
        assert result_0.value >= 0
        assert result_plus.value >= 0

    def test_compute_random_state(self):
        """Test compute on random quantum states."""
        for _ in range(5):
            # Generate random quantum state
            state_vector = np.random.randn(4) + 1j * np.random.randn(4)
            state_vector = state_vector / np.linalg.norm(state_vector)

            state = DenseState(vector=state_vector, n_qubits=2, d=2, backend="quimb")
            result = compute(state)

            assert np.isfinite(result.value)
            assert result.value >= 0
