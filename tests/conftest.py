"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

import pytest

# Add src directory to path so imports work correctly
src_path = Path(__file__).parent.parent.absolute()
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_gate_params():
    """Fixture providing sample gate parameters."""
    return {
        "kind": "haar",
        "wires": (0, 1),
        "d": 2,
        "seed": 123,
        "tags": ("layer", "L0"),
    }


@pytest.fixture
def sample_circuit_params():
    """Fixture providing sample circuit parameters."""
    return {
        "n_qubits": 6,
        "n_layers": 3,
        "d": 2,
        "family": "haar",
        "topology": "line",
        "global_seed": 123,
    }


@pytest.fixture
def small_circuit_config():
    """Fixture for small test circuit configuration."""
    return {
        "n_qubits": 4,
        "n_layers": 2,
        "d": 2,
        "seed": 42,
        "topology": "line",
    }


@pytest.fixture
def medium_circuit_config():
    """Fixture for medium test circuit configuration."""
    return {
        "n_qubits": 8,
        "n_layers": 3,
        "d": 2,
        "seed": 123,
        "topology": "line",
    }


@pytest.fixture
def large_circuit_config():
    """Fixture for large test circuit configuration."""
    return {
        "n_qubits": 16,
        "n_layers": 5,
        "d": 2,
        "seed": 999,
        "topology": "line",
    }
