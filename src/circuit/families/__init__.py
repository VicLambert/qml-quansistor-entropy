"""Circuit family definitions and registry.

This package provides different families of quantum circuits including
Haar random, Clifford, and Quansistor brickwork patterns.
"""
from src.circuit.families.clifford import CliffordBrickwork
from src.circuit.families.haar import HaarBrickwork
from src.circuit.families.quansistor import QuansistorBrickwork

FAMILY_REGISTRY = {
    "haar": HaarBrickwork(),
    "clifford": CliffordBrickwork(),
    "quansistor": QuansistorBrickwork(),
}
