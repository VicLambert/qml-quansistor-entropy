from src.circuit.families.clifford import CliffordBrickwork
from src.circuit.families.haar import HaarBrickwork
from src.circuit.families.quansistor import QuansistorBrickwork

FAMILY_REGISTRY = {
    "haar": HaarBrickwork(),
    "clifford": CliffordBrickwork(),
    "quansistor": QuansistorBrickwork(),
}
