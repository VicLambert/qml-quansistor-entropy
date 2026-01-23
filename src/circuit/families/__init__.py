from .haar import HaarBrickwork
from .clifford import CliffordBrickwork
from .quansistor import QuansistorBrickwork

FAMILY_REGISTRY = {
    "haar": HaarBrickwork(),
    "clifford": CliffordBrickwork(),
    "quansistor": QuansistorBrickwork(),
}