from .clifford import CliffordBrickwork
from .haar import HaarBrickwork
from .quansistor import QuansistorBrickwork

FAMILY_REGISTRY = {
    "haar": HaarBrickwork(),
    "clifford": CliffordBrickwork(),
    "quansistor": QuansistorBrickwork(),
}
