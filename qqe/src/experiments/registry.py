

from __future__ import annotations

from src.backend.pennylane_backend import PennylaneBackend
from src.circuit.families import CliffordBrickwork, HaarBrickwork, QuansistorBrickwork
from src.properties.SRE import sre_fwht
from src.properties.SRE import sre_exact


def default_backend_registry():
    return {
        "pennylane": PennylaneBackend,
    }

def default_family_registry():
    return {
        "clifford": CliffordBrickwork,
        "haar": HaarBrickwork,
        "quansistor": QuansistorBrickwork,
    }

def default_property_registry():
    return {
        "SRE_FWHT": sre_fwht,
        "SRE_exact": sre_exact,
    }
