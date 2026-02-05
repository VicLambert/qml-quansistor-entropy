

from __future__ import annotations

from qqe.backend.pennylane import PennylaneBackend
from qqe.circuit.families.clifford import CliffordBrickwork
from qqe.circuit.families.haar import HaarBrickwork
from qqe.circuit.families.quansistor import QuansistorBrickwork
from qqe.properties.SRE.fwht_sre import compute as sre_fwht
from qqe.properties.SRE.sre_exact_dense import compute as sre_exact


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