import pytest

from rng.seeds import gate_seed


def test_registry_smoke():
    from circuit.families import FAMILY_REGISTRY
    for name, fam in FAMILY_REGISTRY.items():
        spec = fam.make_spec(n_qubits=6, n_layers=3, d=2, seed=123, topology="line")
        gates = list(fam.gates(spec))
        assert len(gates) > 0

def assert_gate_wires_valid(spec, gate):
    assert all(0 <= w < spec.n_qubits for w in gate.wires)
    assert len(set(gate.wires)) == len(gate.wires)  # no duplicates

def test_deterministic_gate_stream_haar():
    from circuit.families.haar import HaarBrickwork
    fam = HaarBrickwork()
    spec1 = fam.make_spec(8, 5, 2, seed=999, topology="line")
    spec2 = fam.make_spec(8, 5, 2, seed=999, topology="line")

    g1 = list(fam.gates(spec1))
    g2 = list(fam.gates(spec2))

    assert g1 == g2

def test_brickwork_line_pairs_even_n():
    from circuit.families.pattern.brickwork import brickwork_pattern
    n=8
    assert brickwork_pattern(n, layer=0, topology="line") == [(0,1),(2,3),(4,5),(6,7)]
    assert brickwork_pattern(n, layer=1, topology="line") == [(1,2),(3,4),(5,6)]

def test_seed_is_deterministic():
    s1 = gate_seed(123, kind="haar2", layer=5, slot=2, wires=(1, 9))
    s2 = gate_seed(123, kind="haar2", layer=5, slot=2, wires=(1, 9))
    assert s1 == s2

def test_seed_changes_with_kind():
    a = gate_seed(123, kind="haar2", layer=1, slot=0, wires=(0, 1))
    b = gate_seed(123, kind="clifford2", layer=1, slot=0, wires=(0, 1))
    assert a != b

def test_seed_changes_with_wires():
    a = gate_seed(123, kind="haar2", layer=1, slot=0, wires=(0, 1))
    b = gate_seed(123, kind="haar2", layer=1, slot=0, wires=(0, 2))
    assert a != b

def test_wire_sorting_default():
    a = gate_seed(123, kind="haar2", layer=1, slot=0, wires=(0, 2))
    b = gate_seed(123, kind="haar2", layer=1, slot=0, wires=(2, 0))
    assert a == b

def test_ordered_wires_option():
    a = gate_seed(123, kind="cx_like", layer=1, slot=0, wires=(0, 2), ordered_wires=True)
    b = gate_seed(123, kind="cx_like", layer=1, slot=0, wires=(2, 0), ordered_wires=True)
    assert a != b

if __name__ == "__main__":
    test_seed_is_deterministic()
    test_seed_changes_with_kind()
    test_seed_changes_with_wires()
    test_wire_sorting_default()
    test_ordered_wires_option()