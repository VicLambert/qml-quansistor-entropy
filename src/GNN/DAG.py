from __future__ import annotations

import numpy as np
import pennylane as qml

from circuit.families.gates import circ_X, circ_Y

_BLOCK_STEPS = (
    (0, 1),  # q0 q1
    (2, 3),  # q2 q3
    (1, 2),  # q1 q2
    (0, 1),  # q0 q1
    (2, 3),  # q2 q3
)


def quansistor_block(
    params: tuple[tuple[float, float, float], ...],
    wires: list[int],
) -> qml.tape.QuantumScript:
    """Create a Quansistor block as a PennyLane QuantumScript.

    Parameters
    ----------
    params : tuple(tuple(float, float, float), ...)
        Per-step parameters (a, b, g) for the 5 two-qubit sub-blocks.
    wires : list(int)
        Four wires on which to apply the block.

    Returns:
    -------
    qml.tape.QuantumScript
        The PennyLane QuantumScript representing the Quansistor block.
    """
    rng = np.random.default_rng()
    wires4 = (wires[0], wires[1], wires[2], wires[3])
    for step_idx, (i, j) in enumerate(_BLOCK_STEPS):
        a_wire, b_wire = wires4[i], wires4[j]
        a, b, g = params[step_idx]
        U = circ_X(a, b, g) if rng.choice(["X", "Y"]) == "X" else circ_Y(a, b, g)
        qml.QubitUnitary(U, wires=[a_wire, b_wire])


@qml.qnode(qml.device("lightning.qubit", wires=8))
def circuit(params):
    qml.TTN(
        wires=range(8),
        n_block_wires=4,
        block=quansistor_block(params, [0, 1, 2, 3]),
        n_params_block=3,
    )
    return qml.state()

rng = np.random.default_rng()
params = (tuple(rng.standard_normal() for _ in range(3)),)
print(params)
circuit(params)
