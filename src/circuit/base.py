import numpy as np
import pennylane as qml
import quimb as qb
import quimb.tensor as qtn

from abc import ABC, abstractmethod

class BaseCircuit(ABC):
    def __init__(self, n_qubits, num_layers, seed, d=2):
        self.n_qubits = n_qubits
        self.num_layers = num_layers
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.phys_dim = d**self.n_qubits


    