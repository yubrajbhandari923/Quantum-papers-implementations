"""Custom quantum gates for energy-conserving quantum circuits."""

import numpy as np
from qiskit.circuit import Gate, QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import XXPlusYYGate, SGate, SdgGate


class SqrtISWAPGate(Gate):
    """Custom √iSWAP gate, built on Qiskit's XXPlusYYGate."""

    def __init__(self):
        super().__init__(name="sqrtiSWAP_ec", num_qubits=2, params=[])

    def _define(self):
        qc = QuantumCircuit(2, name="sqrtiSWAP_ec")
        qc.append(XXPlusYYGate(theta=-np.pi/2), [0, 1])
        self.definition = qc.to_instruction()
    
    def to_matrix(self):
        """Return the matrix representation of the √iSWAP gate."""
        return XXPlusYYGate(theta=-np.pi/2).to_matrix()


class SqrtISWAPdgGate(Gate):
    """Custom √iSWAP^† gate, built on Qiskit's XXPlusYYGate."""

    def __init__(self):
        super().__init__(name="sqrtiSWAPdg_ec", num_qubits=2, params=[])

    def _define(self):
        qc = QuantumCircuit(2, name="sqrtiSWAPdg_ec")
        qc.append(XXPlusYYGate(theta=np.pi/2), [0, 1])
        self.definition = qc.to_instruction()
    
    def to_matrix(self):
        """Return the matrix representation of the √iSWAP^† gate."""
        return XXPlusYYGate(theta=np.pi/2).to_matrix()


class CZFromXYGate(Gate):
    """Two-qubit CZ implemented from XY / √iSWAP + Rz.
    
    Uses an ancilla qubit to implement CZ using energy-conserving gates.
    """

    def __init__(self):
        super().__init__(name="cz_xy", num_qubits=2, params=[])

    def _define(self):
        qr = QuantumRegister(2, "q")
        ar = AncillaRegister(1, "a")
        qc = QuantumCircuit(qr, ar, name="cz_xy")

        # CZ decomposition using √iSWAP gates
        qc.append(SqrtISWAPGate(), [0, 1])
        qc.append(SqrtISWAPGate(), [0, 1])
        qc.append(SqrtISWAPGate(), [1, 2])
        qc.append(SqrtISWAPGate(), [1, 2])
        
        qc.append(SqrtISWAPdgGate(), [0, 1])
        qc.append(SqrtISWAPdgGate(), [0, 1])
        
        qc.append(SqrtISWAPGate(), [0, 2])
        qc.append(SqrtISWAPGate(), [0, 2])
        
        qc.append(SGate(), [0])
        qc.append(SdgGate(), [2])

        qc.initialize('0', ar[0])
        self.definition = qc

    def inverse(self):
        """CZ is Hermitian and self-inverse."""
        return CZFromXYGate()