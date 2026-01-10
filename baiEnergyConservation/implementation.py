"""Energy-conserving unitary decomposition for quantum circuits.

This module provides tools for decomposing energy-conserving (Hamming weight preserving)
unitaries into basis gates that also conserve energy. The implementation follows the
construction from arXiv:2309.11051.

Classes:
    TwoLevelUnitary: Represents a 2-level unitary matrix.
    ControlledTwoLevel: Represents a controlled 2-level unitary.
    GateSynthesizer: Synthesizes complex gates from energy-conserving primitives.
    TwoQubitDecomposer: Decomposes 2-qubit energy-conserving unitaries.
    ThreeQubitDecomposer: Decomposes 3-qubit energy-conserving unitaries.
    NQubitDecomposer: Decomposes n-qubit energy-conserving unitaries.
    EnergyConservingDecompositionPass: Qiskit transpiler pass for decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from qiskit.circuit import AncillaRegister, QuantumCircuit, QuantumRegister, Gate
from qiskit.circuit.library import UnitaryGate, SGate, SdgGate, XXPlusYYGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Operator
from qiskit.synthesis import OneQubitEulerDecomposer
from qiskit.transpiler import TransformationPass
from qiskit.transpiler.target import Target


# Type aliases
ComplexMatrix = NDArray[np.complexfloating]
RealVector = NDArray[np.floating]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TwoLevelUnitary:
    """A 2-level unitary acting on a d-dimensional Hilbert space.
    
    The unitary acts non-trivially only on the subspace spanned by
    basis states |i⟩ and |j⟩.
    
    Attributes:
        dim: Dimension of the full Hilbert space.
        i: First basis state index.
        j: Second basis state index.
        submatrix: 2x2 unitary matrix acting on the {|i⟩, |j⟩} subspace.
    """
    dim: int
    i: int
    j: int
    submatrix: ComplexMatrix = field(default_factory=lambda: np.eye(2, dtype=complex))
    
    def to_full_matrix(self) -> ComplexMatrix:
        """Convert to full d×d matrix representation."""
        full = np.eye(self.dim, dtype=complex)
        full[self.i, self.i] = self.submatrix[0, 0]
        full[self.i, self.j] = self.submatrix[0, 1]
        full[self.j, self.i] = self.submatrix[1, 0]
        full[self.j, self.j] = self.submatrix[1, 1]
        return full
    
    def dagger(self) -> TwoLevelUnitary:
        """Return the Hermitian conjugate."""
        return TwoLevelUnitary(
            dim=self.dim,
            i=self.i,
            j=self.j,
            submatrix=self.submatrix.conj().T.copy()
        )


@dataclass
class ControlledTwoLevel:
    """A controlled 2-level unitary.
    
    Attributes:
        U: 2x2 unitary matrix to apply.
        control_bits: List of control qubit bit positions.
        control_vals: List of control values (0 or 1) for each control.
        target_bits: List of target qubit bit positions (exactly 2).
    """
    U: ComplexMatrix
    control_bits: List[int]
    control_vals: List[int]
    target_bits: List[int]


# =============================================================================
# region Custom Gates
# =============================================================================


class SqrtISWAPGate(Gate):
    """Custom √iSWAP gate, built on Qiskit's XXPlusYYGate."""

    def __init__(self):
        super().__init__(name="sqrtiSWAP_ec", num_qubits=2, params=[])
        self.label = r"$\sqrt{\text{iSWAP}}$"

    def _define(self):
        qc = QuantumCircuit(2, name="sqrtiSWAP_ec")
        qc.append(XXPlusYYGate(theta=-np.pi/2), [0, 1])
        self.definition = qc
    
    def to_matrix(self):
        """Return the matrix representation of the √iSWAP gate."""
        return XXPlusYYGate(theta=-np.pi/2).to_matrix()


class SqrtISWAPdgGate(Gate):
    """Custom √iSWAP^† gate, built on Qiskit's XXPlusYYGate."""

    def __init__(self):
        super().__init__(name="sqrtiSWAPdg_ec", num_qubits=2, params=[])
        self.label = r"$\sqrt{\text{iSWAP}}^\dagger$"  

    def _define(self):
        qc = QuantumCircuit(2, name="sqrtiSWAPdg_ec")
        qc.append(XXPlusYYGate(theta=np.pi/2), [0, 1])
        self.definition = qc
    
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


# class SqrtISWAPGate(UnitaryGate):
#     """The √iSWAP gate."""
    
#     def __init__(self):
#         matrix = np.array([
#             [1, 0, 0, 0],
#             [0, 1/np.sqrt(2), 1j/np.sqrt(2), 0],
#             [0, 1j/np.sqrt(2), 1/np.sqrt(2), 0],
#             [0, 0, 0, 1]
#         ], dtype=complex)
#         super().__init__(matrix, "√iSWAP")


# class SqrtISWAPdgGate(UnitaryGate):
#     """The √iSWAP† gate."""
    
#     def __init__(self):
#         matrix = np.array([
#             [1, 0, 0, 0],
#             [0, 1/np.sqrt(2), -1j/np.sqrt(2), 0],
#             [0, -1j/np.sqrt(2), 1/np.sqrt(2), 0],
#             [0, 0, 0, 1]
#         ], dtype=complex)
#         super().__init__(matrix, "√iSWAP†")


# class SGate(UnitaryGate):
#     """The S (phase) gate."""
    
#     def __init__(self):
#         matrix = np.array([[1, 0], [0, 1j]], dtype=complex)
#         super().__init__(matrix, "S")


# class SdgGate(UnitaryGate):
#     """The S† gate."""
    
#     def __init__(self):
#         matrix = np.array([[1, 0], [0, -1j]], dtype=complex)
#         super().__init__(matrix, "S†")

# endregion


# =============================================================================
# region Utility Functions
# =============================================================================

def is_energy_conserving(U: ComplexMatrix, tol: float = 1e-10) -> bool:
    """Check if a unitary preserves Hamming weight (is energy-conserving).
    
    Args:
        U: Unitary matrix to check.
        tol: Tolerance for numerical comparisons.
        
    Returns:
        True if U preserves Hamming weight, False otherwise.
    """
    dim = U.shape[0]
    n_qubits = int(np.log2(dim))
    
    if 2**n_qubits != dim:
        return False
    
    for i in range(dim):
        hw_i = bin(i).count('1')
        for j in range(dim):
            hw_j = bin(j).count('1')
            if hw_i != hw_j and np.abs(U[i, j]) > tol:
                return False
    return True


def get_hamming_weight_blocks(U: ComplexMatrix) -> Dict[int, ComplexMatrix]:
    """Extract block-diagonal structure by Hamming weight.
    
    Args:
        U: Energy-conserving unitary matrix.
        
    Returns:
        Dictionary mapping Hamming weight to the corresponding block matrix.
    """
    dim = U.shape[0]
    n_qubits = int(np.log2(dim))
    
    hw_to_indices: Dict[int, List[int]] = {}
    for i in range(dim):
        hw = bin(i).count('1')
        hw_to_indices.setdefault(hw, []).append(i)
    
    blocks = {}
    for hw, indices in hw_to_indices.items():
        block_size = len(indices)
        block = np.zeros((block_size, block_size), dtype=complex)
        for ii, idx_i in enumerate(indices):
            for jj, idx_j in enumerate(indices):
                block[ii, jj] = U[idx_i, idx_j]
        blocks[hw] = block
    
    return blocks


def get_nontrivial_rows_cols(U: ComplexMatrix, tol: float = 1e-10) -> Tuple[int, int]:
    """Find the two indices where a 2-level unitary acts non-trivially.
    
    Args:
        U: 2-level unitary matrix.
        tol: Tolerance for numerical comparisons.
        
    Returns:
        Tuple (i, j) of the two non-trivial indices.
        
    Raises:
        ValueError: If U is not a valid 2-level unitary.
    """
    dim = U.shape[0]
    non_identity_indices = []
    
    for i in range(dim):
        if not np.isclose(U[i, i], 1.0, atol=tol):
            non_identity_indices.append(i)
        else:
            for j in range(dim):
                if i != j and np.abs(U[i, j]) > tol:
                    if i not in non_identity_indices:
                        non_identity_indices.append(i)
                    break
    
    if len(non_identity_indices) != 2:
        # Try finding by off-diagonal elements
        for i in range(dim):
            for j in range(i + 1, dim):
                if np.abs(U[i, j]) > tol or np.abs(U[j, i]) > tol:
                    return i, j
        raise ValueError(f"Could not identify 2-level structure. Found indices: {non_identity_indices}")
    
    return non_identity_indices[0], non_identity_indices[1]


def extract_two_level_blocks(
    U: ComplexMatrix,
    atol: float = 1e-10,
    enforce_hamming_distance: Optional[int] = None
) -> List[TwoLevelUnitary]:
    """Extract 2-level unitaries from a unitary matrix using Reck-like decomposition.
    
    Args:
        U: Unitary matrix to decompose.
        atol: Absolute tolerance.
        enforce_hamming_distance: If set, only extract blocks with this Hamming distance.
        
    Returns:
        List of TwoLevelUnitary objects, or empty list if decomposition fails.
    """
    dim = U.shape[0]
    if dim <= 2:
        return []
    
    # Simple case: check if already close to identity
    if np.allclose(U, np.eye(dim), atol=atol):
        return []
    
    return []  # Fall back to SUD decomposition


# endregion
# =============================================================================
# region Core Mathematical Functions
# =============================================================================

def v_su2(a: complex, b: complex) -> ComplexMatrix:
    """Construct V(a,b) ∈ SU(2) such that V(a,b) @ [a, b]^T = [√(|a|²+|b|²), 0]^T.
    
    From Eq. (D1) of arXiv:2309.11051:
        V(a,b) = (|a|² + |b|²)^{-1/2} * [[a*, b*], [-b, a]]
    
    Args:
        a: First component.
        b: Second component.
        
    Returns:
        2x2 SU(2) matrix.
    """
    norm_sq = np.abs(a)**2 + np.abs(b)**2
    
    if norm_sq < 1e-14:
        return np.eye(2, dtype=complex)
    
    norm = np.sqrt(norm_sq)
    return np.array([
        [np.conj(a), np.conj(b)],
        [-b, a]
    ], dtype=complex) / norm


def su2_ab_from_u(U: ComplexMatrix, atol: float = 1e-10) -> Tuple[ComplexMatrix, ComplexMatrix]:
    """Find A, B ∈ SU(2) such that A B A† B† = U.
    
    This construction follows Eq. (64) from arXiv:2309.11051.
    
    Args:
        U: 2x2 unitary matrix.
        atol: Absolute tolerance.
        
    Returns:
        Tuple (A, B) of SU(2) matrices.
        
    Raises:
        ValueError: If U is not a valid 2x2 unitary.
    """
    if U.shape != (2, 2):
        raise ValueError("U must be 2x2")
    
    det_u = np.linalg.det(U)
    if not np.isclose(abs(det_u), 1.0, atol=atol):
        raise ValueError(f"Determinant of U must have magnitude 1, got {det_u}")
    
    # Special case: U = -I
    if np.allclose(U, -np.eye(2), atol=atol):
        A = 1j * np.array([[0, 1], [1, 0]], dtype=complex)  # i * Pauli-X
        B = 1j * np.array([[0, -1j], [1j, 0]], dtype=complex)  # i * Pauli-Y
        return A, B
    
    # General case via eigendecomposition
    evals, evecs = np.linalg.eig(U)
    phi1, phi2 = np.angle(evals[0]), np.angle(evals[1])
    theta = 0.5 * (phi1 - phi2)
    
    D = np.diag([np.exp(1j * theta), np.exp(-1j * theta)])
    W = evecs
    
    # Verify and potentially fix ordering
    if not np.allclose(W @ D @ W.conj().T, U, atol=1e-6):
        W = W[:, ::-1]
        if not np.allclose(W @ D @ W.conj().T, U, atol=1e-6):
            raise ValueError("Eigendecomposition failed to reconstruct U")
    
    # Normalize W to ensure det(A) = det(B) = 1
    det_W = np.linalg.det(W)
    W = W / np.sqrt(det_W)
    
    exp_itheta_z_half = np.diag(np.exp(1j * theta * np.array([0.5, -0.5])))
    A = W @ exp_itheta_z_half @ W.conj().T
    
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    B = 1j * W @ X @ W.conj().T
    
    return A, B


def sud_decompose(U: ComplexMatrix, tol: float = 1e-12) -> List[TwoLevelUnitary]:
    """Decompose SU(d) matrix into product of 2-level unitaries.
    
    Uses the standard recursive algorithm that zeros out the first column.
    
    Args:
        U: Special unitary matrix (det = 1).
        tol: Tolerance for numerical comparisons.
        
    Returns:
        List of TwoLevelUnitary objects whose product equals U.
        
    Raises:
        ValueError: If U is not special unitary.
        RuntimeError: If decomposition fails.
    """
    d = U.shape[0]
    
    # Validate input
    if U.shape != (d, d):
        raise ValueError("Input must be a square matrix")
    if not np.allclose(U @ U.conj().T, np.eye(d), atol=tol):
        raise ValueError("Input matrix is not unitary")
    if not np.isclose(np.linalg.det(U), 1.0, atol=tol):
        raise ValueError(f"Input matrix is not special unitary (det = {np.linalg.det(U)})")
    
    # Identity case
    if np.allclose(U, np.eye(d), atol=tol):
        return []
    
    # Base case: d = 2
    if d == 2:
        return [TwoLevelUnitary(dim=2, i=0, j=1, submatrix=U.copy())]
    
    # Recursive case
    W = U.copy().astype(complex)
    v_gates: List[TwoLevelUnitary] = []
    
    # Zero out first column below diagonal
    for k in range(1, d):
        a, b = W[0, 0], W[k, 0]
        V_2x2 = v_su2(a, b)
        V_gate = TwoLevelUnitary(dim=d, i=0, j=k, submatrix=V_2x2)
        v_gates.append(V_gate)
        
        # Apply V to W
        row_0, row_k = W[0, :].copy(), W[k, :].copy()
        W[0, :] = V_2x2[0, 0] * row_0 + V_2x2[0, 1] * row_k
        W[k, :] = V_2x2[1, 0] * row_0 + V_2x2[1, 1] * row_k
    
    # Verify reduction
    if not np.isclose(np.abs(W[0, 0]), 1.0, atol=tol):
        raise RuntimeError(f"Failed to reduce: |W[0,0]| = {np.abs(W[0, 0])}")
    if not np.allclose(W[1:, 0], 0, atol=tol):
        raise RuntimeError("Failed to zero first column")
    
    # Recursively decompose the (d-1)×(d-1) block
    W_sub = W[1:, 1:].copy()
    sub_gates = sud_decompose(W_sub, tol=tol)
    
    # Embed sub_gates into d-dimensional space
    embedded_sub_gates = [
        TwoLevelUnitary(dim=d, i=g.i + 1, j=g.j + 1, submatrix=g.submatrix.copy())
        for g in sub_gates
        if not np.allclose(g.submatrix, np.eye(2), atol=tol)
    ]
    
    # Build result: U = V_1† @ V_2† @ ... @ V_{d-1}† @ (1 ⊕ W_sub)
    result_gates = [V_gate.dagger() for V_gate in v_gates]
    result_gates.extend(embedded_sub_gates)
    
    return result_gates


def construct_conjugation_gates(
    b: int, 
    b_prime: int, 
    n_qubits: int
) -> Tuple[List[Tuple[int, int, int]], int]:
    """Construct controlled-iSWAP sequence to reduce Hamming distance to 2.
    
    Following Lemma 10 of arXiv:2309.11051, constructs gates K such that
    K maps (b, b') to (b'', b') where d(b'', b') = 2.
    
    Args:
        b: First basis state index.
        b_prime: Second basis state index.
        n_qubits: Number of qubits.
        
    Returns:
        Tuple of (gates_list, b_double_prime) where:
        - gates_list: List of (control_bit, target1, target2) tuples.
        - b_double_prime: Final state index with d(b'', b') = 2.
        
    Raises:
        ValueError: If Hamming distance is odd or partition fails.
    """
    # Find differing bit positions
    diff_positions = [
        bit_idx for bit_idx in range(n_qubits)
        if ((b >> bit_idx) & 1) != ((b_prime >> bit_idx) & 1)
    ]
    
    hamming_dist = len(diff_positions)
    if hamming_dist % 2 != 0:
        raise ValueError("Hamming distance must be even for equal Hamming weight states")
    
    t = hamming_dist // 2
    if t <= 1:
        return [], b
    
    # Partition: l_positions (1 in b, 0 in b'), r_positions (0 in b, 1 in b')
    l_positions = [i for i in diff_positions if ((b >> i) & 1) == 1]
    r_positions = [i for i in diff_positions if ((b >> i) & 1) == 0]
    
    if len(l_positions) != t or len(r_positions) != t:
        raise ValueError(f"Partition error: {len(l_positions)} l's and {len(r_positions)} r's, expected {t}")
    
    gates = []
    current_state = b
    
    for j in range(t - 1):
        lj, rj = l_positions[j], r_positions[j]
        control_bit = l_positions[j + 1]
        
        gates.append((control_bit, lj, rj))
        
        # Swap bits at lj and rj
        bit_lj = (current_state >> lj) & 1
        bit_rj = (current_state >> rj) & 1
        if bit_lj != bit_rj:
            current_state &= ~(1 << lj)
            current_state &= ~(1 << rj)
            current_state |= (bit_rj << lj)
            current_state |= (bit_lj << rj)
    
    return gates, current_state

# endregion


class GateSynthesizer:
    """Synthesizes complex gates from energy-conserving primitives."""
    
    @staticmethod
    def exp_i_alpha_r(qc: QuantumCircuit, theta: float, q0: int, q1: int) -> None:
        """Apply exp(iαR) rotation where R generates XY-plane rotations.
        
        Args:
            qc: Quantum circuit.
            theta: Rotation angle.
            q0, q1: Qubit indices.
        """
        qc.append(SGate(), [q1])
        qc.append(SqrtISWAPGate(), [q0, q1])
        qc.rz(-theta, q0)
        qc.rz(theta, q1)
        qc.append(SqrtISWAPdgGate(), [q0, q1])
        qc.append(SdgGate(), [q1])
    
    @staticmethod
    def controlled_exp_i_alpha_r(
        qc: QuantumCircuit, 
        theta: float, 
        b: int, 
        q0: int, 
        q1: int, 
        control: int
    ) -> None:
        """Apply controlled exp(iαR) rotation.
        
        Args:
            qc: Quantum circuit.
            theta: Rotation angle.
            b: Control bit value (0 or 1).
            q0, q1: Target qubit indices.
            control: Control qubit index.
        """
        q2 = control
        
        qc.append(SqrtISWAPGate(), [q1, q2])
        qc.append(SqrtISWAPGate(), [q1, q2])
        qc.append(SqrtISWAPdgGate(), [q0, q1])
        qc.append(SqrtISWAPdgGate(), [q0, q1])
        qc.append(SqrtISWAPGate(), [q0, q2])
        qc.append(SqrtISWAPGate(), [q0, q2])
        
        qc.append(SGate(), [q0])
        GateSynthesizer.exp_i_alpha_r(qc, ((-1)**b) * (theta/2), q0, q1)
        qc.append(SdgGate(), [q0])
        
        qc.append(SqrtISWAPdgGate(), [q0, q2])
        qc.append(SqrtISWAPdgGate(), [q0, q2])
        qc.append(SqrtISWAPGate(), [q0, q1])
        qc.append(SqrtISWAPGate(), [q0, q1])
        qc.append(SqrtISWAPdgGate(), [q1, q2])
        qc.append(SqrtISWAPdgGate(), [q1, q2])
        
        GateSynthesizer.exp_i_alpha_r(qc, theta/2, q0, q1)
    
    @staticmethod
    def controlled_exp_i_theta_l(
        qc: QuantumCircuit, 
        theta: float, 
        b: int, 
        q0: int, 
        q1: int, 
        control: int
    ) -> None:
        """Apply controlled exp(iθL) rotation.
        
        Args:
            qc: Quantum circuit.
            theta: Rotation angle.
            b: Control bit value (0 or 1).
            q0, q1: Target qubit indices.
            control: Control qubit index.
        """
        qc.append(SGate(), [q0])
        GateSynthesizer.controlled_exp_i_alpha_r(qc, theta, b, q0, q1, control)
        qc.append(SdgGate(), [q0])
    
    @staticmethod
    def apply_cz_decomposition(
        qc: QuantumCircuit, 
        q0: int, 
        q1: int, 
        ancilla: int
    ) -> None:
        """Apply CZ gate using √iSWAP decomposition with ancilla.
        
        Args:
            qc: Quantum circuit.
            q0, q1: Logical qubit indices.
            ancilla: Ancilla qubit index (assumed |0⟩).
        """
        qc.append(SqrtISWAPGate(), [q0, q1])
        qc.append(SqrtISWAPGate(), [q0, q1])
        qc.append(SqrtISWAPGate(), [q1, ancilla])
        qc.append(SqrtISWAPGate(), [q1, ancilla])
        qc.append(SqrtISWAPdgGate(), [q0, q1])
        qc.append(SqrtISWAPdgGate(), [q0, q1])
        qc.append(SqrtISWAPGate(), [q0, ancilla])
        qc.append(SqrtISWAPGate(), [q0, ancilla])
        qc.append(SGate(), [q0])
        qc.append(SdgGate(), [ancilla])
    
    @staticmethod
    def apply_swap_decomposition(
        qc: QuantumCircuit, 
        q0: int, 
        q1: int, 
        ancilla: int
    ) -> None:
        """Apply SWAP gate using √iSWAP decomposition with ancilla.
        
        Args:
            qc: Quantum circuit.
            q0, q1: Logical qubit indices.
            ancilla: Ancilla qubit index (assumed |0⟩).
        """
        GateSynthesizer.apply_cz_decomposition(qc, q0, q1, ancilla)
        qc.append(SdgGate(), [q0])
        qc.append(SdgGate(), [q1])
        qc.append(SqrtISWAPGate(), [q0, q1])
        qc.append(SqrtISWAPGate(), [q0, q1])


class TwoQubitDecomposer:
    """Decomposes 2-qubit energy-conserving unitaries."""
    
    @staticmethod
    def decompose(
        qc: QuantumCircuit, 
        U: ComplexMatrix, 
        qargs: List[int]
    ) -> List[float]:
        """Decompose and apply a 2-qubit energy-conserving unitary.
        
        Args:
            qc: Quantum circuit to append to.
            U: 4x4 unitary matrix.
            qargs: [q0, q1] qubit indices.
            
        Returns:
            Phases [θ_0, θ_1, θ_2] for each Hamming weight block.
        """
        q0, q1 = qargs
        
        if not is_energy_conserving(U):
            qc.append(UnitaryGate(U), [q0, q1])
            return [0.0, 0.0, 0.0]
        
        blocks = get_hamming_weight_blocks(U)
        
        # HW=0 block (1x1)
        theta_0 = np.angle(blocks.get(0, np.eye(1))[0, 0])
        
        # HW=1 block (2x2) - main decomposition
        block_1 = blocks.get(1, np.eye(2))
        
        # Check determinant        
        det_block = np.linalg.det(block_1)
        theta_1 = np.angle(det_block)
        D = np.eye(2, dtype=complex)
        D[0,0] = np.exp(-1j * np.angle(det_block))  # Factor out global phase
        su2_block = D @ block_1
        block_1 = su2_block
        
        
        det_block_1 = np.linalg.det(block_1)
        if not np.isclose(np.real(det_block_1), 1.0, atol=1e-10):
            raise ValueError(f"HW=1 block determinant must have magnitude 1, got {det_block_1}")
        
        
        theta, phi, lam, global_phase = OneQubitEulerDecomposer('ZYZ').angles_and_phase(
            Operator(block_1)
        )
        
        # assert np.allclose(Operator(qc).data, np.eye(8), atol=1e-12), "Circuit must start as identity"
        
        qc.rz(lam/2, q1)
        qc.rz(-lam/2, q0)
        qc.append(SqrtISWAPGate(), [q0, q1])
        qc.rz(-theta/2, q1)
        qc.rz(theta/2, q0)
        qc.append(SqrtISWAPdgGate(), [q0, q1])
        qc.rz(phi/2, q1)
        qc.rz(-phi/2, q0)
        
        # HW=2 block (1x1)
        theta_2 = np.angle(blocks.get(2, np.eye(1))[0, 0])        
        
        return [theta_0, theta_1, theta_2]


class NQubitDecomposer:
    """Decomposes n-qubit energy-conserving unitaries."""
    
    @staticmethod
    def _decompose_n_controlled_2level(
        U: ComplexMatrix,
        control_bits: List[int],
        control_vals: List[int],
        target_bits: List[int]
    ) -> List[ControlledTwoLevel]:
        """Recursively decompose n-controlled 2-level unitary into single-controlled gates.
        
        Uses the commutator construction A B A† B† = U to halve controls at each step.
        
        Args:
            U: 2x2 SU(2) matrix.
            control_bits: Control qubit bit positions.
            control_vals: Control values (0 or 1).
            target_bits: Target qubit bit positions.
            
        Returns:
            List of single-controlled 2-level unitaries.
        """
        k = len(control_bits)
        
        if k == 1:
            return [ControlledTwoLevel(
                U=U,
                control_bits=control_bits.copy(),
                control_vals=control_vals.copy(),
                target_bits=target_bits.copy()
            )]
        
        A, B = su2_ab_from_u(U)
        
        mid = k // 2
        c1, v1 = control_bits[:mid], control_vals[:mid]
        c2, v2 = control_bits[mid:], control_vals[mid:]
        
        gates = []
        gates += NQubitDecomposer._decompose_n_controlled_2level(A, c1, v1, target_bits)
        gates += NQubitDecomposer._decompose_n_controlled_2level(B, c2, v2, target_bits)
        gates += NQubitDecomposer._decompose_n_controlled_2level(A.conj().T, c1, v1, target_bits)
        gates += NQubitDecomposer._decompose_n_controlled_2level(B.conj().T, c2, v2, target_bits)
        
        return gates
    
    @staticmethod
    def _apply_2level_hw2_unitary(
        qc: QuantumCircuit,
        U: ComplexMatrix,
        qargs: List[int]
    ) -> None:
        """Apply a 2-level unitary with Hamming distance 2 between active states.
        
        Args:
            qc: Quantum circuit.
            U: Full unitary (mostly identity except for 2x2 block).
            qargs: Qubit indices.
        """
        n = len(qargs)
        dim = 2**n
        
        if np.allclose(U, np.eye(dim), atol=1e-12):
            return
        
        local_i, local_j = get_nontrivial_rows_cols(U)
        
        submatrix = np.array([
            [U[local_i, local_i], U[local_i, local_j]],
            [U[local_j, local_i], U[local_j, local_j]]
        ], dtype=complex)
        
        # Determine control and target bits
        control_bits, control_vals, target_bits = [], [], []
        for bit in range(n):
            bit_i = (local_i >> bit) & 1
            bit_j = (local_j >> bit) & 1
            if bit_i == bit_j:
                control_bits.append(bit)
                control_vals.append(bit_i)
            else:
                target_bits.append(bit)
        
        if len(target_bits) != 2:
            raise ValueError(f"Expected 2 target bits, got {len(target_bits)}")
        
        # Decompose into single-controlled gates
        controlled_blocks = NQubitDecomposer._decompose_n_controlled_2level(
            submatrix, control_bits, control_vals, target_bits
        )
        controlled_blocks = list(reversed(controlled_blocks))
        
        # Apply each block
        for block in controlled_blocks:
            theta, phi, lam, global_phase = OneQubitEulerDecomposer('XYX').angles_and_phase(
                Operator(block.U)
            )
            
            # Handle global phase for det=1 matrices
            det_u = np.linalg.det(block.U)
            if np.isclose(det_u, 1.0, atol=1e-8):
                phase = (global_phase + np.pi) % (2 * np.pi) - np.pi
                if np.isclose(abs(phase), np.pi, atol=1e-8):
                    theta += 2 * np.pi
            
            ctrl_bit = block.control_bits[0]
            ctrl_val = block.control_vals[0]
            t0_bit, t1_bit = block.target_bits
            
            control = qargs[ctrl_bit]
            target1 = qargs[t0_bit]
            target2 = qargs[t1_bit]
            
            GateSynthesizer.controlled_exp_i_alpha_r(qc, -lam/2, ctrl_val, target1, target2, control)
            GateSynthesizer.controlled_exp_i_theta_l(qc, -theta/2, ctrl_val, target1, target2, control)
            GateSynthesizer.controlled_exp_i_alpha_r(qc, -phi/2, ctrl_val, target1, target2, control)
    
    @staticmethod
    def _apply_controlled_iswap(
        qc: QuantumCircuit,
        control: int,
        target1: int,
        target2: int,
        control_val: int = 1,
        inverse: bool = False
    ) -> None:
        """Apply controlled-iSWAP gate."""
        angle = -np.pi/2 if inverse else np.pi/2
        GateSynthesizer.controlled_exp_i_alpha_r(qc, angle, control_val, target1, target2, control)
    
    @staticmethod
    def _fix_relative_phases(
        qc: QuantumCircuit,
        qargs: List[int],
        phases: List[float]
    ) -> None:
        """Fix relative phases between Hamming weight sectors.
        
        Args:
            qc: Quantum circuit.
            qargs: Logical qubit indices.
            phases: List of phases [θ_0, θ_1, ...] for each HW sector.
        """
        theta_0 = phases[0]
        num_qubits = len(qargs)
        anc = None
        
        for m, theta in enumerate(phases[1:], start=1):
            delta = theta - theta_0
            if np.abs(delta) < 1e-10:
                continue
            
            # Add ancilla if needed
            if anc is None:
                if qc.num_ancillas > 0:
                    anc = qc.num_qubits - qc.num_ancillas
                else:
                    qc.add_register(AncillaRegister(1, 'phase_anc'))
                    anc = qc.num_qubits - 1
            
            qargs_with_anc = qargs + [anc]
            
            # Pick representative basis state with weight m
            for i in range(2**num_qubits):
                if bin(i).count('1') == m:
                    b = bin(i)[2:].zfill(num_qubits)
                    break
            
            # Construct |b'⟩ by flipping one '1' to '0'
            b_list = list(b)
            for idx in range(num_qubits):
                if b_list[idx] == '1':
                    b_list[idx] = '0'
                    break
            b_prime = ''.join(b_list)
            
            # Build phase Hamiltonian
            dim = 2**(num_qubits + 1)
            H = np.zeros((dim, dim), dtype=complex)
            idx_b0 = int(b, 2)
            idx_bp1 = (1 << num_qubits) + int(b_prime, 2)
            H[idx_b0, idx_b0] = +1
            H[idx_bp1, idx_bp1] = -1
            
            U_phase = scipy.linalg.expm(1j * delta * H)
            NQubitDecomposer._apply_2level_hw2_unitary(qc, U_phase, qargs_with_anc)
    
    @staticmethod
    def decompose(
        qc: QuantumCircuit,
        U: ComplexMatrix,
        qargs: List[int]
    ) -> Dict[int, float]:
        """Decompose and apply an n-qubit energy-conserving unitary.
        
        Args:
            qc: Quantum circuit to append gates to.
            U: 2^n × 2^n unitary matrix.
            qargs: Qubit indices.
            
        Returns:
            Dictionary mapping Hamming weight to phase.
        """
        n_qubits = len(qargs)
        dim = 2**n_qubits
        
        if U.shape != (dim, dim):
            raise ValueError(f"Unitary shape {U.shape} incompatible with {n_qubits} qubits")
        if not np.allclose(U.conj().T @ U, np.eye(dim), atol=1e-10):
            raise ValueError("Matrix is not unitary")
        if not is_energy_conserving(U):
            raise ValueError("Matrix is not energy-conserving")
        
        # Group indices by Hamming weight
        hw_to_indices: Dict[int, List[int]] = {}
        for i in range(dim):
            hw = bin(i).count('1')
            hw_to_indices.setdefault(hw, []).append(i)
        
        hw_phases: Dict[int, float] = {}
        blocks = get_hamming_weight_blocks(U)
        
        for hw in sorted(hw_to_indices.keys()):
            indices = hw_to_indices[hw]
            block = blocks[hw]
            block_size = len(indices)
            
            if block_size == 1:
                hw_phases[hw] = np.angle(block[0, 0])
                continue
            
            # Normalize to SU(block_size)
            det_block = np.linalg.det(block)
            D = np.eye(block_size, dtype=complex)
            D[0, 0] = np.exp(-1j * np.angle(det_block))
            su_block = D @ block
            hw_phases[hw] = np.angle(det_block)
            
            if np.allclose(su_block, np.eye(block_size), atol=1e-10):
                continue
            
            # Decompose into 2-level unitaries
            two_level_blocks = sud_decompose(su_block)
            two_level_blocks = list(reversed(two_level_blocks))
            
            for tlb in two_level_blocks:
                global_i, global_j = indices[tlb.i], indices[tlb.j]
                hd = bin(global_i ^ global_j).count('1')
                
                if hd == 2:
                    full_U = np.eye(dim, dtype=complex)
                    full_U[global_i, global_i] = tlb.submatrix[0, 0]
                    full_U[global_i, global_j] = tlb.submatrix[0, 1]
                    full_U[global_j, global_i] = tlb.submatrix[1, 0]
                    full_U[global_j, global_j] = tlb.submatrix[1, 1]
                    NQubitDecomposer._apply_2level_hw2_unitary(qc, full_U, qargs)
                    
                elif hd > 2:
                    conj_gates, b_double_prime = construct_conjugation_gates(
                        global_i, global_j, n_qubits
                    )
                    
                    # Apply conjugation sequence
                    for ctrl, t1, t2 in conj_gates:
                        NQubitDecomposer._apply_controlled_iswap(
                            qc, qargs[ctrl], qargs[t1], qargs[t2]
                        )
                    
                    # Phase correction from Eq. (76)
                    t = hd // 2
                    phase_upper = (1j) ** (t - 1)
                    phase_lower = (1j) ** (1 - t)
                    
                    W_matrix = np.array([
                        [tlb.submatrix[0, 0], phase_upper * tlb.submatrix[0, 1]],
                        [phase_lower * tlb.submatrix[1, 0], tlb.submatrix[1, 1]]
                    ], dtype=complex)
                    
                    full_W = np.eye(dim, dtype=complex)
                    full_W[b_double_prime, b_double_prime] = W_matrix[0, 0]
                    full_W[b_double_prime, global_j] = W_matrix[0, 1]
                    full_W[global_j, b_double_prime] = W_matrix[1, 0]
                    full_W[global_j, global_j] = W_matrix[1, 1]
                    
                    NQubitDecomposer._apply_2level_hw2_unitary(qc, full_W, qargs)
                    
                    # Inverse conjugation
                    for ctrl, t1, t2 in reversed(conj_gates):
                        NQubitDecomposer._apply_controlled_iswap(
                            qc, qargs[ctrl], qargs[t1], qargs[t2], inverse=True
                        )
        
        # Fix relative phases
        NQubitDecomposer._fix_relative_phases(
            qc, qargs,
            [hw_phases.get(i, 0.0) for i in range(n_qubits + 1)]
        )
        
        return hw_phases




class ThreeQubitDecomposer:
    """Specialized decomposer for 3-qubit energy-conserving unitaries."""
    
    @staticmethod
    def decompose(
        qc: QuantumCircuit,
        U: ComplexMatrix,
        qargs: List[int]
    ) -> List[float]:
        """Decompose and apply a 3-qubit energy-conserving unitary.
        
        Args:
            qc: Quantum circuit.
            U: 8x8 unitary matrix.
            qargs: [q0, q1, q2] qubit indices.
            
        Returns:
            List of phases for each Hamming weight block.
        """
        return list(NQubitDecomposer.decompose(qc, U, qargs).values())



class EnergyConservingDecompositionPass(TransformationPass):
    """Transpiler pass that decomposes energy-conserving unitaries.
    
    This pass decomposes arbitrary energy-conserving (Hamming weight preserving)
    unitaries into a basis set of gates that also conserve energy, specifically
    using √iSWAP and single-qubit rotations.
    
    Attributes:
        target: Optional transpiler target for device constraints.
    """
    
    def __init__(self, target: Optional[Target] = None):
        """Initialize the decomposition pass.
        
        Args:
            target: Optional transpiler target.
        """
        super().__init__()
        self.target = target
    
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the decomposition pass on a DAGCircuit.
        
        Args:
            dag: Input DAG circuit.
            
        Returns:
            Transformed DAG circuit with decomposed gates.
        """
        circuit = dag_to_circuit(dag)
        
        # Create new circuit with ancilla
        qregs = list(dag.qregs.values())
        cregs = list(dag.cregs.values())
        anc = AncillaRegister(1, 'phase_anc')
        new_circuit = QuantumCircuit(*qregs, anc, *cregs)
        ancilla_idx = new_circuit.num_qubits - 1
        
        for instruction in circuit.data:
            instr = instruction.operation
            qargs = instruction.qubits
            cargs = instruction.clbits
            
            # Get qubit indices
            qubit_indices = [new_circuit.find_bit(q).index for q in qargs]
            
            if instr.name == 'cz':
                self._decompose_cz(new_circuit, qubit_indices, ancilla_idx)
                
            elif instr.name == 'swap':
                self._decompose_swap(new_circuit, qubit_indices, ancilla_idx)
                
            elif instr.num_qubits == 2:
                self._decompose_2qubit(new_circuit, instr, qubit_indices)
                
            elif instr.num_qubits >= 3:
                self._decompose_nqubit(new_circuit, instr, qubit_indices)
            else:
                new_circuit.append(instr, qargs, cargs)
        
        return circuit_to_dag(new_circuit)
    
    def _decompose_cz(
        self,
        qc: QuantumCircuit,
        qargs: List[int],
        ancilla: int
    ) -> None:
        """Decompose CZ gate."""
        GateSynthesizer.apply_cz_decomposition(qc, qargs[0], qargs[1], ancilla)
    
    def _decompose_swap(
        self,
        qc: QuantumCircuit,
        qargs: List[int],
        ancilla: int
    ) -> None:
        """Decompose SWAP gate."""
        GateSynthesizer.apply_swap_decomposition(qc, qargs[0], qargs[1], ancilla)
    
    def _decompose_2qubit(
        self,
        qc: QuantumCircuit,
        instr,
        qargs: List[int]
    ) -> None:
        """Decompose 2-qubit gate."""
        U = Operator(instr).data
        # remove phase from u[0,0] to make it special unitary
        global_phase = np.angle(U[0,0])
        U = U * np.exp(-1j * global_phase)
        
        if is_energy_conserving(U):
            phases = TwoQubitDecomposer.decompose(qc, U, qargs)
            # self._fix_phases_2qubit(qc, qargs, phases)
            NQubitDecomposer._fix_relative_phases(qc, qargs, phases)
        else:
            qc.append(instr, qargs)
    
    def _decompose_nqubit(
        self,
        qc: QuantumCircuit,
        instr,
        qargs: List[int]
    ) -> None:
        """Decompose n-qubit gate (n >= 3)."""
        U = Operator(instr).data
        
        U_00 = np.angle(U[0,0])
        U = U * np.exp(-1j * U_00)
        
        if is_energy_conserving(U):
            NQubitDecomposer.decompose(qc, U, qargs)
        else:
            qc.append(instr, qargs)
    
    def _fix_phases_2qubit(
        self,
        qc: QuantumCircuit,
        qargs: List[int],
        phases: List[float]
    ) -> None:
        """Apply phase corrections for 2-qubit decomposition.
        
        For 2 qubits, we have 3 Hamming weight sectors (0, 1, 2).
        Phase corrections use single-qubit Z rotations.
        """
        theta_0, theta_1, theta_2 = phases
        
        # Relative phase between HW=0 and HW=1
        delta_01 = theta_1 - theta_0
        if np.abs(delta_01) > 1e-10:
            qc.rz(delta_01, qargs[0])
        
        # Relative phase between HW=1 and HW=2
        delta_12 = theta_2 - theta_1
        if np.abs(delta_12) > 1e-10:
            qc.rz(delta_12, qargs[1])


# =============================================================================
# Convenience Functions
# =============================================================================

def decompose_energy_conserving_unitary(
    U: ComplexMatrix,
    n_qubits: Optional[int] = None
) -> QuantumCircuit:
    """Decompose an energy-conserving unitary into a quantum circuit.
    
    Args:
        U: Energy-conserving unitary matrix.
        n_qubits: Number of qubits (inferred from U if not provided).
        
    Returns:
        QuantumCircuit implementing U.
        
    Raises:
        ValueError: If U is not energy-conserving or dimensions don't match.
    """
    dim = U.shape[0]
    if n_qubits is None:
        n_qubits = int(np.log2(dim))
    
    if 2**n_qubits != dim:
        raise ValueError(f"Dimension {dim} is not a power of 2")
    
    if not is_energy_conserving(U):
        raise ValueError("Unitary is not energy-conserving")
    
    qc = QuantumCircuit(n_qubits)
    qargs = list(range(n_qubits))
    
    if n_qubits == 2:
        TwoQubitDecomposer.decompose(qc, U, qargs)
    else:
        NQubitDecomposer.decompose(qc, U, qargs)
    
    return qc


def verify_decomposition(
    U: ComplexMatrix,
    qc: QuantumCircuit,
    tol: float = 1e-8
) -> bool:
    """Verify that a circuit implements a given unitary.
    
    Args:
        U: Target unitary matrix.
        qc: Quantum circuit to verify.
        tol: Tolerance for comparison.
        
    Returns:
        True if circuit implements U up to global phase, False otherwise.
    """
    circuit_U = Operator(qc).data
    
    # Extract relevant subspace if circuit has ancillas
    n_target = int(np.log2(U.shape[0]))
    if qc.num_qubits > n_target:
        # Project onto computational subspace with ancillas in |0⟩
        dim_target = 2**n_target
        circuit_U = circuit_U[:dim_target, :dim_target]
    
    # Check equality up to global phase
    if np.allclose(circuit_U, U, atol=tol):
        return True
    
    # Try matching with global phase
    for i in range(U.shape[0]):
        if np.abs(U[i, i]) > tol and np.abs(circuit_U[i, i]) > tol:
            phase = circuit_U[i, i] / U[i, i]
            if np.allclose(circuit_U, phase * U, atol=tol):
                return True
            break
    
    return False



def get_effective_unitary(gate, ancilla_indices, ancilla_state=0):
    """
    Extract the effective unitary on logical qubits when ancilla qubits
    are fixed to a specific state (default |0⟩).
    
    Args:
        gate: The full gate/circuit
        ancilla_indices: List of qubit indices that are ancillas (0-indexed)
        ancilla_state: The fixed state of ancillas (usually 0)
    
    Returns:
        The effective unitary on the logical qubits
    """
    full_unitary = Operator(gate).data
    n_qubits = int(np.log2(full_unitary.shape[0]))
    
    # Find which basis states have ancilla in the specified state
    logical_indices = []
    for i in range(2**n_qubits):
        # Check if all ancilla qubits are in the correct state
        ancilla_match = all(
            ((i >> idx) & 1) == ancilla_state 
            for idx in ancilla_indices
        )
        if ancilla_match:
            logical_indices.append(i)
    
    # Extract the submatrix
    effective_U = full_unitary[np.ix_(logical_indices, logical_indices)]
    return effective_U


def verify_effective_gate(gate, target_gate, ancilla_indices, ancilla_state=0):
    """
    Verify that a gate with ancillas implements a target gate effectively.
    """
    effective_U = get_effective_unitary(gate, ancilla_indices, ancilla_state)
    target_U = Operator(target_gate).data
    
    # Check equivalence up to global phase
    if effective_U.shape != target_U.shape:
        return False, "Dimension mismatch"
    
    # Compute U_eff @ U_target^†
    product = effective_U @ target_U.conj().T
    
    # Should be proportional to identity
    phase = product[0, 0]
    if np.abs(phase) < 1e-10:
        return False, "Zero phase - gates not equivalent"
    
    identity_scaled = phase * np.eye(len(product))
    is_equivalent = np.allclose(product, identity_scaled, atol=1e-8)
    
    return is_equivalent, {
        'effective_unitary': effective_U,
        'target_unitary': target_U,
        'global_phase': np.angle(phase),
        'fidelity': np.abs(np.trace(product) / len(product))**2
    }

def _unitaries_close_up_to_phase(U, V, atol=1e-12):
    
    """Check ||e^{-iφ} U - V|| is small for some global phase φ."""
    # Flatten and compute best-fit global phase via inner product
    uv = np.vdot(V.flatten(), U.flatten())  # <V|U>
    if np.abs(uv) < 1e-12:
        # fallback: just do direct allclose (unlikely)
        return np.allclose(U, V, atol=atol)
    phase = np.angle(uv)
    U_phase = np.exp(-1j * phase) * U
    return np.allclose(U_phase, V, atol=atol)



def print_nontrivial_unitary_basis_action(U, n_qubits, is_hamiltonian=False, tol=1e-10, print_phases=False):
    """
    Print the nontrivial actions of a unitary U on n_qubits in the
    computational basis.

    For each basis state |x>, this prints |x> -> sum_j a_j |j>
    only if U|x> is not (within tol) equal to |x>.
    """
    # dim = 2**n_qubits
    U = np.asarray(U, dtype=complex)
    dim = U.shape[0]
    n_qubits = int(np.log2(dim))
    
    if U.shape != (dim, dim):
        raise ValueError(f"U must have shape {(dim, dim)}, got {U.shape}")

    for col_idx in range(dim):
        # Column col_idx is U |col_idx>
        col = U[:, col_idx]

        # Identity action would be exactly the basis vector e_{col_idx}
        e = np.zeros(dim, dtype=complex)
        if is_hamiltonian:
            e[col_idx] = 0.0
        else:
            e[col_idx] = 1.0

        if np.allclose(col, e, atol=tol):
            # Acts as identity on |col_idx>, skip
            continue

        in_state = format(col_idx, f"0{n_qubits}b")

        # Build a readable expression for the output state
        terms = []
        for row_idx, amp in enumerate(col):
            if abs(amp) > tol:
                out_state = format(row_idx, f"0{n_qubits}b")
                # Nice-ish formatting of complex amplitudes
                if not print_phases:
                    if abs(amp.imag) < tol:
                        terms.append(f"{amp.real:+.3f}|{out_state}>")
                    elif abs(amp.real) < tol:
                        terms.append(f"{amp.imag:+.3f}j|{out_state}>")
                    else:
                        terms.append(f"({amp.real:+.3f}{amp.imag:+.3f}j)|{out_state}>")
                else:
                    terms.append(f"{np.angle(amp) * 180 / np.pi:+.3f}°|{out_state}>")
                

        rhs = " + ".join(terms) if terms else "0"
        print(f"|{in_state}> -> {rhs}")
