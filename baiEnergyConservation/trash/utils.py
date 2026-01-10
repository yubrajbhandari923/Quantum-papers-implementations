"""Utility functions for energy-conserving quantum circuits."""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from qiskit.quantum_info import Operator

from numpy.typing import NDArray

ComplexMatrix = NDArray[np.complexfloating]
RealVector = NDArray[np.floating]

@dataclass
class TwoLevelUnitary:
    """
    Represents a 2-level unitary matrix.
    
    A 2-level unitary acts non-trivially on only 2 dimensions (indices i and j)
    and as identity on all other dimensions.
    
    The 2×2 submatrix at positions (i,i), (i,j), (j,i), (j,j) is given by `submatrix`.
    """
    dim: int              # Total dimension
    i: int              # First index (smaller)
    j: int              # Second index (larger)
    submatrix: np.ndarray  # 2×2 unitary submatrix
    
    # __init__ given a nxn matrix U creating the 2-level unitary acting on rows/cols i and j
    @classmethod
    def from_matrix(cls, U: np.ndarray) -> 'TwoLevelUnitary':
        """Create a TwoLevelUnitary from a full matrix U acting on indices i and j."""
        # Find the two indices where U differs from identity
        comp = []
        for idx in range(U.shape[0]):
            if not np.isclose(U[idx, idx], 1.0) or not np.allclose(U[idx, :], np.eye(U.shape[0])[idx, :]):
                comp.append(idx)
        if len(comp) != 2:
            raise ValueError("Input matrix is not a 2-level unitary.")
        i, j = comp[0], comp[1]
        sub = np.zeros((2, 2), dtype=complex)
        sub[0, 0] = U[i, i]
        sub[0, 1] = U[i, j]
        sub[1, 0] = U[j, i]
        sub[1, 1] = U[j, j]
        return cls(dim=U.shape[0], i=i, j=j, submatrix=sub)
    
    def to_full_matrix(self) -> np.ndarray:
        """Convert to full n×n matrix representation."""
        U = np.eye(self.dim, dtype=complex)
        U[self.i, self.i] = self.submatrix[0, 0]
        U[self.i, self.j] = self.submatrix[0, 1]
        U[self.j, self.i] = self.submatrix[1, 0]
        U[self.j, self.j] = self.submatrix[1, 1]
        return U
    
    def dagger(self) -> 'TwoLevelUnitary':
        """Return the conjugate transpose (inverse) of this 2-level unitary."""
        return TwoLevelUnitary(
            dim=self.dim,
            i=self.i,
            j=self.j,
            submatrix=self.submatrix.conj().T
        )
    
    def __repr__(self) -> str:
        return f"TwoLevelUnitary(dim={self.dim}, i={self.i}, j={self.j})"

@dataclass
class ControlledTwoLevel:
    """A 2-level SU(2) acting on a target pair, conditioned on control bits."""
    U: np.ndarray              # 2x2 SU(2) matrix
    control_bits: list[int]    # bit positions of controls
    control_vals: list[int]    # required control values
    target_bits: list[int]     # two target bit positions [t1, t2]
    

def is_energy_conserving(unitary: ComplexMatrix, atol: float = 1e-10) -> bool:
    """Check if a unitary preserves Hamming weight (is energy-conserving).

    An energy-conserving unitary has block-diagonal structure in the
    computational basis, where each block corresponds to states with
    the same Hamming weight.

    Args:
        unitary: The unitary matrix to check.
        atol: Absolute tolerance for numerical comparisons.

    Returns:
        True if the unitary is energy-conserving, False otherwise.
    """
    n = unitary.shape[0]
    n_qubits = int(np.log2(n))

    if 2**n_qubits != n:
        raise ValueError(f"Unitary dimension {n} is not a power of 2")

    # Check that matrix elements between different Hamming weight sectors vanish
    for i in range(n):
        for j in range(n):
            hw_i = bin(i).count('1')
            hw_j = bin(j).count('1')
            if hw_i != hw_j and np.abs(unitary[i, j]) > atol:
                return False
    return True


def get_hamming_weight_blocks(unitary: ComplexMatrix) -> Dict[int, ComplexMatrix]:
    """Extract the block-diagonal structure by Hamming weight sectors.

    Args:
        unitary: Energy-conserving unitary matrix.

    Returns:
        Dictionary mapping Hamming weight to the corresponding unitary block.

    Raises:
        ValueError: If the unitary is not energy-conserving.
    """
    if not is_energy_conserving(unitary):
        raise ValueError("Unitary is not energy-conserving")

    n = unitary.shape[0]
    n_qubits = int(np.log2(n))

    # Group basis states by Hamming weight
    hw_to_indices: Dict[int, List[int]] = {}
    for i in range(n):
        hw = bin(i).count('1')
        if hw not in hw_to_indices:
            hw_to_indices[hw] = []
        hw_to_indices[hw].append(i)

    # Extract blocks
    blocks = {}
    for hw, indices in hw_to_indices.items():
        block_size = len(indices)
        block = np.zeros((block_size, block_size), dtype=complex)
        for new_i, old_i in enumerate(indices):
            for new_j, old_j in enumerate(indices):
                block[new_i, new_j] = unitary[old_i, old_j]
        blocks[hw] = block

    return blocks


def get_nontrivial_rows_cols(U: np.ndarray, atol: float = 1e-10) -> Tuple[int, int]:
    """Get the indices of non-trivial rows/columns in a 2-level unitary.
    
    Args:
        U: Unitary matrix that should be 2-level
        atol: Absolute tolerance for numerical comparisons
        
    Returns:
        Tuple of (row_index_1, row_index_2) for the two non-trivial levels
        
    Raises:
        ValueError: If U is not a valid 2-level unitary
    """
    n = U.shape[0]
    nontrivial_rows = []
    nontrivial_cols = []

    for i in range(n):
        # row i
        nz_row = np.where(np.abs(U[i, :]) > atol)[0]
        if not (len(nz_row) == 1 and nz_row[0] == i and 
                np.isclose(np.real(U[i, i]), 1.0, atol=atol)):
            nontrivial_rows.append(i)

        # col i
        nz_col = np.where(np.abs(U[:, i]) > atol)[0]
        if not (len(nz_col) == 1 and nz_col[0] == i and 
                np.isclose(np.real(U[i, i]), 1.0, atol=atol)):
            nontrivial_cols.append(i)

    if len(nontrivial_rows) > 2 or len(nontrivial_cols) > 2:
        raise ValueError(
            f"Unitary is not a valid 2-level unitary, found "
            f"{len(nontrivial_rows)} non-trivial rows and "
            f"{len(nontrivial_cols)} non-trivial columns."
        )

    return nontrivial_rows[0], nontrivial_rows[1]


def extract_two_level_blocks(U: np.ndarray, atol: float = 1e-10, enforce_hamming_distance: int = 2) -> List[TwoLevelUnitary]:
    """Extract all disjoint 2-level unitaries embedded in a unitary matrix.

    A "2-level" unitary acts nontrivially only on the span of {|i>, |j>} 
    for some i != j. All other basis states are only multiplied by phases.

    Args:
        U: (N x N) unitary matrix.
        atol: numerical tolerance to treat entries as zero.

    Returns:
        List[TwoLevelUnitary], one per 2-level subspace.

    Raises:
        ValueError: if U is not square, not unitary, or contains a 
                   >2-dimensional mixing component.
    """
    if U.shape[0] != U.shape[1]:
        raise ValueError("U must be square")

    n = U.shape[0]

    # Unitarity check
    if not np.allclose(U.conj().T @ U, np.eye(n), atol=atol):
        raise ValueError("Matrix is not unitary within tolerance")

    # Build adjacency graph of coupled basis states
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.abs(U[i, j]) > atol or np.abs(U[j, i]) > atol:
                adj[i].append(j)

    # Find connected components via DFS
    visited = [False] * n
    components: List[List[int]] = []

    for v in range(n):
        if not visited[v]:
            stack = [v]
            comp = []
            visited[v] = True
            while stack:
                x = stack.pop()
                comp.append(x)
                for y in adj[x]:
                    if not visited[y]:
                        visited[y] = True
                        stack.append(y)
            components.append(sorted(comp))

    # Classify components and extract 2-level blocks
    two_level_blocks: List[TwoLevelUnitary] = []

    for comp in components:
        size = len(comp)

        if size == 1:
            # Only a phase on this basis state
            continue

        if size > 2:
            # raise ValueError(
            #     f"Found a {size}-dimensional mixing component {comp}; "
            #     "this is not a 2-level unitary."
            # )
            continue

        # size == 2 → two-level subspace
        i, j = comp
        
        if i > j:
            i, j = j, i
            
        # print(f"Found 2-level block on indices {i},{j}")

        # Extract the 2x2 submatrix on rows/cols (i,j)
        sub = np.zeros((2, 2), dtype=complex)
        sub[0, 0] = U[i, i]
        sub[0, 1] = U[i, j]
        sub[1, 0] = U[j, i]
        sub[1, 1] = U[j, j]
        
        # Enforce that sub has det=1 (SU(2)) skip otherwise
        det_sub = np.linalg.det(sub)
        print(f"2-level block on indices {i},{j} has determinant {det_sub:.3f}")
        if not np.isclose(np.real(det_sub), 1.0, atol=atol):
            continue

        # Sanity check that outside rows/cols look like phases only
        for k in comp:
            row_nz = np.where(np.abs(U[k, :]) > atol)[0]
            col_nz = np.where(np.abs(U[:, k]) > atol)[0]
            if not set(row_nz).issubset({i, j}) or not set(col_nz).issubset({i, j}):
                raise ValueError(
                    f"Basis state {k} has couplings outside {{i,j}}; "
                    "this is not a pure 2-level block."
                )

        two_level_blocks.append(TwoLevelUnitary(dim=n, i=i, j=j, submatrix=sub))

    return two_level_blocks


def su2_AB_from_U(U: np.ndarray, atol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """Find A,B ∈ SU(2) such that A B A† B† = U.
    
    This construction follows Eq. (64) from the paper.
    
    Args:
        U: 2x2 unitary matrix
        atol: Absolute tolerance for numerical comparisons
        
    Returns:
        Tuple of (A, B) matrices in SU(2)
        
    Raises:
        ValueError: If U is not 2x2 or has zero determinant
    """
    if U.shape != (2, 2):
        raise ValueError("U must be 2x2")

    # Remove global phase so that det(U_su2) = 1
    detU = np.linalg.det(U)
    if np.abs(detU) < atol:
        raise ValueError("Determinant of U is ~0, not unitary?")
    U_su2 = U / np.sqrt(detU)
    # U_su2 = U

    # Eigen-decomposition: U_su2 = W diag(e^{iφ1}, e^{iφ2}) W†
    evals, evecs = np.linalg.eig(U_su2)
    φ1, φ2 = np.angle(evals[0]), np.angle(evals[1])

    # For SU(2), φ1 + φ2 ≈ 0 (mod 2π)
    θ = 0.5 * (φ1 - φ2)

    # Build exp(i θ Z/2) and exp(i θ Z)
    # Z = np.diag([1.0, -1.0])
    D = np.diag([np.exp(1j * θ), np.exp(-1j * θ)])
    
    W = evecs  # columns are eigenvectors
    
    if not np.allclose(W @ D @ W.conj().T, U_su2, atol=1e-6):
        # Eigenvalues are in opposite order, swap columns of W
        W = W[:, ::-1]
        # Or equivalently, negate θ
        # θ = -θ
        if not np.allclose(W @ D @ W.conj().T, U_su2, atol=1e-6):
            raise ValueError("Eigen-decomposition failed to reconstruct U_su2")
        
    exp_iθZ_over2 = np.diag(np.exp(1j * θ * np.array([1.0, -1.0]) / 2.0))
    # A(1) = W exp(i θ Z/2) W†
    A1 = W @ exp_iθZ_over2 @ W.conj().T

    # B(1) = i W X W†, where X = [[0,1],[1,0]]
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    B1 = 1j * W @ X @ W.conj().T

    return A1, B1



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

def _unitaries_close_up_to_phase(U, V, atol=1e-8):
    
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

def _unitaries_close_up_to_phase(U, V, atol=1e-8):
    """Check ||e^{-iφ} U - V|| is small for some global phase φ."""
    # Flatten and compute best-fit global phase via inner product
    uv = np.vdot(V.flatten(), U.flatten())  # <V|U>
    if np.abs(uv) < 1e-12:
        # fallback: just do direct allclose (unlikely)
        return np.allclose(U, V, atol=atol)
    phase = np.angle(uv)
    U_phase = np.exp(-1j * phase) * U
    return np.allclose(U_phase, V, atol=atol)