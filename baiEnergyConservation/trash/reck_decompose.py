"""
Reck Decomposition: Decompose an n×n unitary matrix into 2-level unitaries.

Based on: Reck et al., "Experimental realization of any discrete unitary operator"
          Phys. Rev. Lett. 73, 58 (1994)

The algorithm uses Givens rotations to systematically zero out elements of the 
unitary matrix column by column, from bottom to top within each column.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from utils import TwoLevelBlock as TwoLevelUnitary

def create_givens_rotation(a: complex, b: complex) -> np.ndarray:
    """
    Create a 2×2 unitary G such that G @ [a, b]^T = [r, 0]^T.
    
    This is the Givens rotation that zeros out the second element.
    
    Parameters:
        a: First element
        b: Second element (to be zeroed)
    
    Returns:
        2×2 unitary matrix G
    """
    # Handle edge cases
    if np.abs(b) < 1e-14:
        # b is already zero, return identity (or phase if a is negative)
        if np.abs(a) < 1e-14:
            return np.eye(2, dtype=complex)
        phase = np.conj(a) / np.abs(a)
        return np.array([[phase, 0], [0, 1]], dtype=complex)
    
    if np.abs(a) < 1e-14:
        # a is zero, simple rotation
        phase = np.conj(b) / np.abs(b)
        return np.array([[0, phase], [-np.conj(phase), 0]], dtype=complex)
    
    # General case: construct Givens rotation
    r = np.sqrt(np.abs(a)**2 + np.abs(b)**2)
    c = np.conj(a) / r  # cos-like term
    s = np.conj(b) / r  # sin-like term
    
    # G = [[c, s], [-s*, c*]] such that G @ [a, b]^T = [r, 0]^T
    G = np.array([
        [c, s],
        [-np.conj(s), np.conj(c)]
    ], dtype=complex)
    
    return G


def reck_decomposition(U: np.ndarray, tol: float = 1e-12) -> Tuple[List[TwoLevelUnitary], np.ndarray]:
    """
    Decompose an n×n unitary matrix into a sequence of 2-level unitaries.
    
    Uses the Reck et al. method: systematically zero out elements column by column,
    from bottom to top, using Givens rotations.
    
    Parameters:
        U: n×n unitary matrix to decompose
        tol: Tolerance for numerical checks
    
    Returns:
        (two_level_unitaries, diagonal): 
            - List of 2-level unitaries such that U = U_1^† @ U_2^† @ ... @ U_k^† @ D
            - Final diagonal matrix D (phase factors)
    
    The original U can be reconstructed as:
        U = U_1.dagger() @ U_2.dagger() @ ... @ U_k.dagger() @ D
    
    Or equivalently:
        U_k @ ... @ U_2 @ U_1 @ U = D
    """
    n = U.shape[0]
    
    # Verify input is unitary
    if U.shape != (n, n):
        raise ValueError("Input must be a square matrix")
    if not np.allclose(U @ U.conj().T, np.eye(n), atol=tol):
        raise ValueError("Input matrix is not unitary")
    
    # Work with a copy
    W = U.copy().astype(complex)
    
    # Store the 2-level unitaries used
    two_level_unitaries: List[TwoLevelUnitary] = []
    
    # Process column by column, from left to right
    for col in range(n - 1):
        # Within each column, zero out elements from bottom to top
        # We want to zero W[row, col] for row = n-1, n-2, ..., col+1
        for row in range(n - 1, col, -1):
            # Skip if already zero
            if np.abs(W[row, col]) < tol:
                continue
            
            # Create Givens rotation to zero W[row, col]
            # We mix row (row-1) and row (row) to zero element at position [row, col]
            # The pivot is at [row-1, col], the element to zero is at [row, col]
            pivot_row = row - 1
            
            # Get the 2×2 submatrix elements
            a = W[pivot_row, col]
            b = W[row, col]
            
            # Create Givens rotation G such that G @ [a, b]^T = [r, 0]^T
            G = create_givens_rotation(a, b)
            
            # Create the 2-level unitary
            two_level = TwoLevelUnitary(dim=n, i=pivot_row, j=row, submatrix=G)
            two_level_unitaries.append(two_level)
            
            # Apply G to rows pivot_row and row of W
            # W[pivot_row, :] and W[row, :] get mixed
            W_pivot = W[pivot_row, :].copy()
            W_row = W[row, :].copy()
            
            W[pivot_row, :] = G[0, 0] * W_pivot + G[0, 1] * W_row
            W[row, :] = G[1, 0] * W_pivot + G[1, 1] * W_row
    
    # W should now be diagonal (up to numerical precision)
    diagonal = np.diag(W)
    D = np.diag(diagonal)
    
    # Verify W is diagonal
    off_diag = W - D
    if not np.allclose(off_diag, 0, atol=tol):
        print(f"Warning: Residual off-diagonal elements with max magnitude {np.max(np.abs(off_diag))}")
    
    return two_level_unitaries, D


def reconstruct_unitary(two_level_unitaries: List[TwoLevelUnitary], 
                        diagonal: np.ndarray) -> np.ndarray:
    """
    Reconstruct the original unitary from its decomposition.
    
    U = U_1^† @ U_2^† @ ... @ U_k^† @ D
    """
    n = diagonal.shape[0]
    result = diagonal.copy()
    
    # Apply daggered unitaries in reverse order
    for tlu in reversed(two_level_unitaries):
        U_dag = tlu.dagger().to_full_matrix()
        result = U_dag @ result
    
    return result


def count_two_level_unitaries(n: int) -> int:
    """
    Return the number of 2-level unitaries needed to decompose an n×n unitary.
    This is n(n-1)/2.
    """
    return n * (n - 1) // 2


