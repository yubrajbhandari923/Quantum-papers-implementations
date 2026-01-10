"""
Debugging utilities for quantum decomposition.

Provides configurable logging and visualization helpers for
debugging unitary decompositions.
"""

import logging
from typing import Optional
import numpy as np

# Configure module logger
logger = logging.getLogger('quantum_decomposition')


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> None:
    """
    Configure logging for the quantum decomposition package.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format_string: Custom format string for log messages
    """
    if format_string is None:
        format_string = '%(levelname)s - %(name)s - %(message)s'
    
    logging.basicConfig(level=level, format=format_string)
    logger.setLevel(level)


def enable_debug() -> None:
    """Enable verbose debug logging."""
    setup_logging(logging.DEBUG)


def disable_debug() -> None:
    """Disable debug logging, show only warnings and errors."""
    setup_logging(logging.WARNING)


def print_unitary_action(
    U: np.ndarray, 
    n_qubits: Optional[int] = None,
    is_hamiltonian: bool = False, 
    tol: float = 1e-10
) -> None:
    """
    Print nontrivial actions of a unitary on computational basis states.
    
    For each basis state |x⟩, prints |x⟩ → Σ_j a_j |j⟩
    only if U|x⟩ ≠ |x⟩ (within tolerance).
    
    Args:
        U: Unitary matrix
        n_qubits: Number of qubits (inferred from U if not provided)
        is_hamiltonian: If True, identity action is 0 instead of 1
        tol: Tolerance for comparing to identity
    """
    U = np.asarray(U, dtype=complex)
    dim = U.shape[0]
    
    if n_qubits is None:
        n_qubits = int(np.log2(dim))
    
    if U.shape != (dim, dim):
        raise ValueError(f"U must be square, got shape {U.shape}")

    for col_idx in range(dim):
        col = U[:, col_idx]
        
        # Identity reference
        e = np.zeros(dim, dtype=complex)
        e[col_idx] = 0.0 if is_hamiltonian else 1.0

        if np.allclose(col, e, atol=tol):
            continue

        in_state = format(col_idx, f"0{n_qubits}b")
        terms = _format_amplitude_terms(col, n_qubits, tol)
        
        rhs = " + ".join(terms) if terms else "0"
        print(f"|{in_state}⟩ → {rhs}")


def _format_amplitude_terms(col: np.ndarray, n_qubits: int, tol: float) -> list:
    """Format nonzero amplitudes as readable strings."""
    terms = []
    for row_idx, amp in enumerate(col):
        if abs(amp) <= tol:
            continue
            
        out_state = format(row_idx, f"0{n_qubits}b")
        
        if abs(amp.imag) < tol:
            terms.append(f"{amp.real:+.3f}|{out_state}⟩")
        elif abs(amp.real) < tol:
            terms.append(f"{amp.imag:+.3f}j|{out_state}⟩")
        else:
            terms.append(f"({amp.real:+.3f}{amp.imag:+.3f}j)|{out_state}⟩")
    
    return terms


def print_two_level_block(
    submatrix: np.ndarray, 
    local_i: int, 
    local_j: int,
    global_i: Optional[int] = None,
    global_j: Optional[int] = None
) -> None:
    """
    Print information about a two-level block.
    
    Args:
        submatrix: 2x2 submatrix
        local_i, local_j: Local indices within the block
        global_i, global_j: Optional global basis state indices
    """
    det = np.linalg.det(submatrix)
    
    print(f"Two-level block: ({local_i}, {local_j})")
    if global_i is not None and global_j is not None:
        print(f"  Global indices: ({global_i}, {global_j})")
    
    with np.printoptions(precision=3, suppress=True):
        print(f"  Submatrix:\n{submatrix}")
    print(f"  Determinant: {det:.4f}")


def verify_decomposition(
    original: np.ndarray, 
    circuit, 
    qargs: list,
    atol: float = 1e-8
) -> bool:
    """
    Verify that a circuit implements the target unitary.
    
    Args:
        original: Target unitary matrix
        circuit: QuantumCircuit implementation
        qargs: Qubit indices
        atol: Tolerance for comparison
        
    Returns:
        True if implementation matches target
    """
    from qiskit.quantum_info import Operator
    
    implemented = Operator(circuit).data
    
    # May need to reorder qubits or extract subspace
    if implemented.shape != original.shape:
        logger.warning(
            f"Shape mismatch: target {original.shape}, implemented {implemented.shape}"
        )
        return False
    
    if np.allclose(implemented, original, atol=atol):
        logger.info("Verification passed: implementation matches target")
        return True
    
    # Check if they differ only by global phase
    phase_diff = implemented / original
    if np.allclose(phase_diff, phase_diff[0, 0] * np.ones_like(phase_diff), atol=atol):
        phase = np.angle(phase_diff[0, 0])
        logger.info(f"Match up to global phase: {phase:.4f} rad")
        return True
    
    logger.error("Verification failed: implementation does not match target")
    return False