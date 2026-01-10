"""Common utilities for ablation studies.

This module provides shared functionality for all ablation experiments,
including random unitary generation, decomposition utilities, metrics,
noise models, and result I/O.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, random_unitary, average_gate_fidelity, state_fidelity
from qiskit.transpiler import PassManager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    pauli_error,
)

# Import from the parent package
import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from implementation import (
    EnergyConservingDecompositionPass,
    is_energy_conserving,
    SqrtISWAPGate,
    SqrtISWAPdgGate,
    get_effective_unitary,
    _unitaries_close_up_to_phase,
    print_nontrivial_unitary_basis_action,
)


# Type aliases
ComplexMatrix = NDArray[np.complexfloating]


# =============================================================================
# Random Seed Management
# =============================================================================

def set_random_seed(seed: int | None = None) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed. If None, uses current state.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)


# =============================================================================
# Random Energy-Conserving Unitaries
# =============================================================================

def random_energy_conserving_unitary(
    n_qubits: int,
    seed: int | None = None
) -> Operator:
    """Generate a random energy-conserving n-qubit unitary.

    The unitary is block-diagonal in the Hamming weight basis, with each
    block being a random Haar unitary on the corresponding subspace.

    Args:
        n_qubits: Number of qubits.
        seed: Random seed for reproducibility.

    Returns:
        Random energy-conserving unitary as an Operator.
    """
    if seed is not None:
        set_random_seed(seed)

    dim = 2**n_qubits
    U = np.eye(dim, dtype=complex)

    # Group computational basis states by Hamming weight
    from collections import defaultdict
    hw_to_indices: dict[int, list[int]] = defaultdict(list)
    for i in range(dim):
        hw = bin(i).count('1')
        hw_to_indices[hw].append(i)

    # For each Hamming weight subspace, draw a random Haar unitary
    for hw, indices in hw_to_indices.items():
        block_size = len(indices)
        if block_size > 1:
            # Generate random unitary for this block
            block_U = random_unitary(block_size, seed=seed).data

            # Embed into full matrix
            for ii, idx_i in enumerate(indices):
                for jj, idx_j in enumerate(indices):
                    U[idx_i, idx_j] = block_U[ii, jj]

    return Operator(U)


# =============================================================================
# Decomposition Utilities
# =============================================================================

def build_standard_decomposition(
    U: Operator,
    basis_gates: list[str] | None = None
) -> QuantumCircuit:
    """Build standard decomposition using transpiler.

    Embeds U as a unitary gate and transpiles to a standard basis,
    ignoring energy conservation.

    Args:
        U: Unitary operator to decompose.
        basis_gates: Target basis gates. Defaults to ['rz','sx','x','cx'].

    Returns:
        Transpiled quantum circuit.
    """
    if basis_gates is None:
        basis_gates = ['rz', 'sx', 'x', 'cx']

    U_data = U.data
    n_qubits = int(np.log2(U_data.shape[0]))

    # Create circuit with unitary gate
    qc = QuantumCircuit(n_qubits)
    qc.unitary(U_data, range(n_qubits), label='U')

    # Transpile to target basis
    qc_decomposed = transpile(
        qc,
        basis_gates=basis_gates,
        optimization_level=2
    )

    return qc_decomposed


def build_energy_conserving_decomposition(
    U: Operator,
    n_qubits: int | None = None,
    validate: bool = True,
    atol: float = 1e-6
) -> QuantumCircuit:
    """Build energy-conserving decomposition using the custom pass.

    Args:
        U: Unitary operator to decompose.
        n_qubits: Number of qubits (inferred from U if not provided).
        validate: If True, validate that decomposition matches target unitary.
        atol: Absolute tolerance for validation.

    Returns:
        Quantum circuit with energy-conserving gates.

    Raises:
        ValueError: If validation fails and validate=True.
    """
    U_data = U.data
    if n_qubits is None:
        n_qubits = int(np.log2(U_data.shape[0]))

    # Create circuit with unitary gate
    qc = QuantumCircuit(n_qubits)
    qc.unitary(U_data, range(n_qubits), label='U')

    # Apply energy-conserving decomposition pass
    pass_manager = PassManager([EnergyConservingDecompositionPass()])
    qc_decomposed = pass_manager.run(qc)

    # Validate the decomposition if requested
    if validate:
        # Get the unitary of the decomposed circuit
        U_decomposed = Operator(qc_decomposed)

        # Check if circuit has ancilla qubits
        n_qubits_total = qc_decomposed.num_qubits

        if n_qubits_total > n_qubits:
            # Circuit has ancilla qubits - extract effective unitary
            ancilla_indices = list(range(n_qubits, n_qubits_total))
            U_effective = get_effective_unitary(
                U_decomposed.data,
                ancilla_indices,
                ancilla_state=0
            )
        else:
            U_effective = U_decomposed.data

        # Check if unitaries match up to global phase
        if not _unitaries_close_up_to_phase(U_data, U_effective, atol=atol):
            print("\n" + "="*80)
            print("ERROR: Energy-conserving decomposition validation FAILED!")
            print("="*80)
            print(f"Target unitary (n_qubits={n_qubits}):")
            print_nontrivial_unitary_basis_action(U_data, n_qubits)
            print(f"\nDecomposed unitary (effective):")
            print_nontrivial_unitary_basis_action(U_effective, n_qubits)

            # Compute error metrics
            diff_norm = np.linalg.norm(U_data - U_effective, ord='fro')
            print(f"\nFrobenius norm difference: {diff_norm:.6e}")
            print("="*80)

            raise ValueError(
                f"Decomposition validation failed! "
                f"Frobenius norm difference: {diff_norm:.6e} (tolerance: {atol})"
            )

    return qc_decomposed


# =============================================================================
# Metrics and Summaries
# =============================================================================

def gate_counts(circ: QuantumCircuit) -> dict[str, int]:
    """Count gates in a circuit by type.

    Args:
        circ: Quantum circuit.

    Returns:
        Dictionary mapping gate names to counts.
    """
    counts: dict[str, int] = {}
    for instruction in circ.data:
        gate_name = instruction.operation.name
        counts[gate_name] = counts.get(gate_name, 0) + 1

    return counts


def gate_count_summary(circ: QuantumCircuit) -> dict[str, Any]:
    """Get summary statistics for gate counts.

    Args:
        circ: Quantum circuit.

    Returns:
        Dictionary with gate count statistics.
    """
    counts = gate_counts(circ)

    # Categorize gates
    single_qubit_gates = ['rz', 'sx', 'x', 'h', 'y', 'z', 's', 'sdg', 't', 'tdg', 'u', 'u1', 'u2', 'u3']
    two_qubit_gates = ['cx', 'cy', 'cz', 'swap', 'iswap', 'xx_plus_yy', 'sqrtiSWAP_ec', 'sqrtiSWAPdg_ec']
    energy_nonconserving = ['h', 'sx', 'x', 'y', 't', 'tdg']

    total_1q = sum(counts.get(g, 0) for g in single_qubit_gates)
    total_2q = sum(counts.get(g, 0) for g in two_qubit_gates)
    total_nonconserving = sum(counts.get(g, 0) for g in energy_nonconserving)

    return {
        'total_gates': sum(counts.values()),
        'total_1q': total_1q,
        'total_2q': total_2q,
        'total_nonconserving': total_nonconserving,
        'depth': circuit_depth(circ),
        'gate_counts': counts,
    }


def circuit_depth(circ: QuantumCircuit) -> int:
    """Calculate circuit depth.

    Args:
        circ: Quantum circuit.

    Returns:
        Circuit depth.
    """
    return circ.depth()


def commutes_with_total_Z(U: Operator, atol: float = 1e-8) -> bool:
    """Check if unitary commutes with total Z operator.

    Args:
        U: Unitary operator.
        atol: Absolute tolerance.

    Returns:
        True if [U, sum_j Z_j] ≈ 0.
    """
    return is_energy_conserving(U.data, tol=atol)


def process_fidelity(U_ideal: Operator, U_noisy: Operator) -> float:
    """Compute process fidelity between two unitaries.

    Uses average gate fidelity as a measure of closeness.

    Args:
        U_ideal: Ideal unitary operator.
        U_noisy: Noisy unitary operator.

    Returns:
        Process fidelity in [0, 1].
    """
    # Use average gate fidelity
    fidelity = average_gate_fidelity(U_ideal, U_noisy)
    return fidelity


def operator_distance(U: Operator, V: Operator) -> float:
    """Compute operator norm distance ||U - V||.

    Args:
        U, V: Unitary operators.

    Returns:
        Operator norm distance.
    """
    diff = U.data - V.data
    return np.linalg.norm(diff, ord=2)


# =============================================================================
# Noise Models
# =============================================================================

def build_device_like_noise_model(
    p1q: float = 0.001,
    p2q: float = 0.01,
    t1: float = 50e3,  # ns
    t2: float = 70e3,  # ns
    gate_1q_time: float = 50,  # ns
    gate_2q_time: float = 300,  # ns
) -> NoiseModel:
    """Build a device-like noise model with depolarizing and T1/T2 errors.

    Args:
        p1q: Single-qubit depolarizing error rate.
        p2q: Two-qubit depolarizing error rate.
        t1: T1 relaxation time (ns).
        t2: T2 dephasing time (ns).
        gate_1q_time: Single-qubit gate time (ns).
        gate_2q_time: Two-qubit gate time (ns).

    Returns:
        Qiskit NoiseModel.
    """
    noise_model = NoiseModel()

    # Single-qubit gates: depolarizing + thermal relaxation
    error_1q_depol = depolarizing_error(p1q, 1)
    error_1q_thermal = thermal_relaxation_error(t1, t2, gate_1q_time)
    error_1q = error_1q_depol.compose(error_1q_thermal)

    single_qubit_gates = ['rz', 'sx', 'x', 'h', 'y', 'z', 's', 'sdg', 't', 'tdg', 'u']
    for gate in single_qubit_gates:
        noise_model.add_all_qubit_quantum_error(error_1q, gate)

    # Two-qubit gates: depolarizing + thermal relaxation
    error_2q_depol = depolarizing_error(p2q, 2)
    error_2q_thermal = thermal_relaxation_error(t1, t2, gate_2q_time).tensor(
        thermal_relaxation_error(t1, t2, gate_2q_time)
    )
    error_2q = error_2q_depol.compose(error_2q_thermal)

    two_qubit_gates = ['cx', 'cy', 'cz', 'swap', 'iswap', 'xx_plus_yy',
                       'sqrtiSWAP_ec', 'sqrtiSWAPdg_ec']
    for gate in two_qubit_gates:
        noise_model.add_all_qubit_quantum_error(error_2q, gate)

    return noise_model


def build_clock_jitter_noise_model(
    sigma_1q: float = 0.01,
    sigma_2q: float = 0.02,
) -> NoiseModel:
    """Build a coherent over/under-rotation noise model.

    Simulates clock jitter by adding small random angle errors to rotations.
    This is implemented as a small depolarizing channel for simplicity.

    Args:
        sigma_1q: Standard deviation of angle error for 1q gates (radians).
        sigma_2q: Standard deviation of angle error for 2q gates (radians).

    Returns:
        Qiskit NoiseModel.
    """
    noise_model = NoiseModel()

    # Approximate coherent errors with depolarizing for simplicity
    # In practice, would use custom coherent error channels
    p1q = min(sigma_1q**2 / 2, 0.5)  # Rough approximation
    p2q = min(sigma_2q**2 / 2, 0.5)

    error_1q = depolarizing_error(p1q, 1)
    error_2q = depolarizing_error(p2q, 2)

    single_qubit_gates = ['rz', 'sx', 'x', 'h', 'y', 'z', 's', 'sdg', 't', 'tdg', 'u']
    for gate in single_qubit_gates:
        noise_model.add_all_qubit_quantum_error(error_1q, gate)

    two_qubit_gates = ['cx', 'cy', 'cz', 'swap', 'iswap', 'xx_plus_yy',
                       'sqrtiSWAP_ec', 'sqrtiSWAPdg_ec']
    for gate in two_qubit_gates:
        noise_model.add_all_qubit_quantum_error(error_2q, gate)

    return noise_model


# =============================================================================
# Noisy Simulation Utilities
# =============================================================================

def simulate_noisy_fidelity(
    qc: QuantumCircuit,
    U_ideal: Operator,
    noise_model: NoiseModel,
    n_shots: int = 1000,
    method: str = 'density_matrix',
) -> float:
    """Simulate noisy circuit and estimate fidelity to ideal unitary.

    Args:
        qc: Quantum circuit to simulate.
        U_ideal: Ideal target unitary.
        noise_model: Noise model.
        n_shots: Number of shots for estimation.
        method: Simulation method ('density_matrix' or 'statevector').

    Returns:
        Average output state fidelity.
    """
    # Decompose custom gates by calling their definitions
    # This manually expands sqrtiSWAP_ec and other custom gates
    qc_decomposed = qc.decompose()

    # Transpile circuit to decompose all gates into simulator-compatible gates
    qc_transpiled = transpile(
        qc_decomposed,
        basis_gates=['rz', 'sx', 'x', 'cx', 'h', 'y', 'z', 's', 'sdg', 't', 'tdg', 'xx_plus_yy'],
        optimization_level=0  # Don't optimize, just decompose
    )

    # Further transpile to decompose xx_plus_yy into basic gates
    qc_transpiled = transpile(
        qc_transpiled,
        basis_gates=['rz', 'sx', 'x', 'cx', 'h', 'y', 'z', 's', 'sdg', 't', 'tdg'],
        optimization_level=0
    )

    # Use density matrix simulator for noisy simulation
    simulator = AerSimulator(noise_model=noise_model, method=method)

    # Sample random input states
    n_qubits_logical = int(np.log2(U_ideal.data.shape[0]))
    n_test_states = min(10, 2**n_qubits_logical)  # Test on subset of basis states

    fidelities = []

    for i in range(n_test_states):
        # Prepare input state (computational basis state)
        input_state = np.zeros(2**n_qubits_logical, dtype=complex)
        input_state[i] = 1.0

        # Compute ideal output state
        ideal_output = U_ideal.data @ input_state

        # Simulate noisy circuit
        # Create a circuit that prepares the input state
        # Account for potential ancilla qubits in the transpiled circuit
        n_qubits_total = qc_transpiled.num_qubits
        test_qc = QuantumCircuit(n_qubits_total)

        # Prepare input state (for computational basis, just apply X gates)
        # Only prepare the logical qubits, leave ancillas in |0⟩
        for qubit_idx in range(n_qubits_logical):
            if (i >> qubit_idx) & 1:
                test_qc.x(qubit_idx)

        # Compose with the transpiled circuit
        test_qc.compose(qc_transpiled, inplace=True)

        # Save statevector/density matrix
        if method == 'statevector':
            test_qc.save_statevector()
        else:
            test_qc.save_density_matrix()

        # Run simulation
        result = simulator.run(test_qc, shots=1).result()

        if method == 'statevector':
            # Get statevector (extract logical qubits if ancillas present)
            # New Qiskit API: result.data() returns a dictionary
            try:
                noisy_state = result.data()['statevector']
            except (KeyError, AttributeError):
                # Fall back to old API
                try:
                    noisy_state = result.get_statevector()
                except AttributeError:
                    # Skip this fidelity calculation
                    continue

            noisy_state = np.array(noisy_state)
            if len(noisy_state) > len(ideal_output):
                # Project out ancilla qubits (assume they're in |0> state)
                n_ancilla = int(np.log2(len(noisy_state))) - n_qubits_logical
                noisy_state_logical = np.zeros(2**n_qubits_logical, dtype=complex)
                for j in range(2**n_qubits_logical):
                    # Ancilla in |0> corresponds to first 2^n_logical indices
                    noisy_state_logical[j] = noisy_state[j]
                noisy_state = noisy_state_logical
        else:
            # Get density matrix
            try:
                rho = result.data()['density_matrix']
            except (KeyError, AttributeError):
                # Fall back to old API
                try:
                    rho = result.get_density_matrix()
                except AttributeError:
                    # Skip this fidelity calculation
                    continue

            rho = np.array(rho)
            if rho.shape[0] > len(ideal_output):
                # Trace out ancilla qubits
                n_ancilla = int(np.log2(rho.shape[0])) - n_qubits_logical
                # Partial trace (simplified - assumes ancilla is last qubit)
                dim_logical = 2**n_qubits_logical
                rho_logical = np.zeros((dim_logical, dim_logical), dtype=complex)
                for j in range(dim_logical):
                    for k in range(dim_logical):
                        rho_logical[j, k] = rho[j, k]
                rho = rho_logical

            # Convert ideal state to density matrix
            ideal_rho = np.outer(ideal_output, ideal_output.conj())

            # Compute fidelity with error handling
            try:
                # Ensure rho is normalized
                trace = np.trace(rho)
                if abs(trace) > 1e-10:
                    rho = rho / trace
                fid = state_fidelity(rho, ideal_rho)
                fidelities.append(fid)
            except Exception as e:
                # Skip invalid states
                continue
            continue

        # Compute state fidelity with error handling
        try:
            # Normalize if needed
            norm = np.linalg.norm(noisy_state)
            if norm > 1e-10:
                noisy_state = noisy_state / norm
            fid = state_fidelity(noisy_state, ideal_output)
            fidelities.append(fid)
        except Exception as e:
            # Skip invalid states
            continue

    return float(np.mean(fidelities))


# =============================================================================
# Result I/O
# =============================================================================

def save_results(path: str | Path, results: dict[str, Any]) -> None:
    """Save results dictionary to JSON file.

    Args:
        path: Output file path.
        results: Results dictionary.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to native Python types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(item) for item in obj]
        return obj

    results_converted = convert(results)

    with open(path, 'w') as f:
        json.dump(results_converted, f, indent=2)

    print(f"Results saved to {path}")


def load_results(path: str | Path) -> dict[str, Any]:
    """Load results dictionary from JSON file.

    Args:
        path: Input file path.

    Returns:
        Results dictionary.
    """
    path = Path(path)
    with open(path, 'r') as f:
        results = json.load(f)

    return results


# =============================================================================
# Angle Discretization
# =============================================================================

def snap_angle(theta: float, grid: float) -> float:
    """Round angle to nearest multiple of grid spacing.

    Args:
        theta: Angle in radians.
        grid: Grid spacing in radians.

    Returns:
        Snapped angle.
    """
    return np.round(theta / grid) * grid
