"""Ablation 4: Exact vs approximate angle discretization.

This script studies the approximate synthesis setting where rotation angles
are discretized to a fixed grid, similar to Solovay-Kitaev style approximation.

Key metrics:
- Operator norm error ||U - Ũ|| vs grid resolution
- Average gate infidelity vs grid resolution
- Error vs number of gates/depth trade-off
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

# Add parent directory to path
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.circuit import Parameter

from common import (
    random_energy_conserving_unitary,
    build_energy_conserving_decomposition,
    gate_count_summary,
    operator_distance,
    process_fidelity,
    snap_angle,
    save_results,
    set_random_seed,
)


def discretize_circuit_angles(qc: QuantumCircuit, grid: float) -> QuantumCircuit:
    """Discretize all rotation angles in a circuit to a grid.

    Args:
        qc: Input quantum circuit.
        grid: Grid spacing in radians (e.g., π/8, π/16).

    Returns:
        New circuit with discretized angles.
    """
    # Create a new circuit
    qc_discrete = QuantumCircuit(*qc.qregs, *qc.cregs)

    for instruction in qc.data:
        gate = instruction.operation
        qubits = instruction.qubits
        clbits = instruction.clbits

        # Check if gate has rotation parameters
        if gate.name == 'rz' and len(gate.params) > 0:
            # Discretize the angle
            theta = gate.params[0]
            if isinstance(theta, Parameter):
                # Skip parameters
                qc_discrete.append(gate, qubits, clbits)
            else:
                theta_discrete = snap_angle(theta, grid)
                qc_discrete.rz(theta_discrete, qubits[0])

        elif gate.name in ['u', 'u3'] and len(gate.params) >= 3:
            # Discretize all three angles
            theta, phi, lam = gate.params[:3]
            theta_d = snap_angle(theta, grid)
            phi_d = snap_angle(phi, grid)
            lam_d = snap_angle(lam, grid)
            if gate.name == 'u':
                qc_discrete.u(theta_d, phi_d, lam_d, qubits[0])
            else:
                qc_discrete.u3(theta_d, phi_d, lam_d, qubits[0])

        else:
            # Copy gate as-is
            qc_discrete.append(gate, qubits, clbits)

    return qc_discrete


def run_single_trial(
    n_qubits: int,
    seed: int,
    grids: list[float],
) -> dict[str, Any]:
    """Run a single trial with different discretization grids.

    Args:
        n_qubits: Number of qubits.
        seed: Random seed for this trial.
        grids: List of grid spacings to test.

    Returns:
        Dictionary with metrics for each grid.
    """
    # Generate random energy-conserving unitary
    U = random_energy_conserving_unitary(n_qubits, seed=seed)

    # Build exact decomposition
    qc_exact = build_energy_conserving_decomposition(U, n_qubits)
    summary_exact = gate_count_summary(qc_exact)

    results_by_grid = {}

    for grid in grids:
        # Discretize angles
        qc_discrete = discretize_circuit_angles(qc_exact, grid)

        # Get circuit summary
        summary_discrete = gate_count_summary(qc_discrete)

        # Compute unitary
        U_discrete = Operator(qc_discrete)

        # Extract logical subspace if ancillas are present
        if qc_discrete.num_qubits > n_qubits:
            dim_logical = 2**n_qubits
            U_discrete_data = U_discrete.data[:dim_logical, :dim_logical]
            U_discrete = Operator(U_discrete_data)

        # Compute error metrics
        error_norm = operator_distance(U, U_discrete)
        fidelity = process_fidelity(U, U_discrete)

        results_by_grid[grid] = {
            'grid': grid,
            'total_gates': summary_discrete['total_gates'],
            'total_2q': summary_discrete['total_2q'],
            'depth': summary_discrete['depth'],
            'operator_error': error_norm,
            'gate_fidelity': fidelity,
            'gate_infidelity': 1.0 - fidelity,
        }

    return {
        'n_qubits': n_qubits,
        'seed': seed,
        'exact': {
            'total_gates': summary_exact['total_gates'],
            'total_2q': summary_exact['total_2q'],
            'depth': summary_exact['depth'],
        },
        'discretized': results_by_grid,
    }


def run_angle_discretization(
    n_qubits: int,
    n_trials: int,
    grids: list[float],
    results_path: str,
    seed: int | None = None,
) -> None:
    """Run angle discretization ablation.

    Args:
        n_qubits: Number of qubits.
        n_trials: Number of random trials.
        grids: List of grid spacings (in radians).
        results_path: Path to save results JSON.
        seed: Base random seed.
    """
    if seed is not None:
        set_random_seed(seed)

    all_results = []

    print(f"\n=== Testing {n_qubits} qubits with {len(grids)} grid spacings ===")

    for trial in tqdm(range(n_trials), desc=f"{n_qubits} qubits"):
        # Generate unique seed for this trial
        trial_seed = seed + trial if seed is not None else None

        try:
            result = run_single_trial(n_qubits, trial_seed, grids)
            all_results.append(result)
        except Exception as e:
            print(f"Error in trial {trial}: {e}")
            continue

    # Compute aggregate statistics
    aggregated = aggregate_results(all_results, grids)

    # Save results
    output = {
        'description': 'Angle discretization ablation study',
        'parameters': {
            'n_qubits': n_qubits,
            'n_trials': n_trials,
            'grids': grids,
            'seed': seed,
        },
        'individual_results': all_results,
        'aggregated_statistics': aggregated,
    }

    save_results(results_path, output)


def aggregate_results(
    results: list[dict[str, Any]],
    grids: list[float],
) -> dict[str, Any]:
    """Aggregate results across trials.

    Args:
        results: List of individual trial results.
        grids: List of grid spacings.

    Returns:
        Dictionary with mean and std statistics.
    """
    aggregated = {}

    for grid in grids:
        metrics = {
            'operator_error': [],
            'gate_fidelity': [],
            'gate_infidelity': [],
            'total_gates': [],
            'total_2q': [],
            'depth': [],
        }

        for trial in results:
            if grid in trial['discretized']:
                for key in metrics:
                    val = trial['discretized'][grid].get(key)
                    if val is not None:
                        metrics[key].append(val)

        aggregated[grid] = {
            key: {
                'mean': float(np.mean(vals)) if vals else None,
                'std': float(np.std(vals)) if vals else None,
                'min': float(np.min(vals)) if vals else None,
                'max': float(np.max(vals)) if vals else None,
            }
            for key, vals in metrics.items()
        }

    return aggregated


if __name__ == "__main__":
    # Test with different grid spacings
    grids = [
        np.pi / 4,   # Coarse
        np.pi / 8,   # Medium
        np.pi / 16,  # Fine
        np.pi / 32,  # Very fine
        np.pi / 64,  # Extra fine
    ]

    run_angle_discretization(
        n_qubits=3,
        n_trials=50,
        grids=grids,
        results_path="../results/angle_discretization.json",
        seed=1234,
    )
