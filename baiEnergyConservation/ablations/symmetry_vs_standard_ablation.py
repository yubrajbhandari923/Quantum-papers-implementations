"""Ablation 1: Symmetry-preserving vs standard decomposition.

This script compares energy-conserving decomposition against standard
transpilation for the same random energy-conserving unitaries.

Key metrics:
- Gate counts (1q, 2q, non-energy-conserving gates)
- Circuit depth
- Operator distance to original unitary (sanity check)
- Noisy gate fidelity under device-like noise
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

from qiskit.quantum_info import Operator

from common import (
    random_energy_conserving_unitary,
    build_standard_decomposition,
    build_energy_conserving_decomposition,
    gate_count_summary,
    operator_distance,
    build_device_like_noise_model,
    simulate_noisy_fidelity,
    save_results,
    set_random_seed,
)


def run_single_trial(
    n_qubits: int,
    seed: int,
    noise_model,
) -> dict[str, Any]:
    """Run a single trial comparing standard vs energy-conserving decomposition.

    Args:
        n_qubits: Number of qubits.
        seed: Random seed for this trial.
        noise_model: Noise model for simulation.

    Returns:
        Dictionary with metrics for both decompositions.
    """
    # Generate random energy-conserving unitary
    U = random_energy_conserving_unitary(n_qubits, seed=seed)

    # Build standard decomposition
    qc_std = build_standard_decomposition(U)

    # Build energy-conserving decomposition
    qc_sym = build_energy_conserving_decomposition(U, n_qubits)

    # Get gate count summaries
    summary_std = gate_count_summary(qc_std)
    summary_sym = gate_count_summary(qc_sym)

    # Check operator distance (sanity check)
    U_std = Operator(qc_std)
    U_sym = Operator(qc_sym)

    # Extract logical subspace if ancillas are present
    n_qubits_std = qc_std.num_qubits
    n_qubits_sym = qc_sym.num_qubits

    if n_qubits_std > n_qubits:
        # No ancillas in standard decomposition typically
        pass

    if n_qubits_sym > n_qubits:
        # Extract logical subspace (ancilla in |0> state)
        dim_logical = 2**n_qubits
        U_sym_data = U_sym.data[:dim_logical, :dim_logical]
        U_sym = Operator(U_sym_data)

    dist_std = operator_distance(U, U_std)
    dist_sym = operator_distance(U, U_sym)

    # Simulate noisy fidelity (if n_qubits is small enough)
    noisy_fid_std = None
    noisy_fid_sym = None

    if n_qubits <= 4:  # Only simulate for small systems
        try:
            # For standard decomposition
            noisy_fid_std = simulate_noisy_fidelity(
                qc_std, U, noise_model, n_shots=100, method='density_matrix'
            )

            # For energy-conserving decomposition
            # Need to handle ancilla qubits
            noisy_fid_sym = simulate_noisy_fidelity(
                qc_sym, U, noise_model, n_shots=100, method='density_matrix'
            )
        except Exception as e:
            print(f"Warning: Noisy simulation failed for n_qubits={n_qubits}: {e}")

    return {
        'n_qubits': n_qubits,
        'seed': seed,
        'standard': {
            'total_gates': summary_std['total_gates'],
            'total_1q': summary_std['total_1q'],
            'total_2q': summary_std['total_2q'],
            'total_nonconserving': summary_std['total_nonconserving'],
            'depth': summary_std['depth'],
            'operator_distance': dist_std,
            'noisy_fidelity': noisy_fid_std,
            'gate_counts': summary_std['gate_counts'],
        },
        'energy_conserving': {
            'total_gates': summary_sym['total_gates'],
            'total_1q': summary_sym['total_1q'],
            'total_2q': summary_sym['total_2q'],
            'total_nonconserving': summary_sym['total_nonconserving'],
            'depth': summary_sym['depth'],
            'operator_distance': dist_sym,
            'noisy_fidelity': noisy_fid_sym,
            'gate_counts': summary_sym['gate_counts'],
        },
    }


def run_symmetry_vs_standard(
    n_qubits_list: list[int],
    n_trials: int,
    results_path: str,
    seed: int | None = None,
) -> None:
    """Run symmetry vs standard decomposition ablation.

    Args:
        n_qubits_list: List of qubit counts to test.
        n_trials: Number of random trials per qubit count.
        results_path: Path to save results JSON.
        seed: Base random seed.
    """
    if seed is not None:
        set_random_seed(seed)

    # Build noise model
    noise_model = build_device_like_noise_model()

    all_results = []

    for n_qubits in n_qubits_list:
        print(f"\n=== Testing {n_qubits} qubits ===")

        for trial in tqdm(range(n_trials), desc=f"{n_qubits} qubits"):
            # Generate unique seed for this trial
            trial_seed = seed + trial if seed is not None else None

            try:
                result = run_single_trial(n_qubits, trial_seed, noise_model)
                all_results.append(result)
            except Exception as e:
                print(f"Error in trial {trial} for {n_qubits} qubits: {e}")
                continue

    # Compute aggregate statistics
    aggregated = aggregate_results(all_results)

    # Save results
    output = {
        'description': 'Symmetry-preserving vs standard decomposition ablation',
        'parameters': {
            'n_qubits_list': n_qubits_list,
            'n_trials': n_trials,
            'seed': seed,
        },
        'individual_results': all_results,
        'aggregated_statistics': aggregated,
    }

    save_results(results_path, output)


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate results across trials.

    Args:
        results: List of individual trial results.

    Returns:
        Dictionary with mean and std statistics.
    """
    from collections import defaultdict

    # Group by n_qubits
    by_n_qubits: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_n_qubits[r['n_qubits']].append(r)

    aggregated = {}

    for n_qubits, trials in by_n_qubits.items():
        std_metrics = {
            'total_gates': [],
            'total_1q': [],
            'total_2q': [],
            'total_nonconserving': [],
            'depth': [],
            'operator_distance': [],
            'noisy_fidelity': [],
        }

        sym_metrics = {
            'total_gates': [],
            'total_1q': [],
            'total_2q': [],
            'total_nonconserving': [],
            'depth': [],
            'operator_distance': [],
            'noisy_fidelity': [],
        }

        for trial in trials:
            for key in std_metrics:
                val_std = trial['standard'].get(key)
                val_sym = trial['energy_conserving'].get(key)

                if val_std is not None:
                    std_metrics[key].append(val_std)
                if val_sym is not None:
                    sym_metrics[key].append(val_sym)

        # Compute means and stds
        aggregated[n_qubits] = {
            'standard': {
                key: {
                    'mean': float(np.mean(vals)) if vals else None,
                    'std': float(np.std(vals)) if vals else None,
                }
                for key, vals in std_metrics.items()
            },
            'energy_conserving': {
                key: {
                    'mean': float(np.mean(vals)) if vals else None,
                    'std': float(np.std(vals)) if vals else None,
                }
                for key, vals in sym_metrics.items()
            },
        }

    return aggregated


if __name__ == "__main__":
    run_symmetry_vs_standard(
        n_qubits_list=[3, 4, 5],
        n_trials=30,
        results_path="../results/symmetry_vs_standard.json",
        seed=1234,
    )
