"""Ablation 3: Native XY/√iSWAP entanglers vs generic CX-based entanglers.

This script compares:
1. Native implementation using √iSWAP/XY gates directly
2. Generic implementation transpiling to CX + single-qubit gates

Key metrics:
- Increase in 2q gate count and depth
- Noisy fidelity under device-like and clock-jitter noise
- Performance vs number of entangling layers
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

from qiskit import transpile
from qiskit.quantum_info import Operator

from common import (
    random_energy_conserving_unitary,
    build_energy_conserving_decomposition,
    gate_count_summary,
    build_device_like_noise_model,
    build_clock_jitter_noise_model,
    simulate_noisy_fidelity,
    save_results,
    set_random_seed,
)


def decompose_to_cx_basis(qc, basis_gates=None):
    """Further transpile circuit to CX basis.

    Args:
        qc: Quantum circuit.
        basis_gates: Target basis gates.

    Returns:
        Transpiled circuit.
    """
    if basis_gates is None:
        basis_gates = ['rz', 'sx', 'x', 'cx']

    # First decompose custom gates by calling their definitions
    qc_decomposed = qc.decompose()

    # Then transpile to standard gates
    # Add xx_plus_yy to basis gates to handle the underlying gate
    extended_basis = basis_gates + ['xx_plus_yy', 's', 'sdg']

    qc_generic = transpile(
        qc_decomposed,
        basis_gates=extended_basis,
        optimization_level=2
    )

    # Now transpile to final basis without xx_plus_yy
    qc_generic = transpile(
        qc_generic,
        basis_gates=basis_gates,
        optimization_level=2
    )

    return qc_generic


def run_single_trial(
    n_qubits: int,
    seed: int,
    noise_model_device,
    noise_model_jitter,
) -> dict[str, Any]:
    """Run a single trial comparing native vs generic entanglers.

    Args:
        n_qubits: Number of qubits.
        seed: Random seed for this trial.
        noise_model_device: Device-like noise model.
        noise_model_jitter: Clock jitter noise model.

    Returns:
        Dictionary with metrics for both implementations.
    """
    # Generate random energy-conserving unitary
    U = random_energy_conserving_unitary(n_qubits, seed=seed)

    # Build native decomposition (√iSWAP-based)
    qc_native = build_energy_conserving_decomposition(U, n_qubits)

    # Build generic decomposition (CX-based)
    qc_generic = decompose_to_cx_basis(qc_native)

    # Get gate count summaries
    summary_native = gate_count_summary(qc_native)
    summary_generic = gate_count_summary(qc_generic)

    # Simulate noisy fidelity (if n_qubits is small enough)
    noisy_fid_native_device = None
    noisy_fid_generic_device = None
    noisy_fid_native_jitter = None
    noisy_fid_generic_jitter = None

    if n_qubits <= 4:
        try:
            # Device noise
            noisy_fid_native_device = simulate_noisy_fidelity(
                qc_native, U, noise_model_device, n_shots=100
            )
            noisy_fid_generic_device = simulate_noisy_fidelity(
                qc_generic, U, noise_model_device, n_shots=100
            )

            # Jitter noise
            noisy_fid_native_jitter = simulate_noisy_fidelity(
                qc_native, U, noise_model_jitter, n_shots=100
            )
            noisy_fid_generic_jitter = simulate_noisy_fidelity(
                qc_generic, U, noise_model_jitter, n_shots=100
            )
        except Exception as e:
            print(f"Warning: Noisy simulation failed for n_qubits={n_qubits}: {e}")

    return {
        'n_qubits': n_qubits,
        'seed': seed,
        'native': {
            'total_gates': summary_native['total_gates'],
            'total_1q': summary_native['total_1q'],
            'total_2q': summary_native['total_2q'],
            'depth': summary_native['depth'],
            'gate_counts': summary_native['gate_counts'],
            'noisy_fidelity_device': noisy_fid_native_device,
            'noisy_fidelity_jitter': noisy_fid_native_jitter,
        },
        'generic': {
            'total_gates': summary_generic['total_gates'],
            'total_1q': summary_generic['total_1q'],
            'total_2q': summary_generic['total_2q'],
            'depth': summary_generic['depth'],
            'gate_counts': summary_generic['gate_counts'],
            'noisy_fidelity_device': noisy_fid_generic_device,
            'noisy_fidelity_jitter': noisy_fid_generic_jitter,
        },
        'overhead': {
            'gate_count_ratio': (summary_generic['total_2q'] / max(summary_native['total_2q'], 1)),
            'depth_ratio': (summary_generic['depth'] / max(summary_native['depth'], 1)),
        },
    }


def run_native_vs_generic(
    n_qubits_list: list[int],
    n_trials: int,
    results_path: str,
    seed: int | None = None,
) -> None:
    """Run native vs generic entangler ablation.

    Args:
        n_qubits_list: List of qubit counts to test.
        n_trials: Number of random trials per qubit count.
        results_path: Path to save results JSON.
        seed: Base random seed.
    """
    if seed is not None:
        set_random_seed(seed)

    # Build noise models
    noise_model_device = build_device_like_noise_model()
    noise_model_jitter = build_clock_jitter_noise_model()

    all_results = []

    for n_qubits in n_qubits_list:
        print(f"\n=== Testing {n_qubits} qubits ===")

        for trial in tqdm(range(n_trials), desc=f"{n_qubits} qubits"):
            # Generate unique seed for this trial
            trial_seed = seed + trial if seed is not None else None

            try:
                result = run_single_trial(
                    n_qubits,
                    trial_seed,
                    noise_model_device,
                    noise_model_jitter,
                )
                all_results.append(result)
            except Exception as e:
                print(f"Error in trial {trial} for {n_qubits} qubits: {e}")
                continue

    # Compute aggregate statistics
    aggregated = aggregate_results(all_results)

    # Save results
    output = {
        'description': 'Native XY/√iSWAP vs generic CX-based entanglers',
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
        native_metrics = {
            'total_gates': [],
            'total_2q': [],
            'depth': [],
            'noisy_fidelity_device': [],
            'noisy_fidelity_jitter': [],
        }

        generic_metrics = {
            'total_gates': [],
            'total_2q': [],
            'depth': [],
            'noisy_fidelity_device': [],
            'noisy_fidelity_jitter': [],
        }

        overhead_metrics = {
            'gate_count_ratio': [],
            'depth_ratio': [],
        }

        for trial in trials:
            for key in native_metrics:
                val_native = trial['native'].get(key)
                val_generic = trial['generic'].get(key)

                if val_native is not None:
                    native_metrics[key].append(val_native)
                if val_generic is not None:
                    generic_metrics[key].append(val_generic)

            for key in overhead_metrics:
                val = trial['overhead'].get(key)
                if val is not None:
                    overhead_metrics[key].append(val)

        # Compute means and stds
        aggregated[n_qubits] = {
            'native': {
                key: {
                    'mean': float(np.mean(vals)) if vals else None,
                    'std': float(np.std(vals)) if vals else None,
                }
                for key, vals in native_metrics.items()
            },
            'generic': {
                key: {
                    'mean': float(np.mean(vals)) if vals else None,
                    'std': float(np.std(vals)) if vals else None,
                }
                for key, vals in generic_metrics.items()
            },
            'overhead': {
                key: {
                    'mean': float(np.mean(vals)) if vals else None,
                    'std': float(np.std(vals)) if vals else None,
                }
                for key, vals in overhead_metrics.items()
            },
        }

    return aggregated


if __name__ == "__main__":
    run_native_vs_generic(
        n_qubits_list=[3, 4, 5],
        n_trials=30,
        results_path="../results/native_vs_generic-entangler.json",
        seed=1234,
    )
