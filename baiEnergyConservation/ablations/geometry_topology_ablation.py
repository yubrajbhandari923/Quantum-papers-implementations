"""Ablation 5: Geometry - all-to-all vs nearest-neighbor topology.

This script compares:
1. All-to-all connectivity (any qubit pair can interact)
2. Linear chain (nearest-neighbor only, requiring SWAP routing)

Key metrics:
- SWAP gate overhead
- Total 2q gate count and depth
- Noisy fidelity under device-like noise
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

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info import Operator

from common import (
    random_energy_conserving_unitary,
    build_energy_conserving_decomposition,
    gate_count_summary,
    build_device_like_noise_model,
    simulate_noisy_fidelity,
    save_results,
    set_random_seed,
)


def build_linear_coupling_map(n_qubits: int, include_ancilla: bool = True) -> CouplingMap:
    """Build a linear chain coupling map.

    Args:
        n_qubits: Number of logical qubits.
        include_ancilla: If True, add one extra qubit for ancilla.

    Returns:
        CouplingMap for a linear chain.
    """
    # Account for ancilla qubit that may be added
    total_qubits = n_qubits + 1 if include_ancilla else n_qubits

    edges = [(i, i + 1) for i in range(total_qubits - 1)]
    # Make bidirectional
    edges += [(i + 1, i) for i in range(total_qubits - 1)]
    return CouplingMap(edges)


def apply_topology_constraints(
    qc: QuantumCircuit,
    topology: str,
    n_qubits: int,
) -> QuantumCircuit:
    """Apply topology constraints to a circuit using SWAP routing.

    Args:
        qc: Input quantum circuit.
        topology: 'all_to_all' or 'linear'.
        n_qubits: Number of logical qubits.

    Returns:
        Transpiled circuit satisfying topology constraints.
    """
    if topology == 'all_to_all':
        # No constraints, return as-is
        return qc

    elif topology == 'linear':
        # Build linear coupling map (accounting for potential ancilla qubits)
        coupling_map = build_linear_coupling_map(n_qubits, include_ancilla=True)

        # Transpile with topology constraint
        # First decompose custom gates
        qc_decomposed = qc.decompose()

        # Use extended basis to handle custom gates
        qc_routed = transpile(
            qc_decomposed,
            coupling_map=coupling_map,
            basis_gates=['rz', 'sx', 'x', 'cx', 'xx_plus_yy', 's', 'sdg'],
            optimization_level=3,  # Use highest optimization for routing
        )

        return qc_routed

    else:
        raise ValueError(f"Unknown topology: {topology}")


def run_single_trial(
    n_qubits: int,
    seed: int,
    noise_model,
) -> dict[str, Any]:
    """Run a single trial comparing all-to-all vs linear topologies.

    Args:
        n_qubits: Number of qubits.
        seed: Random seed for this trial.
        noise_model: Noise model.

    Returns:
        Dictionary with metrics for both topologies.
    """
    # Generate random energy-conserving unitary
    U = random_energy_conserving_unitary(n_qubits, seed=seed)

    # Build energy-conserving decomposition (all-to-all)
    qc_all_to_all = build_energy_conserving_decomposition(U, n_qubits)

    # Apply linear topology constraints
    qc_linear = apply_topology_constraints(qc_all_to_all, 'linear', n_qubits)

    # Get gate count summaries
    summary_all_to_all = gate_count_summary(qc_all_to_all)
    summary_linear = gate_count_summary(qc_linear)

    # Count SWAP gates
    n_swaps_all_to_all = summary_all_to_all['gate_counts'].get('swap', 0)
    n_swaps_linear = summary_linear['gate_counts'].get('swap', 0)

    # SWAP overhead
    swap_overhead = n_swaps_linear - n_swaps_all_to_all

    # Simulate noisy fidelity (if n_qubits is small enough)
    noisy_fid_all_to_all = None
    noisy_fid_linear = None

    if n_qubits <= 4:
        try:
            noisy_fid_all_to_all = simulate_noisy_fidelity(
                qc_all_to_all, U, noise_model, n_shots=100
            )
            noisy_fid_linear = simulate_noisy_fidelity(
                qc_linear, U, noise_model, n_shots=100
            )
        except Exception as e:
            print(f"Warning: Noisy simulation failed for n_qubits={n_qubits}: {e}")

    return {
        'n_qubits': n_qubits,
        'seed': seed,
        'all_to_all': {
            'total_gates': summary_all_to_all['total_gates'],
            'total_2q': summary_all_to_all['total_2q'],
            'n_swaps': n_swaps_all_to_all,
            'depth': summary_all_to_all['depth'],
            'noisy_fidelity': noisy_fid_all_to_all,
            'gate_counts': summary_all_to_all['gate_counts'],
        },
        'linear': {
            'total_gates': summary_linear['total_gates'],
            'total_2q': summary_linear['total_2q'],
            'n_swaps': n_swaps_linear,
            'depth': summary_linear['depth'],
            'noisy_fidelity': noisy_fid_linear,
            'gate_counts': summary_linear['gate_counts'],
        },
        'overhead': {
            'swap_overhead': swap_overhead,
            'gate_count_ratio': (summary_linear['total_2q'] / max(summary_all_to_all['total_2q'], 1)),
            'depth_ratio': (summary_linear['depth'] / max(summary_all_to_all['depth'], 1)),
        },
    }


def run_geometry_topology(
    n_qubits_list: list[int],
    n_trials: int,
    results_path: str,
    seed: int | None = None,
) -> None:
    """Run geometry/topology ablation.

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
        'description': 'Geometry/topology: all-to-all vs linear chain',
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
        all_to_all_metrics = {
            'total_gates': [],
            'total_2q': [],
            'n_swaps': [],
            'depth': [],
            'noisy_fidelity': [],
        }

        linear_metrics = {
            'total_gates': [],
            'total_2q': [],
            'n_swaps': [],
            'depth': [],
            'noisy_fidelity': [],
        }

        overhead_metrics = {
            'swap_overhead': [],
            'gate_count_ratio': [],
            'depth_ratio': [],
        }

        for trial in trials:
            for key in all_to_all_metrics:
                val_all = trial['all_to_all'].get(key)
                val_linear = trial['linear'].get(key)

                if val_all is not None:
                    all_to_all_metrics[key].append(val_all)
                if val_linear is not None:
                    linear_metrics[key].append(val_linear)

            for key in overhead_metrics:
                val = trial['overhead'].get(key)
                if val is not None:
                    overhead_metrics[key].append(val)

        # Compute means and stds
        aggregated[n_qubits] = {
            'all_to_all': {
                key: {
                    'mean': float(np.mean(vals)) if vals else None,
                    'std': float(np.std(vals)) if vals else None,
                }
                for key, vals in all_to_all_metrics.items()
            },
            'linear': {
                key: {
                    'mean': float(np.mean(vals)) if vals else None,
                    'std': float(np.std(vals)) if vals else None,
                }
                for key, vals in linear_metrics.items()
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
    run_geometry_topology(
        n_qubits_list=[3, 4, 5, 6],
        n_trials=30,
        results_path="../results/geometry_topology.json",
        seed=1234,
    )
