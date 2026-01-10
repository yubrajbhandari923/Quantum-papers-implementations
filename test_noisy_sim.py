#!/usr/bin/env python3
"""Quick test of noisy simulation fix."""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).resolve().parent / "baiEnergyConservation"
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from ablations.common import (
    random_energy_conserving_unitary,
    build_energy_conserving_decomposition,
    build_device_like_noise_model,
    simulate_noisy_fidelity,
)

print("Testing noisy simulation fix for 3 qubits...")

# Generate a random energy-conserving unitary
U = random_energy_conserving_unitary(n_qubits=3, seed=42)

# Build energy-conserving decomposition
qc = build_energy_conserving_decomposition(U)

print(f"Circuit has {qc.num_qubits} qubits and depth {qc.depth()}")

# Build noise model
noise_model = build_device_like_noise_model()

# Test noisy simulation
try:
    fidelity = simulate_noisy_fidelity(
        qc, U, noise_model, n_shots=100, method='density_matrix'
    )
    print(f"✓ Success! Noisy fidelity: {fidelity:.4f}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
