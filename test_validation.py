#!/usr/bin/env python3
"""Test the decomposition validation."""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).resolve().parent / "baiEnergyConservation"
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from ablations.common import (
    random_energy_conserving_unitary,
    build_energy_conserving_decomposition,
)

print("Testing decomposition validation for 3 qubits...")

# Generate a random energy-conserving unitary
U = random_energy_conserving_unitary(n_qubits=3, seed=42)

# Build energy-conserving decomposition with validation
try:
    qc = build_energy_conserving_decomposition(U, validate=True)
    print(f"✓ Validation passed! Circuit has {qc.num_qubits} qubits and depth {qc.depth()}")
except ValueError as e:
    print(f"✗ Validation failed: {e}")

print("\nTesting for 4 qubits...")
U4 = random_energy_conserving_unitary(n_qubits=4, seed=123)
try:
    qc4 = build_energy_conserving_decomposition(U4, validate=True)
    print(f"✓ Validation passed! Circuit has {qc4.num_qubits} qubits and depth {qc4.depth()}")
except ValueError as e:
    print(f"✗ Validation failed: {e}")
