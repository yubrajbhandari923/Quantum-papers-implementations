# Energy-Conserving Decomposition Ablation Studies

This directory contains ablation studies for the energy-conserving unitary decomposition algorithm from Bai & Marvian (arXiv:2309.11051).

## Overview

We implement five comprehensive ablation studies to evaluate different aspects of the energy-conserving decomposition:

1. **Symmetry-preserving vs Standard Decomposition** (`symmetry_vs_standard-ablation.py`)
   - Compares energy-conserving decomposition against standard transpilation
   - Metrics: gate counts, depth, non-conserving gates, noisy fidelity

2. **Ancilla vs Non-Energy-Conserving Implementations** (`ancilla_vs_nonconserving-ablation.py`)
   - Compares ancilla-based energy-conserving gates (SWAP, SWAP⊗SWAP, CCZ) vs standard implementations
   - Metrics: ancilla count, gate counts, depth, fidelity under different noise models

3. **Native vs Generic Entanglers** (`native_vs_generic-entangler-ablation.py`)
   - Compares native √iSWAP/XY gates vs transpiled CX-based gates
   - Metrics: gate count overhead, depth overhead, noisy fidelity

4. **Angle Discretization** (`angle_discretization-ablation.py`)
   - Studies impact of discretizing rotation angles to a fixed grid
   - Metrics: operator error, gate infidelity vs grid resolution

5. **Geometry/Topology** (`geometry_topology-ablation.py`)
   - Compares all-to-all vs linear chain connectivity
   - Metrics: SWAP overhead, gate count ratio, depth ratio, noisy fidelity

## Directory Structure

```
baiEnergyConservation/
├── ablations/
│   ├── __init__.py
│   ├── README.md (this file)
│   ├── common.py                              # Shared utilities
│   ├── symmetry_vs_standard-ablation.py
│   ├── ancilla_vs_nonconserving-ablation.py
│   ├── native_vs_generic-entangler-ablation.py
│   ├── angle_discretization-ablation.py
│   ├── geometry_topology-ablation.py
│   └── run_all_ablations.py                   # Helper script to run all
├── results/                                    # JSON results
├── notebooks/
│   └── energy_conserving_ablations.ipynb      # Visualization notebook
└── implementation.py                           # Main implementation
```

## Installation

Ensure you have the required dependencies:

```bash
pip install qiskit qiskit-aer numpy scipy matplotlib pandas jupyter tqdm
```

Or if using the project's `pyproject.toml`:

```bash
# From the repository root
pip install -e .
```

## Usage

### Running Individual Ablations

Each ablation script can be run standalone:

```bash
# Run from the ablations directory
cd baiEnergyConservation/ablations

# Symmetry vs standard
python symmetry_vs_standard-ablation.py

# Ancilla vs non-conserving
python ancilla_vs_nonconserving-ablation.py

# Native vs generic entanglers
python native_vs_generic-entangler-ablation.py

# Angle discretization
python angle_discretization-ablation.py

# Geometry/topology
python geometry_topology-ablation.py
```

### Running All Ablations

Use the convenience script:

```bash
cd baiEnergyConservation/ablations
python run_all_ablations.py
```

### Visualizing Results

After running the ablations, use the Jupyter notebook to visualize results:

```bash
cd baiEnergyConservation/notebooks
jupyter notebook energy_conserving_ablations.ipynb
```

## Configuration

Each ablation script has configurable parameters in its `if __name__ == "__main__"` block:

- `n_qubits_list`: List of qubit counts to test
- `n_trials`: Number of random trials per configuration
- `seed`: Random seed for reproducibility
- `results_path`: Output path for JSON results

Example:
```python
run_symmetry_vs_standard(
    n_qubits_list=[3, 4, 5],  # Test 3, 4, and 5 qubits
    n_trials=30,               # 30 random unitaries per size
    results_path="../results/symmetry_vs_standard.json",
    seed=1234,                 # Reproducible results
)
```

## Results

Results are saved as JSON files in `../results/` with the following structure:

```json
{
  "description": "...",
  "parameters": {...},
  "individual_results": [...],
  "aggregated_statistics": {...}
}
```

The aggregated statistics include means and standard deviations across trials for each metric.

## Common Utilities

The `common.py` module provides shared functionality:

- **Random unitary generation**: `random_energy_conserving_unitary()`
- **Decomposition utilities**: `build_standard_decomposition()`, `build_energy_conserving_decomposition()`
- **Metrics**: `gate_counts()`, `circuit_depth()`, `operator_distance()`, `process_fidelity()`
- **Noise models**: `build_device_like_noise_model()`, `build_clock_jitter_noise_model()`
- **Noisy simulation**: `simulate_noisy_fidelity()`
- **I/O**: `save_results()`, `load_results()`

## Performance Considerations

- Small systems (3-5 qubits): All experiments are feasible on a laptop
- Noisy simulations are limited to ≤4 qubits by default (configurable)
- Use density matrix simulation for noisy circuits (more accurate but slower)
- Adjust `n_trials` based on available compute time

## Notes

- The implementation follows modern Qiskit practices (compatible with Qiskit 1.0+)
- All scripts use proper random seeding for reproducibility
- Error handling ensures partial results are saved even if some trials fail
- Progress bars (tqdm) provide feedback during long runs

## Citation

If you use these ablation studies, please cite:

```bibtex
@article{bai2023energy,
  title={Energy-conserving and the noise resilience of the quantum approximate optimization algorithm},
  author={Bai, Yuxuan and Marvian, Iman},
  journal={arXiv preprint arXiv:2309.11051},
  year={2023}
}
```
