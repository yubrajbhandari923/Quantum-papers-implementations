# Quick Start Guide: Energy-Conserving Ablation Studies

## What Has Been Implemented

A comprehensive suite of ablation studies for the energy-conserving unitary decomposition algorithm from Bai & Marvian (arXiv:2309.11051).

### Complete File Structure

```
baiEnergyConservation/
├── __init__.py                     # Package initialization
├── implementation.py               # Core decomposition algorithm
├── README_ABLATIONS.md            # Comprehensive documentation
├── QUICKSTART.md                  # This file
│
├── ablations/                      # Ablation study implementations
│   ├── __init__.py
│   ├── README.md                  # Detailed ablation documentation
│   ├── common.py                  # Shared utilities
│   ├── run_all_ablations.py      # Master script
│   │
│   ├── symmetry_vs_standard-ablation.py        # Ablation 1
│   ├── ancilla_vs_nonconserving-ablation.py    # Ablation 2
│   ├── native_vs_generic-entangler-ablation.py # Ablation 3
│   ├── angle_discretization-ablation.py        # Ablation 4
│   └── geometry_topology-ablation.py           # Ablation 5
│
├── results/                        # JSON results (generated)
│   ├── symmetry_vs_standard.json
│   ├── ancilla_vs_nonconserving.json
│   ├── native_vs_generic-entangler.json
│   ├── angle_discretization.json
│   └── geometry_topology.json
│
└── notebooks/                      # Visualization
    └── energy_conserving_ablations.ipynb
```

## Installation (5 minutes)

### Option 1: Using pip

```bash
pip install qiskit qiskit-aer numpy scipy matplotlib pandas jupyter tqdm
```

### Option 2: Using conda

```bash
conda create -n energy-conserving python=3.10
conda activate energy-conserving
pip install qiskit qiskit-aer numpy scipy matplotlib pandas jupyter tqdm
```

## Running Experiments

### Option A: Run All Ablations (Recommended for First Run)

```bash
cd baiEnergyConservation/ablations

# Quick test (5-10 minutes)
python run_all_ablations.py --quick

# Full experiments (30-60 minutes)
python run_all_ablations.py
```

This will:
- Run all 5 ablation studies sequentially
- Save results to `../results/*.json`
- Print progress and timing information
- Handle errors gracefully

### Option B: Run Individual Ablations

```bash
cd baiEnergyConservation/ablations

# Ablation 1: Symmetry vs Standard (~10-15 min)
python symmetry_vs_standard-ablation.py

# Ablation 2: Ancilla vs Non-conserving (~2-3 min)
python ancilla_vs_nonconserving-ablation.py

# Ablation 3: Native vs Generic (~10-15 min)
python native_vs_generic-entangler-ablation.py

# Ablation 4: Angle Discretization (~5-10 min)
python angle_discretization-ablation.py

# Ablation 5: Geometry/Topology (~15-20 min)
python geometry_topology-ablation.py
```

## Visualizing Results

```bash
cd baiEnergyConservation/notebooks
jupyter notebook energy_conserving_ablations.ipynb
```

The notebook will:
- Load all JSON results
- Generate publication-quality plots
- Create comparison tables
- Provide summary statistics

## What Each Ablation Does

### 1. Symmetry vs Standard (`symmetry_vs_standard-ablation.py`)

**Question**: Is preserving energy conservation worth it?

**Method**:
- Generate 30 random energy-conserving unitaries per system size (3-5 qubits)
- Decompose using both methods
- Compare gate counts, depth, and noisy fidelity

**Output**: `results/symmetry_vs_standard.json`

### 2. Ancilla vs Non-conserving (`ancilla_vs_nonconserving-ablation.py`)

**Question**: What's the cost of using ancilla qubits to preserve symmetry?

**Method**:
- Implement SWAP, SWAP⊗SWAP, and CCZ gates
- Compare energy-conserving (with ancilla) vs standard
- Test under two noise models

**Output**: `results/ancilla_vs_nonconserving.json`

### 3. Native vs Generic Entanglers (`native_vs_generic-entangler-ablation.py`)

**Question**: How much overhead do we pay for not having native √iSWAP gates?

**Method**:
- Decompose to native √iSWAP gates
- Further transpile to CX-based gates
- Measure overhead and fidelity loss

**Output**: `results/native_vs_generic-entangler.json`

### 4. Angle Discretization (`angle_discretization-ablation.py`)

**Question**: How does discretizing angles affect accuracy?

**Method**:
- Snap rotation angles to grids: π/4, π/8, π/16, π/32, π/64
- Measure approximation error
- Plot error vs grid resolution

**Output**: `results/angle_discretization.json`

### 5. Geometry/Topology (`geometry_topology-ablation.py`)

**Question**: What's the cost of limited connectivity?

**Method**:
- Compare all-to-all vs linear chain
- Measure SWAP overhead
- Evaluate fidelity degradation

**Output**: `results/geometry_topology.json`

## Customization

### Modify Parameters

Edit the `if __name__ == "__main__"` block in any ablation script:

```python
# Example: symmetry_vs_standard-ablation.py
if __name__ == "__main__":
    run_symmetry_vs_standard(
        n_qubits_list=[3, 4, 5],  # Change qubit counts
        n_trials=30,               # Change number of trials
        results_path="../results/symmetry_vs_standard.json",
        seed=1234,                 # Change seed for different random instances
    )
```

### Add Custom Noise

In `ablations/common.py`, modify noise models:

```python
def build_device_like_noise_model(
    p1q: float = 0.001,  # Adjust error rates
    p2q: float = 0.01,
    t1: float = 50e3,    # Adjust coherence times
    t2: float = 70e3,
    # ...
):
```

## Expected Results

### Quick Mode (~10 minutes total)

- Tests 3-4 qubits
- 10 trials per configuration
- Reduced parameters for discretization study

### Full Mode (~30-60 minutes total)

- Tests 3-5 qubits (6 qubits for geometry)
- 30-50 trials per configuration
- Comprehensive statistics

## Troubleshooting

### "Module not found" errors

Make sure you're running from the correct directory:

```bash
cd baiEnergyConservation/ablations
python run_all_ablations.py
```

### Memory errors for noisy simulations

Noisy simulations are limited to ≤4 qubits by default. To reduce further:

```python
# In ablation scripts, modify:
if n_qubits <= 3:  # Change from 4 to 3
    noisy_fid = simulate_noisy_fidelity(...)
```

### Qiskit version issues

Ensure you have Qiskit 1.0+:

```bash
pip install --upgrade qiskit qiskit-aer
python -c "import qiskit; print(qiskit.__version__)"
```

## Next Steps

1. **Run experiments**: Start with `--quick` mode
2. **Check results**: Look at JSON files in `results/`
3. **Visualize**: Open Jupyter notebook
4. **Customize**: Modify parameters for your specific needs
5. **Extend**: Add new ablations or metrics

## Getting Help

- Read `README_ABLATIONS.md` for detailed documentation
- Read `ablations/README.md` for implementation details
- Check the Jupyter notebook for visualization examples
- Inspect `ablations/common.py` for available utilities

## Citation

```bibtex
@article{bai2023energy,
  title={Energy-conserving and the noise resilience of the quantum approximate optimization algorithm},
  author={Bai, Yuxuan and Marvian, Iman},
  journal={arXiv preprint arXiv:2309.11051},
  year={2023}
}
```

## Summary

You now have a complete, research-ready ablation study framework for energy-conserving unitary decomposition. The code is:

- ✅ Modular and extensible
- ✅ Well-documented
- ✅ Reproducible (with seeds)
- ✅ Publication-quality (plots at 300 dpi)
- ✅ Compatible with modern Qiskit (1.0+)
- ✅ Ready to run on a laptop

Happy experimenting! 🚀
