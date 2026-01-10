# Energy-Conserving Decomposition: Ablation Studies

Comprehensive ablation studies for the energy-conserving unitary decomposition algorithm from **Bai & Marvian (arXiv:2309.11051)**.

## Quick Start

### 1. Installation

```bash
# Install required packages
pip install qiskit qiskit-aer numpy scipy matplotlib pandas jupyter tqdm
```

### 2. Run All Ablations

```bash
cd baiEnergyConservation/ablations

# Quick test (reduced parameters)
python run_all_ablations.py --quick

# Full experiment (may take 30-60 minutes)
python run_all_ablations.py
```

### 3. Visualize Results

```bash
cd ../notebooks
jupyter notebook energy_conserving_ablations.ipynb
```

## Ablation Studies Overview

We implement **5 comprehensive ablation studies** to evaluate different aspects of the energy-conserving decomposition:

### 1. Symmetry-Preserving vs Standard Decomposition

**Question**: How does energy-conserving decomposition compare to standard transpilation?

**Experiments**:
- Generate random energy-conserving unitaries for 3, 4, 5 qubits
- Decompose using both energy-conserving and standard methods
- Compare gate counts, depth, and noisy fidelity

**Key Metrics**:
- Total gates (1q and 2q)
- Non-energy-conserving gate count
- Circuit depth
- Operator distance to target
- Noisy gate fidelity under device-like noise

**Expected Insights**:
- When symmetry-preserving decomposition is more efficient
- Impact of preserving energy conservation on noise resilience

### 2. Ancilla vs Non-Energy-Conserving Implementations

**Question**: What is the cost of implementing gates (SWAP, CCZ) using energy-conserving methods with ancillas?

**Experiments**:
- Implement SWAP, SWAP⊗SWAP, and CCZ gates
- Compare energy-conserving (with ancilla) vs standard implementations
- Test under device-like and clock-jitter noise

**Key Metrics**:
- Number of ancilla qubits required
- Gate count breakdown
- Circuit depth
- Fidelity under different noise models

**Expected Insights**:
- Trade-off between ancilla usage and energy conservation
- Performance under different noise types

### 3. Native vs Generic Entanglers

**Question**: What is the overhead of transpiling native √iSWAP/XY gates to generic CX-based gates?

**Experiments**:
- Decompose random energy-conserving unitaries to native √iSWAP gates
- Further transpile to CX-based gates
- Compare overhead and noisy performance

**Key Metrics**:
- Gate count ratio (generic/native)
- Depth ratio (generic/native)
- Noisy fidelity degradation

**Expected Insights**:
- Benefits of native √iSWAP support
- Impact of gate decomposition on performance

### 4. Angle Discretization

**Question**: How does discretizing rotation angles to a fixed grid affect approximation quality?

**Experiments**:
- Generate random energy-conserving unitaries
- Discretize rotation angles to grids: π/4, π/8, π/16, π/32, π/64
- Measure approximation error vs grid resolution

**Key Metrics**:
- Operator norm error ||U - Ũ||
- Average gate infidelity
- Error vs gate count trade-off

**Expected Insights**:
- Approximation error scaling with grid resolution
- Relevance for fault-tolerant implementations

### 5. Geometry/Topology

**Question**: What is the overhead of restricted connectivity (linear chain) vs all-to-all?

**Experiments**:
- Decompose unitaries for all-to-all connectivity
- Apply linear chain constraints with SWAP routing
- Compare overhead and performance

**Key Metrics**:
- SWAP gate overhead
- Gate count and depth ratios
- Noisy fidelity degradation

**Expected Insights**:
- Cost of topology constraints
- Benefits of all-to-all connectivity for energy-conserving circuits

## Implementation Details

### Code Structure

```
baiEnergyConservation/
├── implementation.py              # Main decomposition implementation
├── ablations/
│   ├── common.py                  # Shared utilities
│   ├── symmetry_vs_standard-ablation.py
│   ├── ancilla_vs_nonconserving-ablation.py
│   ├── native_vs_generic-entangler-ablation.py
│   ├── angle_discretization-ablation.py
│   ├── geometry_topology-ablation.py
│   └── run_all_ablations.py
├── results/                       # JSON results (generated)
└── notebooks/
    └── energy_conserving_ablations.ipynb
```

### Common Utilities (`ablations/common.py`)

The `common.py` module provides:

1. **Random Unitary Generation**
   - `random_energy_conserving_unitary()`: Generate random block-diagonal (Hamming weight) unitaries

2. **Decomposition Methods**
   - `build_standard_decomposition()`: Standard transpilation to {rz, sx, x, cx}
   - `build_energy_conserving_decomposition()`: Energy-conserving decomposition using custom pass

3. **Metrics**
   - `gate_counts()`: Count gates by type
   - `gate_count_summary()`: Categorize gates (1q, 2q, non-conserving)
   - `circuit_depth()`: Circuit depth
   - `commutes_with_total_Z()`: Verify energy conservation
   - `operator_distance()`: ||U - V|| norm
   - `process_fidelity()`: Average gate fidelity

4. **Noise Models**
   - `build_device_like_noise_model()`: Depolarizing + T1/T2 relaxation
   - `build_clock_jitter_noise_model()`: Coherent over/under-rotation errors

5. **Noisy Simulation**
   - `simulate_noisy_fidelity()`: Estimate fidelity under noise using density matrix simulation

6. **I/O**
   - `save_results()`: Save to JSON with proper type conversion
   - `load_results()`: Load from JSON

### Noise Models

#### Device-like Noise
- Single-qubit depolarizing: p = 0.1%
- Two-qubit depolarizing: p = 1%
- T1 = 50 μs, T2 = 70 μs
- Gate times: 50 ns (1q), 300 ns (2q)

#### Clock Jitter Noise
- Coherent angle errors on rotation gates
- σ_1q = 0.01 rad, σ_2q = 0.02 rad

Both models are realistic but not tied to specific hardware backends.

## Results Format

Results are saved as JSON with structure:

```json
{
  "description": "...",
  "parameters": {
    "n_qubits_list": [3, 4, 5],
    "n_trials": 30,
    "seed": 1234
  },
  "individual_results": [
    {
      "n_qubits": 3,
      "seed": 1234,
      "standard": {...},
      "energy_conserving": {...}
    },
    ...
  ],
  "aggregated_statistics": {
    "3": {
      "standard": {
        "total_gates": {"mean": 42.3, "std": 5.1},
        ...
      },
      "energy_conserving": {...}
    },
    ...
  }
}
```

## Performance Notes

- **Small systems** (3-5 qubits): All experiments run on a laptop in minutes
- **Noisy simulations**: Limited to ≤4 qubits (density matrix simulation is memory-intensive)
- **Quick mode**: `--quick` flag reduces parameters for testing
- **Full mode**: Standard parameters, ~30-60 minutes total on a laptop

### Computational Requirements

- **CPU**: Multi-core recommended (parallelized via NumPy/Qiskit)
- **RAM**: 8-16 GB (density matrix simulation for 4 qubits needs ~1 GB)
- **Storage**: <100 MB for all results

## Usage Examples

### Run Individual Ablation

```python
from ablations.symmetry_vs_standard_ablation import run_symmetry_vs_standard

run_symmetry_vs_standard(
    n_qubits_list=[3, 4, 5],
    n_trials=30,
    results_path="../results/symmetry_vs_standard.json",
    seed=1234,
)
```

### Load and Analyze Results

```python
from ablations.common import load_results

results = load_results("../results/symmetry_vs_standard.json")

# Access aggregated statistics
stats = results['aggregated_statistics']
print(f"3-qubit mean 2q gates (standard): {stats['3']['standard']['total_2q']['mean']}")
print(f"3-qubit mean 2q gates (energy-conserving): {stats['3']['energy_conserving']['total_2q']['mean']}")
```

### Custom Experiments

```python
from ablations.common import (
    random_energy_conserving_unitary,
    build_energy_conserving_decomposition,
    gate_count_summary,
)

# Generate random 4-qubit energy-conserving unitary
U = random_energy_conserving_unitary(n_qubits=4, seed=42)

# Decompose
qc = build_energy_conserving_decomposition(U, n_qubits=4)

# Analyze
summary = gate_count_summary(qc)
print(f"Total 2q gates: {summary['total_2q']}")
print(f"Circuit depth: {summary['depth']}")
print(f"Gate breakdown: {summary['gate_counts']}")
```

## Visualization

The Jupyter notebook provides:

- **Bar plots**: Gate count comparisons
- **Line plots**: Scaling with qubit count
- **Error bars**: Mean ± std across trials
- **Tables**: Detailed gate breakdowns
- **Log-log plots**: Discretization error scaling

All plots are publication-quality (300 dpi, saved to `results/` directory).

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`, ensure you're running from the correct directory:

```bash
cd baiEnergyConservation/ablations
python symmetry_vs_standard-ablation.py
```

Or add the parent directory to your Python path:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Memory Issues

For large systems, reduce `n_trials` or skip noisy simulations:

```python
# Modify in ablation script
if n_qubits <= 3:  # Only simulate small systems
    noisy_fid = simulate_noisy_fidelity(...)
```

### Qiskit Version Issues

The code is designed for **Qiskit 1.0+**. If using older versions:

- Replace `qiskit.transpiler.PassManager` with `qiskit.transpiler.PassManager`
- Use `qiskit_aer.AerSimulator` instead of legacy backends
- Check `XXPlusYYGate` availability (added in Qiskit 0.45)

## Citation

If you use these ablation studies in your research, please cite:

```bibtex
@article{bai2023energy,
  title={Energy-conserving and the noise resilience of the quantum approximate optimization algorithm},
  author={Bai, Yuxuan and Marvian, Iman},
  journal={arXiv preprint arXiv:2309.11051},
  year={2023}
}
```

## Contributing

Contributions are welcome! Areas for extension:

- Additional noise models (amplitude damping, crosstalk)
- More gate implementations (Toffoli variants, multi-controlled gates)
- Larger system sizes (with approximations or cluster computing)
- Alternative topologies (2D grids, heavy-hex)
- Fault-tolerant variants with discrete gate sets

## License

This implementation is provided for research and educational purposes.

## Contact

For questions or issues, please open an issue on the repository or contact the maintainers.
