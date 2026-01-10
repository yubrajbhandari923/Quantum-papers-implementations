"""Convenience script to run all ablation studies sequentially."""

from __future__ import annotations

import sys
from pathlib import Path
import time

# Add parent directory to path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))


def run_all_ablations(
    quick_mode: bool = False,
    seed: int = 1234,
) -> None:
    """Run all ablation studies.

    Args:
        quick_mode: If True, use reduced parameters for faster execution.
        seed: Base random seed for all experiments.
    """
    print("=" * 80)
    print("Running All Energy-Conserving Decomposition Ablation Studies")
    print("=" * 80)

    # Configure parameters based on mode
    if quick_mode:
        print("\n[Quick Mode] Using reduced parameters for faster execution\n")
        n_qubits_list = [3, 4]
        n_trials = 10
        n_qubits_disc = 3
        n_trials_disc = 20
    else:
        print("\n[Full Mode] Using standard parameters (this may take a while)\n")
        n_qubits_list = [3, 4, 5]
        n_trials = 30
        n_qubits_disc = 3
        n_trials_disc = 50

    results_dir = Path("../results")
    results_dir.mkdir(exist_ok=True)

    # Track timings
    timings = {}

    # 1. Symmetry vs Standard
    print("\n" + "=" * 80)
    print("1/5: Running Symmetry vs Standard Decomposition Ablation")
    print("=" * 80)
    start_time = time.time()

    import symmetry_vs_standard_ablation as sym_vs_std
    try:
        sym_vs_std.run_symmetry_vs_standard(
            n_qubits_list=n_qubits_list,
            n_trials=n_trials,
            results_path=str(results_dir / "symmetry_vs_standard.json"),
            seed=seed,
        )
        timings['symmetry_vs_standard'] = time.time() - start_time
        print(f"\n✓ Completed in {timings['symmetry_vs_standard']:.1f}s")
    except Exception as e:
        print(f"\n✗ Failed with error: {e}")
        timings['symmetry_vs_standard'] = None

    # 2. Ancilla vs Non-Conserving
    print("\n" + "=" * 80)
    print("2/5: Running Ancilla vs Non-Conserving Ablation")
    print("=" * 80)
    start_time = time.time()

    import ancilla_vs_nonconserving_ablation as anc_vs_nc
    try:
        anc_vs_nc.run_ancilla_vs_nonconserving(
            results_path=str(results_dir / "ancilla_vs_nonconserving.json"),
        )
        timings['ancilla_vs_nonconserving'] = time.time() - start_time
        print(f"\n✓ Completed in {timings['ancilla_vs_nonconserving']:.1f}s")
    except Exception as e:
        print(f"\n✗ Failed with error: {e}")
        timings['ancilla_vs_nonconserving'] = None

    # 3. Native vs Generic Entanglers
    print("\n" + "=" * 80)
    print("3/5: Running Native vs Generic Entanglers Ablation")
    print("=" * 80)
    start_time = time.time()

    import native_vs_generic_entangler_ablation as nat_vs_gen
    try:
        nat_vs_gen.run_native_vs_generic(
            n_qubits_list=n_qubits_list,
            n_trials=n_trials,
            results_path=str(results_dir / "native_vs_generic-entangler.json"),
            seed=seed,
        )
        timings['native_vs_generic'] = time.time() - start_time
        print(f"\n✓ Completed in {timings['native_vs_generic']:.1f}s")
    except Exception as e:
        print(f"\n✗ Failed with error: {e}")
        timings['native_vs_generic'] = None

    # 4. Angle Discretization
    print("\n" + "=" * 80)
    print("4/5: Running Angle Discretization Ablation")
    print("=" * 80)
    start_time = time.time()

    import numpy as np
    import angle_discretization_ablation as angle_disc

    grids = [
        np.pi / 4,
        np.pi / 8,
        np.pi / 16,
        np.pi / 32,
        np.pi / 64,
    ]

    try:
        angle_disc.run_angle_discretization(
            n_qubits=n_qubits_disc,
            n_trials=n_trials_disc,
            grids=grids,
            results_path=str(results_dir / "angle_discretization.json"),
            seed=seed,
        )
        timings['angle_discretization'] = time.time() - start_time
        print(f"\n✓ Completed in {timings['angle_discretization']:.1f}s")
    except Exception as e:
        print(f"\n✗ Failed with error: {e}")
        timings['angle_discretization'] = None

    # 5. Geometry/Topology
    print("\n" + "=" * 80)
    print("5/5: Running Geometry/Topology Ablation")
    print("=" * 80)
    start_time = time.time()

    import geometry_topology_ablation as geo_topo

    # Use slightly larger systems for geometry study
    n_qubits_geo = n_qubits_list + [6] if not quick_mode else n_qubits_list

    try:
        geo_topo.run_geometry_topology(
            n_qubits_list=n_qubits_geo,
            n_trials=n_trials,
            results_path=str(results_dir / "geometry_topology.json"),
            seed=seed,
        )
        timings['geometry_topology'] = time.time() - start_time
        print(f"\n✓ Completed in {timings['geometry_topology']:.1f}s")
    except Exception as e:
        print(f"\n✗ Failed with error: {e}")
        timings['geometry_topology'] = None

    # Summary
    print("\n" + "=" * 80)
    print("All Ablation Studies Complete!")
    print("=" * 80)

    print("\nTimings:")
    total_time = 0
    for name, elapsed in timings.items():
        if elapsed is not None:
            print(f"  {name}: {elapsed:.1f}s")
            total_time += elapsed
        else:
            print(f"  {name}: FAILED")

    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} minutes)")

    print(f"\nResults saved to: {results_dir.resolve()}")
    print("\nNext steps:")
    print("  1. Open the Jupyter notebook to visualize results:")
    print("     cd ../notebooks && jupyter notebook energy_conserving_ablations.ipynb")
    print("  2. Or inspect the JSON files directly in the results/ directory")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run all energy-conserving decomposition ablation studies"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use reduced parameters for quick testing (fewer qubits, fewer trials)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility (default: 1234)"
    )

    args = parser.parse_args()

    run_all_ablations(quick_mode=args.quick, seed=args.seed)
