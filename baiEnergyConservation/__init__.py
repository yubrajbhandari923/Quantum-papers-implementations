"""Energy-conserving unitary decomposition package.

This package provides tools for decomposing energy-conserving unitaries
following the approach from arXiv:2309.11051.
"""

from .implementation import (
    EnergyConservingDecompositionPass,
    decompose_energy_conserving_unitary,
    is_energy_conserving,
    verify_decomposition,
    SqrtISWAPGate,
    SqrtISWAPdgGate,
)

__all__ = [
    "EnergyConservingDecompositionPass",
    "decompose_energy_conserving_unitary",
    "is_energy_conserving",
    "verify_decomposition",
    "SqrtISWAPGate",
    "SqrtISWAPdgGate",
]
