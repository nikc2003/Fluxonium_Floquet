"""Fluxonium-resonator branch analysis package.

This package implements:
- bare fluxonium diagonalization in the EC-EL harmonic-oscillator basis,
- a capacitively coupled single-mode resonator,
- recursive dressed-state branch assignment,
- branch diagnostics useful for fluxonium readout analysis,
- optional driven Hamiltonian / Lindblad RHS utilities for small systems.
"""

from .units import ghz_to_angular, mhz_to_angular, angular_to_ghz, angular_to_mhz
from .parameters import FluxoniumParams, ResonatorParams, TruncationConfig, DriveParams
from .fluxonium_model import BareFluxonium
from .composite_system import CoupledFluxoniumResonator
from .branch_analysis import BranchAnalyzer, BranchAnalysisResult
from .diagnostics import (
    state_distributions,
    reduced_fluxonium_density_matrix,
    reduced_fluxonium_purity,
    state_metrics,
    branch_metrics,
    detect_instabilities,
)
