"""classes for the physical parameters and truncations."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class FluxoniumParams:
    """
    H = 4 E_C n^2 + (1/2) E_L phi^2 - E_J cos(phi + 2*pi*flux), where flux=Phi_ext/Phi_0.
    """
    EJ_GHz: float
    EC_GHz: float
    EL_GHz: float
    flux: float
    cutoff: int = 110

@dataclass(frozen=True)
class ResonatorParams:
    """res parameters."""
    omega_r_GHz: float
    g_qr_GHz: float
    kappa_MHz: float = 5.0

@dataclass(frozen=True)
class TruncationConfig:
    """subspace truncations in the composite system"""
    fluxonium_levels: int = 25
    resonator_levels: int = 65

@dataclass(frozen=True)
class DriveParams:
    """Resonator drive parameters
    The drive Hamiltonian is H_d(t) = -i eps_d sin(omega_d t) (a - a^dagger)
    """
    epsilon_d_GHz: float
    omega_d_GHz: float
