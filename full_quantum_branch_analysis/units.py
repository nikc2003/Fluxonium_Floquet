"""unit conversion helpers.
note that all of the Hamiltonians are built in ang frequency units.
"""

from __future__ import annotations
import numpy as np

TWOPI = 2.0 * np.pi

def ghz_to_angular(value_ghz: float | np.ndarray) -> float | np.ndarray:
    """Convert frequency in GHz to angular frequency units."""
    return TWOPI * value_ghz

def mhz_to_angular(value_mhz: float | np.ndarray) -> float | np.ndarray:
    """Convert frequency in MHz to angular frequency units."""
    return TWOPI * 1e-3 * value_mhz

def angular_to_ghz(value_ang: float | np.ndarray) -> float | np.ndarray:
    """Convert angular frequency to GHz."""
    return value_ang / TWOPI

def angular_to_mhz(value_ang: float | np.ndarray) -> float | np.ndarray:
    """Convert angular frequency to MHz."""
    return 1e3 * value_ang / TWOPI
