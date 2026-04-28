"""
landau_zener/physics/ringup.py
resoantor ring up model chi_dot(chi) used for Landau Zener velocity estimates.
"""

from __future__ import annotations
import numpy as np


def chi_dot_ringup(chi: np.ndarray, chi_max: float, kappa: float) -> np.ndarray:
    """
    Standard resonator ring-up model: 
    - chi is proportional to photon number.
    - and kappa sets linewidth and ring-up time.
    """
    chi = np.array(chi, dtype=float)
    chi_max_eff = max(float(chi_max), 1e-18)
    p = np.clip(chi / chi_max_eff, 0.0, 1.0)
    y = 1.0 - np.sqrt(p)
    return chi_max_eff * kappa * y * (1.0 - y)