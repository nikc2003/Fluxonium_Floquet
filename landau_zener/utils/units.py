"""
landau_zener/utils/units.py
unit conversions, Brillouin-zone wrap/unwrap helpers.
"""

from __future__ import annotations
import numpy as np


def ghz_to_angular(x_ghz: float | np.ndarray) -> float | np.ndarray:
    """GHz -> rad/ns"""
    return 2.0 * np.pi * x_ghz
 

def mhz_to_angular(x_mhz: float | np.ndarray) -> float | np.ndarray:
    """MHz -> rad/ns"""
    return 2.0 * np.pi * (x_mhz / 1e3)


def angular_to_ghz(x: float | np.ndarray) -> float | np.ndarray:
    """rad/ns -> GHz"""
    return x / (2.0 * np.pi)


def angular_to_mhz(x: float | np.ndarray) -> float | np.ndarray:
    """rad/ns -> MHz"""
    return x / (2.0 * np.pi) * 1e3


def wrap_to_bz(delta: np.ndarray, omega_d: float) -> np.ndarray:
    """Wrap to first Brillouin zone: (-omega_d/2, omega_d/2]."""
    half = 0.5 * omega_d
    return (delta + half) % omega_d - half


def unwrap_series_centered(delta_wrapped: np.ndarray, omega_d: float, center_idx: int) -> np.ndarray:
    """
    Unwrap by adding multiples of omega_d, starting at center_idx and going outward.
    """
    y = np.array(delta_wrapped, copy=True, dtype=float)
    out = np.zeros_like(y)
    out[center_idx] = y[center_idx]

    # forward
    for k in range(center_idx + 1, len(y)):
        jump = y[k] - out[k - 1]
        m = np.round(jump / omega_d)
        out[k] = y[k] - m * omega_d

    # backward
    for k in range(center_idx - 1, -1, -1):
        jump = y[k] - out[k + 1]
        m = np.round(jump / omega_d)
        out[k] = y[k] - m * omega_d

    return out


def unwrap_branches_by_continuity(qe_wrapped: np.ndarray, omega_d: float) -> np.ndarray:
    """
    Unwrap each branch along chi by removing integer omega_d jumps.
    Starts from first chi point for each branch.
    """
    out = np.array(qe_wrapped, copy=True, dtype=float)
    K, nb = out.shape
    for j in range(nb):
        for k in range(1, K):
            jump = out[k, j] - out[k - 1, j]
            m = np.round(jump / omega_d)
            out[k, j] -= m * omega_d
    return out