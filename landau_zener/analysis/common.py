"""
landau_zener/analysis/common.py
Helpers for detection/metrics (local minima, edges, sign-crossing).
"""

from __future__ import annotations
import numpy as np
from landau_zener.floquet.sweep import FloquetSweep

def local_minima_indices(y: np.ndarray) -> np.ndarray:
    """
    find indices k where y[k] is a local minimum.
      - includes k=0 if y[0] <= y[1]
      - includes k=n-1 if y[n-1] < y[n-2]
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)

    mins = []

    if y[0] <= y[1]:
        mins.append(0)

    for k in range(1, n - 1):
        if (y[k] < y[k - 1]) and (y[k] <= y[k + 1]):
            mins.append(k)

    if y[-1] < y[-2]:
        mins.append(n - 1)

    return np.array(mins, dtype=int)


def pick_observable(sweep: FloquetSweep, which: str) -> np.ndarray:
    w = which.strip().lower()
    if w == "nbar":
        return sweep.nbar
    if w == "avg_metric_raw":
        return sweep.avg_metric_raw
    raise ValueError(f"Unknown observable '{which}'. Use 'nbar' or 'avg_metric_raw'.")


def compute_edges_from_delta_chi(
    chi: np.ndarray,
    k_center: int,
    chi_star: float,
    delta_chi: float,
    edge_factor: float,
    min_edge_offset_pts: int,
) -> tuple[int, int]:
    nchi = len(chi)
    if np.isfinite(delta_chi) and delta_chi > 0:
        chiL = float(chi_star - edge_factor * delta_chi)
        chiR = float(chi_star + edge_factor * delta_chi)
        kL = int(np.argmin(np.abs(chi - chiL)))
        kR = int(np.argmin(np.abs(chi - chiR)))
    else:
        kL = max(0, k_center - min_edge_offset_pts)
        kR = min(nchi - 1, k_center + min_edge_offset_pts)

    if kL >= kR:
        kL = max(0, k_center - min_edge_offset_pts)
        kR = min(nchi - 1, k_center + min_edge_offset_pts)

    kL = int(np.clip(kL, 0, nchi - 1))
    kR = int(np.clip(kR, 0, nchi - 1))
    if kL == kR:
        kR = min(nchi - 1, kL + 1)
    return kL, kR


def estimate_crossing_chi_from_sign_change(chi: np.ndarray, d: np.ndarray, kL: int, kR: int) -> float:
    """
    Find chi where d crosses 0 between kL..kR by linear interpolation using the first sign change.
    """
    seg = d[kL : kR + 1]
    s = np.sign(seg)
    for t in range(len(seg) - 1):
        if s[t] == 0:
            return float(chi[kL + t])
        if s[t] * s[t + 1] < 0:
            y0 = float(seg[t])
            y1 = float(seg[t + 1])
            x0 = float(chi[kL + t])
            x1 = float(chi[kL + t + 1])
            frac = y0 / (y0 - y1) if (y0 - y1) != 0 else 0.5
            frac = float(np.clip(frac, 0.0, 1.0))
            return x0 + frac * (x1 - x0)

    return float(chi[int(np.argmin(np.abs(seg)) + kL)])