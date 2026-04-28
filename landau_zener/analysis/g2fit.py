"""
landau_zener/analysis/g2fit.py
Fit g^2(chi) quadratic near avoided crossing to extract chi*, Delta, slope parameter s.
"""

from __future__ import annotations
from typing import Any, Dict
import numpy as np
from landau_zener.utils.units import wrap_to_bz, unwrap_series_centered


def fit_g2_quadratic_vertex(
    chi: np.ndarray,
    qe_wrapped: np.ndarray,
    omega_d: float,
    branch_i: int,
    branch_j: int,
    k_min: int,
    half_window: int,
) -> Dict[str, Any]:
    """
    g^2(chi) = (unwrap(wrap(eps_j - eps_i)))^2 to A chi^2 + B chi + C.
    for ideal two-level: g^2 = Delta^2 + s^2(chi-chi*)^2
      => A = s^2, chi* = -B/(2A), Delta^2 = C - B^2/(4A).
    """
    K = len(chi)
    k0 = max(0, k_min - int(half_window))
    k1 = min(K - 1, k_min + int(half_window))

    delta_w = wrap_to_bz(
        qe_wrapped[k0 : k1 + 1, branch_j] - qe_wrapped[k0 : k1 + 1, branch_i],
        float(omega_d),
    )
    center_local = k_min - k0
    delta_u = unwrap_series_centered(delta_w, float(omega_d), int(center_local))

    g2 = delta_u**2
    x = chi[k0 : k1 + 1]

    A_mat = np.vstack([x**2, x, np.ones_like(x)]).T
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, g2, rcond=None)
    A, B, C = [float(v) for v in coeffs]

    g2_fit = A * x**2 + B * x + C
    resid = g2 - g2_fit
    rms = float(np.sqrt(np.mean(resid**2)))
    denom = float(np.sqrt(np.mean(g2**2))) if np.mean(g2**2) > 0 else np.nan
    rel_rms = float(rms / denom) if np.isfinite(denom) and denom > 0 else np.nan

    ss_tot = float(np.sum((g2 - np.mean(g2)) ** 2))
    ss_res = float(np.sum(resid**2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    fit_ok = bool(np.isfinite(A) and A > 0)
    chi_star = float(-B / (2.0 * A)) if fit_ok else np.nan
    Delta2 = float(C - (B**2) / (4.0 * A)) if fit_ok else np.nan
    Delta_fit = float(np.sqrt(max(0.0, Delta2))) if np.isfinite(Delta2) else np.nan
    s_g2 = float(np.sqrt(A)) if fit_ok else np.nan

    #must lie inside window
    if fit_ok and not (float(np.min(x)) <= chi_star <= float(np.max(x))):
        fit_ok = False

    return dict(
        fit_ok=fit_ok,
        A=A,
        B=B,
        C=C,
        chi_star_fit=chi_star,
        Delta_fit=Delta_fit,
        s_g2=s_g2,
        g2_rel_rms=rel_rms,
        g2_r2=r2,
        k0=k0,
        k1=k1,
    )