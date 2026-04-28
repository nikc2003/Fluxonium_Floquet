"""
landau_zener/analysis/slope_target.py
small chi diabatic slope extraction + candidate chi-window prediction (targeting resonances).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Sequence, Tuple
import numpy as np

import floquet as ft

from landau_zener.floquet.sweep import floquet_sweep_vs_chi, diabatic_energies_floquet
from landau_zener.utils.units import mhz_to_angular


def fit_linear_model(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit y[:,m] =  intercept[m] + slope[m]*x
    y is (N, M).
    """
    mean_x = np.mean(x)
    x0 = x - mean_x
    denom = np.dot(x0, x0)
    M = y.shape[1]
    if denom < 1e-18:
        return np.full(M, np.nan), np.full(M, np.nan), np.full(M, np.nan)

    mean_y = np.mean(y, axis=0)
    y0 = y - mean_y[None, :]
    slope = np.sum(x0[:, None] * y0, axis=0) / denom
    intercept = mean_y - slope * mean_x

    yfit = intercept[None, :] + slope[None, :] * x[:, None]
    ss_res = np.sum((y - yfit) ** 2, axis=0)
    ss_tot = np.sum((y - mean_y[None, :]) ** 2, axis=0)
    r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)
    return intercept, slope, r2


@dataclass
class SlopeConfig:
    chi_fit_max_MHz: float = 20.0
    nchi_fit: int = 41
    bare_a_levels: Sequence[int] = (0, 1, 2)
    bare_b_levels: Optional[Sequence[int]] = None
    photon_num: int = 1
    gap_max_MHz: float = 250.0
    window_factor: float = 8.0
    min_window_MHz: float = 2.0
    min_slope_diff: float = 1e-4
    max_candidates: int = 200


@dataclass
class WindowCand:
    bare_a: int
    bare_b: int
    n_photon: int
    chi_pred: float
    half_window: float
    sdiff: float
    detuning_0: float


def low_chi_diabatic_slopes(
    *,
    dim_q: int,
    qubit_params: Dict[str, Any],
    flux: float,
    omega_d: float,
    options: ft.Options,
    slope_cfg: SlopeConfig,
) -> Dict[str, Any]:
    """
    Run small-chi sweep and fit diabatic energies to eps_m(chi) = eps0[m] + slope[m]*chi.
    """
    chi_fit = mhz_to_angular(np.linspace(0.0, slope_cfg.chi_fit_max_MHz, slope_cfg.nchi_fit))
    sweep_low = floquet_sweep_vs_chi(
        dim_q=dim_q,
        qubit_params=qubit_params,
        flux=flux,
        omega_d=omega_d,
        chi_vals=chi_fit,
        options=options,
    )
    eps_dia = diabatic_energies_floquet(sweep_low) 
    intercept, slope, r2 = fit_linear_model(chi_fit, eps_dia)
    return dict(sweep_low_power=sweep_low, eps0=intercept, slopes=slope, r2=r2)


def predict_candidate_windows(
    *,
    eps0: np.ndarray,
    slopes: np.ndarray,
    omega_d: float,
    chi_min: float,
    chi_max: float,
    cfg: SlopeConfig,
) -> List[WindowCand]:
    """
    Predict candidate chi windows for many pairs using low-chi linear model.
    """
    dim = len(eps0)
    bare_a_levels = list(cfg.bare_a_levels)
    bare_b_levels = list(range(dim)) if cfg.bare_b_levels is None else [
        int(b) for b in cfg.bare_b_levels if 0 <= int(b) < dim
    ]

    gap_threshold = mhz_to_angular(cfg.gap_max_MHz)
    min_half = mhz_to_angular(cfg.min_window_MHz)

    cands: List[WindowCand] = []
    for a in bare_a_levels:
        for b in bare_b_levels:
            if a == b:
                continue
            s_diff = slopes[b] - slopes[a]
            if abs(s_diff) < float(cfg.min_slope_diff):
                continue

            det0 = eps0[b] - eps0[a]
            n_guess = int(np.round(det0 / omega_d))

            for n in range(n_guess - int(cfg.photon_num), n_guess + int(cfg.photon_num) + 1):
                chi_pred = (n * omega_d - det0) / s_diff
                if not (np.isfinite(chi_pred) and (chi_min <= chi_pred <= chi_max)):
                    continue

                half = cfg.window_factor * gap_threshold / max(abs(s_diff), cfg.min_slope_diff)
                half = max(half, min_half)

                cands.append(WindowCand(
                    bare_a=int(a),
                    bare_b=int(b),
                    n_photon=int(n),
                    chi_pred=float(chi_pred),
                    half_window=float(half),
                    sdiff=float(s_diff),
                    detuning_0=float(det0),
                ))

    cands.sort(key=lambda c: c.chi_pred)
    if len(cands) > int(cfg.max_candidates):
        cands = cands[: int(cfg.max_candidates)]
    return cands


def predict_candidate_windows_for_pair(
    *,
    eps0: np.ndarray,
    slopes: np.ndarray,
    omega_d: float,
    chi_min: float,
    chi_max: float,
    cfg: SlopeConfig,
    bare_a: int,
    bare_b: int,
    n_photon: Optional[int] = None,
) -> List[WindowCand]:
    """
    Predict chi windows only for a single bare pair (bare_a, bare_b).
    (opyional) lock to a specific n_photon (e.g., 2 for 0->4 two-photon).
    """
    dim = len(eps0)
    a = int(bare_a); b = int(bare_b)
    if not (0 <= a < dim and 0 <= b < dim) or a == b:
        return []

    s_diff = slopes[b] - slopes[a]
    if abs(s_diff) < float(cfg.min_slope_diff):
        return []

    det0 = eps0[b] - eps0[a]
    n_guess = int(np.round(det0 / omega_d))

    ns: List[int]
    if n_photon is not None:
        ns = [int(n_photon)]
    else:
        ns = list(range(n_guess - int(cfg.photon_num), n_guess + int(cfg.photon_num) + 1))

    gap_threshold = mhz_to_angular(cfg.gap_max_MHz)
    min_half = mhz_to_angular(cfg.min_window_MHz)

    out: List[WindowCand] = []
    for n in ns:
        chi_pred = (n * omega_d - det0) / s_diff
        if not (np.isfinite(chi_pred) and (chi_min <= chi_pred <= chi_max)):
            continue
        half = cfg.window_factor * gap_threshold / max(abs(s_diff), cfg.min_slope_diff)
        half = max(half, min_half)
        out.append(WindowCand(
            bare_a=a, bare_b=b, n_photon=int(n),
            chi_pred=float(chi_pred),
            half_window=float(half),
            sdiff=float(s_diff),
            detuning_0=float(det0),
        ))
    out.sort(key=lambda c: c.chi_pred)
    return out