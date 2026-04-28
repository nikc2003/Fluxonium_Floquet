"""
landau_zener/analysis/detuning_fit.py
Linear fit of diabatic detuning slope away from the avoided crossing.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence
import numpy as np

from landau_zener.floquet.sweep import FloquetSweep, basis_weights
from landau_zener.utils.units import mhz_to_angular, wrap_to_bz, unwrap_series_centered


@dataclass
class LinearDetuningFitConfig:
    lin_scale_1: float = 1.5
    lin_scale_2: float = 4.0
    min_points: int = 4
    min_abs_dchi_MHz: float = 1.0
    method: str = "projection"  #projection/carrier
    fit_side: str = "both" #left/right/both
    max_carrier_switches: int = 2


def linear_fit_diabatic_detuning_for_bare_pair(
    sweep: FloquetSweep,
    *,
    bare_a: int,
    bare_b: int,
    chi_star: float,
    delta_chi: float,
    cfg: LinearDetuningFitConfig = LinearDetuningFitConfig(),
) -> Dict[str, Any]:
    """
    linear fit detuning(chi) far from crossing to detuning = slope*(chi-chi*) + offset.
    method:
      - if projection --> eps_dia_m(chi) = sum_b eps_b(chi) |<m|phi_b(chi)>|^2
      -if carrier --> track argmax-weight carrier branch for each bare level
    """
    chi = np.asarray(sweep.chi, dtype=float)
    E_u = np.asarray(sweep.qe_unwrapped, dtype=float)
    omega_d = float(sweep.omega_d)
    nchi = chi.size
    W = basis_weights(sweep.modes)
    _, nb, dim = W.shape

    empty = dict(x_fit=np.array([], dtype=float), y_fit=np.array([], dtype=float), y_model=np.array([], dtype=float))

    if not (0 <= bare_a < dim and 0 <= bare_b < dim):
        return dict(lin_ok=False, reason=f"bare indices out of range: a={bare_a}, b={bare_b}, dim={dim}", **empty)
    if bare_a == bare_b:
        return dict(lin_ok=False, reason="bare_a == bare_b", **empty)
    if not (np.isfinite(delta_chi) and delta_chi > 0):
        return dict(lin_ok=False, reason="invalid delta_chi", **empty)
    if not np.isfinite(chi_star):
        return dict(lin_ok=False, reason="invalid chi_star", **empty)

    method = cfg.method.strip().lower()

    if method == "projection":
        eps_a = np.sum(E_u * W[:, :, bare_a], axis=1)
        eps_b = np.sum(E_u * W[:, :, bare_b], axis=1)
        det = eps_b - eps_a
        center_idx = int(np.argmin(np.abs(chi - float(chi_star))))
        det_w = wrap_to_bz(det, omega_d)
        det_u = unwrap_series_centered(det_w, omega_d, center_idx)
        carrier_a = None
        carrier_b = None
        carrier_switches = None
    elif method == "carrier":
        carrier_a = np.argmax(W[:, :, bare_a], axis=1)
        carrier_b = np.argmax(W[:, :, bare_b], axis=1)
        det = E_u[np.arange(nchi), carrier_b] - E_u[np.arange(nchi), carrier_a]
        center_idx = int(np.argmin(np.abs(chi - float(chi_star))))
        det_w = wrap_to_bz(det, omega_d)
        det_u = unwrap_series_centered(det_w, omega_d, center_idx)
        carrier_switches = None
    else:
        return dict(lin_ok=False, reason=f"unknown method '{cfg.method}'", **empty)

    min_abs = mhz_to_angular(cfg.min_abs_dchi_MHz)
    dchi = chi - float(chi_star)
    dmin = np.maximum(cfg.lin_scale_1 * delta_chi, min_abs)
    dmax = np.maximum(cfg.lin_scale_2 * delta_chi, dmin + 1e-15)

    left_mask = (dchi <= -dmin) & (dchi >= -dmax)
    right_mask = (dchi >= +dmin) & (dchi <= +dmax)

    def fit_mask(mask):
        npts = int(np.count_nonzero(mask))
        if npts < int(cfg.min_points):
            return dict(ok=False, reason="too few points", npts=npts)
        x = dchi[mask]
        y = det_u[mask]
        A = np.vstack([x, np.ones_like(x)]).T
        slope, offset = np.linalg.lstsq(A, y, rcond=None)[0]
        yfit = slope * x + offset
        resid = y - yfit

        rms = float(np.sqrt(np.mean(resid**2)))
        denom = float(np.sqrt(np.mean(y**2))) if np.mean(y**2) > 0 else np.nan
        rel_rms = float(rms / denom) if np.isfinite(denom) and denom > 0 else np.nan

        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        ss_res = float(np.sum(resid**2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

        return dict(ok=True, m=float(slope), c=float(offset), s=float(abs(slope)), r2=r2, rel_rms=rel_rms,
                    x_fit=x, y_fit=y, y_model=yfit, npts=npts)

    outL = fit_mask(left_mask) if cfg.fit_side in ("left", "both") else dict(ok=False, reason="skipped")
    outR = fit_mask(right_mask) if cfg.fit_side in ("right", "both") else dict(ok=False, reason="skipped")

    if method == "carrier":
        mask_diag = left_mask | right_mask
        if np.count_nonzero(mask_diag) > 2:
            ca = carrier_a[mask_diag]
            cb = carrier_b[mask_diag]
            carrier_switches = int(np.count_nonzero(np.diff(ca) != 0) + np.count_nonzero(np.diff(cb) != 0))
        else:
            carrier_switches = 0

        if carrier_switches > int(cfg.max_carrier_switches):
            return dict(lin_ok=False, reason=f"carrier jumps in fit window (switches={carrier_switches})",
                        method=cfg.method, carrier_switches=carrier_switches, **empty)

    good = [r for r in (outL, outR) if r.get("ok", False)]
    if not good:
        return dict(lin_ok=False, reason="no successful side fits", method=cfg.method, left=outL, right=outR, **empty)

    s_vals = [g["s"] for g in good if np.isfinite(g["s"])]
    s_lin = float(np.median(s_vals)) if s_vals else np.nan

    xs, ys, yms = [], [], []
    if outL.get("ok", False):
        xs.append(outL["x_fit"]); ys.append(outL["y_fit"]); yms.append(outL["y_model"])
    if outR.get("ok", False):
        xs.append(outR["x_fit"]); ys.append(outR["y_fit"]); yms.append(outR["y_model"])

    x_fit = np.concatenate(xs)
    y_fit = np.concatenate(ys)
    order = np.argsort(x_fit)
    x_fit = x_fit[order]
    y_fit = y_fit[order]

    y_model = np.concatenate(yms)
    y_model = y_model[order]

    return dict(
        lin_ok=True,
        method=cfg.method,
        fit_side=cfg.fit_side,
        s_lin=float(s_lin),
        lin_r2=float(np.nanmin([g["r2"] for g in good])),
        lin_rel_rms=float(np.nanmax([g["rel_rms"] for g in good])),
        left=outL,
        right=outR,
        x_fit=x_fit,
        y_fit=y_fit,
        y_model=y_model,
        carrier_a=carrier_a,
        carrier_b=carrier_b,
        carrier_switches=carrier_switches,
    )