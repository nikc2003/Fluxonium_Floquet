"""
landau_zener/analysis/detection_nbar_crossing.py
locate the partial transition point as the nbar crossing between two
target branches.  The crossing is where dnbar = nbar_i - nbar_j = 0,
corresponding to 50/50 hybridization (equal bare-level character).

output tthe quasienergy gap at the nbar crossing chi, which should
approximately coincide with the minimum gap location (if it is well approximated by a simple avoided crossing).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from landau_zener.analysis.g2fit import fit_g2_quadratic_vertex
from landau_zener.floquet.sweep import FloquetSweep
from landau_zener.utils.units import wrap_to_bz, angular_to_ghz, angular_to_mhz


def _find_nbar_crossing(
    chi: np.ndarray,
    dnbar: np.ndarray,
    k_start: int = 0,
    k_end: Optional[int] = None,
) -> Dict[str, Any]:
    """where dnbar crosses zero via linear interpolation.
    Arguments:
    1.chi --> array (nchi,)
    2. dnbar --> array (nchi,); the difference nbar_i - nbar_j
    3. k_start, k_end --> indices where to look/search
    Returns:
    dict with chi_cross_nbar, k_cross, crossing_found, closest_dnbar
    """
    if k_end is None:
        k_end = len(chi) - 1
    seg = dnbar[k_start : k_end + 1]
    chi_seg = chi[k_start : k_end + 1]
    signs = np.sign(seg)
    for t in range(len(seg) - 1):
        if signs[t] == 0:
            return dict(
                chi_cross_nbar=float(chi_seg[t]),
                k_cross=int(k_start + t),
                crossing_found=True,
                closest_dnbar=0.0,
            )
        if signs[t] * signs[t + 1] < 0:
            y0, y1 = float(seg[t]), float(seg[t + 1])
            x0, x1 = float(chi_seg[t]), float(chi_seg[t + 1])
            denom = y0 - y1
            frac = y0 / denom if denom != 0 else 0.5
            frac = float(np.clip(frac, 0.0, 1.0))
            chi_cross = x0 + frac * (x1 - x0)
            k_cross = k_start + t if frac < 0.5 else k_start + t + 1
            return dict(
                chi_cross_nbar=float(chi_cross),
                k_cross=int(k_cross),
                crossing_found=True,
                closest_dnbar=0.0,
            )

    # if there is no corssing, then report closest approach
    k_closest = int(np.argmin(np.abs(seg))) + k_start
    return dict(
        chi_cross_nbar=float(chi[k_closest]),
        k_cross=int(k_closest),
        crossing_found=False,
        closest_dnbar=float(dnbar[k_closest]),
    )


def analyze_pair_nbar_crossing(
    sweep: FloquetSweep,
    branch_i: int,
    branch_j: int,
    *,
    g2_fit_half_window: int = 20,
    event_prefix: str = "NBAR_CROSS",
) -> Dict[str, Any] | None:
    """
    grand purpose: find the transition point via the nbar crossing.

        transition point is defined as the chi where
        nbar[branch_i] == nbar[branch_j], so basically where ``dnbar = 0``.
        whcih means 50/50 hybridization between the two branches.
        Also reports the quasienergy gap at the crossing chi (which should
        approximately align with the minimum gap from the quasienergy picture).
        optionanly, we can do a g squared quadratic fit around the gap minimum nearest
        the nbar crossing.
    """
    chi = np.asarray(sweep.chi, dtype=float)
    nbar = np.asarray(sweep.nbar, dtype=float)
    E = np.asarray(sweep.qe_wrapped, dtype=float)
    omega_d = float(sweep.omega_d)
    K, nb = E.shape
    if not (0 <= branch_i < nb and 0 <= branch_j < nb):
        return None
    dnbar = nbar[:, branch_i] - nbar[:, branch_j]
    cross = _find_nbar_crossing(chi, dnbar)
    chi_cross = float(cross["chi_cross_nbar"])
    k_cross = int(cross["k_cross"])
    crossing_found = bool(cross["crossing_found"])
    delta_at_cross = wrap_to_bz(
        E[k_cross, branch_j] - E[k_cross, branch_i], omega_d
    )
    gap_at_cross = float(np.abs(delta_at_cross))
    delta_all = wrap_to_bz(E[:, branch_j] - E[:, branch_i], omega_d)
    gap_all = np.abs(delta_all)
    k_min_gap = int(np.argmin(gap_all))
    gap_min_global = float(gap_all[k_min_gap])
    hw = max(5, int(g2_fit_half_window))
    g2fit = fit_g2_quadratic_vertex(
        chi=chi,
        qe_wrapped=E,
        omega_d=omega_d,
        branch_i=int(branch_i),
        branch_j=int(branch_j),
        k_min=int(k_cross),
        half_window=hw,
    )

    fit_ok = bool(g2fit["fit_ok"])
    if fit_ok:
        s_g2 = float(g2fit["s_g2"])
        if not (np.isfinite(s_g2) and s_g2 > 0):
            fit_ok = False

    if fit_ok:
        chi_star_g2 = float(g2fit["chi_star_fit"])
        s_g2 = float(g2fit["s_g2"])
        delta_chi = float(gap_at_cross / s_g2)
    else:
        chi_star_g2 = float("nan")
        s_g2 = float("nan")
        delta_chi = float("nan")

    freq_GHz = float(angular_to_ghz(omega_d))
    chi_cross_MHz = float(angular_to_mhz(chi_cross))
    gap_at_cross_MHz = float(angular_to_mhz(gap_at_cross))
    gap_min_global_MHz = float(angular_to_mhz(gap_min_global))
    delta_chi_MHz = float(angular_to_mhz(delta_chi)) if np.isfinite(delta_chi) else float("nan")
    nbar_i_at_cross = float(nbar[k_cross, branch_i])
    nbar_j_at_cross = float(nbar[k_cross, branch_j])
    nbar_i_start = float(nbar[0, branch_i])
    nbar_j_start = float(nbar[0, branch_j])
    nbar_i_end = float(nbar[-1, branch_i])
    nbar_j_end = float(nbar[-1, branch_j])
    #how much of the swap has completed: 0 = no swap, 1 = full swap
    nbar_span_start = abs(nbar_i_start - nbar_j_start)
    nbar_span_end = abs(nbar_i_end - nbar_j_end)
    swap_fraction = 1.0 - nbar_span_end / nbar_span_start if nbar_span_start > 1e-10 else float("nan")
    event_id = (
        f"{event_prefix}_phi{float(sweep.flux):.3f}"
        f"_fd{freq_GHz:.6f}GHz"
        f"_b{int(branch_i)}_{int(branch_j)}"
        f"_chicross{chi_cross_MHz:.3f}MHz"
        f"_gap{gap_at_cross_MHz:.3f}MHz"
    )
    return dict(
        event_id=event_id,
        flux=float(sweep.flux),
        omega_d=float(omega_d),
        freq_GHz=freq_GHz,
        branch_i=int(branch_i),
        branch_j=int(branch_j),
        #nbar crossing point 
        chi_cross=float(chi_cross),
        chi_cross_MHz=chi_cross_MHz,
        chi_cross_nbar=float(chi_cross),
        chi_cross_nbar_MHz=chi_cross_MHz,
        k_cross=int(k_cross),
        crossing_found=crossing_found,
        closest_dnbar=float(cross["closest_dnbar"]),
        # gap at the nbar crossing
        gap_at_cross=float(gap_at_cross),
        gap_at_cross_MHz=gap_at_cross_MHz,
        # global minimum gap 
        gap_min_global=float(gap_min_global),
        gap_min_global_MHz=gap_min_global_MHz,
        k_min_gap=int(k_min_gap),
        chi_at_min_gap=float(chi[k_min_gap]),
        chi_at_min_gap_MHz=float(angular_to_mhz(chi[k_min_gap])),
        # chi_star and chi_cross should be close, show and report the offset
        chi_gap_vs_cross_offset_MHz=float(angular_to_mhz(chi[k_min_gap]) - chi_cross_MHz),
        # g2 fit (centered on nbar crossing)
        chi_star=float(chi_cross), #use nbar crossing as the event chi_star
        chi_star_MHz=chi_cross_MHz,
        Delta_min=float(gap_at_cross),  #gap at crossing 
        Delta_min_MHz=gap_at_cross_MHz,
        s_g2=float(s_g2),
        delta_chi=float(delta_chi),
        delta_chi_MHz=delta_chi_MHz,
        g2_r2=float(g2fit["g2_r2"]),
        g2_rel_rms=float(g2fit["g2_rel_rms"]),
        g2_fit_ok=fit_ok,
        chi_star_g2=float(chi_star_g2),
        chi_star_g2_MHz=float(angular_to_mhz(chi_star_g2)) if np.isfinite(chi_star_g2) else float("nan"),
        #nbar diagnostics
        nbar_i_at_cross=nbar_i_at_cross,
        nbar_j_at_cross=nbar_j_at_cross,
        nbar_i_start=nbar_i_start,
        nbar_j_start=nbar_j_start,
        nbar_i_end=nbar_i_end,
        nbar_j_end=nbar_j_end,
        swap_fraction=float(swap_fraction),
        #search window
        k_search_start=0,
        k_search_end=int(K - 1),
    )
