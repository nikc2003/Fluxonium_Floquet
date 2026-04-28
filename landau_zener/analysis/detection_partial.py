"""
landau_zener/analysis/detection_partial.py
find the minimum quasienergy gap between target branch pairs WITHOUT!! requiring
a full observable swap. show the min gap found within a user-specified chi window, even for partial transitions.
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np

from landau_zener.analysis.g2fit import fit_g2_quadratic_vertex
from landau_zener.floquet.sweep import FloquetSweep
from landau_zener.utils.units import wrap_to_bz, angular_to_ghz, angular_to_mhz, mhz_to_angular


def analyze_pair_partial_gap(
    sweep: FloquetSweep,
    branch_i: int,
    branch_j: int,
    *,
    chi_max_gap: Optional[float] = None,
    g2_fit_half_window: int = 20,
    gap_search_half_window_pts: int = 80,
    event_prefix: str = "PARTIAL",
) -> Dict[str, Any] | None:
    """
    the minimum quasienergy gap between two branches, (optionally) limited to
    chi <= chi_max_gap. Does NOT require a full observable swap. If chi_max_gap == None, s
    earches over the entire sweep range.
    """
    chi = np.asarray(sweep.chi, dtype=float)
    E = np.asarray(sweep.qe_wrapped, dtype=float)
    omega_d = float(sweep.omega_d)
    K, nb = E.shape

    if not (0 <= branch_i < nb and 0 <= branch_j < nb):
        return None
    #search range
    if chi_max_gap is not None and np.isfinite(chi_max_gap):
        search_mask = chi <= float(chi_max_gap)
        if not np.any(search_mask):
            search_mask = np.ones(K, dtype=bool)
    else:
        search_mask = np.ones(K, dtype=bool)

    search_indices = np.where(search_mask)[0]
    k_search_start = int(search_indices[0])
    k_search_end = int(search_indices[-1])
    delta_search = wrap_to_bz(
        E[k_search_start : k_search_end + 1, branch_j]
        - E[k_search_start : k_search_end + 1, branch_i],
        omega_d,
    )
    gap_search = np.abs(delta_search)
    k_min_local = int(np.argmin(gap_search))
    k_min = k_min_local + k_search_start
    Delta_min_raw = float(gap_search[k_min_local])

    hw = max(5, int(g2_fit_half_window))
    g2fit = fit_g2_quadratic_vertex(
        chi=chi,
        qe_wrapped=E,
        omega_d=omega_d,
        branch_i=int(branch_i),
        branch_j=int(branch_j),
        k_min=int(k_min),
        half_window=hw,
    )

    fit_ok = bool(g2fit["fit_ok"])
    if fit_ok:
        chi_star = float(g2fit["chi_star_fit"])
        s_g2 = float(g2fit["s_g2"])
        if not (np.isfinite(s_g2) and s_g2 > 0):
            fit_ok = False
    if fit_ok:
        chi_star = float(g2fit["chi_star_fit"])
        s_g2 = float(g2fit["s_g2"])
        Delta_min = Delta_min_raw  
        delta_chi = float(Delta_min / s_g2)
    else:
        chi_star = float(chi[k_min])
        s_g2 = float("nan")
        Delta_min = Delta_min_raw
        delta_chi = float("nan")

    freq_GHz = float(angular_to_ghz(omega_d))
    chi_star_MHz = float(angular_to_mhz(chi_star))
    Delta_min_MHz = float(angular_to_mhz(Delta_min))
    delta_chi_MHz = float(angular_to_mhz(delta_chi)) if np.isfinite(delta_chi) else float("nan")
    chi_max_gap_MHz = float(angular_to_mhz(chi_max_gap)) if (chi_max_gap is not None and np.isfinite(chi_max_gap)) else float("nan")
    chi_max_gap_tag = f"_cmaxgap{chi_max_gap_MHz:.0f}MHz" if np.isfinite(chi_max_gap_MHz) else ""
    event_id = (
        f"{event_prefix}_phi{float(sweep.flux):.3f}"
        f"_fd{freq_GHz:.6f}GHz"
        f"_b{int(branch_i)}_{int(branch_j)}"
        f"_cstar{chi_star_MHz:.3f}MHz"
        f"_gap{Delta_min_MHz:.3f}MHz"
        f"{chi_max_gap_tag}"
    )
    return dict(
        event_id=event_id,
        flux=float(sweep.flux),
        omega_d=float(omega_d),
        freq_GHz=freq_GHz,
        branch_i=int(branch_i),
        branch_j=int(branch_j),
        k_min=int(k_min),
        chi_star=float(chi_star),
        chi_star_MHz=chi_star_MHz,
        Delta_min=float(Delta_min),
        Delta_min_MHz=Delta_min_MHz,
        s_g2=float(s_g2),
        delta_chi=float(delta_chi),
        delta_chi_MHz=delta_chi_MHz,
        g2_r2=float(g2fit["g2_r2"]),
        g2_rel_rms=float(g2fit["g2_rel_rms"]),
        g2_fit_ok=fit_ok,
        chi_max_gap=float(chi_max_gap) if chi_max_gap is not None else float("nan"),
        chi_max_gap_MHz=chi_max_gap_MHz,
        k_search_start=int(k_search_start),
        k_search_end=int(k_search_end),
    )
