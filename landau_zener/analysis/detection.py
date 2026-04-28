"""
landau_zener/analysis/detection.py
hybridizations between Floquet branches and reference-level swaps.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from landau_zener.analysis.common import (
    local_minima_indices,
    pick_observable,
    compute_edges_from_delta_chi,
    estimate_crossing_chi_from_sign_change,
)
from landau_zener.analysis.g2fit import fit_g2_quadratic_vertex
from landau_zener.floquet.sweep import FloquetSweep
from landau_zener.utils.units import wrap_to_bz, angular_to_ghz, angular_to_mhz, mhz_to_angular
from landau_zener.utils.io import fmt_id_num


@dataclass
class BranchHybridizationDetectConfig:
    gap_max_MHz: float = 250.0
    g2_fit_half_window: int = 20
    observable: str = "nbar"
    obs_edge_factor: float = 4.0
    min_edge_offset_pts: int = 12
    obs_swap_min_diff: float = 0.75
    min_separation_pts: int = 10


def detect_branch_hybridizations(
    sweep: FloquetSweep,
    cfg: BranchHybridizationDetectConfig,
    *,
    restrict_branches: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Detect hybridization events between branch pairs (i<j) based on:
      1) local minimum of |wrap(eps_j - eps_i)| below gap threshold
      2) observable swap: O_i - O_j changes sign across edges around chi*
    """
    chi = np.asarray(sweep.chi, dtype=float)
    E = np.asarray(sweep.qe_wrapped, dtype=float)
    omega_d = float(sweep.omega_d)
    O = pick_observable(sweep, cfg.observable)

    K, nb = E.shape
    branches = list(range(nb)) if restrict_branches is None else sorted(set(
        int(b) for b in restrict_branches if 0 <= int(b) < nb
    ))

    gap_thresh = float(mhz_to_angular(cfg.gap_max_MHz))
    events: List[Dict[str, Any]] = []
    last_kept_min: Dict[tuple[int, int], int] = {}

    for ii, bi in enumerate(branches):
        for bj in branches[ii + 1:]:
            delta = wrap_to_bz(E[:, bj] - E[:, bi], omega_d)
            gap = np.abs(delta)
            mins = local_minima_indices(gap)
            if mins.size == 0:
                continue

            for k0 in mins:
                if float(gap[k0]) > gap_thresh:
                    continue

                key = (bi, bj)
                if key in last_kept_min and abs(int(k0) - int(last_kept_min[key])) < int(cfg.min_separation_pts):
                    continue

                hw = max(3, int(cfg.g2_fit_half_window))
                k_left = max(0, int(k0) - hw)
                k_right = min(K - 1, int(k0) + hw)
                k_min = int(np.argmin(gap[k_left:k_right + 1]) + k_left)
                Delta_min = float(gap[k_min])

                g2fit = fit_g2_quadratic_vertex(
                    chi=chi, qe_wrapped=E, omega_d=omega_d,
                    branch_i=int(bi), branch_j=int(bj),
                    k_min=int(k_min), half_window=hw,
                )
                if not g2fit["fit_ok"]:
                    continue

                chi_star = float(g2fit["chi_star_fit"])
                s_g2 = float(g2fit["s_g2"])
                if not (np.isfinite(s_g2) and s_g2 > 0):
                    continue

                delta_chi = float(Delta_min / s_g2)

                kL, kR = compute_edges_from_delta_chi(
                    chi, k_min, chi_star, delta_chi,
                    edge_factor=float(cfg.obs_edge_factor),
                    min_edge_offset_pts=int(cfg.min_edge_offset_pts),
                )

                d = O[:, bi] - O[:, bj]
                dL = float(d[kL])
                dR = float(d[kR])

                if not (dL * dR < 0):
                    continue
                if not (abs(dL) >= cfg.obs_swap_min_diff and abs(dR) >= cfg.obs_swap_min_diff):
                    continue

                chi_cross = estimate_crossing_chi_from_sign_change(chi, d, kL, kR)

                freq_GHz = float(angular_to_ghz(omega_d))
                chi_star_MHz = float(angular_to_mhz(chi_star))
                chi_cross_MHz = float(angular_to_mhz(chi_cross))
                gap_MHz = float(angular_to_mhz(Delta_min))
                dchi_MHz = float(angular_to_mhz(delta_chi))

                event_id = (
                    f"BR_phi{float(sweep.flux):.3f}"
                    f"_fd{freq_GHz:.6f}GHz"
                    f"_b{int(bi)}_{int(bj)}"
                    f"_cstar{fmt_id_num(chi_star_MHz, 3)}MHz"
                    f"_ccross{fmt_id_num(chi_cross_MHz, 3)}MHz"
                    f"_gap{fmt_id_num(gap_MHz, 3)}MHz"
                    f"_dchi{fmt_id_num(dchi_MHz, 3)}MHz"
                )

                events.append(dict(
                    event_type="branch",
                    swap_observable=str(cfg.observable),
                    event_id=event_id,
                    flux=float(sweep.flux),
                    omega_d=float(omega_d),
                    freq_GHz=freq_GHz,
                    branch_i=int(bi),
                    branch_j=int(bj),
                    k_min=int(k_min),
                    chi_star=float(chi_star),
                    chi_cross=float(chi_cross),
                    Delta_min=float(Delta_min),
                    Delta_min_MHz=float(angular_to_mhz(Delta_min)),
                    s_g2=float(s_g2),
                    delta_chi=float(delta_chi),
                    delta_chi_MHz=float(angular_to_mhz(delta_chi)),
                    g2_r2=float(g2fit["g2_r2"]),
                    g2_rel_rms=float(g2fit["g2_rel_rms"]),
                    obs_d_left=float(dL),
                    obs_d_right=float(dR),
                    kL=int(kL),
                    kR=int(kR),
                ))
                last_kept_min[key] = int(k_min)

    return events
@dataclass
class ReferenceHybridizationConfig:
    ref_levels: Sequence[int] = (0, 1, 2)
    min_carrier_weight: float = 0.15
    hysteresis: float = 0.05
    gap_search_half_window_pts: int = 30
    g2_fit_half_window: int = 12
    partner_edge_factor: float = 4.0
    partner_min_edge_offset_pts: int = 12


def detect_reference_level_swaps(
    sweep: FloquetSweep,
    ref_level: int,
    cfg: ReferenceHybridizationConfig,
) -> List[Dict[str, Any]]:
    """
    which branch carries bare level (==ref_level) by argmax weight.
    """
    from landau_zener.floquet.sweep import basis_weights
    W = basis_weights(sweep.modes)
    chi = sweep.chi
    K, nb, dim = W.shape
    w_ref = W[:, :, int(ref_level)]
    carrier = np.argmax(w_ref, axis=1)

    cands: List[Dict[str, Any]] = []
    current = int(carrier[0])
    for k in range(K - 1):
        nxt = int(carrier[k + 1])
        if nxt == current:
            continue

        bi = current
        bj = nxt
        wi0, wj0 = float(w_ref[k, bi]), float(w_ref[k, bj])
        wi1, wj1 = float(w_ref[k + 1, bi]), float(w_ref[k + 1, bj])
        d0 = wi0 - wj0
        d1 = wi1 - wj1

        if cfg.hysteresis > 0:
            passed = (d0 > cfg.hysteresis) and (d1 < -cfg.hysteresis)
        else:
            passed = (d0 > 0) and (d1 < 0)
        if not passed:
            current = nxt
            continue

        if max(wi0, wj0, wi1, wj1) < float(cfg.min_carrier_weight):
            current = nxt
            continue

        frac = d0 / (d0 - d1) if (d0 - d1) != 0 else 0.5
        frac = float(np.clip(frac, 0.0, 1.0))
        chi_cross = float(chi[k] + frac * (chi[k + 1] - chi[k]))

        cands.append(dict(
            event_type="ref",
            swap_observable="ref_weight",
            ref_level=int(ref_level),
            branch_i=int(bi),
            branch_j=int(bj),
            k_hint=int(k),
            chi_cross=float(chi_cross),
        ))

        current = nxt

    return cands


def analyze_pair_event_near_k(
    sweep: FloquetSweep,
    branch_i: int,
    branch_j: int,
    k_center: int,
    *,
    g2_fit_half_window: int,
    gap_search_half_window_pts: int,
    event_prefix: str,
    ref_level: int | None = None,
) -> Dict[str, Any] | None:
    """
    for a sweep and candidate pair (i,j) near k_center,
      - find local minimum of gap in k_center +/- gap_search_half_window_pts
      - fit g^2 to get chi_star, s_g2, Delta_fit
    """
    chi = sweep.chi
    E = sweep.qe_wrapped
    omega_d = float(sweep.omega_d)
    K, nb = E.shape

    k0 = max(0, int(k_center) - int(gap_search_half_window_pts))
    k1 = min(K - 1, int(k_center) + int(gap_search_half_window_pts))

    delta = wrap_to_bz(E[k0:k1 + 1, branch_j] - E[k0:k1 + 1, branch_i], omega_d)
    gap = np.abs(delta)
    k_min = int(np.argmin(gap) + k0)

    Delta_min = float(np.abs(wrap_to_bz(E[k_min, branch_j] - E[k_min, branch_i], omega_d)))

    hw = max(5, int(g2_fit_half_window))
    g2fit = fit_g2_quadratic_vertex(
        chi=chi, qe_wrapped=E, omega_d=omega_d,
        branch_i=int(branch_i), branch_j=int(branch_j),
        k_min=int(k_min), half_window=hw,
    )
    if not g2fit["fit_ok"]:
        return None

    chi_star = float(g2fit["chi_star_fit"])
    s_g2 = float(g2fit["s_g2"])
    if not (np.isfinite(s_g2) and s_g2 > 0):
        return None

    delta_chi = float(Delta_min / s_g2)
    freq_GHz = float(angular_to_ghz(omega_d))

    base = f"{event_prefix}_phi{float(sweep.flux):.3f}_fd{freq_GHz:.3f}_b{int(branch_i)}_{int(branch_j)}_c{chi_star:.6e}"
    if ref_level is not None:
        base = f"{event_prefix}R{int(ref_level)}_" + base

    return dict(
        event_id=base,
        flux=float(sweep.flux),
        omega_d=float(omega_d),
        freq_GHz=freq_GHz,
        branch_i=int(branch_i),
        branch_j=int(branch_j),
        k_min=int(k_min),
        chi_star=float(chi_star),
        chi_star_MHz=float(angular_to_mhz(chi_star)),
        Delta_min=float(Delta_min),
        Delta_min_MHz=float(angular_to_mhz(Delta_min)),
        s_g2=float(s_g2),
        delta_chi=float(delta_chi),
        delta_chi_MHz=float(angular_to_mhz(delta_chi)),
        g2_r2=float(g2fit["g2_r2"]),
        g2_rel_rms=float(g2fit["g2_rel_rms"]),
    )


def infer_partner_bare_level_for_ref_swap(
    sweep: FloquetSweep,
    *,
    ref_level: int,
    branch_i: int,
    branch_j: int,
    chi_star: float,
    delta_chi: float,
    edge_factor: float,
    min_edge_offset_pts: int,
) -> Dict[str, Any]:
    """
    Infer partner bare level for a ref swap: look at the "other branch" bare dominance
    on left/right edges, pick partner with max min(weight_left, weight_right).
    """
    from landau_zener.floquet.sweep import basis_weights

    chi = sweep.chi
    W = basis_weights(sweep.modes)
    K, nb, dim = W.shape

    k_center = int(np.argmin(np.abs(chi - float(chi_star))))
    kL, kR = compute_edges_from_delta_chi(
        chi, k_center, float(chi_star), float(delta_chi),
        edge_factor=float(edge_factor),
        min_edge_offset_pts=int(min_edge_offset_pts),
    )

    def carrier_among_pair(k: int) -> int:
        wi = float(W[k, branch_i, ref_level])
        wj = float(W[k, branch_j, ref_level])
        return branch_i if wi >= wj else branch_j

    carrier_L = carrier_among_pair(kL)
    carrier_R = carrier_among_pair(kR)

    other_L = branch_j if carrier_L == branch_i else branch_i
    other_R = branch_j if carrier_R == branch_i else branch_i

    wL = W[kL, other_L, :].copy()
    wR = W[kR, other_R, :].copy()
    wL[ref_level] = -np.inf
    wR[ref_level] = -np.inf

    min_weight = np.minimum(wL, wR)
    partner = int(np.argmax(min_weight))
    score = float(min_weight[partner]) if np.isfinite(min_weight[partner]) else 0.0

    return dict(
        kL=kL, kR=kR,
        chiL=float(chi[kL]), chiR=float(chi[kR]),
        carrier_L=int(carrier_L), carrier_R=int(carrier_R),
        other_L=int(other_L), other_R=int(other_R),
        partner_bare_level=int(partner),
        min_weight=float(score),
        partner_wL=float(wL[partner]) if np.isfinite(wL[partner]) else 0.0,
        partner_wR=float(wR[partner]) if np.isfinite(wR[partner]) else 0.0,
    )