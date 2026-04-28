"""
landau_zener/analysis/metrics.py
isolation/crowding/LZ metrics, bare-level inference, n-photon inference, merge all this with event detection.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np

from landau_zener.analysis.common import compute_edges_from_delta_chi, estimate_crossing_chi_from_sign_change, pick_observable
from landau_zener.analysis.detuning_fit import LinearDetuningFitConfig, linear_fit_diabatic_detuning_for_bare_pair
from landau_zener.floquet.sweep import FloquetSweep, basis_weights
from landau_zener.physics.ringup import chi_dot_ringup
from landau_zener.utils.units import wrap_to_bz, angular_to_mhz
from landau_zener.utils.units import angular_to_ghz


@dataclass
class IsolationConfig:
    iso_window_factor: float = 1.0
    iso_ratio_thresh: float = 0.0


def isolation_metric_pair_window(
    sweep: FloquetSweep,
    branch_i: int,
    branch_j: int,
    chi_star: float,
    Delta_min: float,
    delta_chi: float,
    iso_cfg: IsolationConfig,
) -> Dict[str, Any]:
    """
    iso_min_pair = min_{chi in window} min_{k != i,j} {g_{ik}(chi), g_{jk}(chi)}
    with window = iso_window_factor * delta_chi.
    """
    chi = sweep.chi
    qe_w = sweep.qe_wrapped
    omega_d = float(sweep.omega_d)

    if (not np.isfinite(delta_chi)) or delta_chi <= 0:
        return dict(is_isolated_pair=False, iso_min_pair=np.nan, iso_ratio_pair=np.nan)

    Wwin = float(iso_cfg.iso_window_factor * delta_chi)
    mask = np.abs(chi - float(chi_star)) <= Wwin
    if not np.any(mask):
        return dict(is_isolated_pair=False, iso_min_pair=np.nan, iso_ratio_pair=np.nan)

    idxs = np.where(mask)[0]
    iso_vals = []
    for k in idxs:
        E = qe_w[k, :]
        gi = np.abs(wrap_to_bz(E - E[int(branch_i)], omega_d))
        gj = np.abs(wrap_to_bz(E - E[int(branch_j)], omega_d))
        gi[int(branch_i)] = np.inf; gi[int(branch_j)] = np.inf
        gj[int(branch_i)] = np.inf; gj[int(branch_j)] = np.inf
        iso_vals.append(min(float(np.min(gi)), float(np.min(gj))))

    iso_min = float(np.min(iso_vals)) if iso_vals else np.nan
    iso_ratio = float(iso_min / Delta_min) if (np.isfinite(iso_min) and Delta_min > 0) else np.nan
    is_iso = bool(np.isfinite(iso_ratio) and iso_ratio >= float(iso_cfg.iso_ratio_thresh))
    return dict(
        is_isolated_pair=is_iso,
        iso_min_pair=float(iso_min),
        iso_ratio_pair=float(iso_ratio),
        iso_window_W=float(Wwin),
        iso_npts=int(len(idxs)),
    )


@dataclass
class CrowdingConfig:
    crowd_window_factor: float = 2.0


def crowding_metric_anypair_window(
    sweep: FloquetSweep,
    chi_star: float,
    Delta_min: float,
    delta_chi: float,
    crowd_cfg: CrowdingConfig,
    *,
    restrict_branches: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """
    crowd_min_anypair = min_{chi in window} min_{a<b} |wrap(eps_a - eps_b)|
    window = crowd_window_factor * delta_chi
    """
    chi = sweep.chi
    qe_w = sweep.qe_wrapped
    omega_d = float(sweep.omega_d)
    nchi, nb = qe_w.shape

    if (not np.isfinite(delta_chi)) or delta_chi <= 0:
        return dict(crowded_anypair=False, crowd_min_anypair=np.nan, crowd_ratio_anypair=np.nan)

    if restrict_branches is None:
        branches = np.arange(nb, dtype=int)
    else:
        branches = np.array(sorted(set(int(b) for b in restrict_branches if 0 <= int(b) < nb)), dtype=int)
        if branches.size < 2:
            return dict(crowded_anypair=False, crowd_min_anypair=np.nan, crowd_ratio_anypair=np.nan)

    Wwin = float(crowd_cfg.crowd_window_factor * delta_chi)
    mask = np.abs(chi - float(chi_star)) <= Wwin
    if not np.any(mask):
        return dict(crowded_anypair=False, crowd_min_anypair=np.nan, crowd_ratio_anypair=np.nan)

    idxs = np.where(mask)[0]
    mins = []
    for k in idxs:
        E = qe_w[k, branches]
        D = np.abs(wrap_to_bz(E[:, None] - E[None, :], omega_d))
        np.fill_diagonal(D, np.inf)
        mins.append(float(np.min(D)))

    crowd_min = float(np.min(mins)) if mins else np.nan
    crowd_ratio = float(crowd_min / Delta_min) if (np.isfinite(crowd_min) and Delta_min > 0) else np.nan
    crowded = bool(np.isfinite(crowd_ratio) and crowd_ratio < 4.0)
    return dict(
        crowded_anypair=crowded,
        crowd_min_anypair=float(crowd_min),
        crowd_ratio_anypair=float(crowd_ratio),
        crowd_window_W=float(Wwin),
        crowd_npts=int(len(idxs)),
    )


def lz_probability(Delta_min: float, s: float, chi_dot_star: float) -> Tuple[float, float, float]:
    """
    P_dia = exp(-pi*Delta^2/(2*v)), v = s*|chi_dot|
    => (P_dia, P_ad, v).
    """
    if (not np.isfinite(Delta_min)) or Delta_min <= 0:
        return np.nan, np.nan, np.nan
    if (not np.isfinite(s)) or s <= 0:
        return np.nan, np.nan, np.nan
    if (not np.isfinite(chi_dot_star)) or chi_dot_star <= 0:
        return np.nan, np.nan, np.nan

    v = float(s * abs(chi_dot_star))
    P_dia = float(np.exp(-np.pi * (Delta_min ** 2) / (2.0 * v)))
    return P_dia, (1.0 - P_dia), v


def infer_bare_levels_for_branch_event(
    sweep: FloquetSweep,
    event: Dict[str, Any],
    *,
    edge_factor: float = 4.0,
    min_edge_offset_pts: int = 12,
) -> Dict[str, Any]:
    """
    Infer which bare levels swap by looking at dominant bare components on branches at edges.
    """
    out = dict(event)
    chi = sweep.chi
    W = basis_weights(sweep.modes)
    nchi, nb, dim = W.shape

    bi = int(event["branch_i"])
    bj = int(event["branch_j"])
    chi_star = float(event.get("chi_star", np.nan))
    delta_chi = float(event.get("delta_chi", np.nan))
    k_center = int(event.get("k_min", 0))

    kL, kR = compute_edges_from_delta_chi(
        chi, k_center, chi_star, delta_chi,
        edge_factor=float(edge_factor),
        min_edge_offset_pts=int(min_edge_offset_pts),
    )

    dom_bi_L = int(np.argmax(W[kL, bi, :]))
    dom_bj_L = int(np.argmax(W[kL, bj, :]))
    dom_bi_R = int(np.argmax(W[kR, bi, :]))
    dom_bj_R = int(np.argmax(W[kR, bj, :]))

    if dom_bi_L == dom_bj_R and dom_bj_L == dom_bi_R:
        bare_a, bare_b = sorted([dom_bi_L, dom_bj_L])
    else:
        bare_a, bare_b = sorted([dom_bi_L, dom_bj_L])

    out["bare_a"] = int(bare_a)
    out["bare_b"] = int(bare_b)
    out["dom_bi_L"] = int(dom_bi_L)
    out["dom_bj_L"] = int(dom_bj_L)
    out["dom_bi_R"] = int(dom_bi_R)
    out["dom_bj_R"] = int(dom_bj_R)
    return out


def add_reference_overlap_metrics(
    sweep: FloquetSweep,
    event: Dict[str, Any],
    *,
    ref_levels: Sequence[int] = (0, 1, 2),
    window_pts: int = 5,
) -> Dict[str, Any]:
    """
 for branch event (bi,bj,k_min), compute max overlap with ref_levels
    in a neighborhood around k_min.
    """
    out = dict(event)
    W = basis_weights(sweep.modes)
    K, nb, dim = W.shape

    bi = int(event["branch_i"])
    bj = int(event["branch_j"])
    k0 = int(event["k_min"])
    a = max(0, k0 - int(window_pts))
    b = min(K - 1, k0 + int(window_pts))

    bests = []
    for r in ref_levels:
        if not (0 <= int(r) < dim):
            continue
        wi = float(np.max(W[a:b + 1, bi, int(r)]))
        wj = float(np.max(W[a:b + 1, bj, int(r)]))
        out[f"w_ref{int(r)}_max_pair"] = float(max(wi, wj))
        out[f"w_ref{int(r)}_max_bi"] = float(wi)
        out[f"w_ref{int(r)}_max_bj"] = float(wj)
        bests.append(max(wi, wj))

    out["w_ref_max_over_set"] = float(np.max(bests)) if bests else np.nan
    return out


def compute_event_chi_cross(
    sweep: FloquetSweep,
    event: Dict[str, Any],
    *,
    detect_observable: str,
    obs_edge_factor: float,
    min_edge_offset_pts: int,
) -> Dict[str, Any]:
    """
    does chi_cross exists using the event's swap_observable?
    if the event already has a finite chi_cross (for ex. from nbar crossing
    detection), it is preserved.  Only kL/kR and swap diagnostics are
    recomputed.
    """
    out = dict(event)
    chi = sweep.chi

    bi = int(event["branch_i"])
    bj = int(event["branch_j"])
    chi_star = float(event.get("chi_star", np.nan))
    delta_chi = float(event.get("delta_chi", np.nan))
    k_center = int(event.get("k_min", event.get("k_hint", 0)))

    kL, kR = compute_edges_from_delta_chi(
        chi, k_center, chi_star, delta_chi,
        edge_factor=float(obs_edge_factor),
        min_edge_offset_pts=int(min_edge_offset_pts),
    )

    swap_obs = str(event.get("swap_observable", detect_observable)).strip().lower()
    if swap_obs == "ref_weight":
        ref_level = int(event["ref_level"])
        W = basis_weights(sweep.modes)
        d = W[:, bi, ref_level] - W[:, bj, ref_level]
    else:
        O = pick_observable(sweep, detect_observable)
        d = O[:, bi] - O[:, bj]

    existing = float(event.get("chi_cross", np.nan))
    if not np.isfinite(existing):
        chi_cross = float(estimate_crossing_chi_from_sign_change(chi, d, kL, kR))
        out["chi_cross"] = float(chi_cross)

    out["kL"] = int(kL)
    out["kR"] = int(kR)
    out["swap_d_left"] = float(d[kL])
    out["swap_d_right"] = float(d[kR])
    return out


def infer_n_photon_at_chi_star(
    sweep: FloquetSweep,
    *,
    bare_a: int,
    bare_b: int,
    chi_star: float,
) -> int | None:
    """
    Infer n_photon by computing diabatic energies at chi_star and rounding detuning/omega_d.
    """
    W = basis_weights(sweep.modes)         
    E = np.asarray(sweep.qe_unwrapped)        
    chi = sweep.chi
    k = int(np.argmin(np.abs(chi - float(chi_star))))
    dim = W.shape[-1]
    if not (0 <= bare_a < dim and 0 <= bare_b < dim):
        return None

    eps_a = float(np.sum(E[k, :] * W[k, :, int(bare_a)]))
    eps_b = float(np.sum(E[k, :] * W[k, :, int(bare_b)]))
    n = int(np.round((eps_b - eps_a) / float(sweep.omega_d)))
    return n


def event_with_metrics(
    sweep: FloquetSweep,
    event: Dict[str, Any],
    *,
    detect_observable: str,
    obs_edge_factor: float,
    min_edge_offset_pts: int,
    iso_cfg: Optional[IsolationConfig] = None,
    crowd_cfg: Optional[CrowdingConfig] = None,
    kappa_res: Optional[float] = None,
    restrict_branches_for_crowd: Optional[Sequence[int]] = None,
    ref_levels_for_overlap: Sequence[int] = (0, 1, 2),
    lin_cfg: Optional[LinearDetuningFitConfig] = LinearDetuningFitConfig(),
    chi_max_global: Optional[float] = None,
) -> Dict[str, Any]:
    """
    reutrn an event dict that adds various metrics to the input event dict. Metrics include:
      - chi_cros
      - isolation + crowding
      - LZ probability via chi_dot_ringup
      - ref overlap metrics (branch events)
      - bare_a/b inference (if branch)
      - linear diabatic detuning slope (optional)
      - n_photon inference (if bare labels exist)
    """
    out = compute_event_chi_cross(
        sweep, event,
        detect_observable=detect_observable,
        obs_edge_factor=obs_edge_factor,
        min_edge_offset_pts=min_edge_offset_pts,
    )

    bi = int(out["branch_i"])
    bj = int(out["branch_j"])
    chi_star = float(out.get("chi_star", np.nan))
    Delta_min = float(out.get("Delta_min", np.nan))
    delta_chi = float(out.get("delta_chi", np.nan))

    if iso_cfg is not None:
        out.update(isolation_metric_pair_window(
            sweep, branch_i=bi, branch_j=bj,
            chi_star=chi_star, Delta_min=Delta_min, delta_chi=delta_chi,
            iso_cfg=iso_cfg
        ))

    if crowd_cfg is not None:
        out.update(crowding_metric_anypair_window(
            sweep,
            chi_star=chi_star, Delta_min=Delta_min, delta_chi=delta_chi,
            crowd_cfg=crowd_cfg,
            restrict_branches=restrict_branches_for_crowd,
        ))

    if kappa_res is not None and np.isfinite(chi_star):
        chi_max = float(chi_max_global) if chi_max_global is not None else float(sweep.chi[-1])
        chi_dot_star = float(chi_dot_ringup(np.array([chi_star]), chi_max=chi_max, kappa=float(kappa_res))[0])
        P_dia, P_ad, v = lz_probability(Delta_min=Delta_min, s=float(out["s_g2"]), chi_dot_star=abs(chi_dot_star))
        out.update(dict(chi_dot_star=chi_dot_star, v=v, P_dia=P_dia, P_ad=P_ad))
    if str(out.get("event_type", "")).lower() == "branch":
        out = add_reference_overlap_metrics(sweep, out, ref_levels=ref_levels_for_overlap, window_pts=5)
    et = str(out.get("event_type", "")).lower()
    if et == "ref":
        out["bare_a"] = int(out.get("ref_level", 0))
        out["bare_b"] = int(out.get("partner_bare_level", 0))
    elif et == "branch":
        if ("bare_a" not in out) or ("bare_b" not in out):
            out = infer_bare_levels_for_branch_event(
                sweep, out,
                edge_factor=float(obs_edge_factor),
                min_edge_offset_pts=int(min_edge_offset_pts),
            )
    if "bare_a" in out and "bare_b" in out:
        bare_a = int(out["bare_a"])
        bare_b = int(out["bare_b"])
        if bare_a != bare_b and np.isfinite(delta_chi) and delta_chi > 0 and lin_cfg is not None:
            lin = linear_fit_diabatic_detuning_for_bare_pair(
                sweep,
                bare_a=bare_a,
                bare_b=bare_b,
                chi_star=chi_star,
                delta_chi=delta_chi,
                cfg=lin_cfg,
            )
            if lin.get("lin_ok", False):
                out["s_lin"] = float(lin["s_lin"])
                out["lin_r2"] = float(lin["lin_r2"])
                out["lin_rel_rms"] = float(lin["lin_rel_rms"])
        if "n_photon" not in out and np.isfinite(chi_star):
            nph = infer_n_photon_at_chi_star(sweep, bare_a=bare_a, bare_b=bare_b, chi_star=chi_star)
            if nph is not None:
                out["n_photon"] = int(nph)
    out["freq_GHz"] = float(out.get("freq_GHz", angular_to_ghz(float(out["omega_d"])) if "omega_d" in out else np.nan))
    out["chi_star_MHz"] = float(out.get("chi_star_MHz", angular_to_mhz(float(out["chi_star"])) if "chi_star" in out else np.nan))
    out["Delta_min_MHz"] = float(out.get("Delta_min_MHz", angular_to_mhz(float(out["Delta_min"])) if "Delta_min" in out else np.nan))
    out["delta_chi_MHz"] = float(out.get("delta_chi_MHz", angular_to_mhz(float(out["delta_chi"])) if "delta_chi" in out else np.nan))
    return out