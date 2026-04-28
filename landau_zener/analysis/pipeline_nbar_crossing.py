"""
landau_zener/analysis/pipeline_nbar_crossing.py
targeted transition sweep that locates the partial transition point
via the nbar crossing between target branches.as mentioned, crossing is where
nbar_i == nbar_j (50/50 hybridization), and the quasienergy gap at
that chi is reported.
"""

from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import inspect

import numpy as np
import pandas as pd
import floquet as ft

from landau_zener.floquet.sweep import (
    FloquetSweep,
    floquet_sweep_vs_chi,
    diabatic_energies_floquet,
    basis_weights,
)
from landau_zener.analysis.detection import BranchHybridizationDetectConfig
from landau_zener.analysis.detection_nbar_crossing import analyze_pair_nbar_crossing
from landau_zener.analysis.metrics import (
    IsolationConfig,
    CrowdingConfig,
    event_with_metrics,
)
from landau_zener.analysis.detuning_fit import LinearDetuningFitConfig
from landau_zener.analysis.slope_target import SlopeConfig
from landau_zener.analysis.photon import (
    bare_energies_rad_ns,
    physical_photon_number,
    floquet_replica_label,
)
from landau_zener.utils.units import ghz_to_angular, wrap_to_bz, angular_to_mhz
from landau_zener.utils.io import ensure_dir, save_df_csv
from landau_zener.utils.parallel import process_pool_map


def _to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if is_dataclass(obj):
        return asdict(obj)
    try:
        return dict(vars(obj))
    except TypeError:
        return {"value": obj}


def _filter_kwargs_for_ctor(cls: Any, d: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(d)
    try:
        sig = inspect.signature(cls)
        params = sig.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return d
        allowed = set(params.keys())
        return {k: v for k, v in d.items() if k in allowed}
    except Exception:
        return d


def _make_ft_options(options_like: Any) -> ft.Options:
    d = _filter_kwargs_for_ctor(ft.Options, _to_dict(options_like))
    return ft.Options(**d)


def run_targeted_pair_nbar_crossing_single_point(
    *,
    dim_q: int,
    qubit_params: Dict[str, Any],
    flux: float,
    omega_d: float,
    chi_range_GHz: Tuple[float, float],
    options: ft.Options,
    detect_cfg: BranchHybridizationDetectConfig,
    lin_cfg: LinearDetuningFitConfig,
    iso_cfg: Optional[IsolationConfig] = None,
    crowd_cfg: Optional[CrowdingConfig] = None,
    kappa_res: Optional[float] = None,
    target_bare_pair: Tuple[int, int],
    target_n_photon: Optional[int] = None,
    n_cross: int = 10,
    nchi_sweep: int = 1201,
    carrier_avg_half_window_pts: int = 5,
    min_target_branch_weight: float = 0.05,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """nbar-crossing detector for a requested bare pair (a, b).
    rtaher of searching for the minimum quasienergy gap within a chi window,
    this finds where the nbar of the two target branches crosses (50/50
    hybridization) and reports the gap at that chi value.
    """
    a, b = sorted(map(int, target_bare_pair))
    chi_vals = ghz_to_angular(
        np.linspace(chi_range_GHz[0], chi_range_GHz[1], int(nchi_sweep))
    )
    sweep = floquet_sweep_vs_chi(
        dim_q=dim_q,
        qubit_params=qubit_params,
        flux=float(flux),
        omega_d=float(omega_d),
        chi_vals=chi_vals,
        options=options,
    )
    E_bare = bare_energies_rad_ns(
        qubit_params=qubit_params, flux=float(flux), dim_q=int(dim_q)
    )
    pinfo = physical_photon_number(
        E_bare=E_bare, bare_a=a, bare_b=b, omega_d=float(omega_d)
    )
    if target_n_photon is not None:
        if int(pinfo["n_photon_phys_abs"]) != int(target_n_photon):
            return pd.DataFrame(), dict(
                reason="physical photon order mismatch at this frequency",
                sweep=sweep,
                physical_photon_info=pinfo,
            )
    eps_dia = diabatic_energies_floquet(sweep)
    det_target = wrap_to_bz(eps_dia[:, b] - eps_dia[:, a], float(omega_d))
    abs_det = np.abs(det_target)
    k_det = int(np.argmin(abs_det))
    W = basis_weights(sweep.modes)
    K = W.shape[0]
    k0 = max(0, k_det - int(carrier_avg_half_window_pts))
    k1 = min(K, k_det + int(carrier_avg_half_window_pts) + 1)
    branch_score = np.mean(W[k0:k1, :, a] + W[k0:k1, :, b], axis=0)
    top2 = np.argsort(branch_score)[-2:]
    bi, bj = sorted(int(t) for t in top2)
    if (
        float(branch_score[top2[0]]) < float(min_target_branch_weight)
        or float(branch_score[top2[1]]) < float(min_target_branch_weight)
    ):
        return pd.DataFrame(), dict(
            reason="target branches never carry enough weight",
            sweep=sweep,
            physical_photon_info=pinfo,
            k_det=int(k_det),
            branch_score=branch_score,
        )
    ev = analyze_pair_nbar_crossing(
        sweep,
        branch_i=int(bi),
        branch_j=int(bj),
        g2_fit_half_window=int(detect_cfg.g2_fit_half_window),
        event_prefix="TNBAR",
    )
    if ev is None:
        return pd.DataFrame(), dict(
            reason="branch indices out of range",
            sweep=sweep,
            physical_photon_info=pinfo,
            k_det=int(k_det),
            branch_score=branch_score,
            branch_i=int(bi),
            branch_j=int(bj),
        )
    ev["event_type"] = "nbar_crossing"
    ev["swap_observable"] = str(detect_cfg.observable)
    ev["bare_a"] = int(a)
    ev["bare_b"] = int(b)

    ev["k_det"] = int(k_det)
    ev["target_detuning_center"] = float(det_target[k_det])
    ev["target_detuning_center_MHz"] = float(angular_to_mhz(det_target[k_det]))
    ev["target_branch_i_score"] = float(branch_score[bi])
    ev["target_branch_j_score"] = float(branch_score[bj])
    ev["target_support_sum"] = float(branch_score[bi] + branch_score[bj])

    g2_ok = ev.get("g2_fit_ok", False)
    ev = event_with_metrics(
        sweep,
        ev,
        detect_observable=detect_cfg.observable,
        obs_edge_factor=detect_cfg.obs_edge_factor,
        min_edge_offset_pts=detect_cfg.min_edge_offset_pts,
        iso_cfg=iso_cfg if g2_ok else None,
        crowd_cfg=crowd_cfg if g2_ok else None,
        kappa_res=kappa_res if g2_ok else None,
        restrict_branches_for_crowd=None,
        ref_levels_for_overlap=(0, 1, 2),
        lin_cfg=lin_cfg if g2_ok else None,
        chi_max_global=float(chi_vals[-1]),
    )

    ev.update(pinfo)
    ev["floquet_transition_label"] = floquet_replica_label(
        bare_a=a,
        bare_b=b,
        n_photon_phys=int(pinfo["n_photon_phys"]),
        n_cross=int(n_cross),
    )

    df = pd.DataFrame([ev])
    return df, dict(
        sweep=sweep,
        physical_photon_info=pinfo,
        k_det=int(k_det),
        branch_score=branch_score,
        branch_i=int(bi),
        branch_j=int(bj),
    )


def _drive_sweep_nbar_worker(job: Dict[str, Any]) -> pd.DataFrame:
    from landau_zener.utils.units import ghz_to_angular as _g2a
    fGHz = float(job["fGHz"])
    omega_d = _g2a(fGHz)
    outdir = str(job["outdir"])
    odir = os.path.join(outdir, f"fd_{fGHz:.6f}GHz")
    options = _make_ft_options(job["options"])
    detect_cfg = BranchHybridizationDetectConfig(**_to_dict(job["detect_cfg"]))
    lin_cfg = LinearDetuningFitConfig(**_to_dict(job["lin_cfg"]))
    iso_cfg = (
        IsolationConfig(**_to_dict(job["iso_cfg"]))
        if job.get("iso_cfg") is not None
        else None
    )
    crowd_cfg = (
        CrowdingConfig(**_to_dict(job["crowd_cfg"]))
        if job.get("crowd_cfg") is not None
        else None
    )

    df_b, _dbg = run_targeted_pair_nbar_crossing_single_point(
        dim_q=int(job["dim_q"]),
        qubit_params=dict(job["qubit_params"]),
        flux=float(job["flux"]),
        omega_d=float(omega_d),
        chi_range_GHz=tuple(job["chi_range_GHz"]),
        options=options,
        detect_cfg=detect_cfg,
        lin_cfg=lin_cfg,
        iso_cfg=iso_cfg,
        crowd_cfg=crowd_cfg,
        kappa_res=(
            float(job["kappa_res"]) if job.get("kappa_res") is not None else None
        ),
        target_bare_pair=tuple(job["target_bare_pair"]),
        target_n_photon=(
            int(job["target_n_photon"])
            if job.get("target_n_photon") is not None
            else None
        ),
        n_cross=int(job.get("n_cross", 10)),
        nchi_sweep=int(job.get("nchi_sweep", 1201)),
    )

    if df_b is not None and len(df_b) > 0:
        ensure_dir(odir)
        save_df_csv(df_b, os.path.join(odir, "events_nbar_crossing.csv"))

    return df_b


def run_drive_sweep_nbar_crossing(
    *,
    drive_freqs_GHz: Sequence[float],
    dim_q: int,
    qubit_params: Dict[str, Any],
    flux: float,
    chi_range_GHz: Tuple[float, float],
    options: ft.Options,
    outdir: str,
    detect_cfg: BranchHybridizationDetectConfig,
    iso_cfg: Optional[IsolationConfig],
    crowd_cfg: Optional[CrowdingConfig],
    kappa_res: Optional[float],
    lin_cfg: LinearDetuningFitConfig,
    slope_cfg: SlopeConfig,
    target_bare_pair: Tuple[int, int],
    target_n_photon: Optional[int],
    parallel: bool = False,
    max_workers: int = 4,
    n_cross: int = 10,
    nchi_sweep: int = 1201,
) -> pd.DataFrame:
    """Drive-frequency sweep with nbar-crossing detection."""
    ensure_dir(outdir)
    drive_freqs_GHz = list(map(float, drive_freqs_GHz))

    jobs: List[Dict[str, Any]] = []
    for fGHz in drive_freqs_GHz:
        jobs.append(
            dict(
                fGHz=float(fGHz),
                outdir=str(outdir),
                dim_q=int(dim_q),
                qubit_params=dict(qubit_params),
                flux=float(flux),
                chi_range_GHz=tuple(chi_range_GHz),
                options=_to_dict(options),
                detect_cfg=_to_dict(detect_cfg),
                iso_cfg=_to_dict(iso_cfg) if iso_cfg is not None else None,
                crowd_cfg=_to_dict(crowd_cfg) if crowd_cfg is not None else None,
                kappa_res=(
                    float(kappa_res) if kappa_res is not None else None
                ),
                lin_cfg=_to_dict(lin_cfg),
                slope_cfg=_to_dict(slope_cfg),
                target_bare_pair=tuple(map(int, target_bare_pair)),
                target_n_photon=(
                    int(target_n_photon) if target_n_photon is not None else None
                ),
                nchi_window=801,
                n_cross=int(n_cross),
                nchi_sweep=int(nchi_sweep),
            )
        )

    if parallel:
        dfs = process_pool_map(
            _drive_sweep_nbar_worker,
            jobs,
            max_workers=int(max_workers),
            mp_start_method="spawn",
        )
    else:
        dfs = [_drive_sweep_nbar_worker(j) for j in jobs]

    rows = [d for d in dfs if d is not None and len(d) > 0]
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    if len(out) > 0:
        save_df_csv(
            out, os.path.join(outdir, "drive_sweep_nbar_crossing_all.csv")
        )

    return out
