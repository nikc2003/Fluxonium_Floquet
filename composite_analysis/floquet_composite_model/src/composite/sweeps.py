from __future__ import annotations

from typing import Sequence, Tuple, Optional, Dict, Any

import numpy as np
import floquet as ft
from .config import CompositeSysConfig, Label
from .model import build_composite_model
from .analysis import run_floquet_analysis_composite, extract_scar_maps_from_displaced_overlaps
from ..utils.timing import Timing

"""
Floquet analysis across frequency or flux. This is where the heart of our parallelization process will be. 
For the flux sweep, we will loop over flux points sequentially, but each sequential step requires 
rebuilding the undriven dressed static composite model and re-running the Floquet analysis, which probably eats up more time than drive sweeps. 

timing checks instituted for each point in the sweep!
"""


def drive_sweep_scar(
    cfg: CompositeSysConfig,
    *,
    flux_value: float,
    omega_d_values: np.ndarray,
    chi_ac_values: np.ndarray,
    label_states: Sequence[Label] = ((0, 0), (1, 0)),
    options: ft.Options,
    state_slots: Sequence[int] = (0, 1),
    clip_max: float = 0.5,
    timer: Optional[Timing] = None,
) -> Tuple[Dict[int, np.ndarray], Timing]:
    tr = timer or Timing()
    qparams = dict(cfg.qubit_params)
    qparams["flux"] = float(flux_value)
    cfg2 = CompositeSysConfig(
        qubit_params=qparams,
        dim_q=cfg.dim_q,
        dim_r=cfg.dim_r,
        omega_r=cfg.omega_r,
        g=cfg.g,
        coupling_op=cfg.coupling_op,
        resonator_quadrature=cfg.resonator_quadrature,
    )

    with tr.span("build_composite_model"):
        build = build_composite_model(cfg2, omega_d_values=omega_d_values, chi_ac_values=chi_ac_values, label_states=label_states, timer=tr)

    with tr.span("floquet_analysis"):
        data = run_floquet_analysis_composite(build, options=options, timer=tr)

    with tr.span("extract_scar"):
        scar_maps = extract_scar_maps_from_displaced_overlaps(data, state_slots=state_slots, clip_max=clip_max)
 
    return scar_maps, tr


def flux_sweep_scar(
    cfg: CompositeSysConfig,
    *,
    flux_grid: np.ndarray,
    chi_ac_values: np.ndarray,
    omega_d_bare: float,
    chi_v_flux_GHz: Optional[np.ndarray] = None,
    label_states: Sequence[Label] = ((0, 0), (1, 0)),
    options: ft.Options,
    state_slots: Sequence[int] = (0,),
    clip_max: float = 0.5,
    timer: Optional[Timing] = None,
) -> Tuple[Dict[int, np.ndarray], np.ndarray, Timing]:

    tr = timer or Timing()
    flux_grid = np.asarray(flux_grid, dtype=float)
    chi_ac_values = np.asarray(chi_ac_values, dtype=float)

    omega_used = np.zeros(len(flux_grid), dtype=float)
    scar_maps: Dict[int, np.ndarray] = {int(s): np.zeros((len(flux_grid), len(chi_ac_values)), dtype=np.float32) for s in state_slots}

    for i, flx in enumerate(flux_grid):
        qparams = dict(cfg.qubit_params)
        qparams["flux"] = float(flx)

        omega_d = float(omega_d_bare)
        if chi_v_flux_GHz is not None:
            omega_d = float(omega_d_bare) + 2 * np.pi * float(chi_v_flux_GHz[i])
        omega_used[i] = omega_d

        cfg2 = CompositeSysConfig(
            qubit_params=qparams,
            dim_q=cfg.dim_q,
            dim_r=cfg.dim_r,
            omega_r=cfg.omega_r,
            g=cfg.g,
            coupling_op=cfg.coupling_op,
            resonator_quadrature=cfg.resonator_quadrature,
        )

        with tr.span("per_flux.build_model"):
            build = build_composite_model(cfg2, omega_d_values=np.array([omega_d]), chi_ac_values=chi_ac_values, label_states=label_states, timer=tr)

        with tr.span("per_flux.floquet_run"):
            data = run_floquet_analysis_composite(build, options=options, timer=tr)

        with tr.span("per_flux.extract_scar"):
            maps = extract_scar_maps_from_displaced_overlaps(data, state_slots=state_slots, clip_max=clip_max)
            for slot, arr in maps.items():
                # arr shape (1,Nchi)
                scar_maps[int(slot)][i, :] = arr[0, :]

    return scar_maps, omega_used, tr
