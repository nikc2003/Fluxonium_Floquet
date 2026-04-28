"""
landau_zener/plotting/events.py
Purpose:
  - Plot a detected avoided crossing event with annotations:
      1. chi_star (gap minimum location)
      2. Delta_min (minimum gap)
      3. delta_chi (half-width), shown as 2*delta_chi on x-axis
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from landau_zener.floquet.sweep import FloquetSweep
from landau_zener.utils.io import ensure_dir

_TWO_PI = 2.0 * np.pi

def _angular_to_ghz(x):
    return np.asarray(x, dtype=float) / _TWO_PI

def _angular_to_mhz(x):
    return np.asarray(x, dtype=float) / _TWO_PI * 1e3


def _event_dict(event: Union[Dict[str, Any], pd.Series]) -> Dict[str, Any]:
    return event.to_dict() if isinstance(event, pd.Series) else dict(event)

def _safe_float(v, default=np.nan) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)

def _safe_int(v, default: int = -1) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)

def _photon_label(ev: Dict[str, Any], *, n_cross: int = 10) -> str:
    """
wrirte this later
    """
    parts = []

    ba = ev.get("bare_a", None)
    bb = ev.get("bare_b", None)
    if ba is not None and bb is not None:
        parts.append(f"{int(ba)}→{int(bb)}")

    nabs = ev.get("n_photon_phys_abs", None)
    nphys = ev.get("n_photon_phys", None)

    if nabs is not None and np.isfinite(_safe_float(nabs)):
        parts.append(f"{int(abs(_safe_int(nabs)))}-photon resonance")
    elif nphys is not None and np.isfinite(_safe_float(nphys)):
        parts.append(f"{int(abs(_safe_int(nphys)))}-photon resonance")
    else:
        nzone = ev.get("n_zone_pred", ev.get("n_photon_pred", ev.get("n_photon", None)))
        if nzone is not None and np.isfinite(_safe_float(nzone)):
            parts.append(f"{int(_safe_int(nzone))}-photon (ZONE/FOLDED)")

    floq = ev.get("floquet_transition_label", None)
    if floq:
        parts.append(str(floq))
    else:
        if (ba is not None) and (bb is not None) and (nphys is not None) and np.isfinite(_safe_float(nphys)):
            n = int(_safe_int(nphys))
            k = int(n_cross)
            parts.append(f"|{int(ba)},{k}> ↔ |{int(bb)},{k-n}>")

    return " | ".join(parts) if parts else ""


def _clip_xlim_to_data(x: np.ndarray, xL: float, xR: float) -> Tuple[float, float]:

    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    xL2 = max(float(xL), xmin)
    xR2 = min(float(xR), xmax)
    if xmin >= 0.0:
        xL2 = max(xL2, 0.0)
    if not (xR2 > xL2):
        return xmin, xmax
    return xL2, xR2




def plot_event_annotated_gap(
    sweep: FloquetSweep,
    event: Union[Dict[str, Any], pd.Series],
    *,
    chi_units: str = "MHz",
    energy_units: str = "MHz",
    use_unwrapped: bool = False,
    observable: str = "nbar",
    include_observable_panel: bool = True,
    zoom: bool = True,
    zoom_x_factor: float = 6.0,
    zoom_y_factor: float = 12.0,
    clip_x_to_data: bool = True,

    n_cross: int = 10,

    outdir: Optional[str] = None,
    filename: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,

    **_ignored,
) -> Optional[str]:
    """
    Plot an event with explicit Delta_min and 2*delta_chi annotations.
    """
    ev = _event_dict(event)

    bi = _safe_int(ev.get("branch_i", 0))
    bj = _safe_int(ev.get("branch_j", 1))
    k_min = _safe_int(ev.get("k_min", 0))
    chi_star = _safe_float(ev.get("chi_star", np.nan))
    delta_chi = _safe_float(ev.get("delta_chi", np.nan))
    chi_cross = _safe_float(ev.get("chi_cross", np.nan))
    Delta_min = _safe_float(ev.get("Delta_min", np.nan))

    event_id = str(ev.get("event_id", "event"))
    fd_GHz = _safe_float(ev.get("freq_GHz", _angular_to_ghz(sweep.omega_d)))

    chi = np.asarray(sweep.chi, dtype=float)
    if chi_units.lower() == "ghz":
        x = _angular_to_ghz(chi)
        x_star = _angular_to_ghz(chi_star) if np.isfinite(chi_star) else np.nan
        dx = _angular_to_ghz(delta_chi) if np.isfinite(delta_chi) else np.nan
        x_cross = _angular_to_ghz(chi_cross) if np.isfinite(chi_cross) else np.nan
        xlab = r"$\chi/2\pi$ [GHz]"
        chi_unit_str = "GHz"
    elif chi_units.lower() == "mhz":
        x = _angular_to_mhz(chi)
        x_star = _angular_to_mhz(chi_star) if np.isfinite(chi_star) else np.nan
        dx = _angular_to_mhz(delta_chi) if np.isfinite(delta_chi) else np.nan
        x_cross = _angular_to_mhz(chi_cross) if np.isfinite(chi_cross) else np.nan
        xlab = r"$\chi/2\pi$ [MHz]"
        chi_unit_str = "MHz"
    else:
        x = chi
        x_star = chi_star
        dx = delta_chi
        x_cross = chi_cross
        xlab = r"$\chi$ [rad/ns]"
        chi_unit_str = "rad/ns"

    Eraw = np.asarray(sweep.qe_unwrapped if use_unwrapped else sweep.qe_wrapped, dtype=float)
    if energy_units.lower() == "ghz":
        E = _angular_to_ghz(Eraw)
        dE = _angular_to_ghz(Delta_min) if np.isfinite(Delta_min) else np.nan
        ylabE = r"$\epsilon/2\pi$ [GHz]"
        e_unit_str = "GHz"
    elif energy_units.lower() == "mhz":
        E = _angular_to_mhz(Eraw)
        dE = _angular_to_mhz(Delta_min) if np.isfinite(Delta_min) else np.nan
        ylabE = r"$\epsilon/2\pi$ [MHz]"
        e_unit_str = "MHz"
    else:
        E = Eraw
        dE = Delta_min
        ylabE = r"$\epsilon$ [rad/ns]"
        e_unit_str = "rad/ns"

    if observable.strip().lower() == "nbar":
        O = np.asarray(sweep.nbar, dtype=float)
        ylabO = r"$\langle N_q\rangle$"
    else:
        O = np.asarray(sweep.nbar, dtype=float)
        ylabO = r"$\langle N_q\rangle$"

    if include_observable_panel:
        fig, (axE, axO) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    else:
        fig, axE = plt.subplots(1, 1, figsize=(11, 4))
        axO = None

    axE.plot(x, E[:, bi], lw=2.4, label=f"branch {bi}")
    axE.plot(x, E[:, bj], lw=2.4, label=f"branch {bj}")
    axE.set_ylabel(ylabE)
    axE.legend(loc="upper right")

    if include_observable_panel and axO is not None:
        axO.plot(x, O[:, bi], lw=2.4, label=f"{observable} {bi}")
        axO.plot(x, O[:, bj], lw=2.4, label=f"{observable} {bj}")
        axO.set_xlabel(xlab)
        axO.set_ylabel(ylabO)
        axO.legend(loc="upper right")
    else:
        axE.set_xlabel(xlab)

    if np.isfinite(x_star):
        axE.axvline(x_star, ls="-.", lw=1.2)
        if axO is not None:
            axO.axvline(x_star, ls="-.", lw=1.2)

    if np.isfinite(dx) and np.isfinite(x_star):
        axE.axvline(x_star - dx, ls=":", lw=1.2)
        axE.axvline(x_star + dx, ls=":", lw=1.2)
        if axO is not None:
            axO.axvline(x_star - dx, ls=":", lw=1.2)
            axO.axvline(x_star + dx, ls=":", lw=1.2)

    if np.isfinite(x_cross):
        axE.axvline(x_cross, ls="--", lw=1.2)
        if axO is not None:
            axO.axvline(x_cross, ls="--", lw=1.2)

    if 0 <= k_min < len(x) and np.isfinite(x_star):
        y1 = float(E[k_min, bi])
        y2 = float(E[k_min, bj])
        ylo, yhi = (y1, y2) if y1 <= y2 else (y2, y1)

        axE.annotate(
            "",
            xy=(x_star, ylo),
            xytext=(x_star, yhi),
            arrowprops=dict(arrowstyle="<->", lw=1.4),
        )

        if np.isfinite(dE):
            axE.text(
                x_star,
                yhi + 0.03 * (axE.get_ylim()[1] - axE.get_ylim()[0]),
                f"Delta min = {dE:.3f} {e_unit_str}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        if np.isfinite(dx):
            y_span = axE.get_ylim()[1] - axE.get_ylim()[0]
            y_ann = ylo - 0.10 * y_span
            axE.annotate(
                "",
                xy=(x_star - dx, y_ann),
                xytext=(x_star + dx, y_ann),
                arrowprops=dict(arrowstyle="<->", lw=1.4),
            )
            axE.text(
                x_star,
                y_ann,
                f"2*delta_chi = {2*dx:.3f} {chi_unit_str}",
                ha="center",
                va="top",
                fontsize=10,
            )

    if zoom and np.isfinite(x_star) and np.isfinite(dx):
        xL = x_star - zoom_x_factor * dx
        xR = x_star + zoom_x_factor * dx
        if clip_x_to_data:
            xL, xR = _clip_xlim_to_data(x, xL, xR)
        axE.set_xlim(xL, xR)

        if np.isfinite(dE) and dE > 0 and (0 <= k_min < len(x)):
            y_center = 0.5 * (float(E[k_min, bi]) + float(E[k_min, bj]))
            y_half = float(max(1e-9, zoom_y_factor * dE))
            axE.set_ylim(y_center - y_half, y_center + y_half)

    plab = _photon_label(ev, n_cross=n_cross)
    fig.suptitle(f"{event_id} | fd={fd_GHz:.6f} GHz | {plab}", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    savepath = None
    if outdir and filename:
        ensure_dir(outdir)
        savepath = os.path.join(outdir, filename)
        fig.savefig(savepath, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return savepath


plot_event_branches_and_weight_transfer = plot_event_annotated_gap