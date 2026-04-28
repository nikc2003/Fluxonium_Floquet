"""
landau_zener/plotting/fan.py
fan plots for quasienergies and nbar, plus truncation diagnostics.
"""

from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt

from landau_zener.floquet.sweep import FloquetSweep, basis_weights
from landau_zener.plotting.colors import build_branch_colors_shaded, normalize_branch_list, normalize_highlight_list
from landau_zener.utils.io import ensure_dir
from landau_zener.utils.units import angular_to_ghz, angular_to_mhz


def plot_nbar_all_branches(
    sweep: FloquetSweep,
    *,
    chi_units: str = "GHz",
    branches: Optional[Sequence[int]] = None,
    highlight_branches: Optional[Sequence[int]] = None,
    gray_others: bool = True,
    show_legend: bool = True,
    legend_mode: str = "highlight",
    legend_max_items: int = 18,
    title: Optional[str] = None,
    outdir: Optional[str] = None,
    filename: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
    branch_colors: Optional[Sequence] = None,
    other_alpha: float = 0.15,
    highlight_alpha: float = 0.95,
    other_lw: float = 1.0,
    highlight_lw: float = 2.4,
) -> Optional[str]:
    chi = np.asarray(sweep.chi, dtype=float)
    nb = int(sweep.nbar.shape[1])

    branches = normalize_branch_list(branches, nb)
    hi = normalize_highlight_list(highlight_branches, nb)
    colors = list(branch_colors) if branch_colors is not None else build_branch_colors_shaded(nb)

    if chi_units.lower() == "ghz":
        x = angular_to_ghz(chi); xlab = r"$\chi/2\pi$ [GHz]"
    elif chi_units.lower() == "mhz":
        x = angular_to_mhz(chi); xlab = r"$\chi/2\pi$ [MHz]"
    else:
        x = chi; xlab = r"$\chi$ [rad/ns]"

    fig, ax = plt.subplots(figsize=(10, 5))

    for b in branches:
        if hi and (b not in hi):
            ax.plot(x, sweep.nbar[:, b], color=("0.7" if gray_others else colors[b]), alpha=other_alpha, lw=other_lw, zorder=1)

    for b in branches:
        if (not hi) or (b in hi):
            ax.plot(x, sweep.nbar[:, b], color=colors[b], alpha=highlight_alpha,
                    lw=(highlight_lw if hi else other_lw), label=str(b), zorder=2)

    ax.set_xlabel(xlab)
    ax.set_ylabel(r"$\bar n$ (avg bare level index)")
    ax.set_title(title or "nbar vs chi")

    if show_legend:
        items = sorted(list(hi)) if (hi and legend_mode == "highlight") else branches
        if len(items) <= legend_max_items:
            ax.legend(ncol=2, fontsize=9, title="branch")

    plt.tight_layout()

    savepath = None
    if outdir and filename:
        ensure_dir(outdir)
        savepath = f"{outdir}/{filename}"
        plt.savefig(savepath, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return savepath


def plot_quasienergies_all_branches(
    sweep: FloquetSweep,
    *,
    chi_units: str = "GHz",
    energy_units: str = "GHz",
    use_unwrapped: bool = False,
    branches: Optional[Sequence[int]] = None,
    highlight_branches: Optional[Sequence[int]] = None,
    gray_others: bool = True,
    show_legend: bool = True,
    legend_mode: str = "highlight",
    legend_max_items: int = 18,
    show_bz_bounds: bool = True,
    title: Optional[str] = None,
    outdir: Optional[str] = None,
    filename: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
    branch_colors: Optional[Sequence] = None,
    other_alpha: float = 0.15,
    highlight_alpha: float = 0.95,
    other_lw: float = 1.0,
    highlight_lw: float = 2.4,
) -> Optional[str]:
    chi = np.asarray(sweep.chi, dtype=float)
    Eraw = np.asarray(sweep.qe_unwrapped if use_unwrapped else sweep.qe_wrapped, dtype=float)
    nb = int(Eraw.shape[1])

    branches = normalize_branch_list(branches, nb)
    hi = normalize_highlight_list(highlight_branches, nb)
    colors = list(branch_colors) if branch_colors is not None else build_branch_colors_shaded(nb)

    if chi_units.lower() == "ghz":
        x = angular_to_ghz(chi); xlab = r"$\chi/2\pi$ [GHz]"
    elif chi_units.lower() == "mhz":
        x = angular_to_mhz(chi); xlab = r"$\chi/2\pi$ [MHz]"
    else:
        x = chi; xlab = r"$\chi$ [rad/ns]"

    if energy_units.lower() == "ghz":
        E = angular_to_ghz(Eraw); ylab = r"$\epsilon/2\pi$ [GHz]"
        bz = angular_to_ghz(0.5 * sweep.omega_d)
    elif energy_units.lower() == "mhz":
        E = angular_to_mhz(Eraw); ylab = r"$\epsilon/2\pi$ [MHz]"
        bz = angular_to_mhz(0.5 * sweep.omega_d)
    else:
        E = Eraw; ylab = r"$\epsilon$ [rad/ns]"
        bz = 0.5 * sweep.omega_d

    fig, ax = plt.subplots(figsize=(10, 5))

    for b in branches:
        if hi and (b not in hi):
            ax.plot(x, E[:, b], color=("0.7" if gray_others else colors[b]), alpha=other_alpha, lw=other_lw, zorder=1)

    for b in branches:
        if (not hi) or (b in hi):
            ax.plot(x, E[:, b], color=colors[b], alpha=highlight_alpha,
                    lw=(highlight_lw if hi else other_lw), label=str(b), zorder=2)

    if show_bz_bounds and (not use_unwrapped):
        ax.axhline(+bz, linestyle="--", linewidth=1.0, color="0.5")
        ax.axhline(-bz, linestyle="--", linewidth=1.0, color="0.5")

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title or ("Quasienergy fan (unwrapped)" if use_unwrapped else "Quasienergy fan (wrapped)"))

    if show_legend:
        items = sorted(list(hi)) if (hi and legend_mode == "highlight") else branches
        if len(items) <= legend_max_items:
            ax.legend(ncol=2, fontsize=9, title="branch")

    plt.tight_layout()

    savepath = None
    if outdir and filename:
        ensure_dir(outdir)
        savepath = f"{outdir}/{filename}"
        plt.savefig(savepath, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return savepath


def truncation_leakage(sweep: FloquetSweep, *, top_m: int = 3) -> np.ndarray:
    """
    leakage[k,b] = sum_{m >= dim-top_m} |<m|phi_b(chi_k)>|^2
    """
    W = basis_weights(sweep.modes)
    K, nb, dim = W.shape
    top_m = int(min(max(1, top_m), dim))
    return np.sum(W[:, :, dim - top_m : dim], axis=2)


def plot_truncation_fan_diagnostic(
    sweep: FloquetSweep,
    *,
    chi_units: str = "GHz",
    energy_units: str = "GHz",
    use_unwrapped: bool = True,
    branches: Optional[Sequence[int]] = None,
    highlight_top_n_branches: int = 12,
    top_m_leak: int = 3,
    leak_thresh: float = 0.10,
    turning_point_thresh: int = 1,
    title: Optional[str] = None,
    outdir: Optional[str] = None,
    filename: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
) -> Optional[str]:
    """

    """
    chi = np.asarray(sweep.chi, dtype=float)
    Eraw = np.asarray(sweep.qe_unwrapped if use_unwrapped else sweep.qe_wrapped, dtype=float)
    K, nb = Eraw.shape

    branches = normalize_branch_list(branches, nb)
    if len(branches) == 0:
        branches = list(range(nb))

    sorted_br = sorted(branches)
    hi = set(sorted_br[-int(min(highlight_top_n_branches, len(sorted_br))):])

    colors = build_branch_colors_shaded(nb)
    leak = truncation_leakage(sweep, top_m=top_m_leak)

    def count_turning_points(y: np.ndarray, eps: float = 1e-12) -> int:
        dy = np.diff(np.asarray(y, dtype=float))
        s = np.sign(dy)
        s[np.abs(dy) < eps] = 0.0
        s = s[s != 0.0]
        if len(s) < 2:
            return 0
        return int(np.sum(s[1:] * s[:-1] < 0))

    suspicious = []
    for b in hi:
        tp = count_turning_points(Eraw[:, b])
        lk = float(np.max(leak[:, b]))
        if (tp >= turning_point_thresh) or (lk >= leak_thresh):
            suspicious.append(b)

    if chi_units.lower() == "ghz":
        x = angular_to_ghz(chi); xlab = r"$\chi/2\pi$ [GHz]"
    elif chi_units.lower() == "mhz":
        x = angular_to_mhz(chi); xlab = r"$\chi/2\pi$ [MHz]"
    else:
        x = chi; xlab = r"$\chi$ [rad/ns]"

    if energy_units.lower() == "ghz":
        E = angular_to_ghz(Eraw); ylab = r"$\epsilon/2\pi$ [GHz]"
    elif energy_units.lower() == "mhz":
        E = angular_to_mhz(Eraw); ylab = r"$\epsilon/2\pi$ [MHz]"
    else:
        E = Eraw; ylab = r"$\epsilon$ [rad/ns]"

    fig, ax = plt.subplots(figsize=(10, 6))
    for b in sorted_br:
        if b not in hi:
            ax.plot(x, E[:, b], color="0.7", alpha=0.55, lw=1.0, zorder=1)
    for b in sorted(hi):
        ax.plot(x, E[:, b], color=colors[b], alpha=0.95, lw=1.6, label=f"{b}", zorder=2)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title or "Truncation diagnostic fan diagram")
    ax.legend(ncol=2, fontsize=9, title="highlighted branches")

    plt.tight_layout()

    savepath = None
    if outdir and filename:
        ensure_dir(outdir)
        savepath = f"{outdir}/{filename}"
        plt.savefig(savepath, dpi=dpi)

    if suspicious:
        print("[truncation_diagnostic] suspicious branches:", suspicious)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return savepath