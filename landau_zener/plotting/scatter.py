"""
Path: landau_zener/plotting/scatter.py
Scatter plots (Delta_min vs delta_chi)
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from landau_zener.utils.io import ensure_dir


def plot_gap_vs_deltachi_scatter(
    df: pd.DataFrame,
    outdir: str,
    *,
    annotate_topk: int = 50,
    color_by: Optional[str] = None,
    annotate_by: str = "branch", #branch/event_id
    filename: str = "gap_vs_deltachi_scatter.png",
    title: str = r"$\Delta_{\min}$ vs $\delta\chi$",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_table: bool = False,
    table_columns: Optional[List[str]] = None,
    dpi: int = 300,
) -> str:
    ensure_dir(outdir)
    if df is None or len(df) == 0:
        raise ValueError("No events to plot")
    x = df["delta_chi_MHz"].to_numpy()
    y = df["Delta_min_MHz"].to_numpy()
    if show_table:
        fig = plt.figure(figsize=(16, 6))
        ax = plt.subplot(1, 2, 1)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))

    if color_by is None:
        ax.scatter(x, y, alpha=0.75)
    else:
        c = df[color_by].to_numpy()
        sc = ax.scatter(x, y, c=c, alpha=0.75, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(color_by)

    ax.set_xlabel(r"$\delta\chi$ [MHz]")
    ax.set_ylabel(r"$\Delta_{\min}/2\pi$ [MHz]")
    ax.set_title(title)

    #annotate the smallest gaps
    order = np.argsort(y)[: min(int(annotate_topk), len(df))]
    for idx in order:
        row = df.iloc[idx]
        if annotate_by == "branch":
            lab = f"b({int(row['branch_i'])},{int(row['branch_j'])})"
        else:
            lab = str(row.get("event_id", "event"))
        ax.annotate(lab, (x[idx], y[idx]), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7)

    if show_table:
        ax_table = plt.subplot(1, 2, 2)
        ax_table.axis("off")
        if table_columns is None:
            table_columns = ["branch_i", "branch_j", "bare_a", "bare_b", "n_photon", "Delta_min_MHz", "delta_chi_MHz", "P_ad", "chi_star_MHz"]
        cols = [c for c in table_columns if c in df.columns]
        table_df = df[cols].copy()

        # format a bit
        for col in cols:
            if col in ["Delta_min_MHz", "delta_chi_MHz", "chi_star_MHz"]:
                table_df[col] = table_df[col].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "NA")
            if col in ["P_ad", "P_dia"]:
                table_df[col] = table_df[col].apply(lambda v: f"{v:.4f}" if pd.notna(v) else "NA")

        table = ax_table.table(
            cellText=table_df.values.tolist(),
            colLabels=table_df.columns.tolist(),
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)

    plt.tight_layout()
    savepath = os.path.join(outdir, filename)
    plt.savefig(savepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[scatter] saved: {savepath}")
    return savepath