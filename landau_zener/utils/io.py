"""
landau_zener/utils/io.py
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_df_csv(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    df.to_csv(path, index=False)
    print(f"[save_df_csv] Saved: {path} (rows={len(df)})")


def fmt_id_num(x: float, nd: int = 3) -> str:
    if not np.isfinite(x):
        return "nan"
    s = f"{float(x):.{nd}f}"
    return s.replace("-", "m").replace(".", "p")


def make_outdir(base_name: str, qubit: str, dim: int, freq_ghz: float | None = None, flux_val: float | None = None) -> str:
    parts = [base_name, qubit, f"dim{dim}"]
    if freq_ghz is not None:
        parts.append(f"f{freq_ghz:.3f}GHz")
    if flux_val is not None:
        parts.append(f"flux{flux_val:.3f}")
    return "_".join(parts)