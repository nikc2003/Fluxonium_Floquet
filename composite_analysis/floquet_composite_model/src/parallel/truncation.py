from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, Sequence, Tuple, List, Optional

import numpy as np
import h5py

from ..composite.config import CompositeSysConfig, Label
from ..composite.truncation import truncation_metrics_point
from ..utils.timing import Timing

"""
truncation is probably not well 
"""
_METRIC_KEYS = (
    "amp",
    "Nr_avg",
    "p_r_top_avg",
    "p_q_top_avg",
    "purity_avg",
    "entropy_avg",
    "comm_error_ptop_avg",
    "comm_error_analytic_avg",
)


def save_truncation_chunk_h5(
    path: str | Path,
    *,
    dim_q: np.ndarray,
    dim_r: np.ndarray,
    metrics: Dict[str, np.ndarray],
    pair_wall_s: np.ndarray,
    metadata: Dict[str, Any],
    timing: Dict[str, Any],
) -> str:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.create_dataset("dim_q", data=np.asarray(dim_q, dtype=int))
        f.create_dataset("dim_r", data=np.asarray(dim_r, dtype=int))
        f.create_dataset("pair_wall_s", data=np.asarray(pair_wall_s, dtype=float), compression="gzip", compression_opts=4, shuffle=True)

        for k, arr in metrics.items():
            f.create_dataset(str(k), data=np.asarray(arr), compression="gzip", compression_opts=4, shuffle=True)
        g = f.create_group("_metadata")
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool, np.integer, np.floating)):
                g.attrs[str(k)] = v
            else:
                g.attrs[str(k)] = json.dumps(v, default=str)

        f.attrs["timing_json"] = json.dumps(timing, sort_keys=True)

    return str(path)


def worker_truncation_chunk(
    *,
    base_cfg_dict: Dict[str, Any],
    pairs: Sequence[Tuple[int, int]],
    flux: float,
    omega_d: float,
    chi_ac: float,
    target_label: Label,
    nsteps: int,
    ntimes: int,
    outdir: str,
    chunk_id: int,
) -> Dict[str, Any]:
    """Compute truncation metrics for a list of (dim_q, dim_r) pairs."""
    tr = Timing()
    t0 = time.perf_counter()

    pairs = [(int(dq), int(dr)) for dq, dr in pairs]
    dim_q_arr = np.array([dq for dq, _ in pairs], dtype=int)
    dim_r_arr = np.array([dr for _, dr in pairs], dtype=int)

    metrics = {k: np.zeros(len(pairs), dtype=float) for k in _METRIC_KEYS}
    pair_wall = np.zeros(len(pairs), dtype=float)

    for i, (dq, dr) in enumerate(pairs):
        cfg = CompositeSysConfig(
            qubit_params=dict(base_cfg_dict["qubit_params"]),
            dim_q=int(dq),
            dim_r=int(dr),
            omega_r=float(base_cfg_dict["omega_r"]),
            g=float(base_cfg_dict["g"]),
            coupling_op=str(base_cfg_dict.get("coupling_op", "n")),
            resonator_quadrature=str(base_cfg_dict.get("resonator_quadrature", "x")),
        )

        t_pair0 = time.perf_counter()
        with tr.span("pair.total"):
            out = truncation_metrics_point(
                cfg,
                flux=float(flux),
                omega_d=float(omega_d),
                chi_ac=float(chi_ac),
                # options_like={},  
                target_label=tuple(target_label),
                nsteps=int(nsteps),
                ntimes=int(ntimes),
                timer=tr,
            )
        pair_wall[i] = time.perf_counter() - t_pair0
        row = out["row"]
        for k in _METRIC_KEYS:
            metrics[k][i] = float(row[k])

    outpath = Path(outdir) / f"chunk_{int(chunk_id):04d}.h5"
    meta = {
        "kind": "truncation_metrics",
        "chunk_id": int(chunk_id),
        "flux": float(flux),
        "omega_d": float(omega_d),
        "chi_ac": float(chi_ac),
        "target_label": list(target_label),
        "base_cfg_json": base_cfg_dict,
        "nsteps": int(nsteps),
        "ntimes": int(ntimes),
    }

    with tr.span("chunk.save_h5"):
        file_path = save_truncation_chunk_h5(
            outpath,
            dim_q=dim_q_arr,
            dim_r=dim_r_arr,
            metrics=metrics,
            pair_wall_s=pair_wall,
            metadata=meta,
            timing=tr.to_dict(),
        )

    t1 = time.perf_counter()
    return {
        "chunk_id": int(chunk_id),
        "file": file_path,
        "n_pairs": int(len(pairs)),
        "t_total_s": float(t1 - t0),
        "timing": tr.to_dict(),
    }


def merge_truncation_chunks(
    files: Sequence[str | Path],
    out_path: str | Path,
    *,
    dim_q_list: Sequence[int],
    dim_r_list: Sequence[int],
) -> str:
    """Merge truncation chunks into a single HDF5 with both long table and grid arrays."""
    files = [Path(p).expanduser().resolve() for p in files]
    out_path = Path(out_path).expanduser().resolve()
    if not files:
        raise ValueError("No chunk files to merge.")

    dim_q_list = list(map(int, dim_q_list))
    dim_r_list = list(map(int, dim_r_list))
    dq_set = set(dim_q_list)
    dr_set = set(dim_r_list)

    rows = []
    meta0 = None
    for fp in files:
        with h5py.File(fp, "r") as f:
            dq = f["dim_q"][:].astype(int)
            dr = f["dim_r"][:].astype(int)
            pair_wall = f["pair_wall_s"][:].astype(float)
            metrics = {k: f[k][:].astype(float) for k in _METRIC_KEYS if k in f}
            if meta0 is None and "_metadata" in f:
                meta0 = dict(f["_metadata"].attrs.items())
        for i in range(len(dq)):
            rows.append({
                "dim_q": int(dq[i]),
                "dim_r": int(dr[i]),
                "pair_wall_s": float(pair_wall[i]),
                **{k: float(metrics[k][i]) for k in metrics.keys()},
            })

    # build grids
    grid = {k: np.full((len(dim_q_list), len(dim_r_list)), np.nan, dtype=float) for k in ("pair_wall_s",) + _METRIC_KEYS}
    dq_index = {dq: i for i, dq in enumerate(dim_q_list)}
    dr_index = {dr: j for j, dr in enumerate(dim_r_list)}

    for r in rows:
        dq = r["dim_q"]
        dr = r["dim_r"]
        if dq not in dq_index or dr not in dr_index:
            continue
        i = dq_index[dq]
        j = dr_index[dr]
        for k in grid.keys():
            grid[k][i, j] = r.get(k, np.nan)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as out:
        out.create_dataset("dim_q_list", data=np.asarray(dim_q_list, dtype=int))
        out.create_dataset("dim_r_list", data=np.asarray(dim_r_list, dtype=int))

        g_table = out.create_group("table")
        g_table.create_dataset("dim_q", data=np.asarray([r["dim_q"] for r in rows], dtype=int))
        g_table.create_dataset("dim_r", data=np.asarray([r["dim_r"] for r in rows], dtype=int))
        for k in ("pair_wall_s",) + _METRIC_KEYS:
            g_table.create_dataset(k, data=np.asarray([r.get(k, np.nan) for r in rows], dtype=float))

        g_grid = out.create_group("grid")
        for k, arr in grid.items():
            g_grid.create_dataset(k, data=np.asarray(arr, dtype=float))

        if meta0:
            g_meta = out.create_group("_metadata")
            for k, v in meta0.items():
                g_meta.attrs[str(k)] = v

    return str(out_path)
