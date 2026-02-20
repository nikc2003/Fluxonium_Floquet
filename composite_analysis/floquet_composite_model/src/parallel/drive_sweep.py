from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, Sequence, Tuple, List, Optional

import numpy as np
import h5py
import floquet as ft

from ..composite.config import CompositeSysConfig, Label
from ..composite.model import build_composite_model
from ..composite.analysis import run_floquet_analysis_composite, extract_scar_maps_from_displaced_overlaps
from ..utils.timing import Timing


"""Parallelization! Assign chunk of drive frequencies to each worker, loop over drive frequencies within each chunk sequentially
    Each worker saves its own chunk of data to disk, and we aggregate and analyze after the fact.
"""

quantities: Dict[str, List[str]] = {
    "avg_excitation": [
        "avg_excitation",
    ],
    "quasienergies": [
        "quasienergies",

    ],
}


def _extract_selected_arrays(data: Dict[str, Any]) -> tuple[Dict[str, np.ndarray], Dict[str, str]]:
    arrays: Dict[str, np.ndarray] = {}
    sources: Dict[str, str] = {}
    for i, aliases in quantities.items():
        for key in aliases:
            if key in data:
                arrays[i] = np.asarray(data[key])
                sources[i] = key
                break
    return arrays, sources


def save_drive_chunk_h5(
    path: str | Path,
    *,
    omega_d_values: np.ndarray,
    chi_ac_values: np.ndarray,
    scar_maps: Dict[int, np.ndarray],
    extra_arrays: Optional[Dict[str, np.ndarray]] = None,
    metadata: Dict[str, Any],
    timing: Dict[str, Any],
) -> str:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)


    with h5py.File(path, "w") as f:
        f.create_dataset("omega_d_values", data=np.asarray(omega_d_values, dtype=float))
        f.create_dataset("chi_ac_values", data=np.asarray(chi_ac_values, dtype=float))

        for slot, arr in scar_maps.items():
            f.create_dataset(
                f"scar_state{int(slot)}",
                data=np.asarray(arr, dtype=np.float32),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

        if extra_arrays:
            for name, arr in extra_arrays.items():
                if name in ("omega_d_values", "chi_ac_values"):
                    continue
                A = np.asarray(arr)
                f.create_dataset(
                    str(name),
                    data=A,
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )

        g = f.create_group("_metadata")
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool, np.integer, np.floating)):
                g.attrs[str(k)] = v
            else:
                g.attrs[str(k)] = json.dumps(v, default=str)

        f.attrs["timing_json"] = json.dumps(timing, sort_keys=True)

    return str(path)


def worker_drive_sweep_chunk(
    *,
    cfg_dict: Dict[str, Any],
    flux_value: float,
    omega_chunk: Sequence[float],
    chi_ac_values: Sequence[float],
    label_states: Sequence[Label],
    options_dict: Dict[str, Any],
    state_slots: Sequence[int],
    clip_max: float,
    outdir: str,
    chunk_id: int,
) -> Dict[str, Any]:
    tr = Timing()
    t0 = time.perf_counter()

    omega_chunk = np.asarray(list(omega_chunk), dtype=float)
    chi_ac_values = np.asarray(list(chi_ac_values), dtype=float)

    qparams = dict(cfg_dict["qubit_params"])
    qparams["flux"] = float(flux_value)

    cfg = CompositeSysConfig(
        qubit_params=qparams,
        dim_q=int(cfg_dict["dim_q"]),
        dim_r=int(cfg_dict["dim_r"]),
        omega_r=float(cfg_dict["omega_r"]),
        g=float(cfg_dict["g"]),
        coupling_op=str(cfg_dict.get("coupling_op", "n")),
        resonator_quadrature=str(cfg_dict.get("resonator_quadrature", "x")),
    )
    #just checking time for everything for each chunk, including config build, analysis, and extraction. 
    #analysis needs more thorough breakdown in timing (and obviously is the one that takes the most time)

    with tr.span("chunk.build_model"):
        build = build_composite_model(
            cfg,
            omega_d_values=omega_chunk,
            chi_ac_values=chi_ac_values,
            label_states=label_states,
            timer=tr,
        )

    with tr.span("chunk.floquet_run"):
        options = ft.Options(**options_dict)
        data = run_floquet_analysis_composite(build, options=options, timer=tr)

    with tr.span("chunk.extract_branch_arrays"):
        extra_arrays, extra_sources = _extract_selected_arrays(data)

    with tr.span("chunk.extract_scar"):
        scar_maps = extract_scar_maps_from_displaced_overlaps(
            data,
            state_slots=state_slots,
            clip_max=float(clip_max),
        )

    #save
    outpath = Path(outdir) / f"chunk_{int(chunk_id):04d}.h5"
    meta = {
        "kind": "drive_sweep",
        "flux_value": float(flux_value),
        "chunk_id": int(chunk_id),
        "omega_min": float(omega_chunk[0]),
        "omega_max": float(omega_chunk[-1]),
        "dim_q": int(cfg.dim_q),
        "dim_r": int(cfg.dim_r),
        "label_states": [list(x) for x in label_states],
        "state_slots": [int(s) for s in state_slots],
        "saved_arrays": sorted(list(extra_arrays.keys())),
        "saved_arrays_source_keys": extra_sources,
        "cfg_json": cfg.to_dict(),
        "options_json": options_dict,
    }

    with tr.span("chunk.save_h5"):
        file_path = save_drive_chunk_h5(
            outpath,
            omega_d_values=omega_chunk,
            chi_ac_values=chi_ac_values,
            scar_maps=scar_maps,
            extra_arrays=extra_arrays,
            metadata=meta,
            timing=tr.to_dict(),
        )

    t1 = time.perf_counter()
    return {
        "chunk_id": int(chunk_id),
        "file": file_path,
        "omega_min": float(omega_chunk[0]),
        "omega_max": float(omega_chunk[-1]),
        "t_total_s": float(t1 - t0),
        "timing": tr.to_dict(),
    }
