from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, Sequence, Optional, List

import numpy as np
import h5py
import floquet as ft

from ..composite.config import CompositeSysConfig, Label
from ..composite.model import build_composite_model
from ..composite.analysis import run_floquet_analysis_composite, extract_scar_maps_from_displaced_overlaps
from ..utils.timing import Timing
"""
Module for running flux sweeps in parallel. Each worker gets assigned a chunk of flux points, and loops through them sequentially, saving results for each point to disk.
Should save: scar maps for each flux point, drive frequencies and chi ac values used in the analysis, and avg excitation and quasienergy arrays.
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
    for canon, aliases in quantities.items():
        for key in aliases:
            if key in data:
                arrays[canon] = np.asarray(data[key])
                sources[canon] = key
                break
    return arrays, sources

def save_flux_chunk_h5(
    path: str | Path,
    *,
    flux_values: np.ndarray,
    chi_ac_values: np.ndarray,
    omega_used: np.ndarray,
    scar_maps: Dict[int, np.ndarray],
    extra_arrays: Optional[Dict[str, np.ndarray]] = None,
    metadata: Dict[str, Any],
    timing: Dict[str, Any],
) -> str:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.create_dataset("flux_values", data=np.asarray(flux_values, dtype=float))
        f.create_dataset("chi_ac_values", data=np.asarray(chi_ac_values, dtype=float))
        f.create_dataset("omega_used", data=np.asarray(omega_used, dtype=float), compression="gzip", compression_opts=4, shuffle=True)


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
                A = np.asarray(arr)
                if name in ("flux_values", "chi_ac_values", "omega_used"):
                    continue
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


def worker_flux_sweep_chunk(
    *,
    cfg_dict: Dict[str, Any],
    flux_chunk: Sequence[float],
    chi_ac_values: Sequence[float],
    omega_d_bare: float,
    chi_v_flux_GHz_chunk: Optional[Sequence[float]],
    label_states: Sequence[Label],
    options_dict: Dict[str, Any],
    state_slots: Sequence[int],
    clip_max: float,
    outdir: str,
    chunk_id: int,
) -> Dict[str, Any]:
    tr = Timing()
    t0 = time.perf_counter()

    flux_chunk = np.asarray(list(flux_chunk), dtype=float)
    chi_ac_values = np.asarray(list(chi_ac_values), dtype=float)

    if chi_v_flux_GHz_chunk is not None:
        chi_v_flux_GHz_chunk = np.asarray(list(chi_v_flux_GHz_chunk), dtype=float)
        if len(chi_v_flux_GHz_chunk) != len(flux_chunk):
            raise ValueError("chi_v_flux_GHz_chunk must match flux_chunk length")

    scar_maps = {int(s): np.zeros((len(flux_chunk), len(chi_ac_values)), dtype=np.float32) for s in state_slots}
    omega_used = np.zeros(len(flux_chunk), dtype=float)

    options = ft.Options(**options_dict)

    for i, flx in enumerate(flux_chunk):
        qparams = dict(cfg_dict["qubit_params"])
        qparams["flux"] = float(flx)
        cfg = CompositeSysConfig(
            qubit_params=qparams,
            dim_q=int(cfg_dict["dim_q"]),
            dim_r=int(cfg_dict["dim_r"]),
            omega_r=float(cfg_dict["omega_r"]),
            g=float(cfg_dict["g"]),
            coupling_op=str(cfg_dict.get("coupling_op", "n")),
            resonator_quadrature=str(cfg_dict.get("resonator_quadrature", "x")),
        )

        omega_d = float(omega_d_bare)
        if chi_v_flux_GHz_chunk is not None:
            omega_d = float(omega_d_bare) + 2 * np.pi * float(chi_v_flux_GHz_chunk[i])
        omega_used[i] = omega_d

        with tr.span("per_flux.build_model"):
            build = build_composite_model(
                cfg,
                omega_d_values=np.array([omega_d], dtype=float),
                chi_ac_values=chi_ac_values,
                label_states=label_states,
                timer=tr,
            )
            

        with tr.span("per_flux.floquet_run"):
            data = run_floquet_analysis_composite(build, options=options, timer=tr)

        with tr.span("per_flux.extract_branch_arrays"):
            extra_arrays, extra_sources = _extract_selected_arrays(data)

        with tr.span("per_flux.extract_scar"):
            maps_i = extract_scar_maps_from_displaced_overlaps(data, state_slots=state_slots, clip_max=float(clip_max))
            for slot, arr in maps_i.items():
                scar_maps[int(slot)][i, :] = arr[0, :]

    outpath = Path(outdir) / f"chunk_{int(chunk_id):04d}.h5"
    meta = {
        "kind": "flux_sweep",
        "chunk_id": int(chunk_id),
        "flux_min": float(flux_chunk[0]),
        "flux_max": float(flux_chunk[-1]),
        "omega_d_bare": float(omega_d_bare),
        "dim_q": int(cfg_dict["dim_q"]),
        "dim_r": int(cfg_dict["dim_r"]),
        "label_states": [list(x) for x in label_states],
        "state_slots": [int(s) for s in state_slots],
        "cfg_json": cfg_dict,
        "options_json": options_dict,    }

    with tr.span("chunk.save_h5"):
        file_path = save_flux_chunk_h5(
            outpath,
            flux_values=flux_chunk,
            chi_ac_values=chi_ac_values,
            omega_used=omega_used,
            scar_maps=scar_maps,
            extra_arrays=extra_arrays,
            extra_sources=extra_sources,
            metadata=meta,
            timing=tr.to_dict(),
        )

    t1 = time.perf_counter()
    return {
        "chunk_id": int(chunk_id),
        "file": file_path,
        "flux_min": float(flux_chunk[0]),
        "flux_max": float(flux_chunk[-1]),
        "t_total_s": float(t1 - t0),
        "timing": tr.to_dict(),
    }
