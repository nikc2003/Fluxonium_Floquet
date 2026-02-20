
from __future__ import annotations

import os
from floquet_composite_model.src.utils.env import set_single_thread_env
from floquet_composite_model.src.utils.env import CPUPlan
set_single_thread_env(1, force=True)

import argparse
import csv
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import h5py

from floquet_composite_model.src.utils.partitioning import partition_slices
from floquet_composite_model.src.utils.timing import fmt_seconds
from floquet_composite_model.src.parallel.drive_sweep import worker_drive_sweep_chunk
from floquet_composite_model.src.parallel.merge import merge_axis0_chunks

import datetime as dt
today = dt.date.today()
def _stage_seconds(timing: dict, stage: str) -> float:
    return float(timing.get(stage, {}).get("seconds", 0.0))

#change sweep parameters here
def main() -> None:
    path = Path("C:\\Users\\slab\\floquet\\floquet\\floquet_composite_model")
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default= path / f"{today}_results_drive_sweep", help="output directory")
    ap.add_argument("--max-workers", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=25, help="omega points per chunk")
    ap.add_argument("--n-omega", type=int, default=301, help="number of omega points")
    ap.add_argument("--n-chi", type=int, default=251, help="number of chi points")
    ap.add_argument("--omega-min", type=float, default=3.8, help="GHz")
    ap.add_argument("--omega-max", type=float, default=4.2, help="GHz ")
    ap.add_argument("--chi-max", type=float, default=0.2, help="GHz")
    ap.add_argument("--flux", type=float, default=0.501)
    ap.add_argument("--clip-max", type=float, default=0.5)
    args = ap.parse_args()

    plan = CPUPlan.recommend_cpu_plan(max_workers=args.max_workers)
    max_workers = plan.max_workers
    print(f"Using max_workers={max_workers} with CPUPlan: {plan}")
    options_dict = dict(
        fit_range_fraction=0.6,
        fit_cutoff=4,
        overlap_cutoff=0.8,
        nsteps=30000,
        num_cpus=int(plan.inner_num_cpus), 
        save_floquet_modes=False,
    )
    qA_params = {
        "EJ": 4.1184,
        "EC": 0.8283,
        "EL": 1.1929,
        "flux": float(args.flux),
        "cutoff": 110,
    }
    omega_r = 2 * np.pi * 5.097
    g_qr    = 2 * np.pi * 0.026

    cfg_dict = {
        "qubit_params": qA_params,
        "dim_q": 20,
        "dim_r": 8,
        "omega_r": float(omega_r),
        "g": float(g_qr),
        "coupling_op": "n",
        "resonator_quadrature": "x",
    }
    label_states = [(0, 0), (1, 0)]
    state_slots = [0, 1]
    omega_d_values = 2.0 * np.pi * np.linspace(args.omega_min, args.omega_max, args.n_omega)
    chi_ac_values  = 2.0 * np.pi * np.linspace(0.0, args.chi_max, args.n_chi)
    outdir = Path(args.outdir).resolve()
    chunks_dir = outdir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    (outdir / "run_config.json").write_text(json.dumps({
        "cfg": cfg_dict,
        "options": options_dict,
        "label_states": [list(x) for x in label_states],
        "state_slots": state_slots,
        "omega_min_GHz": args.omega_min,
        "omega_max_GHz": args.omega_max,
        "n_omega": args.n_omega,
        "chi_max_GHz": args.chi_max,
        "n_chi": args.n_chi,
        "flux": args.flux,
        "chunk_size": args.chunk_size,
        "max_workers": max_workers,
    }, indent=2, sort_keys=True))

    futures = {}
    ok_files = []
    chunk_rows = []

    t_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        chunk_id = 0
        for s, e in partition_slices(len(omega_d_values), args.chunk_size):
            omega_chunk = omega_d_values[s:e]
            fut = ex.submit(
                worker_drive_sweep_chunk,
                cfg_dict=cfg_dict,
                flux_value=float(args.flux),
                omega_chunk=omega_chunk.tolist(),
                chi_ac_values=chi_ac_values.tolist(),
                label_states=label_states,
                options_dict=options_dict,
                state_slots=state_slots,
                clip_max=float(args.clip_max),
                outdir=str(chunks_dir),
                chunk_id=int(chunk_id),
            )
            futures[fut] = (chunk_id, float(omega_chunk[0]), float(omega_chunk[-1]))
            chunk_id += 1

        done = 0
        durations = []
        for fut in as_completed(futures):
            cid, omin, omax = futures[fut]
            r = fut.result()
            ok_files.append(r["file"])
            done += 1
            durations.append(r["t_total_s"])

            timing = r.get("timing", {})
            t_build = _stage_seconds(timing, "chunk.build_model")
            t_run   = _stage_seconds(timing, "chunk.floquet_run")
            t_branch = _stage_seconds(timing, "chunk.extract_branch_arrays")
            t_save  = _stage_seconds(timing, "chunk.save_h5")
            t_wall  = float(r["t_total_s"])

            t_eigs   = _stage_seconds(timing, "fluxonium.eigensys")
            t_c2a    = _stage_seconds(timing, "ChiacToAmp.compute")
            t_cminit = _stage_seconds(timing, "CompositeModel.init")
            t_farun  = _stage_seconds(timing, "FloquetAnalysis.run")

            chunk_rows.append({
                "chunk_id": cid,
                "omega_min": r["omega_min"],
                "omega_max": r["omega_max"],
                "t_total_s": t_wall,
                "t_build_model_s": t_build,
                "t_floquet_run_s": t_run,
                "t_extract_branch_arrays_s": t_branch,
                "t_save_h5_s": t_save,
                "sec__fluxonium_eigensys": t_eigs,
                "sec__CompositeModel_init": t_cminit,
                "sec__ChiacToAmp_compute": t_c2a,
                "sec__FloquetAnalysis_run": t_farun,
            })

            elapsed = time.perf_counter() - t_start
            mean = sum(durations) / len(durations)
            print(
                f"[{done}/{len(futures)}] chunk {cid:04d} "
                fr"$\omega \in$[{r['omega_min']/(2*np.pi):.3f},{r['omega_max']/(2*np.pi):.3f}]GHz "
                f"t={fmt_seconds(t_wall)} (build={fmt_seconds(t_build)}, run={fmt_seconds(t_run)}, save={fmt_seconds(t_save)}) "
                f"mean={fmt_seconds(mean)} elapsed={fmt_seconds(elapsed)}"
            )
    merged_path = outdir / "merged.h5"
    merged = merge_axis0_chunks(
        ok_files,
        merged_path,
        x_key="omega_d_values",
        y_key="chi_ac_values",
    )
    print("merged ->", merged)

    timing_csv = outdir / "chunk_timings.csv"
    with open(timing_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(chunk_rows[0].keys()))
        writer.writeheader()
        for row in sorted(chunk_rows, key=lambda d: d["chunk_id"]):
            writer.writerow(row)
    print("timing ->", timing_csv)
    with h5py.File(merged_path, "a") as f:
        g = f.require_group("_timing")
        for k in chunk_rows[0].keys():
            g.create_dataset(k, data=np.asarray([row[k] for row in sorted(chunk_rows, key=lambda d: d["chunk_id"])], dtype=float if (k.endswith("_s") or k.startswith("t_") or k.startswith("sec__")) else int))
        f.attrs["timing_note"] = "chunk_timings.csv contains per-chunk stage timing."

    total = time.perf_counter() - t_start
    print("done. total:", fmt_seconds(total))


if __name__ == "__main__":
    main()
