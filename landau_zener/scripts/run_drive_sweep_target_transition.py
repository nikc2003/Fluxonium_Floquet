"""
/scripts/run_drive_sweep_target_transition.py
targeted drive sweep for a known bare transition (forex. 0->4, 2-photon).
"""

from __future__ import annotations
import os
import argparse
import importlib.util
import numpy as np
import json
import datetime
import platform
import subprocess
from pathlib import Path
from dataclasses import asdict, is_dataclass
import sys

from landau_zener.analysis.pipeline import run_drive_sweep_target_transition
from landau_zener.plotting.scatter import plot_gap_vs_deltachi_scatter
from landau_zener.plotting.events import plot_event_annotated_gap
from landau_zener.plotting.fan import plot_truncation_fan_diagnostic
from landau_zener.utils.units import ghz_to_angular
import json
import datetime
import platform
import subprocess
from pathlib import Path
from dataclasses import asdict, is_dataclass
import numpy as np
import sys


def _jsonable(x):
    if x is None or isinstance(x, (str, int, float, bool)):
        return x

    if isinstance(x, Path):
        return str(x)

    if isinstance(x, range):
        return list(x)

    if isinstance(x, np.ndarray):
        return x.tolist()

    if isinstance(x, (np.integer,)):
        return int(x)

    if isinstance(x, (np.floating,)):
        return float(x)

    if isinstance(x, (np.bool_,)):
        return bool(x)

    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}

    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]

    if is_dataclass(x):
        return _jsonable(asdict(x))

    if hasattr(x, "__dict__"):
        return {k: _jsonable(v) for k, v in vars(x).items() if not k.startswith("_")}

    return repr(x)


def _git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def save_run_config(args, cfg, drive_freqs_GHz):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    qubit_params = getattr(cfg, "QUBIT_PARAMS", getattr(cfg, "qA_params", None))

    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "cwd": str(Path.cwd()),
        "platform": platform.platform(),
        "git_commit": _git_hash(),
        "command": " ".join(sys.argv),
        "config_path": str(Path(args.config).resolve()),
        "cli_args": vars(args),
        "resolved": {
            "drive_freqs_GHz": [float(f) for f in drive_freqs_GHz],
            "f_center_GHz": float(args.f_center),
            "f_span_GHz": float(args.f_span),
            "nfreq": int(args.nfreq),
            "target_bare_pair": [int(args.bare_a), int(args.bare_b)],
            "target_n_photon": None if args.n_photon is None else int(args.n_photon),
            "chi_range_GHz": [float(args.chi_min), float(args.chi_max)],
            "flux": float(args.flux),
            "dim_q": int(args.dim),
            "parallel": bool(args.parallel),
            "workers": int(args.workers),
            "plot_topk": int(args.plot_topk),
        },
        "module_config": {
            "QUBIT_PARAMS": qubit_params,
            "OPTIONS": getattr(cfg, "OPTIONS", None),
            "DETECT_CFG": getattr(cfg, "DETECT_CFG", None),
            "ISO_CFG": getattr(cfg, "ISO_CFG", None),
            "CROWD_CFG": getattr(cfg, "CROWD_CFG", None),
            "LIN_CFG": getattr(cfg, "LIN_CFG", None),
            "SLOPE_CFG": getattr(cfg, "SLOPE_CFG", None),
        },
    }

    record = _jsonable(record)

    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    print(f"Saved run config: {outdir / 'run_config.json'}")

def load_config_module(path: str):
    spec = importlib.util.spec_from_file_location("user_cfg", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load config from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config python file (devices/config_qA.py)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--flux", type=float, default=None)
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--chi-min", type=float, default=None)
    ap.add_argument("--chi-max", type=float, default=None)
    ap.add_argument("--nchi-sweep", type=int, default=1201)

    ap.add_argument("--f-center", type=float, required=True)
    ap.add_argument("--f-span", type=float, default=0.09)
    ap.add_argument("--nfreq", type=int, default=9)

    ap.add_argument("--bare-a", type=int, required=True)
    ap.add_argument("--bare-b", type=int, required=True)
    ap.add_argument("--n-photon", type=int, default=None)

    ap.add_argument("--kappa-ghz", type=float, default=0.005)

    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--workers", type=int, default=4)

    ap.add_argument("--plot-topk", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config_module(args.config)

    dim_q = args.dim if args.dim is not None else cfg.DEFAULT_DIM_Q
    flux = args.flux if args.flux is not None else cfg.DEFAULT_FLUX
    chi_range = (
        args.chi_min if args.chi_min is not None else cfg.DEFAULT_CHI_RANGE_GHZ[0],
        args.chi_max if args.chi_max is not None else cfg.DEFAULT_CHI_RANGE_GHZ[1],
    )

    if args.kappa_ghz is None:
        kappa_res = cfg.kappa_res_angular()
    else:
        kappa_res = ghz_to_angular(float(args.kappa_ghz))

    f0 = float(args.f_center)
    span = float(args.f_span)
    nfreq = int(args.nfreq)
    drive_freqs = np.linspace(f0 - 0.5 * span, f0 + 0.5 * span, nfreq).tolist()
    save_run_config(args, cfg, drive_freqs)

    os.makedirs(args.outdir, exist_ok=True)

    df = run_drive_sweep_target_transition(
        drive_freqs_GHz=drive_freqs,
        dim_q=dim_q,
        qubit_params=cfg.QUBIT_PARAMS,
        flux=flux,
        chi_range_GHz=chi_range,
        options=cfg.OPTIONS,
        outdir=args.outdir,
        detect_cfg=cfg.DETECT_CFG,
        iso_cfg=cfg.ISO_CFG,
        crowd_cfg=None,
        kappa_res=kappa_res,
        lin_cfg=cfg.LIN_CFG,
        slope_cfg=cfg.SLOPE_CFG,
        target_bare_pair=(args.bare_a, args.bare_b),
        target_n_photon=args.n_photon,
        parallel=args.parallel,
        max_workers=int(args.workers),
        nchi_sweep=int(args.nchi_sweep),
    )
    if df is None or len(df) == 0:
        print("No target events found.")
        return

    #
    plot_gap_vs_deltachi_scatter(
        df,
        outdir=args.outdir,
        color_by="freq_GHz" if "freq_GHz" in df.columns else None,
        title=f"Target transition {args.bare_a}->{args.bare_b}  Δmin vs δχ",
        filename="target_scatter_Delta_vs_dchi.png",
        show_table=True,
    )

    from landau_zener.floquet.sweep import floquet_sweep_vs_chi
    chi_vals = ghz_to_angular(np.linspace(chi_range[0], chi_range[1], 2001))
    omega_d0 = ghz_to_angular(f0)
    sweep = floquet_sweep_vs_chi(
        dim_q=dim_q,
        qubit_params=cfg.QUBIT_PARAMS,
        flux=flux,
        omega_d=omega_d0,
        chi_vals=chi_vals,
        options=cfg.OPTIONS,
    )
    plot_truncation_fan_diagnostic(
        sweep,
        chi_units="GHz",
        energy_units="GHz",
        use_unwrapped=False,
        highlight_top_n_branches=10,
        top_m_leak=3,
        leak_thresh=0.10,
        outdir=args.outdir,
        filename="truncation_fan_diag.png",
        title=f"Truncation diagnostic fan (fd={f0:.4f} GHz)",
        show=False,
    )

    dsort = df.sort_values("Delta_min_MHz", ascending=True).head(int(args.plot_topk))
    for k, (_, row) in enumerate(dsort.iterrows()):
        plot_event_annotated_gap(
            sweep=sweep,
            event=row,
            chi_units="MHz",
            energy_units="MHz",
            use_unwrapped=False,
            observable="nbar",
            outdir=args.outdir,
            filename=f"annotated_gap_{k}.png",
            show=False,
        )

    print("Done. Outputs in:", args.outdir)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    
    main()