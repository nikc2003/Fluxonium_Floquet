"""fluxonium branch analysis using Aggron qA"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from fluxonium_branch.parameters import FluxoniumParams, ResonatorParams, TruncationConfig
from fluxonium_branch.fluxonium_model import BareFluxonium
from fluxonium_branch.composite_system import CoupledFluxoniumResonator
from fluxonium_branch.branch_analysis import BranchAnalyzer
from fluxonium_branch.diagnostics import branch_metrics, detect_instabilities
from fluxonium_branch.plotting import plot_single_branch, plot_branch_population_distribution
output_dir = Path("qA_branch_analysis_outputs")
output_dir.mkdir(exist_ok=True)

def print_transition_summary(system: CoupledFluxoniumResonator, max_rows: int = 8) -> None:
    table = system.transition_scan_from_state(reference_state=0)
    table = sorted(table, key=lambda row: abs(row["detuning_to_resonator_GHz"]))
    print("\nBare fluxonium transitions from |0_f>:")
    print("  j   omega_0j [GHz]   |n_0j|     |g_0j| [GHz]   detuning to resonator [GHz]")
    for row in table[:max_rows]:
        print(
            f"  {row['j']:2d}   "
            f"{row['omega_ij_GHz']:10.6f}   "
            f"{row['abs_n_ij']:8.5f}   "
            f"{row['abs_g_ij_GHz']:12.6f}   "
            f"{row['detuning_to_resonator_GHz']:11.6f}"
        )

def summarize_branch(metrics: list[dict], label: str) -> None:
    print(f"\nSummary for {label}:")
    header = (
        " rung   eig_idx   <n_r>      avg_flux_level   P_comp      PR_flux   purity"
    )
    print(header)
    for m in metrics[: min(12, len(metrics))]:
        print(
            f" {m['rung']:4d}   {m['eigenstate_index']:7d}   "
            f"{m['avg_resonator_number']:7.3f}   "
            f"{m['avg_fluxonium_level']:14.6f}   "
            f"{m['computational_population']:8.5f}   "
            f"{m['fluxonium_participation_ratio']:8.5f}   "
            f"{m['reduced_fluxonium_purity']:8.5f}"
        )

def main() -> None:
    fluxonium_params = FluxoniumParams(
        EJ_GHz=4.1184,
        EC_GHz=0.8283,
        EL_GHz=1.1929,
        flux=0.501,
        cutoff=110,
    )
    resonator_params = ResonatorParams(
        omega_r_GHz=5.097,
        g_qr_GHz=0.026,
        kappa_MHz=5.0,
    )

    #increase resonator_levels later
    truncation = TruncationConfig(
        fluxonium_levels=12,
        resonator_levels=30,
    )

    print("phase 1: build bare fluxonium")
    fluxonium = BareFluxonium(fluxonium_params)
    evals_GHz = fluxonium.eigenvalues_GHz(truncation.fluxonium_levels, relative_to_ground=True)
    print("\nLowest bare fluxonium energies relative to ground [GHz]:")
    for i, val in enumerate(evals_GHz[:10]):
        print(f"  E_{i} - E_0 = {val:.6f} GHz")

    print("\nphase 2: coupled system...")
    system = CoupledFluxoniumResonator(
        fluxonium=fluxonium,
        resonator_params=resonator_params,
        truncation=truncation,
    )

    print_transition_summary(system)

    print("\nphase 3: diagonalizing undriven coupled Hamiltonian...")
    dressed_evals_GHz = system.eigenvalues_GHz(relative_to_ground=True)
    print(f"Composite Hilbert-space dimension: {system.dim}")
    print("Lowest dressed energies relative to ground [GHz]:")
    for i, val in enumerate(dressed_evals_GHz[:12]):
        print(f"  E_dressed[{i}] - E_0 = {val:.6f} GHz")

    print("\nphase 4: running branch analysis...")
    analyzer = BranchAnalyzer(system)
    result = analyzer.build_branches()
    b0 = branch_metrics(result, branch=0, comp_levels=(0, 1))
    b1 = branch_metrics(result, branch=1, comp_levels=(0, 1))
    summarize_branch(b0, "branch B_0")
    summarize_branch(b1, "branch B_1")

    b0_events = detect_instabilities(b0)
    b1_events = detect_instabilities(b1)

    print("\ndetected branch instability flags for branch 0:")
    print(json.dumps(b0_events[:10], indent=2))
    print("\ndetected branch instability flags for branch 1:")
    print(json.dumps(b1_events[:10], indent=2))

    plot_single_branch(b0, title="Fluxonium branch B_0", savepath=str(output_dir / "branch_B0_metrics.png"))
    plot_single_branch(b1, title="Fluxonium branch B_1", savepath=str(output_dir / "branch_B1_metrics.png"))
    
    #population distributions;  have claude finish later
    snapshot_rungs = [0, min(10, truncation.resonator_levels - 1), truncation.resonator_levels - 1]
    for rung in snapshot_rungs:
        plot_branch_population_distribution(
            b0, rung=rung, savepath=str(output_dir / f"branch_B0_pop_rung_{rung}.png")
        )
        plot_branch_population_distribution(
            b1, rung=rung, savepath=str(output_dir / f"branch_B1_pop_rung_{rung}.png")
        )

    #claude to get json summary
    summary = {
        "fluxonium_params": fluxonium_params.__dict__,
        "resonator_params": resonator_params.__dict__,
        "truncation": truncation.__dict__,
        "b0_first12": [
            {
                key: value
                for key, value in m.items()
                if key not in {"p_fluxonium", "p_resonator"}
            }
            for m in b0[:12]
        ],
        "b1_first12": [
            {
                key: value
                for key, value in m.items()
                if key not in {"p_fluxonium", "p_resonator"}
            }
            for m in b1[:12]
        ],
        "b0_instabilities": b0_events,
        "b1_instabilities": b1_events,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\noutputs in: {output_dir.resolve()}")
if __name__ == "__main__":
    main()
