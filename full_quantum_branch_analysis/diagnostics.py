"""State and branch diagnostics."""

from __future__ import annotations

from typing import Iterable, Sequence
import numpy as np

from .branch_analysis import BranchAnalysisResult
from .units import angular_to_ghz

def _reshape_state(psi: np.ndarray, fluxonium_levels: int, resonator_levels: int) -> np.ndarray:
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    expected_dim = fluxonium_levels * resonator_levels
    if psi.size != expected_dim:
        raise ValueError(
            f"State dimension {psi.size} incompatible with "
            f"{fluxonium_levels} x {resonator_levels} composite basis"
        )
    return psi.reshape((fluxonium_levels, resonator_levels))

def state_distributions(
    psi: np.ndarray,
    fluxonium_levels: int,
    resonator_levels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """marginal populations over fluxonium and resonator bare bases."""
    psi_fr = _reshape_state(psi, fluxonium_levels, resonator_levels)
    prob = np.abs(psi_fr) ** 2
    p_fluxonium = prob.sum(axis=1)
    p_resonator = prob.sum(axis=0)
    return p_fluxonium, p_resonator

def reduced_fluxonium_density_matrix(
    psi: np.ndarray,
    fluxonium_levels: int,
    resonator_levels: int,
) -> np.ndarray:
    """Trace out the resonator from a pure composite state"""
    psi_fr = _reshape_state(psi, fluxonium_levels, resonator_levels)
    return psi_fr @ psi_fr.conj().T

def reduced_fluxonium_purity(
    psi: np.ndarray,
    fluxonium_levels: int,
    resonator_levels: int,
) -> float:
    """Purity of the reduced fluxonium state"""
    rho_f = reduced_fluxonium_density_matrix(psi, fluxonium_levels, resonator_levels)
    return float(np.real(np.trace(rho_f @ rho_f)))

def state_metrics(
    psi: np.ndarray,
    fluxonium_levels: int,
    resonator_levels: int,
    comp_levels: Sequence[int] = (0, 1),
    seed_fluxonium_level: int | None = None,
) -> dict:
    """Compute diagnostics for a single dressed state."""
    p_fluxonium, p_resonator = state_distributions(
        psi=psi,
        fluxonium_levels=fluxonium_levels,
        resonator_levels=resonator_levels,
    )

    avg_fluxonium_level = float(np.dot(np.arange(fluxonium_levels, dtype=float), p_fluxonium))
    avg_resonator_number = float(np.dot(np.arange(resonator_levels, dtype=float), p_resonator))
    comp_population = float(np.sum([p_fluxonium[i] for i in comp_levels if 0 <= i < fluxonium_levels]))
    fluxonium_participation_ratio = float(1.0 / np.sum(p_fluxonium**2))
    purity = reduced_fluxonium_purity(
        psi=psi,
        fluxonium_levels=fluxonium_levels,
        resonator_levels=resonator_levels,
    )

    seed_level_population = None
    seed_bare_zero_photon_overlap = None
    if seed_fluxonium_level is not None:
        seed_level_population = float(p_fluxonium[seed_fluxonium_level])  # summed over all resonator numbers
        psi_fr = _reshape_state(psi, fluxonium_levels, resonator_levels)
        seed_bare_zero_photon_overlap = float(np.abs(psi_fr[seed_fluxonium_level, 0]) ** 2)

    dominant_fluxonium_level = int(np.argmax(p_fluxonium))
    dominant_fluxonium_population = float(np.max(p_fluxonium))
    dominant_resonator_fock = int(np.argmax(p_resonator))
    dominant_resonator_population = float(np.max(p_resonator))

    return {
        "avg_fluxonium_level": avg_fluxonium_level,
        "avg_resonator_number": avg_resonator_number,
        "computational_population": comp_population,
        "fluxonium_participation_ratio": fluxonium_participation_ratio,
        "reduced_fluxonium_purity": purity,
        "seed_fluxonium_population": seed_level_population,
        "seed_bare_zero_photon_overlap": seed_bare_zero_photon_overlap,
        "dominant_fluxonium_level": dominant_fluxonium_level,
        "dominant_fluxonium_population": dominant_fluxonium_population,
        "dominant_resonator_fock": dominant_resonator_fock,
        "dominant_resonator_population": dominant_resonator_population,
        "p_fluxonium": p_fluxonium,
        "p_resonator": p_resonator,
    }

def branch_metrics(
    result: BranchAnalysisResult,
    branch: int,
    comp_levels: Sequence[int] = (0, 1),
) -> list[dict]:
    """Compute diagnostics for every state along one branch."""
    metrics = []
    seed_level = int(result.seed_fluxonium_levels[branch])

    for rung in range(result.resonator_levels):
        lam = int(result.branch_state_indices[branch, rung])
        psi = result.eigenvectors[:, lam]
        state_dict = state_metrics(
            psi=psi,
            fluxonium_levels=result.fluxonium_levels,
            resonator_levels=result.resonator_levels,
            comp_levels=comp_levels,
            seed_fluxonium_level=seed_level,
        )
        state_dict.update(
            {
                "branch": int(branch),
                "seed_fluxonium_level": seed_level,
                "rung": int(rung),
                "eigenstate_index": lam,
                "energy_angular": float(result.eigenvalues[lam]),
                "energy_GHz": float(angular_to_ghz(result.eigenvalues[lam] - result.eigenvalues[0])),
                "seed_assignment_score": float(result.seed_assignment_scores[branch]) if rung == 0 else np.nan,
                "step_assignment_score": np.nan if rung == 0 else float(result.step_assignment_scores[branch, rung - 1]),
            }
        )
        metrics.append(state_dict)
    return metrics

def detect_instabilities(
    metrics: list[dict],
    avg_fluxonium_jump_threshold: float = 1.0,
    comp_population_drop_threshold: float = 0.2,
    purity_drop_threshold: float = 0.1,
) -> list[dict]:
    """ detect abrupt changes along a branch."""
    events = []
    for prev, curr in zip(metrics[:-1], metrics[1:]):
        d_avg_level = curr["avg_fluxonium_level"] - prev["avg_fluxonium_level"]
        d_comp = curr["computational_population"] - prev["computational_population"]
        d_purity = curr["reduced_fluxonium_purity"] - prev["reduced_fluxonium_purity"]

        if (
            abs(d_avg_level) >= avg_fluxonium_jump_threshold
            or d_comp <= -abs(comp_population_drop_threshold)
            or d_purity <= -abs(purity_drop_threshold)
        ):
            events.append(
                {
                    "branch": curr["branch"],
                    "seed_fluxonium_level": curr["seed_fluxonium_level"],
                    "between_rungs": (prev["rung"], curr["rung"]),
                    "delta_avg_fluxonium_level": float(d_avg_level),
                    "delta_computational_population": float(d_comp),
                    "delta_reduced_fluxonium_purity": float(d_purity),
                    "prev_avg_resonator_number": float(prev["avg_resonator_number"]),
                    "curr_avg_resonator_number": float(curr["avg_resonator_number"]),
                }
            )
    return events
