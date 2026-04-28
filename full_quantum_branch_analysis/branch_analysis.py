"""this will handle branch-assignment (recursively aka blais qubit--resoantor)"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment

from .composite_system import CoupledFluxoniumResonator

@dataclass
class BranchAnalysisResult:
    """for the branch analysis output"""
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    branch_state_indices: np.ndarray 
    seed_fluxonium_levels: np.ndarray
    seed_assignment_scores: np.ndarray  
    step_assignment_scores: np.ndarray  
    composite_dim: int
    fluxonium_levels: int
    resonator_levels: int

    def state_index(self, branch: int, rung: int) -> int:
        return int(self.branch_state_indices[branch, rung])

    def state_vector(self, branch: int, rung: int) -> np.ndarray:
        idx = self.state_index(branch, rung)
        return self.eigenvectors[:, idx].copy()

    def state_energy(self, branch: int, rung: int) -> float:
        idx = self.state_index(branch, rung)
        return float(self.eigenvalues[idx])

    def branch_energies(self, branch: int) -> np.ndarray:
        return self.eigenvalues[self.branch_state_indices[branch, :]].copy()

class BranchAnalyzer:
    """
    dressed-state branches using recursive resonator raising overlaps.
    Given the dressed eigenstates |lambda>, seeds are chosen by overlap with the
    bare states |i_f, 0_r>. The next state in each branch is assigned by
    maximizing |<lambda| a^dagger |psi_current>|^2 among unassigned dressed states.

    ** According to gemini: "grow all branches in parallel" ==> each
    family is assigned with a linear sum assignment as opposed to the greedy one 
    branch-at-a-time .
    """

    def __init__(self, system: CoupledFluxoniumResonator) -> None:
        self.system = system
        self.evals, self.evecs = system.eigensystem()
        self.dim = self.evecs.shape[0]

    def _seed_score_matrix(self, seed_fluxonium_levels: np.ndarray) -> np.ndarray:
        """overlaps with bare |i,0> states."""
        scores = np.zeros((len(seed_fluxonium_levels), self.dim), dtype=float)
        for row, i_f in enumerate(seed_fluxonium_levels):
            bare_index = self.system.bare_state_index(int(i_f), 0)
            scores[row, :] = np.abs(self.evecs[bare_index, :]) ** 2
        return scores

    @staticmethod
    def _solve_unique_assignment(score_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ max-sum unique assignmen; basically just the linear sum assignment/Hungarian on -score_matrix. Returns the chosen column for each row and
        the outputs:
        1. chosen_columns: array of selected column indices, one per row
        2. chosen_scores: the corres. score for each row
        """
        if score_matrix.ndim != 2:
            raise ValueError("score_matrix must be 2D")
        n_rows, n_cols = score_matrix.shape
        if n_rows > n_cols:
            raise ValueError("you cant so unique assignment: more rows than columns")
        row_ind, col_ind = linear_sum_assignment(-score_matrix)
        order = np.argsort(row_ind)
        col_ind = col_ind[order]
        chosen_scores = score_matrix[np.arange(n_rows), col_ind]
        return col_ind.astype(int), chosen_scores.astype(float)

    def build_branches(self, seed_fluxonium_levels: list[int] | None = None) -> BranchAnalysisResult:
        """form all requested branches.

        use one seed per kept fluxonium level.The default is seed_fluxonium_levels = [0,Nf-1]
        """
        if seed_fluxonium_levels is None:
            seed_fluxonium_levels = list(range(self.system.Nf))

        seed_fluxonium_levels_arr = np.asarray(seed_fluxonium_levels, dtype=int)
        if np.any(seed_fluxonium_levels_arr < 0) or np.any(seed_fluxonium_levels_arr >= self.system.Nf):
            raise ValueError("seed_fluxonium_levels contains an out-of-range level index")

        n_branches = len(seed_fluxonium_levels_arr)
        n_steps = self.system.Nr

        branch_state_indices = -np.ones((n_branches, n_steps), dtype=int)
        step_assignment_scores = np.full((n_branches, n_steps - 1), np.nan, dtype=float)
        assigned = np.zeros(self.dim, dtype=bool)

        #overlaps with |i_f, 0_r>
        seed_scores_full = self._seed_score_matrix(seed_fluxonium_levels_arr)
        seed_cols, seed_scores = self._solve_unique_assignment(seed_scores_full)
        branch_state_indices[:, 0] = seed_cols
        assigned[seed_cols] = True

        #now lets recursively try iterating a^dag and assigning the best overlaps with unassigned states. We do this in parallel for all branches using a linear sum assignment at each step.
        a_dag_full = self.system.ad_full
        current_vectors = self.evecs[:, seed_cols] 

        for rung in range(n_steps - 1):
            targets = a_dag_full @ current_vectors  
            overlap_matrix = np.abs(self.evecs.conj().T @ targets) ** 2  
            available = np.flatnonzero(~assigned)
            if len(available) < n_branches:
                raise RuntimeError("insufficient unassigned states remain to continue branch assignment")

            score_matrix = overlap_matrix[available, :].T  #dim = (#branches, # of available)
            chosen_local_cols, chosen_scores = self._solve_unique_assignment(score_matrix)
            chosen_global_cols = available[chosen_local_cols]

            branch_state_indices[:, rung + 1] = chosen_global_cols
            step_assignment_scores[:, rung] = chosen_scores
            assigned[chosen_global_cols] = True
            current_vectors = self.evecs[:, chosen_global_cols]

        return BranchAnalysisResult(
            eigenvalues=self.evals.copy(),
            eigenvectors=self.evecs.copy(),
            branch_state_indices=branch_state_indices,
            seed_fluxonium_levels=seed_fluxonium_levels_arr.copy(),
            seed_assignment_scores=seed_scores.copy(),
            step_assignment_scores=step_assignment_scores.copy(),
            composite_dim=self.dim,
            fluxonium_levels=self.system.Nf,
            resonator_levels=self.system.Nr,
        )
