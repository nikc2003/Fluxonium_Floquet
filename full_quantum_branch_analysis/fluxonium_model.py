"""Bare fluxonium model and diagonalization."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
from scipy.linalg import eigh, expm

from .operators import destroy
from .parameters import FluxoniumParams
from .units import ghz_to_angular, angular_to_ghz

class BareFluxonium:
    """i think its a simple and good exercise for me to do this by hand! going to use the harmonic-oscillator basis.
        also double checked with the scqubits repo to make sure i do the right things
    The Hamiltonian convention matches the scqubits form:
    H = 4 E_C n^2 + (1/2) E_L phi^2 - E_J cos(phi + phi_ext), phi_ext = 2*pi*flux
    """
    def __init__(self, params: FluxoniumParams) -> None:
        self.params = params
        self.cutoff = params.cutoff
        self.EJ = ghz_to_angular(params.EJ_GHz)
        self.EC = ghz_to_angular(params.EC_GHz)
        self.EL = ghz_to_angular(params.EL_GHz)
        self.phi_ext = 2.0 * np.pi * params.flux
        self._eigensystem_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._projected_operator_cache: Dict[Tuple[str, int], np.ndarray] = {}
        self._build_harmonic_basis_representation()

    def _build_harmonic_basis_representation(self) -> None:
        b = destroy(self.cutoff)
        bd = b.conj().T
        self.phi_zpf = (2.0 * self.EC / self.EL) ** 0.25
        self.n_zpf = (self.EL / (32.0 * self.EC)) ** 0.25
        self.phi_operator_ho = self.phi_zpf * (b + bd)
        self.n_operator_ho = 1j * self.n_zpf * (bd - b)
        self.H_lc_ho = np.sqrt(8.0 * self.EC * self.EL) * (bd @ b + 0.5 * np.eye(self.cutoff))
        #josephson term -->  matrix exponential of phi
        e_iphi = expm(1j * self.phi_operator_ho)
        cos_phi_plus_ext = 0.5 * (
            np.exp(1j * self.phi_ext) * e_iphi
            + np.exp(-1j * self.phi_ext) * e_iphi.conj().T
        )
        self.H_ho = self.H_lc_ho - self.EJ * cos_phi_plus_ext

    def eigensystem(self, levels: int) -> Tuple[np.ndarray, np.ndarray]:
        """"
        eigenvectors are columns in the returned matrix and are expressed in the
        harmonic-oscillator basis.
        """
        if levels < 1:
            raise ValueError("levels must be >= 1")
        if levels > self.cutoff:
            raise ValueError(
                f" {levels} levels but cutoff is only {self.cutoff}. "
                "so increase the harmonic-basis cutoff."
            )
        if levels not in self._eigensystem_cache:
            evals, evecs = eigh(self.H_ho)
            self._eigensystem_cache[levels] = (evals[:levels].copy(), evecs[:, :levels].copy())
        return self._eigensystem_cache[levels]

    def eigenvalues(self, levels: int, relative_to_ground: bool = False) -> np.ndarray:
        """eigenvalues in angular units."""
        evals, _ = self.eigensystem(levels)
        if relative_to_ground:
            return evals - evals[0]
        return evals.copy()

    def eigenvalues_GHz(self, levels: int, relative_to_ground: bool = False) -> np.ndarray:
        """eigenvalues in GHz."""
        return angular_to_ghz(self.eigenvalues(levels, relative_to_ground=relative_to_ground))

    def operator_in_eigenbasis(self, operator_name: str, levels: int) -> np.ndarray:
        """
        project a bare operator into the lowes enerrgy levels of fluxonium subspace.
        - operator names: phi, n, H
        """
        key = (operator_name, levels)
        if key in self._projected_operator_cache:
            return self._projected_operator_cache[key].copy()

        _, evecs = self.eigensystem(levels)

        if operator_name == "phi":
            op_ho = self.phi_operator_ho
        elif operator_name == "n":
            op_ho = self.n_operator_ho
        elif operator_name == "H":
            op_ho = self.H_ho
        else:
            raise ValueError(f"unknown operator_name={operator_name!r}")

        projected = evecs.conj().T @ op_ho @ evecs
        self._projected_operator_cache[key] = projected.copy()
        return projected

    def charge_operator(self, levels: int) -> np.ndarray:
        """ the charge operator n in the eigenbasis."""
        return self.operator_in_eigenbasis("n", levels)

    def phase_operator(self, levels: int) -> np.ndarray:
        """ the phase operator phi in the eigenbasis."""
        return self.operator_in_eigenbasis("phi", levels)

    def transition_frequency(self, i: int, j: int, levels: int | None = None) -> float:
        """omega_ij = E_j - E_i in angular units."""
        needed = max(i, j) + 1 if levels is None else levels
        evals, _ = self.eigensystem(needed)
        return float(evals[j] - evals[i])

    def transition_frequency_GHz(self, i: int, j: int, levels: int | None = None) -> float:
        """omega_ij = E_j - E_i in GHz."""
        return float(angular_to_ghz(self.transition_frequency(i, j, levels=levels)))

    def charge_matrix_element(self, i: int, j: int, levels: int | None = None) -> complex:
        """<i|n|j> in the bare fluxonium eigenbasis."""
        needed = max(i, j) + 1 if levels is None else levels
        n_op = self.charge_operator(needed)
        return complex(n_op[i, j])

    def coupling_matrix_element_GHz(self, i: int, j: int, g_qr_GHz: float, levels: int | None = None) -> complex:
        """g_ij = g_qr * <i|n|j> in GHz."""
        return g_qr_GHz * self.charge_matrix_element(i, j, levels=levels)

    def transition_table(self, levels: int, reference_state: int = 0) -> list[dict]:
        """transition data from `reference_state` to all other kept states."""
        if not (0 <= reference_state < levels):
            raise ValueError("reference_state must be within the kept subspace")
        table = []
        n_op = self.charge_operator(levels)
        evals_GHz = self.eigenvalues_GHz(levels, relative_to_ground=False)
        Ei = evals_GHz[reference_state]
        for j in range(levels):
            if j == reference_state:
                continue
            table.append(
                {
                    "i": reference_state,
                    "j": j,
                    "omega_ij_GHz": float(evals_GHz[j] - Ei),
                    "n_ij_real": float(np.real(n_op[reference_state, j])),
                    "n_ij_imag": float(np.imag(n_op[reference_state, j])),
                    "abs_n_ij": float(np.abs(n_op[reference_state, j])),
                }
            )
        return table
