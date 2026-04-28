"""capacitively coupled fluxonium-resonator composite model (use the Y quadrature for the resoantor coupling)"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
from scipy.linalg import eigh

from .fluxonium_model import BareFluxonium
from .operators import destroy, number, identity, basis_vector
from .parameters import ResonatorParams, TruncationConfig, DriveParams
from .units import ghz_to_angular, mhz_to_angular, angular_to_ghz

class CoupledFluxoniumResonator:
    """building the fluxonium-resonator model.

    Hamiltonian:
        H0 = omega_r a^dagger a + H_fluxonium - i g_qr n (a - a^dagger)
    The composite Hilbert-space basis ordering is: |fluxonium_level i> x |resonator_fock n>
    so the bare basis index -> i*N_r + n
    """

    def __init__(
        self,
        fluxonium: BareFluxonium,
        resonator_params: ResonatorParams,
        truncation: TruncationConfig,
    ) -> None:
        self.fluxonium = fluxonium
        self.res_params = resonator_params
        self.truncation = truncation

        self.Nf = truncation.fluxonium_levels
        self.Nr = truncation.resonator_levels
        self.dim = self.Nf * self.Nr

        self.omega_r = ghz_to_angular(resonator_params.omega_r_GHz)
        self.g_qr = ghz_to_angular(resonator_params.g_qr_GHz)
        self.kappa = mhz_to_angular(resonator_params.kappa_MHz)

        self._eigensystem_cache: Tuple[np.ndarray, np.ndarray] | None = None

        self._build_subspace_operators()
        self._build_undriven_hamiltonian()

    def _build_subspace_operators(self) -> None:
        self.Ef, self.Uf = self.fluxonium.eigensystem(self.Nf)
        self.n_flux = self.fluxonium.charge_operator(self.Nf)
        self.phi_flux = self.fluxonium.phase_operator(self.Nf)

        self.a_res = destroy(self.Nr)
        self.ad_res = self.a_res.conj().T
        self.num_res = self.ad_res @ self.a_res

        self.I_flux = identity(self.Nf)
        self.I_res = identity(self.Nr)

        self.a_full = np.kron(self.I_flux, self.a_res)
        self.ad_full = self.a_full.conj().T
        self.num_full = self.ad_full @ self.a_full

    def _build_undriven_hamiltonian(self) -> None:
        H_flux = np.diag(self.Ef)
        H_res = self.omega_r * self.num_res

        self.H0 = np.kron(H_flux, self.I_res) + np.kron(self.I_flux, H_res)
        self.H0 += -1j * self.g_qr * np.kron(self.n_flux, self.a_res - self.ad_res)

    def undriven_hamiltonian(self) -> np.ndarray:
        """ H0 in angular units."""
        return self.H0.copy()

    def drive_operator(self, epsilon_d_GHz: float) -> np.ndarray:
        """
        H_d(t) = sin(omega_d t) * drive_operator(epsilon_d)
        """
        eps = ghz_to_angular(epsilon_d_GHz)
        return -1j * eps * np.kron(self.I_flux, self.a_res - self.ad_res)

    def hamiltonian_t(self, t: float, drive: DriveParams | None = None) -> np.ndarray:
        """H(t)."""
        if drive is None:
            return self.undriven_hamiltonian()
        omega_d = ghz_to_angular(drive.omega_d_GHz)
        return self.H0 + np.sin(omega_d * t) * self.drive_operator(drive.epsilon_d_GHz)

    def collapse_operators(self) -> list[np.ndarray]:
        """collapse operators for resonator decay."""
        if self.kappa <= 0.0:
            return []
        return [np.sqrt(self.kappa) * self.a_full]

    def eigensystem(self) -> Tuple[np.ndarray, np.ndarray]:
        """the full undriven eigensystem."""
        if self._eigensystem_cache is None:
            evals, evecs = eigh(self.H0)
            self._eigensystem_cache = (evals, evecs)
        return self._eigensystem_cache[0].copy(), self._eigensystem_cache[1].copy()

    def eigenvalues_GHz(self, relative_to_ground: bool = False) -> np.ndarray:
        """undriven dressed eigenvalues in GHz."""
        evals, _ = self.eigensystem()
        if relative_to_ground:
            evals = evals - evals[0]
        return angular_to_ghz(evals)

    def bare_state_index(self, fluxonium_level: int, resonator_n: int) -> int:
        """basis index of |fluxonium_level> x |resonator_n>."""
        if not (0 <= fluxonium_level < self.Nf):
            raise IndexError("fluxonium_level out of range")
        if not (0 <= resonator_n < self.Nr):
            raise IndexError("resonator_n out of range")
        return fluxonium_level * self.Nr + resonator_n

    def bare_state_vector(self, fluxonium_level: int, resonator_n: int) -> np.ndarray:
        """basis vector of |fluxonium_level> x |resonator_n>"""
        return basis_vector(self.dim, self.bare_state_index(fluxonium_level, resonator_n))

    def transition_scan_from_state(self, reference_state: int = 0) -> list[dict]:
        """bare fluxonium transition data with coupling strengths."""
        base_table = self.fluxonium.transition_table(self.Nf, reference_state=reference_state)
        for row in base_table:
            nij = self.fluxonium.charge_matrix_element(row["i"], row["j"], levels=self.Nf)
            row["g_ij_GHz_real"] = float(np.real(self.res_params.g_qr_GHz * nij))
            row["g_ij_GHz_imag"] = float(np.imag(self.res_params.g_qr_GHz * nij))
            row["abs_g_ij_GHz"] = float(np.abs(self.res_params.g_qr_GHz * nij))
            row["detuning_to_resonator_GHz"] = float(row["omega_ij_GHz"] - self.res_params.omega_r_GHz)
        return base_table
