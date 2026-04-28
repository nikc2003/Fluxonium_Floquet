"""
mist/floquet/sweep.py
floquet sweeps vs chi, track branches by continuity, store modes/quasienergies.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np

import qutip as qt
import floquet as ft  

from landau_zener.physics.hamiltonian import get_H0_H1_for_flux
from landau_zener.utils.units import wrap_to_bz, unwrap_branches_by_continuity


def basis_weights(modes: np.ndarray) -> np.ndarray:
    """modes: (nchi, nbranch, dim of hs)"""
    return np.abs(modes) ** 2


def avg_bare_level_index_from_modes(modes: np.ndarray) -> np.ndarray:
    """
    avg_n_branch(chi) = sum_m m * |<m|phi_b>|^2
    """
    W = np.abs(modes) ** 2
    dim = W.shape[-1]
    levels = np.arange(dim, dtype=float)
    return W @ levels


@dataclass
class FloquetSweep:
    chi: np.ndarray              
    qe_wrapped: np.ndarray    
    qe_unwrapped: np.ndarray     
    modes: np.ndarray             
    avg_metric_raw: np.ndarray    
    nbar: np.ndarray             
    omega_d: float
    flux: float


def floquet_sweep_vs_chi(
    dim_q: int,
    qubit_params: Dict[str, Any],
    flux: float,
    omega_d: float,
    chi_vals: np.ndarray,
    options: ft.Options,
    *,
    state_indices_for_chi: Sequence[int] = (0, 1),
    analysis_state_indices: Optional[Sequence[int]] = None,
    init_prev_modes: Optional[np.ndarray] = None,
) -> FloquetSweep:
    """
    Sweep chi -> drive amplitude via ChiacToAmp, run Floquet at each point,
    and order branches using the floquet package's Blais stepping.
    """
    chi_vals = np.array(chi_vals, dtype=float)
    if analysis_state_indices is None:
        analysis_state_indices = list(range(int(dim_q)))
    else:
        analysis_state_indices = list(analysis_state_indices)

    H0, H1 = get_H0_H1_for_flux(float(flux), qubit_params, int(dim_q))
    chi_to_amp = ft.ChiacToAmp(H0, H1, list(state_indices_for_chi), np.array([omega_d]))
    drive_amps = chi_to_amp.amplitudes_for_omega_d(chi_vals)  # (nchi,1)
    amps = np.array(drive_amps[:, 0], dtype=float)
    model = ft.Model(
        H0=H0,
        H1=H1,
        omega_d_values=np.array([omega_d]),
        drive_amplitudes=drive_amps,
    )
    fa = ft.FloquetAnalysis(model=model, state_indices=analysis_state_indices, options=options)

    prev_f_modes = init_prev_modes
    if prev_f_modes is None:
        prev_f_modes = fa.bare_state_array()  

    K = len(chi_vals)
    N = int(dim_q)

    modes = np.zeros((K, N, N), dtype=complex)
    qe = np.zeros((K, N), dtype=float)
    avg_raw = np.zeros((K, N), dtype=float)

    for k in range(K):
        f_modes_t, f_energies = fa.run_one_floquet((omega_d, float(amps[k])))
        avg_k, qe_k, modes_k = fa._step_in_amp((f_modes_t, f_energies), prev_f_modes)

        qe_k = np.array(qe_k, dtype=float)
        if qe_k.shape[0] != N:
            raise RuntimeError(
                f"floquet returned {qe_k.shape[0]} quasienergies but dim_q={N}. "
                f"check truncation/state_indices usage."
            )

        avg_raw[k, :] = np.real(avg_k)
        qe[k, :] = np.real(qe_k)
        modes[k, :, :] = modes_k
        prev_f_modes = modes_k

    qe_wrapped = wrap_to_bz(qe, float(omega_d))
    qe_unwrapped = unwrap_branches_by_continuity(qe_wrapped, float(omega_d))
    nbar = avg_bare_level_index_from_modes(modes)

    return FloquetSweep(
        chi=chi_vals,
        qe_wrapped=qe_wrapped,
        qe_unwrapped=qe_unwrapped,
        modes=modes,
        avg_metric_raw=avg_raw,
        nbar=nbar,
        omega_d=float(omega_d),
        flux=float(flux),
    )


def diabatic_energies_floquet(sweep: FloquetSweep) -> np.ndarray:
    """
    eps_dia_m(chi) = sum_b eps_b(chi)*|<m|phi_b(chi)>|^2  (spectral decomposition)
    """
    E = np.asarray(sweep.qe_unwrapped, dtype=float)        
    W = basis_weights(sweep.modes)                        
    eps = np.sum(E[:, :, None] * W, axis=1)               
    return eps