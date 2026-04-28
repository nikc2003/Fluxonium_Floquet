"""
landau_zener/analysis/photon.py
1. gett the physical photon order n_phys from undriven bare energies (H0) and drive frequency.
2. put consistent labels like: |a,n_cross> <-> |b,n_cross-n_phys>
3. keep this separate from Floquet zone or wrapped-quasienergy indices.
"""

from __future__ import annotations
from typing import Any, Dict, Tuple

import numpy as np

from landau_zener.physics.hamiltonian import get_H0_H1_for_flux


def bare_energies_rad_ns(*, qubit_params: Dict[str, Any], flux: float, dim_q: int) -> np.ndarray:
    """
    undriven bare energies in rad/ns for levels [0..dim_q-1], with E0=0.
    """
    H0, _H1 = get_H0_H1_for_flux(float(flux), qubit_params, int(dim_q))
    E = np.real(np.diag(H0.full())).astype(float)
    return E


def physical_photon_number(
    *,
    E_bare: np.ndarray,
    bare_a: int,
    bare_b: int,
    omega_d: float,
) -> Dict[str, Any]:
    """
    photon number based on undriven bare energy gap:
     n_phys = round((E_b - E_a)/omega_d)
    and returns the folded detuning:
        det_folded = (E_b - E_a) - n_phys*omega_d
    which should be near 0 at resonance.
    """
    a = int(bare_a)
    b = int(bare_b)
    if omega_d == 0:
        return dict(n_photon_phys=np.nan, detuning_folded=np.nan, dE_bare=np.nan)

    dE = float(E_bare[b] - E_bare[a])
    n_phys = int(np.round(dE / float(omega_d)))
    det_folded = float(dE - n_phys * float(omega_d))
    return dict(
        dE_bare=float(dE),
        n_photon_phys=int(n_phys),          
        n_photon_phys_abs=int(abs(n_phys)), 
        detuning_folded=float(det_folded),
    )


def floquet_replica_label(
    *,
    bare_a: int,
    bare_b: int,
    n_photon_phys: int,
    n_cross: int = 10,
) -> str:
    """
    label floquet replicas in the common convention:
    so E_b - E_a = n*omega_d  =>  |a,k> couples to |b,k-n|
    so show: |a,n_cross> <-> |b,n_cross-n|
    """
    a = int(bare_a)
    b = int(bare_b)
    n = int(n_photon_phys)
    k = int(n_cross)
    return f"|{a},{k}> <-> |{b},{k - n}>"