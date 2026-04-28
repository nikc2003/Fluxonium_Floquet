"""
time-evolution 
not needed for branch analysis itself but i tried to include it so that eventually the same package can also be used for direct driven simulations of the
fluxonium-resonator model with resonator decay (branching off of Google IST). I would take this section as hand-wavy and not super polished...

Because the full density matrix scales as (Nf*Nr)^2, this is computationally very expensive! hence why i havent really tried to test or optimize it much yet
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

from .composite_system import CoupledFluxoniumResonator
from .parameters import DriveParams

def lindblad_rhs(
    t: float,
    rho_vec: np.ndarray,
    system: CoupledFluxoniumResonator,
    drive: DriveParams | None = None,
) -> np.ndarray:
    """right hand side of the Lindblad master equation in vectorized form"""
    dim = system.dim
    rho = np.asarray(rho_vec, dtype=complex).reshape((dim, dim))
    Ht = system.hamiltonian_t(t, drive=drive)

    drho = -1j * (Ht @ rho - rho @ Ht)
    for c in system.collapse_operators():
        cd = c.conj().T
        drho += c @ rho @ cd - 0.5 * (cd @ c @ rho + rho @ cd @ c)
    return drho.reshape(dim * dim)

def solve_master_equation(
    system: CoupledFluxoniumResonator,
    rho0: np.ndarray,
    t_span: tuple[float, float],
    t_eval: np.ndarray,
    drive: DriveParams | None = None,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    method: str = "RK45",
):
    """integrate the Lindblad equation for a density matrix rho(t), use runge kutta"""
    dim = system.dim
    rho0 = np.asarray(rho0, dtype=complex)
    if rho0.shape != (dim, dim):
        raise ValueError(f"rho0 must have shape {(dim, dim)}")
    rhs = lambda t, y: lindblad_rhs(t, y, system=system, drive=drive)
    return solve_ivp(
        rhs,
        t_span=t_span,
        y0=rho0.reshape(dim * dim),
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    )

def pure_state_density_matrix(psi: np.ndarray) -> np.ndarray:
    """Return |psi><psi|."""
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    return np.outer(psi, psi.conj())
