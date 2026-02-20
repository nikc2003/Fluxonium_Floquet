from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple, Optional, Any, List

import numpy as np
import qutip as qt

from .config import BuildComposite, CompositeSysConfig, Label
from .model import build_composite_model
from ..utils.timing import Timing

"""Module for computing truncation-related metrics for the composite model, such as commutator deviations (overall commutator infidelity) and population of the top Fock state. 
    This is used for diagnosing truncation errors in the linear mode, and can be used to inform how to choose the truncation dimension.
    Also attempts to look into entanglement and mixedness of the qubit reduced state, but I need to rigorously check that the metrics are computed correctly and make sense.
"""


def _psi_diag_to_bare_matrix(model, psi_diag: qt.Qobj, dim_q: int, dim_r: int) -> np.ndarray:
    """Convert a state expressed in the model's diagonal basis into bare tensor basis matrix (dim_q, dim_r)."""
    U = model.U.full()
    v = psi_diag.full().reshape(-1)
    v_bare = U @ v
    return v_bare.reshape((dim_q, dim_r))


def reduced_qubit_density_matrix(psi_bare_mat: np.ndarray) -> np.ndarray:
    return psi_bare_mat @ psi_bare_mat.conj().T


def purity_from_rho(rho: np.ndarray) -> float:
    return float(np.real(np.trace(rho @ rho)))


def von_neumann_entropy(rho: np.ndarray, *, eps: float = 1e-10) -> float:
    weights = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
    weights = np.clip(np.real(weights), 0.0, 1.0)
    weights = weights[weights > eps]
    return float(-np.sum(weights * np.log(weights))) if len(weights) else 0.0


def expectations_from_psi_bare(psi_bare_mat: np.ndarray) -> Dict[str, float]:
    dim_q, dim_r = psi_bare_mat.shape
    p = np.abs(psi_bare_mat) ** 2
    p_q = np.sum(p, axis=1)
    p_r = np.sum(p, axis=0)

    Nq = float(np.sum(np.arange(dim_q) * p_q))
    Nr = float(np.sum(np.arange(dim_r) * p_r))

    return {
        "Nq": Nq,
        "Nr": Nr,
        "p_r0": float(p_r[0]),
        "p_r_nonzero": float(1.0 - p_r[0]),
        "p_r_top": float(p_r[-1]),
        "p_q_top": float(p_q[-1]),
    }


def commutator_error(dim_r: int, p_r_top: float) -> float:
    return float(dim_r) * float(p_r_top)


def _as_density_qobj(x: qt.Qobj) -> qt.Qobj:
    if x.isket:
        return x.proj()
    if x.isoper:
        return x
    raise ValueError(f"expected a state ket or density operator, got {type(x)}")


def _rewrap_with_dims(op: qt.Qobj, dims) -> qt.Qobj:
    return qt.Qobj(op.full(), dims=dims)


def commutator_error_analytic(state_or_rho: qt.Qobj, a: qt.Qobj, *, mode: str = "expectation") -> float:
    """commutator deviation for truncated oscillator.
    mode:
        'expectation': |Tr(rho*(aa^dag - a^daga - I))|
        'op_norm': norm  (aa^dag - a^dag a - I)
    state_or_rho: can be a state ket or density operator; if ket, will be converted to density operator.
    """
    rho = _as_density_qobj(state_or_rho)
    if a.shape != rho.shape:
        raise ValueError(f"mismatch: a is {a.shape}, rho is {rho.shape}")
    a_match = a if (a.dims == rho.dims) else _rewrap_with_dims(a, rho.dims)
    comm = a_match * a_match.dag() - a_match.dag() * a_match
    I = qt.Qobj(np.eye(comm.shape[0]), dims=rho.dims)
    dev = comm - I
    if mode == "expectation":
        return float(abs((rho * dev).tr()))
    elif mode == "op_norm":
        X = dev.full()
        return float(np.linalg.norm(X, ord="fro"))


def floquet_select_mode_by_overlap(modes_at_t: Sequence[qt.Qobj] | qt.Qobj, target_basis_index: int) -> int:
    """floquet mode index with maximum overlap to a bare basis ket e_target."""
    if isinstance(modes_at_t, qt.Qobj):
        modes_arr = modes_at_t.full()
        dim = modes_arr.shape[0]
        ket = np.zeros(dim, dtype=complex)
        ket[target_basis_index] = 1.0
        overlaps = np.array([abs(np.vdot(ket, modes_arr[:, i])) for i in range(modes_arr.shape[1])], dtype=float)
    else:
        dim = modes_at_t[0].shape[0]
        ket = np.zeros(dim, dtype=complex)
        ket[target_basis_index] = 1.0
        overlaps = np.array([abs(np.vdot(ket, m.full().ravel())) for m in modes_at_t], dtype=float)
    return int(np.argmax(overlaps))


def floquet_metrics_single_pt(
    build_composite: BuildComposite,
    *,
    omega_d: float,
    amp: float,
    target_label: Label = (0, 0),
    nsteps: int = 10_000,
    ntimes: int = 21,
    timer: Optional[Timing] = None,
) -> Dict[str, Any]:
    tr = timer or Timing()
    model = build_composite.model
    dim_q = int(build_composite.cfg.dim_q)
    dim_r = int(build_composite.cfg.dim_r)
    T = 2 * np.pi / float(omega_d)
    tlist = np.linspace(0.0, T, int(ntimes), endpoint=False)
    with tr.span("FloquetBasis.init"):
        try:
            fbasis = qt.FloquetBasis(model.hamiltonian((omega_d, amp)), T, options={"nsteps": int(nsteps)}, precompute=tlist)
            modes_t = [fbasis.mode(t) for t in tlist]
        except TypeError:
            fbasis = qt.FloquetBasis(model.hamiltonian((omega_d, amp)), T, options={"nsteps": int(nsteps)})
            modes_t = [fbasis.mode(t) for t in tlist]

    modes_0 = modes_t[0]
    target_index = int(model.label_to_index(tuple(target_label)))
    alpha = floquet_select_mode_by_overlap(modes_0, target_index)

    purities: List[float] = []
    entropies: List[float] = []
    Nq_list: List[float] = []
    Nr_list: List[float] = []
    p_r_top_list: List[float] = []
    p_q_top_list: List[float] = []
    comm_ptop_list: List[float] = []
    comm_analytic_list: List[float] = []

    a_diag = build_composite.diag_ops["a"]

    with tr.span("metrics.loop"):
        for modes in modes_t:
            psi_diag = modes[alpha]
            psi_bare_mat = _psi_diag_to_bare_matrix(model, psi_diag, dim_q, dim_r)
            rho_q = reduced_qubit_density_matrix(psi_bare_mat)

            P = purity_from_rho(rho_q)
            S = von_neumann_entropy(rho_q)
            ex = expectations_from_psi_bare(psi_bare_mat)
            comm_ptop = commutator_error(dim_r, ex["p_r_top"])
            comm_an = commutator_error_analytic(psi_diag, a_diag, mode="expectation")

            purities.append(P)
            entropies.append(S)
            Nq_list.append(ex["Nq"])
            Nr_list.append(ex["Nr"])
            p_r_top_list.append(ex["p_r_top"])
            p_q_top_list.append(ex["p_q_top"])
            comm_ptop_list.append(comm_ptop)
            comm_analytic_list.append(comm_an)

    result: Dict[str, Any] = {
        "alpha_index": int(alpha),
        "target_label": tuple(target_label),
        "omega_d": float(omega_d),
        "amp": float(amp),
        "T": float(T),

        "purity_avg": float(np.mean(purities)),
        "entropy_avg": float(np.mean(entropies)),
        "Nq_avg": float(np.mean(Nq_list)),
        "Nr_avg": float(np.mean(Nr_list)),
        "p_r_top_avg": float(np.mean(p_r_top_list)),
        "p_q_top_avg": float(np.mean(p_q_top_list)),
        "commutator_error_avg": float(np.mean(comm_ptop_list)),
        "commutator_error_analytic_avg": float(np.mean(comm_analytic_list)),

        "purity_t": np.asarray(purities, dtype=float),
        "entropy_t": np.asarray(entropies, dtype=float),
        "Nq_t": np.asarray(Nq_list, dtype=float),
        "Nr_t": np.asarray(Nr_list, dtype=float),
        "commutator_error_analytic_t": np.asarray(comm_analytic_list, dtype=float),
    }
    return result


def truncation_metrics_point(
    cfg: CompositeSysConfig,
    *,
    flux: float,
    omega_d: float,
    chi_ac: float,
    options_like: Dict[str, Any],
    target_label: Label = (0, 0),
    nsteps: int = 10_000,
    ntimes: int = 21,
    timer: Optional[Timing] = None,
) -> Dict[str, Any]:
    tr = timer or Timing()
    qparams = dict(cfg.qubit_params)
    qparams["flux"] = float(flux)
    cfg2 = CompositeSysConfig(
        qubit_params=qparams,
        dim_q=int(cfg.dim_q),
        dim_r=int(cfg.dim_r),
        omega_r=float(cfg.omega_r),
        g=float(cfg.g),
        coupling_op=str(cfg.coupling_op),
        resonator_quadrature=str(cfg.resonator_quadrature),
    )


    with tr.span("build_model"):
        build = build_composite_model(
            cfg2,
            omega_d_values=np.array([float(omega_d)]),
            chi_ac_values=np.array([float(chi_ac)]),
            label_states=(tuple(target_label),),
            timer=tr,
        )

    amp = float(build.model.drive_amplitudes[0, 0])
    with tr.span("metrics_single_pt"):
        m = floquet_metrics_single_pt(
            build,
            omega_d=float(omega_d),
            amp=amp,
            target_label=tuple(target_label),
            nsteps=int(nsteps),
            ntimes=int(ntimes),
            timer=tr,
        )

    row = {
        "dim_q": int(cfg2.dim_q),
        "dim_r": int(cfg2.dim_r),
        "flux": float(flux),
        "omega_d": float(omega_d),
        "chi_ac": float(chi_ac),
        "amp": float(amp),
        "Nr_avg": float(m["Nr_avg"]),
        "p_r_top_avg": float(m["p_r_top_avg"]),
        "p_q_top_avg": float(m["p_q_top_avg"]),
        "purity_avg": float(m["purity_avg"]),
        "entropy_avg": float(m["entropy_avg"]),
        "comm_error_ptop_avg": float(m["commutator_error_avg"]),
        "comm_error_analytic_avg": float(m["commutator_error_analytic_avg"]),
    }
    return {"row": row, "timing": tr.to_dict()}
