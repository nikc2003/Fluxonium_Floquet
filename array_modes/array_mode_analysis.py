from __future__ import annotations

import json
import math
import datetime as dt
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import (Dict, Tuple, Sequence, Optional, Literal, Union, List, Iterable)
import numpy as np
import qutip as qt
import scqubits as scq
import h5py
import matplotlib.pyplot as plt
from itertools import cycle
from cycler import cycler
from scipy.optimize import linear_sum_assignment

#we can either use the linearized ver. of zero point fluctuations
#(harmonic approx.) or calculate the matrix elements directly to get vaccum fluctuations
ZPFMode = Literal['linearized', 'override']
DriveParam = Literal["nbar", "chi_ac"]
Bare_Label = Tuple[int, int]  # (excitation number in qubit, array mode number)

#helpers for saving/loading data
def _today_str() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d")

def _now_str() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def make_outdir(path: Union[str, Path]):
    savepath = Path(path).expanduser().resolve()
    savepath.mkdir(parents=True, exist_ok=True)
    return savepath

def as_1d(x: Union[float, Sequence[float], np.ndarray]) -> np.ndarray:
    if isinstance(x, (float, int)):
        return np.array([x])
    return np.asarray(x).ravel()

#CONFIGURATION
@dataclass(frozen=True)
class FluxoniumArrayCfg:
    #cfg for fluxonium
    EJ: float
    EC: float
    EL: float
    flux: float  # in units of flux quanta
    cutoff: int
    dim_q: int

    #cfg for array modes
    freq_array: float  # GHz
    N_mu_zpf: float
    dim_array: int

    #cfg for coupling and resonator
    freq_r: float  # GHz
    g_phi_r: float  # GHz
    g_mu_r: float  # GHz
    g_phi_mu: float  # GHz

    #zero point fluctuation normalization
    Nphi_zpf_mode: ZPFMode = "linearized"
    Nphi_zpf_override: Optional[float] = None

    #scqubits in GHz, but qutip in angular freq.
    use_angular_units: bool = True #for consistency

    def to_json(self) -> dict:
        return asdict(self)
    
def Nphi_zpf_linearized(cfg: FluxoniumArrayCfg) -> float:
    """Calculate the zero point fluctuations of the phase across the junction in the linearized approx."""
    return (cfg.EL / (32*cfg.EC))**0.25

def get_Nphi_zpf(cfg: FluxoniumArrayCfg) -> float:
    if cfg.Nphi_zpf_mode == "linearized":
        return Nphi_zpf_linearized(cfg)
    elif cfg.Nphi_zpf_mode == "override":
        if cfg.Nphi_zpf_override is None:
            raise ValueError("Nphi_zpf_override must be set when using 'override' mode.")
        return cfg.Nphi_zpf_override
    
def scale_to_angular(cfg: FluxoniumArrayCfg) -> float:
    return 2*np.pi if cfg.use_angular_units else 1.0

    
#HAMILTONIAN AND OPERATOR CONSTRUCTION
@dataclass
class SystemOps:
    H0_GHz: qt.Qobj
    #couplig ops.
    Xphi: qt.Qobj
    Xmu: Optional[qt.Qobj]
    nphi_excitation: qt.Qobj
    nmu_number: Optional[qt.Qobj]
    Nphi_zpf: float
    #truncation metrics for array mode
    a_mu_composite: Optional[qt.Qobj] = None
    commutator_infidelity: Optional[qt.Qobj] = None
    P_mu_top: Optional[qt.Qobj] = None



def include_trunc_metrics(cfg: FluxoniumArrayCfg, ops: SystemOps) -> SystemOps:
    if cfg.dim_array <= 0 or ops.Xmu is None:
        return ops

    N_q = int(cfg.dim_q)
    N_mu = int(cfg.dim_array)

    Iq = qt.qeye(N_q)
    a_mu = qt.destroy(N_mu)
    a_mu_composite = qt.tensor(Iq, a_mu)

    comm = a_mu_composite * a_mu_composite.dag() - a_mu_composite.dag() * a_mu_composite
    I_full = qt.tensor(Iq, qt.qeye(N_mu))

    #deviation from ideal bosonic commutator
    comm_minus_I = comm - I_full

    top = qt.basis(N_mu, N_mu - 1)
    P_mu_top = qt.tensor(Iq, top * top.dag())

    ops.a_mu_composite = a_mu_composite
    ops.commutator_infidelity = comm_minus_I
    ops.P_mu_top = P_mu_top
    return ops

def array_commutator_metrics(ops: SystemOps, state: qt.Qobj) -> Dict[str, float]:
    if ops.commutator_infidelity is None or ops.P_mu_top is None:
        return {"commutator_infidelity": 0.0, "P_mu_top": 0.0}

    if state.isket:
        comm_val = qt.expect(ops.commutator_infidelity, state)
        p_top = qt.expect(ops.P_mu_top, state)
    else:
        # density matrix
        comm_val = (ops.commutator_infidelity * state).tr()
        p_top = (ops.P_mu_top * state).tr()

    return {
        "commutator_infidelity": float(abs(comm_val)),
        "P_mu_top": float(np.real(p_top)),
    }
def _unitary_from_evecs(evecs: Sequence[qt.Qobj], op_dims=None) -> qt.Qobj:
    """
    build unitary U with columns that are eigenkets in the computational basis,
    and preserve tensor dims so U.dag() * H * U works with tensor operators.
    """
    U_mat = np.hstack([psi.full() for psi in evecs])  
    if op_dims is None:
        subsys = evecs[0].dims[0]
        op_dims = [subsys, subsys]
    U = qt.Qobj(U_mat, dims=op_dims)
    return U

def _columns_from_evecs(evecs_q, dim_q: int) -> np.ndarray:
    """
    scqubits eigensys eigenvectors --> a matrix U whose columns are eigenvectors
    in the basis
    """
    if isinstance(evecs_q, (list, tuple)):
        U = np.column_stack([np.asarray(v).reshape(-1) for v in evecs_q])
    else:
        M = np.asarray(evecs_q)

        if M.ndim != 2:
            raise ValueError(f"Unexpected evecs_q with ndim={M.ndim}, shape={M.shape}")
        candidates = []
        for A in (M, M.T):
            if A.ndim == 2 and A.shape[1] == dim_q:
                err = np.linalg.norm(A.conj().T @ A - np.eye(dim_q))
                candidates.append((err, A))

        err, U = min(candidates, key=lambda x: x[0])
        if err > 1e-6:
            print(f"eigenvector orthonormality error = {err:.3e}")
    return np.asarray(U, dtype=complex)


def build_fluxoniumarray_hamiltonian(cfg: FluxoniumArrayCfg) -> SystemOps:
    """this will construct the hamiltonian of either the fluxonium alone or fluxonium + array"""
    fluxonium = scq.Fluxonium(
        EJ=cfg.EJ,
        EC=cfg.EC,
        EL=cfg.EL,
        flux=cfg.flux,
        cutoff=cfg.cutoff,
        truncated_dim=cfg.dim_q,)
    evals_q, evecs_q = fluxonium.eigensys(evals_count=cfg.dim_q)
    #zero the ground state energy
    evals_q = evals_q - evals_q[0]
    Hq = qt.Qobj(np.diag(evals_q))
    Nphi_mat_bare = np.asarray(fluxonium.n_operator())
    U = _columns_from_evecs(evecs_q, cfg.dim_q)
    Nphi_mat = U.conj().T @ Nphi_mat_bare @ U
    if Nphi_mat.shape != (cfg.dim_q, cfg.dim_q):
        raise ValueError(f"Nphi_mat has shape {Nphi_mat.shape}, but we need {(cfg.dim_q, cfg.dim_q)}.")
    Nphi = qt.Qobj(Nphi_mat, dims=[[cfg.dim_q], [cfg.dim_q]])
    Nphi_zpf = get_Nphi_zpf(cfg)
    if cfg.dim_array == 0:
        Xphi = Nphi / Nphi_zpf
        nq_index = qt.Qobj(np.diag(np.arange(cfg.dim_q)))
        ops = SystemOps(
            H0_GHz=Hq,
            Xphi=Xphi,
            Xmu=None,
            nphi_excitation=nq_index,
            nmu_number=None,
            Nphi_zpf=float(Nphi_zpf),
        )
        return ops
    #parasitic array mode operators
    N_mu = int(cfg.dim_array)
    a_mu = qt.destroy(N_mu)
    num_mu = a_mu.dag() * a_mu
    Nmu_op = -1j * cfg.N_mu_zpf * (a_mu - a_mu.dag()) 
    Hmu = cfg.freq_array * num_mu

    #promote everything to composite space
    Iq = qt.qeye(cfg.dim_q)
    Imu = qt.qeye(N_mu)

    Hq_full = qt.tensor(Hq, Imu)
    Hmu_full = qt.tensor(Iq, Hmu)

    Nphi_full = qt.tensor(Nphi, Imu)
    Nmu_full = qt.tensor(Iq, Nmu_op)

    Xphi = Nphi_full / Nphi_zpf
    Xmu = Nmu_full / cfg.N_mu_zpf

    Hint = cfg.g_phi_mu * (Xphi * Xmu)
    H0 = Hq_full + Hmu_full + Hint    #static hamiltonian

    hermitian_check = (H0 - H0.dag()).norm()
    if hermitian_check > 1e-10:
        raise ValueError(f"H0 is not Hermitian! ||H0 - H0.dag|| = {hermitian_check}")
    nq_index = qt.Qobj(np.diag(np.arange(cfg.dim_q)))
    nphi_excitation = qt.tensor(nq_index, Imu)
    nmu_number = qt.tensor(Iq, num_mu)
    ops = SystemOps(
        H0_GHz=H0,
        Xphi=Xphi,
        Xmu=Xmu,
        nphi_excitation=nphi_excitation,
        nmu_number=nmu_number,
        Nphi_zpf=float(Nphi_zpf),
    )
    ops = include_trunc_metrics(cfg, ops)
    return ops

def bare_ket(cfg: FluxoniumArrayCfg, i: int, n_mu: int) -> qt.Qobj:
    if cfg.dim_array <= 0:
        return qt.basis(cfg.dim_q, i)
    return qt.tensor(qt.basis(cfg.dim_q, i), qt.basis(cfg.dim_array, n_mu))

def seed_labels_cfg(cfg: FluxoniumArrayCfg, max_q: Optional[int] = None, max_mu: Optional[int] = None) -> List[Bare_Label]:
    nq = int(max_q) if max_q is not None else int(cfg.dim_q)
    if cfg.dim_array <= 0:
        return [(i, 0) for i in range(nq)]
    nmu = int(max_mu) if max_mu is not None else int(cfg.dim_array)
    return [(i, n) for i in range(nq) for n in range(nmu)]


def seed_indices_v2(H0_GHz: qt.Qobj,
                    cfg: FluxoniumArrayCfg,
                    seed_labels: Sequence[Bare_Label],
                    overlap_warn: float = 0.8,
                    unique: bool = True,
                  ):
    evals, evecs = H0_GHz.eigenstates()
    idx: Dict[Bare_Label, int] = {}
    overlaps_best: Dict[Bare_Label, float] = {}
    used_indices: set[int] = set()

    for (i, n) in seed_labels:
        if cfg.dim_array == 0 and n != 0:
            continue
        bare = bare_ket(cfg, i, n)
        overlaps = np.array([abs(bare.overlap(psi))**2 for psi in evecs])
        order = np.argsort(overlaps)[::-1]
        chosen = None
        for j in order:
            if not unique or j not in used_indices:
                chosen = j
                break
        if chosen is None:
            chosen = order[0]
        idx[(i, n)] = chosen
        overlaps_best[(i, n)] = float(overlaps[chosen])
        used_indices.add(chosen)
        # if overlaps_best[(i, n)] < overlap_warn:
        #     print(f"overlap for {(i,n)} is only {overlaps_best[(i,n)]:.3f} "
        #           f"(<{overlap_warn}). label may not be accurate...")
    return idx, evecs, overlaps_best, np.asarray(evals)

def wrap_quasi(eps: np.ndarray, omega: float) -> np.ndarray:
    """wrap quasienergies to first Floquet zone [0, omega)"""
    return np.mod(eps, omega)

#drive operator config
def drive_op_dimless(cfg: FluxoniumArrayCfg, ops: SystemOps) -> qt.Qobj:
    """
    dimensionless drive operator such that physical drive term can be expressed as,
    H_d(t) = Omega_d * O_drive * cos(omega_d t); Omega_d [GHz]

    then, choose O_drive such that nbar parametrization matches:
    2 sqrt(nbar) (g_phi_r Xphi + g_mu_r Xmu)
    =  (2 g_phi_r sqrt(nbar)) (Xphi + (g_mu_r/g_phi_r) Xmu)
    = Omega_d * O_drive, where Omega_d = 2 g_phi_r sqrt(nbar)
    =====> O_drive = Xphi + (g_mu_r/g_phi_r) Xmu !<======
    """
    if ops.Xmu is None:
        return ops.Xphi
    return ops.Xphi + (cfg.g_mu_r / cfg.g_phi_r) * ops.Xmu

#amplitude conversion so we can parametrize drive strength in terms of chi_ac
class ChiacToAmp:
    """
    same thing as danny's floquet package but ensure that:
    1. H0 is passes as diagonal in the eigenbasis of our static hamiltonian in consideration
    2. O_drive (i call it H1 here) passed in the same basis
    3. take omega_d and chi_ac in angular units
    """
    def __init__(self, H0_diag:qt.Qobj, H1_eigbasis: qt.Qobj,
                 state_indices: list[int], omega_d_values: np.ndarray):
        self.H0 = H0_diag
        self.H1 = H1_eigbasis
        self.state_indices = list(state_indices)
        self.omega_d_values = omega_d_values  # angular freq.
    
    def amplitudes_for_omega_d(self, chi_ac_values: np.ndarray) -> np.ndarray:
        chi_ac_values = as_1d(chi_ac_values)
        chis_for_omega_d = self.compute_chis_for_omega_d() #num. of omegas
        return np.einsum(
            "a,w->aw",
            2.0 * np.sqrt(chi_ac_values),
            1.0 / np.sqrt(chis_for_omega_d),
        ) 
    
    def compute_chis_for_omega_d(self) -> np.ndarray:
        energies = np.diag(self.H0.full()).astype(np.float64)
        H1 = self.H1.full()
        chi_0 = self.chi_ell(energies, H1, self.omega_d_values, self.state_indices[0])
        chi_1 = self.chi_ell(energies, H1, self.omega_d_values, self.state_indices[1])
        return np.abs(chi_1 - chi_0)
    @staticmethod
    def chi_ell_ellp(energies: np.ndarray, H1: np.ndarray, E_osc: np.ndarray, ell: int, ellp:int) -> np.ndarray:
        """compute: 
        chi_ell_ellp(E_osc) = sum_{ellp != ell} |<ell|H1|ellp>|^2 / (E_ell - E_ellp - E_osc)
        """
        E_ell_ellp = energies[ell] - energies[ellp]
        return (np.abs(H1[ell, ellp])**2) / (E_ell_ellp - E_osc)
    def chi_ell(self, energies: np.ndarray, H1: np.ndarray, E_osc: np.ndarray, ell: int) -> np.ndarray:
        """
        return both positive and negative contributions to chi_ell (counter rotating terms and exc. number preserving terms)
        """
        n = len(energies)
        out = 0.0
        for ellp in range(n):
            if ellp == ell:
                continue
            out += self.chi_ell_ellp(energies, H1, E_osc, ell, ellp)
            out -= self.chi_ell_ellp(energies, H1, E_osc, ellp, ell)
        return out
    

def build_amplitude_calibrator(
        cfg: FluxoniumArrayCfg,
        ops: SystemOps,
        omega_d_list_GHz: Sequence[float],
        *, state_labels_for_chi: Tuple[Bare_Label, Bare_Label] = ((0,0), (1,0)),
        overlap_warn: float = 0.8, 
    ) -> Tuple[ChiacToAmp,qt.Qobj]:
    
    ang_units = scale_to_angular(cfg)
    omega_d_list_GHz = np.asarray(omega_d_list_GHz)
    omega_d_list_ang = omega_d_list_GHz * ang_units
    evals0, evecs0 = ops.H0_GHz.eigenstates()
    U = _unitary_from_evecs(evecs0, op_dims=ops.H0_GHz.dims)
    H1_dimless = drive_op_dimless(cfg, ops)
    H1_eigbasis = U.dag() * H1_dimless * U
    H0_diag_ang = qt.Qobj(np.diag(np.asarray(evals0) * ang_units), dims=ops.H0_GHz.dims)
    #dressed labels corresponding to |0,0>, |1,0>
    idx_map, _, _, _ = seed_indices_v2(ops.H0_GHz, cfg, seed_labels=list(state_labels_for_chi), overlap_warn=overlap_warn, unique=True)
    i0 = idx_map[state_labels_for_chi[0]]
    i1 = idx_map[state_labels_for_chi[1]]

    amplitude_calibrator =  ChiacToAmp(H0_diag_ang, H1_eigbasis, [i0, i1], omega_d_list_ang)

    return amplitude_calibrator, H1_dimless

def time_dependent_H(cfg: FluxoniumArrayCfg, ops: SystemOps, omega_d_GHz: float, 
                     amplitude_angular: float, H1_dimless: qt.Qobj) -> Tuple[list, float, dict]:
    """
    construct the time dependent hamiltonian (ang units since we use QuTip)
    H(t) = H0_ang + amplitude_angular * H1_dimless * cos(omega_d * t)
    """
    ang_units = scale_to_angular(cfg)
    H0_ang = ops.H0_GHz * ang_units
    omega_d_ang = omega_d_GHz * ang_units
    T = 2*np.pi / omega_d_ang

    def coeff(t, args):
        return args["amp"] *np.cos(omega_d_ang * t)
    Ht = [H0_ang, [H1_dimless, coeff]]
    args = {"w": omega_d_ang, "amp": amplitude_angular}
    return Ht, T, args

def floquet_modes(cfg: FluxoniumArrayCfg, ops: SystemOps, omega_d_GHz: float,
                  amplitude_angular: float, H1_dimless: qt.Qobj):
    Ht, T, args = time_dependent_H(cfg, ops, omega_d_GHz, amplitude_angular, H1_dimless)
    modes, eps = qt.floquet_modes(Ht, T, args=args)
    return modes, np.asarray(eps), T, args

def overlap_matrix(prev_states: Sequence[qt.Qobj], curr_states: Sequence[qt.Qobj]) -> np.ndarray:
    """O_ij = |<prev_i | curr_j>|^2"""
    prev_mat = np.hstack([psi.full() for psi in prev_states])
    curr_mat = np.hstack([psi.full() for psi in curr_states])
    overlap_mat = np.abs(prev_mat.conj().T @ curr_mat)**2
    return overlap_mat

def assignment_from_overlaps(overlaps: np.ndarray) -> Dict[int, int]:
    overlaps = np.asarray(overlaps, dtype=float)
    nrow, ncol = overlaps.shape
    if nrow > ncol:
        raise ValueError(f"need nrow <= ncol for unique assignment, but {nrow} > {ncol}")
    if linear_sum_assignment is not None:
        #hungarian says to maximize overlap = minimize negative overlap
        row_ind, col_ind = linear_sum_assignment(-overlaps)
        return {ri: ci for ri, ci in zip(row_ind, col_ind)}
    pairs = [(overlaps[i,j], i, j) for i in range(nrow) for j in range(ncol)]
    pairs.sort(reverse = True, key = lambda x: x[0])
    used_rows, used_cols = set(), set()
    assignment: Dict[int, int] = {}
    for val, i, j in pairs:
        if i in used_rows or j in used_cols:
            continue
        assignment[i] = j
        used_rows.add(i)
        used_cols.add(j)
        if len(assignment) == nrow:
            break
    return assignment

@dataclass 
class BranchMapping:
    label: Bare_Label
    eps: np.ndarray
    n_phi: np.ndarray
    n_mu: np.ndarray
    seed_population: np.ndarray
    commutation_infidelity_mu: np.ndarray
    p_mu_top: np.ndarray
    seed_H0_overlap: float
    states: Optional[List[qt.Qobj]] = None

def track_branch_drive(
    cfg: FluxoniumArrayCfg,
    ops: SystemOps,
    omega_d_GHz: float,
    *,
    drive_axis: DriveParam,
    drive_values: np.ndarray,
    seed_labels: Sequence[Bare_Label],
    scar_labels: Sequence[Bare_Label] = ((0,0), (1,0), (2,0)),
    state_labels_for_chi: Tuple[Bare_Label, Bare_Label] = ((0,0), (1,0)),
    track_unique: bool = True,
    store_states: bool = False,
    overlap_warn: float = 0.8,
) -> Tuple[Dict[Bare_Label, BranchMapping], Dict[str, np.ndarray]]:

    ang_units = scale_to_angular(cfg)
    drive_values = np.asarray(drive_values, dtype=float)
    H1_dimless = drive_op_dimless(cfg, ops)
    if drive_axis == "nbar":
        ang_amps = ang_units * (2.0 * cfg.g_phi_r * np.sqrt(np.maximum(drive_values, 0.0)))
    elif drive_axis == "chi_ac":
        cal, H1_dimless = build_amplitude_calibrator(
            cfg, ops, omega_d_list_GHz=[omega_d_GHz],
            state_labels_for_chi=state_labels_for_chi,
            overlap_warn=overlap_warn,
        )
        chi_ac_ang = ang_units * drive_values
        amps = cal.amplitudes_for_omega_d(chi_ac_ang)   
        ang_amps = amps[:, 0]
    
    idx_map, evecs0, overlaps0, evals0_GHz = seed_indices_v2(
        ops.H0_GHz, cfg, seed_labels=seed_labels, overlap_warn=overlap_warn, unique=True
    )
    omega_d_ang = float(omega_d_GHz) * ang_units
    order_labels = [lab for lab in seed_labels if not (cfg.dim_array <= 0 and lab[1] != 0)]
    branch_states_last: List[qt.Qobj] = []
    branch_data: Dict[Bare_Label, dict] = {}
    for label in order_labels:
        idx0 = idx_map[label]
        psi0 = evecs0[idx0]
        eps0 = wrap_quasi(np.array([evals0_GHz[idx0] * ang_units]), omega_d_ang)[0]
        seed = bare_ket(cfg, label[0], label[1])
        seed_pop0 = float(abs(seed.overlap(psi0))**2)
        m0 = array_commutator_metrics(ops, psi0)
        branch_data[label] = {
            "eps": [float(eps0)],
            "n_phi": [float(np.real(qt.expect(ops.nphi_excitation, psi0)))],
            "n_mu": [0.0 if ops.nmu_number is None else float(np.real(qt.expect(ops.nmu_number, psi0)))],
            "seed_population": [seed_pop0],
            "commutation_infidelity_mu": [float(m0["commutator_infidelity"])],
            "p_mu_top": [float(m0["P_mu_top"])],
            "seed_H0_overlap": float(overlaps0[label]),
            "states": [psi0] if store_states else None,
        }
        branch_states_last.append(psi0)
    for ii in range(1, len(drive_values)):
        modes, eps, T, args = floquet_modes(cfg, ops, float(omega_d_GHz), float(ang_amps[ii]), H1_dimless)
        wrapped_eps = wrap_quasi(eps, args["w"])
        if track_unique:
            O = overlap_matrix(branch_states_last, modes)  
            assignment = assignment_from_overlaps(O)
        else:
            assignment = {}
            for bi, prev in enumerate(branch_states_last):
                ovs = np.array([abs(prev.overlap(psi))**2 for psi in modes], dtype=float)
                assignment[bi] = int(np.argmax(ovs))
        new_last: List[qt.Qobj] = []
        for bi, label in enumerate(order_labels):
            j = assignment[bi]
            psi = modes[j]
            new_last.append(psi)
            seed = bare_ket(cfg, label[0], label[1])
            seed_pop = float(abs(seed.overlap(psi))**2)
            branch_data[label]["eps"].append(float(wrapped_eps[j]))
            branch_data[label]["n_phi"].append(float(np.real(qt.expect(ops.nphi_excitation, psi))))
            branch_data[label]["n_mu"].append(
                0.0 if ops.nmu_number is None else float(np.real(qt.expect(ops.nmu_number, psi)))
            )
            branch_data[label]["seed_population"].append(seed_pop)
            comm_metrics = array_commutator_metrics(ops, psi)
            branch_data[label]["commutation_infidelity_mu"].append(float(comm_metrics["commutator_infidelity"]))
            branch_data[label]["p_mu_top"].append(float(comm_metrics["P_mu_top"]))
            if store_states and branch_data[label]["states"] is not None:
                branch_data[label]["states"].append(psi)
        branch_states_last = new_last
    branches: Dict[Bare_Label, BranchMapping] = {}
    for label in order_labels:
        data = branch_data[label]
        branches[label] = BranchMapping(
            label=label,
            eps=np.asarray(data["eps"], dtype=float),
            n_phi=np.asarray(data["n_phi"], dtype=float),
            n_mu=np.asarray(data["n_mu"], dtype=float),
            seed_population=np.asarray(data["seed_population"], dtype=float),
            commutation_infidelity_mu=np.asarray(data["commutation_infidelity_mu"], dtype=float),
            p_mu_top=np.asarray(data["p_mu_top"], dtype=float),
            seed_H0_overlap=float(data["seed_H0_overlap"]),
            states=data["states"],
        )
    more_data = {
        "drive_values": np.asarray(drive_values, dtype=float),
        "ang_amps": np.asarray(ang_amps, dtype=float),
        "omega_d_GHz": np.array([float(omega_d_GHz)], dtype=float),
    }
    return branches, more_data


#2d swewps

@dataclass
class ScarAnalysis:
    x_values: np.ndarray
    y_values: np.ndarray
    nphi: Dict[Bare_Label, np.ndarray]
    nmu: Dict[Bare_Label, np.ndarray]
    meta: dict

@dataclass
class SweepResult:
    device: str
    cfg_base: dict
    x_name: str
    y_name: str
    scar_maps: ScarAnalysis
    branches_x: Dict[float, Dict[Bare_Label, BranchMapping]]
    omega_d_used_x: Dict[float, float]

def run_sweep(
        cfg: FluxoniumArrayCfg,
        *,
        device: str,
        x_axis_name: Literal["omega_d", "flux"],
        x_values: Sequence[float],
        y_axis_mode: DriveParam,
        y_values: Sequence[float],
        seed_labels: Union[Literal['all'], Sequence[Bare_Label]] = 'all',
        seed_max_q: Optional[int] = None,
        seed_max_mu: Optional[int] = None,
        scar_labels: Sequence[Bare_Label] = ((0,0), (1,0), (2,0)),
        omega_d_fixed_GHz: Optional[float] = None,
        omega_d_used_flux: Optional[Sequence[float]] = None,
        track_unique: bool = True,
        store_states: bool = False,) -> SweepResult:
    """ 
    will run branch analysis over all seed labels or a specfiic set and then 
    only use scar labels to build the scar analysis maps. the x axis name will set 
    whether we sweep flux or drive frequency. and then for flux sweeps, we should pass
    omega_d_used_flux if we want to use a specific drive frequency corresponding to the flux val.
    """
    if seed_labels == 'all':
        seed_labels_list = seed_labels_cfg(cfg, max_q=seed_max_q, max_mu=seed_max_mu)
    else:
        seed_labels_list = list(seed_labels)
    
    Nx, Ny = len(x_values), len(y_values)
    maps_nphi = {label: np.zeros((Nx, Ny)) for label in scar_labels}
    maps_nmu = {label: np.zeros((Nx, Ny)) for label in scar_labels}
    branches_x: Dict[float, Dict[Bare_Label, BranchMapping]] = {}
    omega_d_used_x: Dict[float, float] = {}
    
    if x_axis_name == "omega_d":
        system = build_fluxoniumarray_hamiltonian(cfg)
        for ix, omega_d in enumerate(x_values):
            branches, surplus = track_branch_drive(
                cfg, system, omega_d, 
                drive_axis=y_axis_mode,
                drive_values=y_values,
                seed_labels=seed_labels_list,
                scar_labels=scar_labels,
                track_unique=track_unique,
                store_states=store_states,
            )
            branches_x[omega_d] = branches
            omega_d_used_x[omega_d] = omega_d

            for label in scar_labels:
                maps_nphi[label][ix,:] = branches[label].n_phi
                maps_nmu[label][ix,:] = branches[label].n_mu
            print(f"[{ix+1}/{Nx}], omega_d/2pi = {omega_d:.3f} [GHz]")
        scar_map = ScarAnalysis(
            x_values=np.asarray(x_values),
            y_values=np.asarray(y_values),
            nphi=maps_nphi,
            nmu=maps_nmu,
            meta={
                "x_axis": "omega_d",
                "y_axis": y_axis_mode,
            }
        )
        return SweepResult(
            device=device,
            cfg_base=cfg.to_json(),
            x_name="omega_d",
            y_name=y_axis_mode,
            scar_maps=scar_map,
            branches_x=branches_x,
            omega_d_used_x=omega_d_used_x,
        )
    if omega_d_used_flux is not None:
        omega_list = np.asarray(omega_d_used_flux, dtype=float)
        if omega_list.ndim == 0:
            omega_list = np.full(Nx, float(omega_list))
        if len(omega_list) != Nx:
            raise ValueError("omega_d_used_flux must be length Nx (same as x_values).")
    else:
        if omega_d_fixed_GHz is None:
            omega_d_fixed_GHz = cfg.freq_r  
        omega_list = np.full(Nx, float(omega_d_fixed_GHz), dtype=float)
    for ix, flux in enumerate(x_values):
        cfg_flux = replace(cfg, flux=flux)
        system_flux = build_fluxoniumarray_hamiltonian(cfg_flux)
        omega_d = omega_list[ix]
        branches, surplus = track_branch_drive(
            cfg_flux, system_flux, omega_d,
            drive_axis=y_axis_mode,
            drive_values=y_values,
            seed_labels=seed_labels_list,
            scar_labels=scar_labels,
            track_unique=track_unique,
            store_states=store_states,
        )
        branches_x[flux] = branches
        omega_d_used_x[flux] = omega_d
        for label in scar_labels:
            maps_nphi[label][ix,:] = branches[label].n_phi
            maps_nmu[label][ix,:] = branches[label].n_mu
        print(f"[{ix+1}/{Nx}], flux = {flux:.3f} [Phi0], omega_d/2pi = {omega_d:.5f} [GHz]")
    scar_map = ScarAnalysis(
        x_values=np.asarray(x_values),
        y_values=np.asarray(y_values),
        nphi=maps_nphi,
        nmu=maps_nmu,
        meta={
            "x_axis": "flux",
            "y_axis": y_axis_mode,
        }
    )
    return SweepResult(
        device=device,
        cfg_base=cfg.to_json(),
        x_name="flux",
        y_name=y_axis_mode,
        scar_maps=scar_map,
        branches_x=branches_x,
        omega_d_used_x=omega_d_used_x,
    )

def fmt_float(x: float, ndigits: int = 4) -> str:
    """
    Make float filename-safe: 0.4725 -> '0p4725'
    """
    s = f"{x:.{ndigits}f}"
    return s.replace(".", "p").replace("-", "m")

@dataclass
class SaveData:
    base_dir: Union[str, Path] = "results"
    date: Optional[str] = None
    device: str = "device" #change to put "qA" or "qB".....
    tag: Optional[str] = None 

    def outdir(self) -> Path:
        dt = self.date or _today_str()
        parts = [str(self.base_dir), dt, self.device or "device"]
        if self.tag:
            parts.append(self.tag)
        return make_outdir(Path(*parts))
    
    def _save_metadata(self, cfg:FluxoniumArrayCfg, surplus: Optional[dict] = None) -> Path:
        out = self.outdir() / "metadata.json"
        data = {"cfg": cfg.to_json(), "surplus": surplus or {}, "time": _now_str()}
        out.write_text(json.dumps(data, indent= 2, sort_keys=True))
        return out
    
    def filename_2D_scan(
            self, *, kind: str, x_name: str, x_min: float, x_max: float, x_pts: int,
            y_name: str, y_min: float, y_max: float, y_pts: int, omega_note: str = "",
            ext: str = "h5",
    ):
        return (
            f"{self.device}__{kind}"
            f"__x_{x_name}_{fmt_float(x_min)}_{fmt_float(x_max)}_{x_pts}pts"
            f"__y_{y_name}_{fmt_float(y_min)}_{fmt_float(y_max)}_{y_pts}pts"
            f"{omega_note}"
            f"__{_now_str()}.{ext}"
        )

    def filename_1d(
            self,
            *,
            kind: str,
            flux: Optional[float] = None,
            omega_d_GHz: Optional[float] = None,
            y_name: str = "",
            y_min: Optional[float] = None,
            y_max: Optional[float] = None,
            y_pts: Optional[int] = None,
            ext: str = "h5",
        ) -> str:
        parts = [self.device, kind]
        if flux is not None:
            parts.append(f"flux_{fmt_float(flux, 3)}")
        if omega_d_GHz is not None:
            parts.append(f"omega_d_{fmt_float(omega_d_GHz, 3)}GHz")
        if y_name and y_min is not None and y_max is not None and y_pts is not None:
            parts.append(f"{y_name}_{fmt_float(y_min)}_{fmt_float(y_max)}_{y_pts}pts")
        return "__".join(parts) + f"__{_now_str()}.{ext}"
    
    def save_fig(self, fig: plt.Figure, name: str, dpi: int = 300) -> Path:
        out = self.outdir() / name
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        return out
    
def save_scar_maps_h5(path: Union[str, Path], scar: ScarAnalysis, *, cfg_dict: dict, omega_used_by_x: dict):
    path = Path(path).expanduser().resolve()
    with h5py.File(path, "w") as f:
        f.create_dataset("x_values", data=np.asarray(scar.x_values))
        f.create_dataset("y_values", data=np.asarray(scar.y_values))
        gphi = f.create_group("nphi")
        gmu  = f.create_group("nmu")
        for (k, n), arr in scar.nphi.items():
            gphi.create_dataset(f"label_{k}_{n}", data=np.asarray(arr))
        for (k, n), arr in scar.nmu.items():
            gmu.create_dataset(f"label_{k}_{n}", data=np.asarray(arr))
        f.attrs["cfg_json"] = json.dumps(cfg_dict)
        f.attrs["meta_json"] = json.dumps(scar.meta)
        f.attrs["omega_used_by_x_json"] = json.dumps({str(k): float(v) for k, v in omega_used_by_x.items()})
    return path

def save_branches_h5(path: Union[str, Path], branches: Dict[Bare_Label, BranchMapping], *,
                     x_value: float, omega_d_GHz: float, y_values: np.ndarray, cfg_dict: dict):
    path = Path(path).expanduser().resolve()
    with h5py.File(path, "w") as f:
        f.create_dataset("y_values", data=np.asarray(y_values))
        f.attrs["x_value"] = float(x_value)
        f.attrs["omega_d_GHz"] = float(omega_d_GHz)
        f.attrs["cfg_json"] = json.dumps(cfg_dict)

        grp = f.create_group("branches")
        for (k, n), tr in branches.items():
            g = grp.create_group(f"label_{k}_{n}")
            g.create_dataset("eps", data=tr.eps)
            g.create_dataset("n_phi", data=tr.n_phi)
            g.create_dataset("n_mu", data=tr.n_mu)
            g.create_dataset("seed_population", data=tr.seed_population)
            g.create_dataset("comm_mu_err", data=tr.comm_mu_err)
            g.create_dataset("p_mu_top", data=tr.p_mu_top)
            g.attrs["seed_H0_overlap"] = float(tr.seed_H0_overlap)
    return path

def plot_scar_maps(
        scar: ScarAnalysis, labels: Sequence[Bare_Label] = ((0,0), (1,0), (2,0)),
        *, x_label:Optional[str] = None, y_label:Optional[str] = None,
        logscale: bool = False, floor = 1e-4, cmaps: Sequence[str] = ("Reds", "Greens", "Blues"),
        title_prefix: str = "",):
    x = np.asarray(scar.x_values)
    y = np.asarray(scar.y_values)
    fig, axes = plt.subplots(2, len(labels), figsize = (4 * len(labels), 7),
                              constrained_layout=True, sharex=True, sharey=True)

    if x_label is None:
        x_label = r"$\omega_d/2\pi$ [GHz]" if scar.meta.get("x_axis") == "omega_d" else r"$\Phi_\mathrm{ext}/\Phi_0$"
    if y_label is None:
        if scar.meta.get("y_axis") == "nbar":
            y_label = r"$\bar n_r$"
        elif scar.meta.get("y_axis") == "chi_ac":
            y_label = r"$\chi_{\rm ac}/2\pi$ [GHz]"
    for col, (label, cmap) in enumerate(zip(labels, cmaps)):
        Zphi = np.asarray(scar.nphi[label]).T  # (Ny, Nx)
        Zmu  = np.asarray(scar.nmu[label]).T

        if logscale:
            Zphi_plot = np.log10(np.clip(Zphi, floor, None))
            Zmu_plot  = np.log10(np.clip(Zmu, floor, None))
            cbphi = r"$\log_{10}\langle n_\phi\rangle$"
            cbmu  = r"$\log_{10}\langle n_\mu\rangle$"
        else:
            Zphi_plot, Zmu_plot = Zphi, Zmu
            cbphi = r"$\langle n_\phi\rangle$"
            cbmu  = r"$\langle n_\mu\rangle$"

        ax = axes[0, col]
        pm = ax.pcolormesh(x, y, Zphi_plot, shading="none", cmap=cmap, edgecolors='face')
        ax.set_title(rf"{title_prefix} $\langle n_\phi\rangle$ branch {label}")
        if col == 0:
            ax.set_ylabel(y_label)
        fig.colorbar(pm, ax=ax, pad=0.02, label=cbphi)

        ax = axes[1, col]
        pm = ax.pcolormesh(x, y, Zmu_plot, shading="none", cmap=cmap, edgecolors='face')
        ax.set_title(rf"{title_prefix} $\langle n_\mu\rangle$ branch {label}") 
        ax.set_xlabel(x_label)
        if col == 0:
            ax.set_ylabel(y_label)
        fig.colorbar(pm, ax=ax, pad=0.02, label=cbmu)
    return fig, axes

def label_str(lab: Bare_Label) -> str:
    return rf"$|{lab[0]},{lab[1]}\rangle$"

def plot_branch_quantities(
        branches: Dict[Bare_Label, BranchMapping],
        y_values: np.ndarray, *, omega_d_GHz: float, flux: Optional[float] = None,
        quantities: Sequence[Literal["n_phi", "n_mu", "seed_population", "eps"]] = ("n_phi", "n_mu", "seed_population", "eps"),
        eps_units: Literal["rad", "GHz", "omega_d"] = "GHz",
        highlight: Optional[Union[Sequence[Bare_Label], Literal["all"]]] = "all",
        x_label: Optional[str] = "drive axis",
        max_gray_curves: int = 25,
        gray_alpha: float = 0.35,
        show_gray_labels: bool = False,
        gray_color: str = "0.70",
        linewidth: float = 1.5,
        figsize: Tuple[float, float] = (10, 6),
        show_grid: bool = False,
        prop_cycler=None,
):
    all_labels = sorted(branches.keys(), key = lambda x: (x[0], x[1]))
    if highlight is None or highlight == "all":
        highlighted_labels = all_labels
    else:
        highlight_set = set(tuple(x) for x in highlight)
        highlighted_labels = [label for label in all_labels if tuple(label) in highlight_set]
    base_gray = all_labels[: min(max_gray_curves, len(all_labels))]
    gray_labels = [lab for lab in base_gray if lab not in highlighted_labels]

    if prop_cycler is None:
        if cycler is not None:
            prop_cycler = cycler(color=["maroon", "royalblue", "darkseagreen", "darkorange", "mediumpurple"]) * cycler(linestyle=["-", "--", "-.", ":", "-"])
        else:
            prop_cycler = None
    style_iter = cycle(prop_cycler) if prop_cycler is not None else cycle([{"color": "C0"}])
    if eps_units == "rad":
        eps_scale = 1.0
        eps_ylabel = r"$\varepsilon\ [\mathrm{rad/ns}]$"
    elif eps_units == "GHz":
        eps_scale = 1.0 / (2 * np.pi)
        eps_ylabel = r"$\varepsilon/2\pi\ [\mathrm{GHz}]$"
    else:
        eps_scale = 1.0 / (2 * np.pi * omega_d_GHz)
        eps_ylabel = r"$\varepsilon/\omega_d$"

    figs = {}
    for qty in quantities:
        fig, ax = plt.subplots(figsize=figsize)
        for label in gray_labels: 
            trace = branches[label]
            ydata = getattr(trace, qty)
            if qty == "eps":
                ydata = ydata * eps_scale
            ax.plot(y_values, ydata, color=gray_color, alpha=gray_alpha, lw=max(0.8, linewidth * 0.7), zorder=5)

        handles, labels_legend = [], []
        for label in highlighted_labels:
            trace = branches[label]
            ydata = getattr(trace, qty)
            if qty == "eps":
                ydata = ydata * eps_scale
            sty = next(style_iter)
            (ln,) = ax.plot(y_values, ydata, lw=linewidth, zorder=50, **dict(sty))
            handles.append(ln)
            labels_legend.append(label_str(label))
        if qty == "n_phi":
            ax.set_ylabel(r"$\langle n_\phi\rangle$")
        elif qty == "n_mu":
            ax.set_ylabel(r"$\langle n_\mu\rangle$")
        elif qty == "seed_population":
            ax.set_ylabel(r"$|\langle k,n_\mu|\psi\rangle|^2$")
        else:
            ax.set_ylabel(eps_ylabel)

        ax.set_xlabel(x_label or "drive axis")

        title_bits = [rf"$\omega_d/2\pi$={omega_d_GHz:.4f} GHz"]
        if flux is not None:
            title_bits.insert(0, f"flux={flux:.6f}")
        ax.set_title(", ".join(title_bits))

        if show_grid:
            ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)

        if handles:
            ax.legend(handles, labels_legend, fontsize=9, ncol=2, loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        figs[qty] = (fig, ax)

    return figs

def chi_vs_flux_sweep(
        qubit_params: dict, flux_grid: Sequence[float], 
        *, dim_q: int, dim_r: int, omega_r_GHz: float, g_qr_GHz: float,
        coupling_op: Literal["n", "phi"] = "n", evals_count: int = 30,
        num_cpus = None, return_transition_chi: bool = False,
):
    """
    if return transition chi -> true then return chi01 = chi(level1) - chi(level0)
    else return chi1 = chi(level1)
    """
    flux_grid = np.asarray(flux_grid, dtype=float)
    qparams0 = dict(qubit_params)
    qparams0["flux"] = float(flux_grid[0])

    q = scq.Fluxonium(**qparams0, truncated_dim=int(dim_q))
    r = scq.Oscillator(E_osc=float(omega_r_GHz), truncated_dim=int(dim_r))
    hilbert_space = scq.HilbertSpace([q, r])

    if coupling_op == "n":
        op_q = q.n_operator
    elif coupling_op == "phi":
        op_q = q.phi_operator


    hilbert_space.add_interaction(
        g_strength=float(g_qr_GHz),
        op1=op_q,
        op2=r.annihilation_operator,
        add_hc=True,
        id_str="q-r",
    )

    def update_hilbertspace(flux):
        q.flux = float(flux)

    sweep = scq.ParameterSweep(
        hilbertspace=hilbert_space,
        paramvals_by_name={"flux": flux_grid},
        update_hilbertspace=update_hilbertspace,
        subsys_update_info={"flux": [q]},
        evals_count=int(evals_count),
        num_cpus=num_cpus,
    )
    i_q = hilbert_space.subsys_list.index(q)
    i_r = hilbert_space.subsys_list.index(r)
    chi = sweep["chi"]
    chi_qr = chi["subsys1": i_q, "subsys2": i_r]  
    chi0 = np.asarray(chi_qr[:, 0], dtype=float)
    chi1 = np.asarray(chi_qr[:, 1], dtype=float)
    if return_transition_chi:
        return (chi1 - chi0), sweep
    return chi1, sweep
    









    



