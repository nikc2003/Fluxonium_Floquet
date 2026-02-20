from __future__ import annotations

from typing import Sequence, Tuple, Dict, Optional

import numpy as np
import qutip as qt
import floquet as ft

from .config import CompositeSysConfig, BuildComposite, Label
from .basis import build_fluxonium_eigenbasis, enforce_hermiticity, bosonic_quadrature
from ..utils.timing import Timing



"""
Build the composite model, dress the operators, and promote everything to the extended Hilbert space> This is the first step in the drive/flux sweep protocol

the setup of the composite model is generalized for qubit + bosonic mode, where the static hamiltonian has form:
omega_r adag a + sum_j E_j |j><j| + sum_{j,k} g_{jk}_eff |j><k| (a + adag) 

[or -ij (a - adag) depending on if coupling quadrature in x or p)]

timing checks + operation description:
- build_fluxonium_eigenbasis: diagonalization of the fluxonium Hamiltonian, and projection of the coupling operator into the truncated eigenbasis.
- build_bare_operators + ops.build: constructing the bare operators for the qubit and resonator in their respective bases, and tensoring them up to the full Hilbert space.
- CompositeModel.init: negligible time, just sets up the object
- CompositeModel.labels_to_indices: negligible time, just maps the provided labels to indices in the full Hilbert space via overlap
- ChiacToAmp.compute: computes the drive amplitudes corresponding to the desired chi_ac values at each drive frequency. This involves some linear algebra but is still relatively fast.
- ops.to_dressed: transforming the operators to the dressed basis using the unitary transformation from the diagonalization. 

"""


def build_composite_model(
    cfg: CompositeSysConfig,
    omega_d_values: np.ndarray,
    chi_ac_values: np.ndarray,
    *,
    label_states: Sequence[Label] = ((0, 0), (1, 0)),
    timer: Optional[Timing] = None,
) -> BuildComposite:

    tr = timer or Timing()

    omega_d_values = np.atleast_1d(np.array(omega_d_values, dtype=float))
    chi_ac_values = np.atleast_1d(np.array(chi_ac_values, dtype=float))

    dim_q = int(cfg.dim_q)
    dim_r = int(cfg.dim_r)

    with tr.span("build_fluxonium_eigenbasis"):
        Hq, n_op, _ = build_fluxonium_eigenbasis(
            cfg.qubit_params,
            dim_q,
            coupling_op=cfg.coupling_op,
            timer=tr,
        )

    with tr.span("build_bare_operators"):
        a = qt.destroy(dim_r)
        Iq = qt.qeye(dim_q)
        Ir = qt.qeye(dim_r)
        Xr = bosonic_quadrature(a, cfg.resonator_quadrature)

        H_r = cfg.omega_r * qt.tensor(Iq, a.dag() * a)
        H_q = qt.tensor(Hq, Ir)
        H_int = cfg.g * qt.tensor(n_op, Xr)
        H0_bare = enforce_hermiticity(H_q + H_r + H_int)

        H1_drive_bare = enforce_hermiticity(qt.tensor(n_op, Ir))

    placeholder = np.zeros((len(chi_ac_values), len(omega_d_values)), dtype=float)

    with tr.span("CompositeModel.init"):
        model = ft.model.CompositeModel(
            H0_bare=H0_bare,
            H1_drive=H1_drive_bare,
            omega_d_values=omega_d_values,
            drive_amplitudes=placeholder,
            subsystem_dims=(dim_q, dim_r),
            state_labels_dims=(dim_q, dim_r),
        )

    with tr.span("CompositeModel.labels_to_indices"):
        dressed_indices = model.labels_to_indices(list(label_states))

    with tr.span("ChiacToAmp.compute"):
        chi2amp = ft.ChiacToAmp(model.H0, model.H1, dressed_indices, model.omega_d_values)
        model.drive_amplitudes = chi2amp.amplitudes_for_omega_d(chi_ac_values)

    # bare operators
    with tr.span("ops.build"):
        Nq_op = qt.Qobj(np.diag(np.arange(dim_q)), dims=[[dim_q], [dim_q]])
        Nr_op = a.dag() * a

        ops_bare: Dict[str, qt.Qobj] = {
            "a": qt.tensor(Iq, a),
            "adag": qt.tensor(Iq, a.dag()),
            "Nr": qt.tensor(Iq, Nr_op),
            "Nq": qt.tensor(Nq_op, Ir),
            "nq": qt.tensor(n_op, Ir),
            "H0": H0_bare,
            "H1_drive_bare": H1_drive_bare,
        }

    U = model.U
    ops_diag: Dict[str, qt.Qobj] = {}
    with tr.span("ops.to_dressed"):
        for name, op in ops_bare.items():
            dressed = U.dag() * op * U
            ops_diag[name] = enforce_hermiticity(dressed) if op.isherm else dressed

    return BuildComposite(
        cfg=cfg,
        model=model,
        dressed_indices=list(map(int, dressed_indices)),
        bare_ops=ops_bare,
        diag_ops=ops_diag,
    )
