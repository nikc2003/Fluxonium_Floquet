from __future__ import annotations

from typing import Dict, Tuple, Optional

import numpy as np
import qutip as qt
import scqubits as scq

from ..utils.timing import Timing

"""
Constructs the static qubit Hamiltonian + drive coupling operators within subsystem eigenbasis and sets up bosonic coupling mechanism,
where Uq is the change of basis matrix for the qubit coupling operator to truncate to the lowest dim_q eigenstates of the qubit, and the 
bosonic operators are left in the Fock basis.

timing checks:
build_fluxonium_eigenbasis 
    - diagonalization of the fluxonium Hamiltonian, and projection of the coupling operator into the truncated eigenbasis.
"""

def enforce_hermiticity(H: qt.Qobj) -> qt.Qobj:
    return 0.5 * (H + H.dag())

def _columns_from_evecs(evecs, cutoff: int, dim_q: int) -> np.ndarray:
    if isinstance(evecs, (list, tuple)):
        U = np.column_stack([np.asarray(v).reshape(-1) for v in evecs])
    else:
        M = np.asarray(evecs)
        candidates = []
        for A in (M, M.T):
            if A.shape == (cutoff, dim_q):
                err = np.linalg.norm(A.conj().T @ A - np.eye(dim_q))
                candidates.append((err, A))
        if not candidates:
            raise ValueError(f"incompatible eigenvector array shape {M.shape}; cutoff={cutoff}, dim_q={dim_q}")
        _, U = min(candidates, key=lambda x: x[0])

   
    return np.asarray(U, dtype=complex)


def build_fluxonium_eigenbasis(
    qubit_params: Dict[str, float],
    dim_q: int,
    *,
    coupling_op: str = "n",
    timer: Optional[Timing] = None,
) -> Tuple[qt.Qobj, qt.Qobj, np.ndarray]:

    tr = timer or Timing()
    with tr.span("fluxonium.eigensys"):
        fluxonium = scq.Fluxonium(**qubit_params, truncated_dim=int(dim_q))
        evals_GHz, evecs = fluxonium.eigensys(evals_count=int(dim_q))
    evals_GHz = np.asarray(evals_GHz, dtype=float)

    Hq = 2 * np.pi * qt.Qobj(np.diag(evals_GHz - evals_GHz[0]), dims=[[dim_q], [dim_q]])

    if coupling_op == "n":
        op = fluxonium.n_operator()
    elif coupling_op == "phi":
        op = fluxonium.phi_operator()
    else:
        raise ValueError(f"unknown coupling_op: {coupling_op}")

    with tr.span("fluxonium.op_project"):
        op_arr = np.array(op, dtype=complex)
        cutoff = op_arr.shape[0]
        Uq = _columns_from_evecs(evecs, cutoff, dim_q)
        op_eig = Uq.conj().T @ op_arr @ Uq
        op_eig = enforce_hermiticity(qt.Qobj(op_eig, dims=[[dim_q], [dim_q]]))
        qubit_coupling_op = qt.Qobj(op_eig.full(), dims=[[dim_q], [dim_q]])

    return Hq, qubit_coupling_op, evals_GHz


def bosonic_quadrature(a: qt.Qobj, quadrature: str) -> qt.Qobj:
    if quadrature == "x":
        return (a + a.dag())
    elif quadrature == "y":
        return -1j * (a - a.dag())
    else:
        raise ValueError(f"unknown bosonic_quadrature: {quadrature}")
