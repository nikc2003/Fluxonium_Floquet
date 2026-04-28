"""
physics/hamiltonian.py
build H0 and H1 in fluxonium eigenbasis using scqubits + qutip.
"""

from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np
import qutip as qt
import scqubits as scq


def get_H0_H1_for_flux(flux_value: float, qubit_params: Dict[str, Any], dim_q: int) -> Tuple[qt.Qobj, qt.Qobj]:
    """
    Build H0 and drive operator H1 in fluxonium eigenbasis.
    scqubits energies are in GHz -> convert to rad/s (ns?) (multiply by 2pi).
    """
    qparams = dict(qubit_params)
    qparams["flux"] = float(flux_value)
    fluxonium = scq.Fluxonium(**qparams, truncated_dim=int(dim_q))
    hs = scq.HilbertSpace([fluxonium])
    hs.generate_lookup()
    evals = hs["evals"][0][:dim_q]
    H0 = 2.0 * np.pi * qt.Qobj(np.diag(evals - evals[0]))  
    H1 = hs.op_in_dressed_eigenbasis(fluxonium.n_operator)
    return H0, H1