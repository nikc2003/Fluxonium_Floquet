from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Sequence, Optional, List, Any

"""
Module for defining configuration dataclasses and related utilities for the composite model (qubit + bosonic mode).
    Sets dimension, frequencies, coupling parameters, etc. for the composite system, as well as the preliminary configuration for 
    floquet analysis such as which states to label and compute overlaps for. 
"""
Label = Tuple[int, int]
@dataclass(frozen=True)
class CompositeSysConfig:
    qubit_params: Dict[str, float]
    dim_q: int
    dim_r: int
    omega_r: float
    g: float
    coupling_op: str = "n"             
    resonator_quadrature: str = "x"    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class BuildComposite:
    cfg: CompositeSysConfig
    model: Any  #ft.model.CompositeModel
    dressed_indices: List[int]
    bare_ops: Dict[str, Any]#qt.Qobj
    diag_ops: Dict[str, Any]#qt.Qobj
