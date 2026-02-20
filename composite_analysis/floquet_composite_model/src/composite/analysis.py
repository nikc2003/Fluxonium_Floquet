from __future__ import annotations

from typing import Dict, Optional, Sequence, List, Any

import numpy as np
import floquet as ft

from .config import BuildComposite
from ..utils.timing import Timing
"""
Module for running Floquet analysis post building the composite model.
This protocol is what will ultimately be computed for each point in the drive/flux sweep, 
and the results are what get saved to disk for later analysis and plotting.
Bulk of computation time is in the heart of 'fa.run() in floquet/floquet.py, FloquetAnalysis class.


timing checks:
- FloquetAnalysis.init: negligible time, just sets up the object
- FloquetAnalysis.run: where we solve for the Floquet states and compute all the quantities of interest. 
"""
def run_floquet_analysis_composite(
    build: BuildComposite,
    options: ft.Options,
    *,
    filepath: Optional[str] = None,
    timer: Optional[Timing] = None,
) -> Dict[str, Any]:
    tr = timer or Timing()
    #note: need to config. timing records to witin the floquet.py code. all the computation really lies in the `fa.run()` call

    with tr.span("FloquetAnalysis.init"):
        fa = ft.FloquetAnalysis(build.model, state_indices=build.dressed_indices, options=options)
    with tr.span("FloquetAnalysis.run"):
        data = fa.run(filepath=filepath)
    return data

def extract_scar_maps_from_displaced_overlaps(
    data: Dict[str, Any],
    *,
    state_slots: Sequence[int] = (0,),
    clip_max: float = 0.5,
) -> Dict[int, np.ndarray]:
    ovlp = np.asarray(data.get("displaced_state_overlaps"))

    out: Dict[int, np.ndarray] = {}
    for slot in state_slots:
        slot = int(slot)
        if slot < 0 or slot >= ovlp.shape[2]:
            raise IndexError(f"state_slot {slot} out of range for ovlp with Nstates={ovlp.shape[2]}")
        scar = 1.0 - np.abs(ovlp[:, :, slot]) ** 2
        out[slot] = np.clip(np.real(scar), 0.0, float(clip_max)).astype(np.float32, copy=False)
    return out
