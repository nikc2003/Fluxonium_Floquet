"""
landau_zener/utils/parallel.py
process-pool helpers,  parallel map with progress.
"""

from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from typing import Callable, Iterable, Any, List, Optional


def process_pool_map(
    fn: Callable[[Any], Any],
    items: Iterable[Any],
    *,
    max_workers: int,
    mp_start_method: str = "spawn",
    chunksize: int = 1,
) -> List[Any]:

    items = list(items)
    if len(items) == 0:
        return []

    ctx = mp.get_context(mp_start_method)

    results = [None] * len(items)
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        fut_to_idx = {}
        for i, it in enumerate(items):
            fut = ex.submit(fn, it)
            fut_to_idx[fut] = i

        for fut in as_completed(fut_to_idx):
            i = fut_to_idx[fut]
            results[i] = fut.result()

    return results