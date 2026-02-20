from __future__ import annotations
import os
from dataclasses import dataclass

thread_env_vars = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]

def set_single_thread_env(n: int = 1, force: bool = True) -> None:
    n_str = str(int(n))
    for k in thread_env_vars:
        if force or (k not in os.environ):
            os.environ[k] = n_str


@dataclass(frozen=True)
class CPUPlan:
    max_workers: int 
    blas_threads_per_worker: int 
    inner_num_cpus: int #for ft.Options

    def recommend_cpu_plan(*, max_workers: int |None = None) -> CPUPlan:
        """use a fraction of logical CPUs for process workers."""
        cpu_count = os.cpu_count() or 2
        if max_workers is None:
            max_workers = max(1, cpu_count // 2)
        return CPUPlan(
            max_workers=max_workers,
            blas_threads_per_worker=1,
            inner_num_cpus=1)
