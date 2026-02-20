from __future__ import annotations
from typing import Iterable, Iterator, Sequence, Tuple, TypeVar, List

T = TypeVar("T")

def partition_slices(n: int, chunk_size: int) -> Iterator[Tuple[int, int]]:
    """(start, end) indices for partitioning a range(n) into chunks."""
    for i0 in range(0, n, chunk_size):
        yield i0, min(n, i0 + chunk_size)


def partition_list(lst: Sequence[T], chunk_size: int) -> Iterator[List[T]]:
    """successive chunk-sized partitions of the input list."""
    for i0, i1 in partition_slices(len(lst), chunk_size):
        yield list(lst[i0:i1])