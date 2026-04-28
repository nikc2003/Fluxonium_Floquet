"""
landau_zener/plotting/colors.py
branch color mapping + highlight helpers.
"""

from __future__ import annotations
from typing import Optional, Sequence, List, Set
import numpy as np
import matplotlib.pyplot as plt
import colorsys
import matplotlib.colors as mcolors


def _adjust_lightness(rgb, factor: float):
    r, g, b = mcolors.to_rgb(rgb)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, min(1.0, l * factor))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return (r2, g2, b2)


def build_branch_colors_shaded(nb: int, *, min_factor: float = 0.65, max_factor: float = 1.15) -> list:
    cmap = plt.get_cmap("Paired")
    base = list(getattr(cmap, "colors", [cmap(x) for x in np.linspace(0, 1, 9)]))
    n_base = len(base)
    n_layers = int(np.ceil(nb / n_base))
    factors = np.linspace(max_factor, min_factor, n_layers)

    colors = []
    for b in range(nb):
        c0 = base[b % n_base]
        layer = b // n_base
        f = float(factors[layer])
        colors.append(_adjust_lightness(c0, f))
    return colors


def normalize_branch_list(branches: Optional[Sequence[int]], nb: int) -> list[int]:
    if branches is None:
        return list(range(nb))
    out = []
    for b in branches:
        bi = int(b)
        if 0 <= bi < nb:
            out.append(bi)
    return sorted(set(out))


def normalize_highlight_list(highlight_branches: Optional[Sequence[int]], nb: int) -> Set[int]:
    if highlight_branches is None:
        return set()
    s = set()
    for b in highlight_branches:
        bi = int(b)
        if 0 <= bi < nb:
            s.add(bi)
    return s