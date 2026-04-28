"""some basic plotting helpers for branch diagnostics"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable
import matplotlib.pyplot as plt

def plot_single_branch(metrics: list[dict], title: str | None = None, savepath: str | None = None) -> None:
    """common diagnostics for a single branch."""
    x = [m["avg_resonator_number"] for m in metrics]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    axes[0].plot(x, [m["avg_fluxonium_level"] for m in metrics], marker="o", markersize=3)
    axes[0].set_xlabel(r"$\langle a^\dagger a \rangle$")
    axes[0].set_ylabel("avg fluxonium level")
    axes[1].plot(x, [m["computational_population"] for m in metrics], marker="o", markersize=3)
    axes[1].set_xlabel(r"$\langle a^\dagger a \rangle$")
    axes[1].set_ylabel("computational population")
    axes[2].plot(x, [m["fluxonium_participation_ratio"] for m in metrics], marker="o", markersize=3)
    axes[2].set_xlabel(r"$\langle a^\dagger a \rangle$")
    axes[2].set_ylabel("fluxonium participation ratio")
    axes[3].plot(x, [m["reduced_fluxonium_purity"] for m in metrics], marker="o", markersize=3)
    axes[3].set_xlabel(r"$\langle a^\dagger a \rangle$")
    axes[3].set_ylabel("reduced fluxonium purity")

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    if savepath is not None:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_branch_population_distribution(metrics: list[dict], rung: int, savepath: str | None = None) -> None:
    """bare fluxonium-level populations for one branch state."""
    if not (0 <= rung < len(metrics)):
        raise IndexError("rung out of range")
    m = metrics[rung]
    p = m["p_fluxonium"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(p)), p)
    ax.set_xlabel("bare fluxonium level")
    ax.set_ylabel("population")
    ax.set_title(f"Branch {m['branch']} rung {m['rung']} population distribution")
    fig.tight_layout()
    if savepath is not None:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
