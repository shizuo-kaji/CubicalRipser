"""Lightweight plotting helpers for persistence diagrams.

This module intentionally avoids a dependency on `persim`.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _as_diagram_array(diagram: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(diagram, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Each diagram must have shape (n, 2).")
    return arr


def _normalize_diagrams(
    diagrams: Iterable[np.ndarray | Sequence[Sequence[float]]] | np.ndarray,
) -> list[np.ndarray]:
    if isinstance(diagrams, np.ndarray):
        return [_as_diagram_array(diagrams)]
    out: list[np.ndarray] = []
    for d in diagrams:
        out.append(_as_diagram_array(d))
    return out


def plot_diagrams(
    diagrams: Iterable[np.ndarray | Sequence[Sequence[float]]] | np.ndarray,
    *,
    labels: Sequence[str] | None = None,
    ax=None,
    title: str | None = None,
    legend: bool = True,
    diagonal: bool = True,
    marker_size: float = 18.0,
    alpha: float = 0.8,
    show: bool = False,
):
    """Plot one or more persistence diagrams.

    Parameters
    - diagrams: list of arrays (n_i, 2) or a single (n, 2) array.
    - labels: optional labels for the legend.
    - ax: optional matplotlib axis.
    - title: optional axis title.
    - diagonal: whether to draw y=x.
    - legend: whether to show legend.
    """
    import matplotlib.pyplot as plt

    diag_list = _normalize_diagrams(diagrams)
    if labels is not None and len(labels) != len(diag_list):
        raise ValueError("labels length must match number of diagrams")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    finite_vals: list[np.ndarray] = []
    has_inf = False
    for d in diag_list:
        if d.size == 0:
            continue
        finite_vals.append(d[:, 0])
        finite_deaths = d[np.isfinite(d[:, 1]), 1]
        if finite_deaths.size:
            finite_vals.append(finite_deaths)
        if np.any(~np.isfinite(d[:, 1])):
            has_inf = True

    if finite_vals:
        all_finite = np.concatenate(finite_vals)
        lo = float(np.min(all_finite))
        hi = float(np.max(all_finite))
    else:
        lo, hi = 0.0, 1.0

    span = max(hi - lo, 1e-8)
    pad = 0.08 * span
    inf_y = hi + 0.15 * span

    for i, d in enumerate(diag_list):
        if d.size == 0:
            continue
        plot_d = d.copy()
        inf_mask = ~np.isfinite(plot_d[:, 1])
        if np.any(inf_mask):
            plot_d[inf_mask, 1] = inf_y
        label = labels[i] if labels is not None else f"H{i}"
        ax.scatter(
            plot_d[:, 0],
            plot_d[:, 1],
            s=marker_size,
            alpha=alpha,
            label=label,
        )

    if diagonal:
        diag_lo = lo - pad
        diag_hi = inf_y + pad if has_inf else hi + pad
        ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], "k--", linewidth=1.0, alpha=0.6)

    if has_inf:
        ax.axhline(inf_y, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.text(
            lo - pad * 0.7,
            inf_y,
            "inf",
            va="bottom",
            ha="left",
            fontsize=9,
            color="gray",
        )

    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, (inf_y + pad) if has_inf else (hi + pad))
    ax.grid(True, alpha=0.2)
    if title is not None:
        ax.set_title(title)
    if legend is True:
        ax.legend(loc="lower right")
    if show:
        plt.show()
