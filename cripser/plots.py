"""Lightweight plotting helpers for persistence diagrams and cycles.

This module intentionally avoids a dependency on `persim`.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .utils import to_gudhi_diagrams


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
        # Allow passing raw CubicalRipser output of shape (n, 9).
        if diagrams.ndim == 2 and diagrams.shape[1] == 9:
            return [np.asarray(d, dtype=np.float64) for d in to_gudhi_diagrams(diagrams)]
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
    - diagrams: list of arrays (n_i, 2), a single (n, 2) array, or
      CubicalRipser output of shape (n, 9).
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


def _as_planar_h1_cycle(cycle: Sequence[Sequence[int | float]] | np.ndarray) -> np.ndarray:
    """Validate the compact H_1 cell encoding returned by ``compute_ph``."""
    cells = np.asarray(cycle)
    if cells.ndim != 2 or cells.shape[1] != 4:
        raise ValueError(
            "A 2-D H_1 cycle must have shape (n, 4): [x, y, z, cell_type]."
        )
    if cells.shape[0] == 0:
        raise ValueError("A cycle must contain at least one cell.")
    if not np.all(np.isfinite(cells)) or not np.all(cells == np.floor(cells)):
        raise ValueError("Cycle cells must use finite integer coordinates and cell types.")

    cells = cells.astype(np.int64, copy=False)
    if np.any(cells[:, 2] != 0):
        raise ValueError("Only planar H_1 cycles (z == 0) can be drawn on a 2-D image.")
    if np.any((cells[:, 3] != 0) & (cells[:, 3] != 1)):
        raise ValueError("Planar H_1 cycles may contain only x-edges (0) and y-edges (1).")
    return cells


def _plot_image_background(
    ax,
    image: np.ndarray | Sequence[Sequence[float]],
    *,
    cmap: str,
    alpha: float,
    interpolation: str,
) -> np.ndarray:
    """Draw an ``image[x, y]`` array in the cycle coordinate convention."""
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError("image must be a two-dimensional array.")
    if array.size == 0:
        raise ValueError("image must not be empty.")

    size_x, size_y = array.shape
    ax.imshow(
        array.T,
        cmap=cmap,
        alpha=alpha,
        interpolation=interpolation,
        origin="lower",
        extent=(-0.5, size_x - 0.5, -0.5, size_y - 0.5),
    )
    ax.set_xlim(-0.5, size_x - 0.5)
    ax.set_ylim(-0.5, size_y - 0.5)
    return array


def _cycle_segments(cells: np.ndarray) -> list[np.ndarray]:
    """Convert x/y cubical edges into Matplotlib line segments."""
    segments: list[np.ndarray] = []
    for x, y, _z, cell_type in cells:
        if cell_type == 0:  # x-edge
            segments.append(np.array(((x, y), (x + 1, y)), dtype=np.float64))
        else:  # y-edge
            segments.append(np.array(((x, y), (x, y + 1)), dtype=np.float64))
    return segments


def plot_cycle(
    cycle: Sequence[Sequence[int | float]] | np.ndarray,
    *,
    image: np.ndarray | Sequence[Sequence[float]] | None = None,
    overlay: bool = True,
    ax=None,
    color: str = "tab:red",
    linewidth: float = 2.5,
    alpha: float = 1.0,
    label: str | None = None,
    image_cmap: str = "gray",
    image_alpha: float = 0.85,
    interpolation: str = "nearest",
    show: bool = False,
):
    """Plot one planar H₁ representative cycle.

    ``cycle`` uses the compact cell encoding returned by
    ``compute_ph(..., representatives=True)``: each row is
    ``[x, y, z, cell_type]``.  In 2-D, cell type ``0`` is an x-edge and ``1``
    is a y-edge.  Supply ``image`` to draw the cycle over an ``image[x, y]``
    background; set ``overlay=False`` to draw only the cycle.

    The function returns the Matplotlib axis, making repeated calls with the
    same ``ax`` convenient for comparing or layering cycles.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    cells = _as_planar_h1_cycle(cycle)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    if image is not None and overlay:
        _plot_image_background(
            ax,
            image,
            cmap=image_cmap,
            alpha=image_alpha,
            interpolation=interpolation,
        )

    line_collection = LineCollection(
        _cycle_segments(cells),
        colors=color,
        linewidths=linewidth,
        alpha=alpha,
        label=label,
    )
    ax.add_collection(line_collection)
    if image is None or not overlay:
        ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if label is not None:
        ax.legend(loc="best")
    if show:
        plt.show()
    return ax


def plot_cycles(
    cycles: Iterable[Sequence[Sequence[int | float]] | np.ndarray],
    *,
    image: np.ndarray | Sequence[Sequence[float]] | None = None,
    overlay: bool = True,
    ax=None,
    colors: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
    linewidth: float = 2.5,
    alpha: float = 1.0,
    image_cmap: str = "gray",
    image_alpha: float = 0.85,
    interpolation: str = "nearest",
    show: bool = False,
):
    """Plot multiple planar H₁ representative cycles, optionally over an image."""
    import matplotlib.pyplot as plt

    cycle_list = list(cycles)
    if colors is not None and len(colors) != len(cycle_list):
        raise ValueError("colors length must match the number of cycles.")
    if labels is not None and len(labels) != len(cycle_list):
        raise ValueError("labels length must match the number of cycles.")
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    if image is not None and overlay:
        _plot_image_background(
            ax,
            image,
            cmap=image_cmap,
            alpha=image_alpha,
            interpolation=interpolation,
        )
        image_xlim = ax.get_xlim()
        image_ylim = ax.get_ylim()
    else:
        image_xlim = None
        image_ylim = None

    default_colors = plt.get_cmap("tab10")
    for index, cycle in enumerate(cycle_list):
        color = colors[index] if colors is not None else default_colors(index % 10)
        label = labels[index] if labels is not None else None
        plot_cycle(
            cycle,
            overlay=False,
            ax=ax,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
        )

    if image_xlim is not None and image_ylim is not None:
        # Each ``plot_cycle`` call autoscales its LineCollection. Restore the
        # image extent so pixel/edge alignment remains exact.
        ax.set_xlim(image_xlim)
        ax.set_ylim(image_ylim)
    elif cycle_list:
        ax.autoscale_view()
    if show:
        plt.show()
    return ax
