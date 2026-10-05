"""Zigzag persistence of time-varying binary images and volumes."""

from __future__ import annotations

import numpy as np

from ._cripser import computeZigzag  # type: ignore


def compute_zigzag(
    masks: np.ndarray,
    *,
    filtration: str = "V",
    connect: str = "intersection",
    maxdim: int = 3,
    exhaustive: bool = True,
) -> np.ndarray:
    """Compute zigzag persistent homology of a sequence of binary masks.

    Frame ``t`` of ``masks`` defines a cubical complex ``K_t``.  With
    ``connect="intersection"`` the zigzag is

        K_0 <- K_0 ∩ K_1 -> K_1 <- K_1 ∩ K_2 -> ... -> K_{T-1},

    and with ``connect="union"`` it is

        K_0 -> K_0 ∪ K_1 <- K_1 -> K_1 ∪ K_2 <- ... <- K_{T-1}.

    Coefficients are in F₂.

    Parameters
    - masks: array of shape ``(T, n1)``, ``(T, n1, n2)`` or
      ``(T, n1, n2, n3)``; the first axis is time.  Boolean or integer
      (nonzero means present).  Threshold real-valued data first, e.g.
      ``arr <= level`` for sublevel sets.  C- and Fortran-ordered arrays
      are both used without copying.
    - filtration: ``"V"`` (mask entries are vertices) or ``"T"`` (mask
      entries are top-dimensional cells), as in :func:`compute_ph`.
    - connect: ``"intersection"`` (default) or ``"union"``.
    - maxdim: highest homology dimension to compute; bounded by the number
      of spatial axes minus one.
    - exhaustive: for two spatial axes, reduce cached columns exhaustively in
      the dimension-1 computation (default).  This is faster on noisy
      sequences and somewhat slower on smooth ones; the result is the same
      either way.  It has no effect for one or three spatial axes.

    Returns
    - np.ndarray of shape (n, 9):
      ``[dim, birth, death, b_x, b_y, b_z, d_x, d_y, d_z]``.
      ``birth`` and ``death`` are the first and last positions at which the
      class is alive, in frame units: ``t`` is frame ``t`` and ``t + 0.5`` is
      the complex connecting frames ``t`` and ``t + 1``.  An integer end is a
      closed end at that frame and a half-integer end is an open end.  A class
      alive in the last frame has ``death = T - 1``.  ``(b_x, b_y, b_z)`` and
      ``(d_x, d_y, d_z)`` are the anchor voxels of the cells whose addition or
      deletion creates and destroys the class; unused axes are 0.  The
      destroyer of a class alive in the last frame is ``-1`` on every axis.
    """
    masks = np.asarray(masks)
    if masks.ndim < 2 or masks.ndim > 4:
        raise ValueError(
            "Expected masks of shape (T, n1[, n2[, n3]]): a time axis followed "
            "by 1-3 spatial axes"
        )
    if masks.dtype != np.bool_:
        if not np.issubdtype(masks.dtype, np.integer):
            raise TypeError(
                "compute_zigzag expects boolean or integer masks; threshold "
                "real-valued data first, e.g. arr <= level"
            )
        masks = masks != 0  # keeps the memory layout
    if not (masks.flags.c_contiguous or masks.flags.f_contiguous):
        masks = np.ascontiguousarray(masks)

    construction = filtration.upper()
    if construction not in ("V", "T"):
        raise ValueError('filtration must be "V" or "T"')
    if connect not in ("intersection", "union"):
        raise ValueError('connect must be "intersection" or "union"')
    if maxdim < 0:
        raise ValueError("maxdim must be non-negative")

    return computeZigzag(
        masks,
        tconstruction=construction == "T",
        union_connect=connect == "union",
        maxdim=int(maxdim),
        exhaustive=bool(exhaustive),
    )
