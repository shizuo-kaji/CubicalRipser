"""
Utilities for CubicalRipser's pybind modules under `cripser.utils`.

This submodule provides helpers to:
- Compute persistent homology via `cripser` or `tcripser`.
- Convert the returned array (n, 9) into GUDHI-compatible representations.

Columns of `computePH` output:
    [dim, birth, death, b_x, b_y, b_z, d_x, d_y, d_z]

GUDHI-compatible formats supported here:
- diagrams: List[np.ndarray] of shape (k_i, 2) per homology dimension i.
- persistence: List[Tuple[int, Tuple[float, float]]]

Infinite deaths (encoded internally as DBL_MAX) are converted to numpy.inf.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union
from itertools import product

import numpy as np
from ._cripser import computePH, __version__  # type: ignore
try:
    from tcripser import computePH as computePH_T
except ImportError:
    ValueError(
        "tcripser is not installed. Please install it to use the T-construction."
    )

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


_INF_CUTOFF = np.finfo(np.float64).max / 2.0  # heuristic to detect DBL_MAX


def dual_embedding(arr: ArrayLike) -> np.ndarray:
    """Embed values onto the dual grid by taking local minima.

    Given an input array ``arr`` of shape ``(n1, ..., nd)``, return an array
    of shape ``(n1+1, ..., nd+1)`` where each output entry is the minimum of
    incident input cells (boundary handled naturally with fewer incidents).

    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim < 1 or a.ndim > 4:
        raise ValueError("dual_embedding expects a 1D-4D array")

    out_shape = tuple(s + 1 for s in a.shape)
    out = np.full(out_shape, np.inf, dtype=np.float64)

    # Each input cell contributes to one of the 2^d corners in the dual grid.
    for offset in product((0, 1), repeat=a.ndim):
        out_slice = tuple(slice(o, o + s) for o, s in zip(offset, a.shape))
        out[out_slice] = np.minimum(out[out_slice], a)

    return out


def compute_ph(
    arr: np.ndarray,
    *,
    filtration: str = "V",
    maxdim: int = 3,
    top_dim: bool = False,
    embedded: bool = False,
    location: str = "yes",
    inf_cutoff: bool = True,
) -> np.ndarray:
    """Compute persistent homology using `cripser` or `tcripser`.

    Parameters
    - arr: numpy array (1D/2D/3D/4D) of dtype float64
    - maxdim, top_dim, embedded, location: forwarded to the pybind function
    - inf_cutoff: value to use as cutoff for detecting DBL_MAX (default is _INF_CUTOFF)

    Returns
    - np.ndarray of shape (n, 9) (or (n,11) in 4D case): columns are
      [dim, birth, death, b_x, b_y, b_z, d_x, d_y, d_z]
    """
    if arr.dtype != np.float64:
        arr = arr.astype(np.float64, copy=False)
    #mod = importlib.import_module(module)
    func = computePH_T if filtration.upper() == "T" else computePH
    out = func(arr, maxdim=maxdim, top_dim=top_dim, embedded=embedded, location=location)
    # Convert CubicalRipser's DBL_MAX to np.inf
    if inf_cutoff:
        mask = out[:, 2] >= _INF_CUTOFF  # death column is index 2
        if mask.any():
            out[mask, 2] = np.inf
    return out


def _as_2col_pairs(bd: np.ndarray) -> np.ndarray:
    """Ensure an array of shape (k, 2) with inf conversion."""
    out = np.asarray(bd, dtype=np.float64)
    if out.ndim != 2 or out.shape[1] != 2:
        raise ValueError("Expected array of shape (k, 2)")
    # Convert CubicalRipser's DBL_MAX to np.inf
    mask = out[:, 1] >= _INF_CUTOFF
    if mask.any():
        out[mask, 1] = np.inf
    return out


def to_gudhi_diagrams(
    ph: np.ndarray,
    *,
    maxdim: int | None = None,
    include_empty: bool = True,
) -> List[np.ndarray]:
    """Convert CubicalRipser output to GUDHI diagrams (list of arrays).

    Parameters
    - ph: array of shape (n, 9) from `computePH`.
    - maxdim: if given, include dimensions [0..maxdim]; otherwise infer from data.
    - include_empty: if True, include empty arrays for missing dimensions.

    Returns
    - List[np.ndarray], where entry i is an array of shape (k_i, 2)
      of (birth, death) pairs for H_i. Deaths at DBL_MAX are converted to np.inf.
    """
    a = np.asarray(ph)
    if a.ndim != 2 or a.shape[1] < 3:
        raise ValueError("Expected (n, 9) or at least (n, 3) array from computePH")

    dims = a[:, 0].astype(int, copy=False)
    b = a[:, 1]
    d = a[:, 2]

    if maxdim is None:
        maxdim = int(dims.max()) if a.size else 0

    diagrams: List[np.ndarray] = []
    for k in range(maxdim + 1):
        sel = dims == k
        if not sel.any():
            if include_empty:
                diagrams.append(np.empty((0, 2), dtype=np.float64))
            continue
        pairs = np.column_stack((b[sel], d[sel]))
        diagrams.append(_as_2col_pairs(pairs))
    return diagrams


def to_gudhi_persistence(ph: np.ndarray) -> List[Tuple[int, Tuple[float, float]]]:
    """Convert CubicalRipser output to GUDHI's `SimplexTree.persistence()`-like list.

    Returns
    - List of tuples: (dim, (birth, death)), with death converted to np.inf when appropriate.
    """
    a = np.asarray(ph)
    if a.ndim != 2 or a.shape[1] < 3:
        raise ValueError("Expected (n, 9) or at least (n, 3) array from computePH")
    out: List[Tuple[int, Tuple[float, float]]] = []
    for row in a:
        dim = int(row[0])
        birth = float(row[1])
        death = float(row[2])
        if death >= _INF_CUTOFF:
            death = float("inf")
        out.append((dim, (birth, death)))
    return out


def group_by_dim(ph: np.ndarray) -> List[np.ndarray]:
    """Group full rows by homology dimension.

    Parameters
    - ph: array of shape (n, 9) from `computePH`.

    Returns
    - List of subarrays (each with the same 9 columns), ordered by increasing dimension.
    """
    a = np.asarray(ph)
    if a.ndim != 2 or a.shape[1] != 9:
        raise ValueError("Expected (n, 9) array from computePH")
    dims = a[:, 0].astype(int, copy=False)
    maxdim = int(dims.max()) if a.size else 0
    groups: List[np.ndarray] = []
    for k in range(maxdim + 1):
        groups.append(a[dims == k])
    return groups
