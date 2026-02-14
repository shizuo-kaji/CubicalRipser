"""Persistent-homology vectorization utilities."""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np

LocationMode = Literal["birth", "death"]


def _infer_image_shape(
    image_shape: Sequence[int] | None,
    reference_volume: np.ndarray | Sequence[float] | None,
) -> tuple[int, ...]:
    if image_shape is not None:
        shape = tuple(int(v) for v in image_shape)
    elif reference_volume is not None:
        shape = tuple(int(v) for v in np.asarray(reference_volume).shape)
    else:
        raise ValueError("Provide either `image_shape` or `reference_volume`.")
    if len(shape) < 1:
        raise ValueError("`image_shape` must contain at least one spatial axis.")
    if any(v <= 0 for v in shape):
        raise ValueError("All spatial dimensions in `image_shape` must be > 0.")
    return shape


def _resolve_ranges(
    births: np.ndarray,
    lifetimes: np.ndarray,
    *,
    birth_range: tuple[float, float] | None,
    life_range: tuple[float, float] | None,
    quantile_range: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    if births.size == 0 or lifetimes.size == 0:
        if birth_range is None or life_range is None:
            raise ValueError("No finite PH pairs available to estimate histogram ranges.")
        b_min, b_max = (float(birth_range[0]), float(birth_range[1]))
        l_min, l_max = (float(life_range[0]), float(life_range[1]))
        if not np.isfinite([b_min, b_max, l_min, l_max]).all():
            raise ValueError("Histogram ranges must be finite.")
        if b_min == b_max or l_min == l_max:
            raise ValueError("Histogram ranges must have non-zero span.")
        if b_min > b_max or l_min > l_max:
            raise ValueError("Histogram ranges must satisfy min <= max.")
        return (b_min, b_max), (l_min, l_max)

    q_low, q_high = quantile_range
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError("`quantile_range` must satisfy 0 <= low < high <= 1.")

    if birth_range is None:
        b_min, b_max = (float(v) for v in np.quantile(births, [q_low, q_high]))
    else:
        b_min, b_max = (float(birth_range[0]), float(birth_range[1]))

    if life_range is None:
        l_min, l_max = (float(v) for v in np.quantile(lifetimes, [q_low, q_high]))
    else:
        l_min, l_max = (float(life_range[0]), float(life_range[1]))

    if not np.isfinite([b_min, b_max, l_min, l_max]).all():
        raise ValueError("Histogram ranges must be finite.")
    if b_min == b_max or l_min == l_max:
        raise ValueError("Histogram ranges must have non-zero span.")
    if b_min > b_max or l_min > l_max:
        raise ValueError("Histogram ranges must satisfy min <= max.")

    return (b_min, b_max), (l_min, l_max)


def _interior_edges(value_range: tuple[float, float], n_bins: int) -> np.ndarray:
    # Use interior split points so values in [min, max] are spread across bins.
    if n_bins <= 1:
        return np.empty((0,), dtype=np.float64)
    lo, hi = value_range
    return np.linspace(lo, hi, n_bins + 1, dtype=np.float64)[1:-1]


def create_PH_histogram_volume(
    ph: np.ndarray | Sequence[Sequence[float]],
    *,
    image_shape: Sequence[int] | None = None,
    reference_volume: np.ndarray | Sequence[float] | None = None,
    homology_dims: Sequence[int] | None = None,
    n_life_bins: int = 4,
    n_birth_bins: int = 4,
    birth_range: tuple[float, float] | None = None,
    life_range: tuple[float, float] | None = None,
    quantile_range: tuple[float, float] = (0.1, 0.9),
    location: LocationMode = "birth",
    drop_nonfinite: bool = True,
    dtype: np.dtype[Any] | str = np.float32,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Create a channels-first PH histogram volume from CubicalRipser output.

    Parameters
    - ph:
      Array of shape `(n_pairs, >=3)` following CubicalRipser format:
      `[dim, birth, death, x1, y1, z1, ..., x2, y2, z2, ...]`.
    - image_shape / reference_volume:
      Spatial shape for the output histogram. Provide one of them.
    - homology_dims:
      Homology dimensions to include. Default: `range(len(image_shape))`.
    - n_life_bins / n_birth_bins:
      Histogram bins for lifetime `(death-birth)` and birth value.
    - birth_range / life_range:
      Explicit `(min, max)` ranges. If omitted, estimated from finite pairs using
      `quantile_range`.
    - location:
      `"birth"` uses creator coordinates, `"death"` uses destroyer coordinates.
    - drop_nonfinite:
      If `True`, remove rows with non-finite birth/death before counting.
    - dtype:
      Output dtype.
    - return_metadata:
      If `True`, also return bin edges, ranges, and counting stats.

    Returns
    - volume:
      NumPy array of shape `(channels, *image_shape)` where
      `channels = len(homology_dims) * n_life_bins * n_birth_bins`.
    """
    ph_arr = np.asarray(ph, dtype=np.float64)
    if ph_arr.ndim != 2 or ph_arr.shape[1] < 3:
        raise ValueError("`ph` must be a 2D array with at least 3 columns.")
    if n_life_bins < 1 or n_birth_bins < 1:
        raise ValueError("`n_life_bins` and `n_birth_bins` must be >= 1.")
    if location not in ("birth", "death"):
        raise ValueError("`location` must be either 'birth' or 'death'.")

    spatial_shape = _infer_image_shape(image_shape, reference_volume)
    spatial_ndim = len(spatial_shape)

    total_coord_cols = ph_arr.shape[1] - 3
    if total_coord_cols < 2:
        raise ValueError(
            "CubicalRipser output with coordinates is required (expected >= 5 columns)."
        )
    spatial_from_ph = total_coord_cols // 2
    if spatial_from_ph < spatial_ndim:
        raise ValueError(
            f"`ph` appears to contain {spatial_from_ph}D coordinates, "
            f"but requested image_shape has {spatial_ndim} dims."
        )

    birth_coord_slice = slice(3, 3 + spatial_from_ph)
    death_coord_slice = slice(3 + spatial_from_ph, 3 + 2 * spatial_from_ph)
    coord_slice = birth_coord_slice if location == "birth" else death_coord_slice

    births = ph_arr[:, 1]
    deaths = ph_arr[:, 2]
    lifetimes = deaths - births

    finite_mask = np.isfinite(births) & np.isfinite(deaths) & np.isfinite(lifetimes)
    if drop_nonfinite:
        work = ph_arr[finite_mask]
        births_work = births[finite_mask]
        lifetimes_work = lifetimes[finite_mask]
    else:
        work = ph_arr
        births_work = births
        lifetimes_work = lifetimes

    if homology_dims is None:
        homology_dims_tuple = tuple(range(spatial_ndim))
    else:
        homology_dims_tuple = tuple(int(d) for d in homology_dims)
        if len(homology_dims_tuple) == 0:
            raise ValueError("`homology_dims` must not be empty.")
        if len(set(homology_dims_tuple)) != len(homology_dims_tuple):
            raise ValueError("`homology_dims` must not contain duplicates.")

    dim_to_index = {dim: i for i, dim in enumerate(homology_dims_tuple)}

    (b_min, b_max), (l_min, l_max) = _resolve_ranges(
        births_work[np.isfinite(births_work)],
        lifetimes_work[np.isfinite(lifetimes_work)],
        birth_range=birth_range,
        life_range=life_range,
        quantile_range=quantile_range,
    )

    birth_edges = _interior_edges((b_min, b_max), n_birth_bins)
    life_edges = _interior_edges((l_min, l_max), n_life_bins)

    dims = work[:, 0].astype(np.int64, copy=False)
    in_dim = np.array([d in dim_to_index for d in dims], dtype=bool)

    coords = work[:, coord_slice][:, :spatial_ndim].astype(np.int64, copy=False)
    in_bounds = np.ones((coords.shape[0],), dtype=bool)
    for axis, axis_size in enumerate(spatial_shape):
        axis_coords = coords[:, axis]
        in_bounds &= (axis_coords >= 0) & (axis_coords < axis_size)

    valid = in_dim & in_bounds

    hist = np.zeros(
        (*spatial_shape, len(homology_dims_tuple), n_life_bins, n_birth_bins),
        dtype=np.int64,
    )

    if valid.any():
        dims_v = dims[valid]
        births_v = work[valid, 1]
        lifetimes_v = work[valid, 2] - work[valid, 1]
        coords_v = coords[valid]

        dim_idx = np.array([dim_to_index[int(d)] for d in dims_v], dtype=np.int64)
        life_idx = np.searchsorted(life_edges, lifetimes_v, side="left")
        birth_idx = np.searchsorted(birth_edges, births_v, side="left")

        # searchsorted already returns [0, n_bins-1] for finite values against interior edges,
        # but clip for safety.
        life_idx = np.clip(life_idx, 0, n_life_bins - 1)
        birth_idx = np.clip(birth_idx, 0, n_birth_bins - 1)

        indices = tuple(coords_v[:, k] for k in range(spatial_ndim)) + (
            dim_idx,
            life_idx,
            birth_idx,
        )
        np.add.at(hist, indices, 1)

    volume = np.moveaxis(hist.reshape(*spatial_shape, -1), -1, 0).astype(dtype, copy=False)

    if not return_metadata:
        return volume

    metadata = {
        "image_shape": spatial_shape,
        "homology_dims": homology_dims_tuple,
        "n_birth_bins": int(n_birth_bins),
        "n_life_bins": int(n_life_bins),
        "birth_range": (b_min, b_max),
        "life_range": (l_min, l_max),
        "birth_edges": birth_edges,
        "life_edges": life_edges,
        "location": location,
        "n_input_pairs": int(ph_arr.shape[0]),
        "n_finite_pairs": int(work.shape[0]),
        "n_counted_pairs": int(valid.sum()),
    }
    return volume, metadata


__all__ = ["create_PH_histogram_volume"]
