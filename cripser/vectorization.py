"""Persistent-homology vectorization utilities."""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

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


def _validate_resolved_ranges(
    b_min: float,
    b_max: float,
    l_min: float,
    l_max: float,
    *,
    name: str,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if not np.isfinite([b_min, b_max, l_min, l_max]).all():
        raise ValueError(f"{name} ranges must be finite.")
    if b_min == b_max or l_min == l_max:
        raise ValueError(f"{name} ranges must have non-zero span.")
    if b_min > b_max or l_min > l_max:
        raise ValueError(f"{name} ranges must satisfy min <= max.")
    return (b_min, b_max), (l_min, l_max)


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
        return _validate_resolved_ranges(
            b_min,
            b_max,
            l_min,
            l_max,
            name="Histogram",
        )

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

    return _validate_resolved_ranges(
        b_min,
        b_max,
        l_min,
        l_max,
        name="Histogram",
    )


def _resolve_ranges_torch(
    births: "torch.Tensor",
    lifetimes: "torch.Tensor",
    *,
    birth_range: tuple[float, float] | None,
    life_range: tuple[float, float] | None,
    quantile_range: tuple[float, float],
    range_name: str,
) -> tuple[tuple[float, float], tuple[float, float]]:
    births_det = births.detach().to(dtype=torch.float64)
    lifetimes_det = lifetimes.detach().to(dtype=torch.float64)

    if births_det.numel() == 0 or lifetimes_det.numel() == 0:
        if birth_range is None or life_range is None:
            raise ValueError(f"No finite PH pairs available to estimate {range_name.lower()} ranges.")
        b_min, b_max = (float(birth_range[0]), float(birth_range[1]))
        l_min, l_max = (float(life_range[0]), float(life_range[1]))
        return _validate_resolved_ranges(b_min, b_max, l_min, l_max, name=range_name)

    q_low, q_high = quantile_range
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError("`quantile_range` must satisfy 0 <= low < high <= 1.")

    if birth_range is None:
        b_q = torch.quantile(births_det, torch.tensor([q_low, q_high], device=births_det.device))
        b_min, b_max = (float(b_q[0]), float(b_q[1]))
    else:
        b_min, b_max = (float(birth_range[0]), float(birth_range[1]))

    if life_range is None:
        l_q = torch.quantile(
            lifetimes_det, torch.tensor([q_low, q_high], device=lifetimes_det.device)
        )
        l_min, l_max = (float(l_q[0]), float(l_q[1]))
    else:
        l_min, l_max = (float(life_range[0]), float(life_range[1]))

    return _validate_resolved_ranges(b_min, b_max, l_min, l_max, name=range_name)


def _interior_edges(value_range: tuple[float, float], n_bins: int) -> np.ndarray:
    # Use interior split points so values in [min, max] are spread across bins.
    if n_bins <= 1:
        return np.empty((0,), dtype=np.float64)
    lo, hi = value_range
    return np.linspace(lo, hi, n_bins + 1, dtype=np.float64)[1:-1]


def _validate_homology_dims(
    homology_dims: Sequence[int] | None,
    *,
    default_dims: Sequence[int],
) -> tuple[int, ...]:
    if homology_dims is None:
        dims = tuple(int(d) for d in default_dims)
        if len(dims) == 0:
            return (0,)
        return dims

    dims = tuple(int(d) for d in homology_dims)
    if len(dims) == 0:
        raise ValueError("`homology_dims` must not be empty.")
    if len(set(dims)) != len(dims):
        raise ValueError("`homology_dims` must not contain duplicates.")
    return dims


def _resolve_torch_float_dtype(
    dtype: Any | None,
    *,
    fallback: "torch.dtype",
) -> "torch.dtype":
    if torch is None:
        raise RuntimeError("PyTorch is required to resolve torch dtypes.")
    if dtype is None:
        resolved = fallback
    elif isinstance(dtype, torch.dtype):
        resolved = dtype
    else:
        np_dtype = np.dtype(dtype)
        mapping = {
            np.dtype(np.float16): torch.float16,
            np.dtype(np.float32): torch.float32,
            np.dtype(np.float64): torch.float64,
        }
        if np_dtype not in mapping:
            raise TypeError("`dtype` must be a floating torch or numpy dtype.")
        resolved = mapping[np_dtype]

    if resolved not in {torch.float16, torch.float32, torch.float64, torch.bfloat16}:
        raise TypeError("`dtype` must be a floating torch dtype.")
    return resolved


def _default_dims_from_numpy(dims_raw: np.ndarray) -> tuple[int, ...]:
    finite = np.isfinite(dims_raw)
    if not np.any(finite):
        return (0,)
    return tuple(int(v) for v in np.unique(dims_raw[finite].astype(np.int64, copy=False)))


def _default_dims_from_torch(dims_raw: "torch.Tensor") -> tuple[int, ...]:
    finite = torch.isfinite(dims_raw)
    if not bool(torch.any(finite)):
        return (0,)
    return tuple(
        int(v)
        for v in torch.unique(dims_raw[finite].to(dtype=torch.int64)).detach().cpu().tolist()
    )


def _persistence_image_numpy(
    ph: np.ndarray | Sequence[Sequence[float]],
    *,
    homology_dims: Sequence[int] | None,
    n_birth_bins: int,
    n_life_bins: int,
    birth_range: tuple[float, float] | None,
    life_range: tuple[float, float] | None,
    quantile_range: tuple[float, float],
    sigma: float,
    weight_power: float,
    drop_nonfinite: bool,
    normalize: bool,
    dtype: Any | None,
    return_metadata: bool,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    ph_arr = np.asarray(ph, dtype=np.float64)
    if ph_arr.ndim != 2 or ph_arr.shape[1] < 3:
        raise ValueError("`ph` must be a 2D array with at least 3 columns.")

    dims_raw = ph_arr[:, 0]
    births = ph_arr[:, 1]
    deaths = ph_arr[:, 2]
    lifetimes = deaths - births

    finite_mask = (
        np.isfinite(dims_raw) & np.isfinite(births) & np.isfinite(deaths) & np.isfinite(lifetimes)
    )
    if drop_nonfinite:
        work = ph_arr[finite_mask]
    else:
        work = ph_arr

    dims_raw_work = work[:, 0]
    births_work = work[:, 1]
    lifetimes_work = work[:, 2] - work[:, 1]
    dim_finite_work = np.isfinite(dims_raw_work)
    dims_work = np.zeros(work.shape[0], dtype=np.int64)
    if np.any(dim_finite_work):
        dims_work[dim_finite_work] = dims_raw_work[dim_finite_work].astype(np.int64, copy=False)

    homology_dims_tuple = _validate_homology_dims(
        homology_dims,
        default_dims=_default_dims_from_numpy(dims_raw_work),
    )
    dim_to_index = {dim: i for i, dim in enumerate(homology_dims_tuple)}

    (b_min, b_max), (l_min, l_max) = _resolve_ranges(
        births_work[np.isfinite(births_work)],
        lifetimes_work[np.isfinite(lifetimes_work)],
        birth_range=birth_range,
        life_range=life_range,
        quantile_range=quantile_range,
    )

    birth_centers = np.linspace(b_min, b_max, n_birth_bins, dtype=np.float64)
    life_centers = np.linspace(l_min, l_max, n_life_bins, dtype=np.float64)
    birth_grid = birth_centers[None, None, :]
    life_grid = life_centers[None, :, None]

    valid = (
        dim_finite_work
        & np.array([int(d) in dim_to_index for d in dims_work], dtype=bool)
        & np.isfinite(births_work)
        & np.isfinite(lifetimes_work)
    )

    image = np.zeros((len(homology_dims_tuple), n_life_bins, n_birth_bins), dtype=np.float64)
    if np.any(valid):
        dims_v = dims_work[valid]
        births_v = births_work[valid]
        lifetimes_v = lifetimes_work[valid]
        sigma_sq = float(sigma) * float(sigma)

        for channel, dim in enumerate(homology_dims_tuple):
            mask = dims_v == int(dim)
            if not np.any(mask):
                continue
            births_c = births_v[mask][:, None, None]
            lifes_c = lifetimes_v[mask][:, None, None]
            sqdist = (births_c - birth_grid) ** 2 + (lifes_c - life_grid) ** 2
            kernels = np.exp(-0.5 * sqdist / sigma_sq)
            weights = np.maximum(lifetimes_v[mask], 0.0) ** float(weight_power)
            image[channel] = np.sum(weights[:, None, None] * kernels, axis=0)

    if normalize:
        sums = image.sum(axis=(1, 2), keepdims=True)
        image = np.divide(image, sums, out=image, where=sums > 0)

    out_dtype = np.dtype(np.float32 if dtype is None else dtype)
    image = image.astype(out_dtype, copy=False)

    if not return_metadata:
        return image

    metadata = {
        "homology_dims": homology_dims_tuple,
        "n_birth_bins": int(n_birth_bins),
        "n_life_bins": int(n_life_bins),
        "birth_range": (b_min, b_max),
        "life_range": (l_min, l_max),
        "birth_centers": birth_centers,
        "life_centers": life_centers,
        "sigma": float(sigma),
        "weight_power": float(weight_power),
        "drop_nonfinite": bool(drop_nonfinite),
        "normalize": bool(normalize),
        "n_input_pairs": int(ph_arr.shape[0]),
        "n_finite_pairs": int(finite_mask.sum()),
        "n_counted_pairs": int(valid.sum()),
    }
    return image, metadata


def _persistence_image_torch(
    ph: "torch.Tensor",
    *,
    homology_dims: Sequence[int] | None,
    n_birth_bins: int,
    n_life_bins: int,
    birth_range: tuple[float, float] | None,
    life_range: tuple[float, float] | None,
    quantile_range: tuple[float, float],
    sigma: float,
    weight_power: float,
    drop_nonfinite: bool,
    normalize: bool,
    dtype: Any | None,
    return_metadata: bool,
) -> "torch.Tensor" | tuple["torch.Tensor", dict[str, Any]]:
    if torch is None:
        raise RuntimeError("PyTorch is required for torch tensor inputs.")
    if ph.ndim != 2 or ph.shape[1] < 3:
        raise ValueError("`ph` must be a 2D tensor with at least 3 columns.")
    if not ph.is_floating_point():
        raise TypeError(f"Expected floating-point PH tensor, got dtype {ph.dtype}.")

    dims_raw = ph[:, 0]
    births = ph[:, 1]
    deaths = ph[:, 2]
    lifetimes = deaths - births

    finite_mask = (
        torch.isfinite(dims_raw)
        & torch.isfinite(births)
        & torch.isfinite(deaths)
        & torch.isfinite(lifetimes)
    )
    if drop_nonfinite:
        work = ph[finite_mask]
    else:
        work = ph

    dims_raw_work = work[:, 0]
    births_work = work[:, 1]
    lifetimes_work = work[:, 2] - work[:, 1]
    dim_finite_work = torch.isfinite(dims_raw_work)
    dims_work = torch.zeros(work.shape[0], dtype=torch.int64, device=work.device)
    if bool(torch.any(dim_finite_work)):
        dims_work[dim_finite_work] = dims_raw_work[dim_finite_work].to(dtype=torch.int64)

    homology_dims_tuple = _validate_homology_dims(
        homology_dims,
        default_dims=_default_dims_from_torch(dims_raw_work),
    )

    finite_births = births_work[torch.isfinite(births_work)].detach()
    finite_lifetimes = lifetimes_work[torch.isfinite(lifetimes_work)].detach()
    (b_min, b_max), (l_min, l_max) = _resolve_ranges_torch(
        finite_births,
        finite_lifetimes,
        birth_range=birth_range,
        life_range=life_range,
        quantile_range=quantile_range,
        range_name="Persistence-image",
    )

    out_dtype = _resolve_torch_float_dtype(dtype, fallback=ph.dtype)
    birth_centers = torch.linspace(
        b_min, b_max, n_birth_bins, device=ph.device, dtype=out_dtype
    )
    life_centers = torch.linspace(
        l_min, l_max, n_life_bins, device=ph.device, dtype=out_dtype
    )
    birth_grid = birth_centers.view(1, 1, -1)
    life_grid = life_centers.view(1, -1, 1)

    dim_values = torch.tensor(homology_dims_tuple, device=ph.device, dtype=torch.int64)
    in_dim = torch.any(dims_work[:, None] == dim_values[None, :], dim=1)
    valid = dim_finite_work & in_dim & torch.isfinite(births_work) & torch.isfinite(lifetimes_work)

    image = torch.zeros(
        (len(homology_dims_tuple), n_life_bins, n_birth_bins),
        device=ph.device,
        dtype=out_dtype,
    )
    if bool(torch.any(valid)):
        dims_v = dims_work[valid]
        births_v = births_work[valid].to(dtype=out_dtype)
        lifetimes_v = lifetimes_work[valid].to(dtype=out_dtype)
        sigma_sq = float(sigma) * float(sigma)

        for channel, dim in enumerate(homology_dims_tuple):
            mask = dims_v == int(dim)
            if not bool(torch.any(mask)):
                continue
            births_c = births_v[mask].view(-1, 1, 1)
            lifes_c = lifetimes_v[mask].view(-1, 1, 1)
            sqdist = (births_c - birth_grid).pow(2) + (lifes_c - life_grid).pow(2)
            kernels = torch.exp(-0.5 * sqdist / sigma_sq)
            weights = torch.clamp(lifetimes_v[mask], min=0.0).pow(float(weight_power)).view(
                -1, 1, 1
            )
            image[channel] = (weights * kernels).sum(dim=0)

    if normalize:
        sums = image.sum(dim=(1, 2), keepdim=True)
        image = torch.where(sums > 0, image / sums, image)

    if not return_metadata:
        return image

    metadata = {
        "homology_dims": homology_dims_tuple,
        "n_birth_bins": int(n_birth_bins),
        "n_life_bins": int(n_life_bins),
        "birth_range": (b_min, b_max),
        "life_range": (l_min, l_max),
        "birth_centers": birth_centers,
        "life_centers": life_centers,
        "sigma": float(sigma),
        "weight_power": float(weight_power),
        "drop_nonfinite": bool(drop_nonfinite),
        "normalize": bool(normalize),
        "n_input_pairs": int(ph.shape[0]),
        "n_finite_pairs": int(finite_mask.sum().item()),
        "n_counted_pairs": int(valid.sum().item()),
    }
    return image, metadata


def persistence_image(
    ph: np.ndarray | Sequence[Sequence[float]] | "torch.Tensor",
    *,
    homology_dims: Sequence[int] | None = None,
    n_life_bins: int = 16,
    n_birth_bins: int = 16,
    birth_range: tuple[float, float] | None = None,
    life_range: tuple[float, float] | None = None,
    quantile_range: tuple[float, float] = (0.1, 0.9),
    sigma: float = 1.0,
    weight_power: float = 1.0,
    drop_nonfinite: bool = True,
    normalize: bool = False,
    dtype: Any | None = None,
    return_metadata: bool = False,
) -> (
    np.ndarray
    | "torch.Tensor"
    | tuple[np.ndarray, dict[str, Any]]
    | tuple["torch.Tensor", dict[str, Any]]
):
    """Create a differentiable persistence image from PH pairs.

    Parameters
    - ph:
      PH pairs with shape `(n_pairs, >=3)` and columns `[dim, birth, death, ...]`.
      Accepts NumPy-like input, or a torch tensor when PyTorch is installed.
    - homology_dims:
      Dimensions to include. Defaults to finite dimensions observed in `ph`.
    - n_life_bins / n_birth_bins:
      Output resolution in lifetime and birth directions.
    - birth_range / life_range:
      Explicit `(min, max)` ranges. If omitted, estimated from finite pairs via
      `quantile_range`.
    - sigma:
      Gaussian kernel width.
    - weight_power:
      Weight each pair by `max(lifetime, 0) ** weight_power`.
    - drop_nonfinite:
      If `True`, rows with non-finite dim/birth/death/lifetime are removed.
    - normalize:
      If `True`, each homology-channel image is L1-normalized.
    - dtype:
      Output dtype (`numpy` dtype for NumPy input, floating torch dtype for torch input).
    - return_metadata:
      If `True`, also return image ranges, centers, and counting stats.
    """
    if n_life_bins < 1 or n_birth_bins < 1:
        raise ValueError("`n_life_bins` and `n_birth_bins` must be >= 1.")
    if sigma <= 0:
        raise ValueError("`sigma` must be > 0.")
    if weight_power < 0:
        raise ValueError("`weight_power` must be >= 0.")

    if torch is not None and isinstance(ph, torch.Tensor):
        return _persistence_image_torch(
            ph,
            homology_dims=homology_dims,
            n_birth_bins=n_birth_bins,
            n_life_bins=n_life_bins,
            birth_range=birth_range,
            life_range=life_range,
            quantile_range=quantile_range,
            sigma=sigma,
            weight_power=weight_power,
            drop_nonfinite=drop_nonfinite,
            normalize=normalize,
            dtype=dtype,
            return_metadata=return_metadata,
        )

    return _persistence_image_numpy(
        ph,
        homology_dims=homology_dims,
        n_birth_bins=n_birth_bins,
        n_life_bins=n_life_bins,
        birth_range=birth_range,
        life_range=life_range,
        quantile_range=quantile_range,
        sigma=sigma,
        weight_power=weight_power,
        drop_nonfinite=drop_nonfinite,
        normalize=normalize,
        dtype=dtype,
        return_metadata=return_metadata,
    )


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

    homology_dims_tuple = _validate_homology_dims(
        homology_dims,
        default_dims=tuple(range(spatial_ndim)),
    )

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


__all__ = ["create_PH_histogram_volume", "persistence_image"]
