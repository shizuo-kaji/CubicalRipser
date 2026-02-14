"""Image transformation helpers used before persistent homology computation."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

TransformName = str

SUPPORTED_TRANSFORMS = (
    "binarisation",
    "distance",
    "signed_distance",
    "distance_inv",
    "signed_distance_inv",
    "radial",
    "radial_inv",
    "geodesic",
    "geodesic_inv",
    "upward",
    "downward",
)


def _require_distance_transform_edt():
    try:
        from scipy.ndimage import distance_transform_edt  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("scipy is required for distance-based transforms.") from exc
    return distance_transform_edt


def _require_threshold_otsu():
    try:
        from skimage.filters import threshold_otsu  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("scikit-image is required for automatic Otsu thresholding.") from exc
    return threshold_otsu


def _require_rescale():
    try:
        from skimage.transform import rescale  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("scikit-image is required for rescaling.") from exc
    return rescale


def _require_skfmm():
    try:
        import skfmm  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("scikit-fmm is required for geodesic transforms.") from exc
    return skfmm


def _normalise_origin(origin: Sequence[int] | None, ndim: int) -> np.ndarray:
    if origin is None:
        return np.zeros(ndim, dtype=np.int64)
    if len(origin) != ndim:
        raise ValueError(f"origin must have length {ndim}, got {len(origin)}")
    return np.asarray(origin, dtype=np.int64)


def binarize(
    arr: np.ndarray | Sequence[float],
    *,
    threshold: float | None = None,
    threshold_upper_limit: float | None = None,
) -> np.ndarray:
    """Binarize an image with optional lower/upper thresholds."""
    data = np.asarray(arr)
    if threshold is not None:
        if threshold_upper_limit is not None:
            return np.logical_and(data >= threshold, data <= threshold_upper_limit)
        return data >= threshold
    if threshold_upper_limit is not None:
        return data <= threshold_upper_limit

    otsu = _require_threshold_otsu()
    return data >= otsu(data)


def _height_axis(shape: Sequence[int]) -> np.ndarray:
    h = np.arange(shape[0])
    return h.reshape((shape[0],) + (1,) * (len(shape) - 1))


def apply_transform(
    arr: np.ndarray | Sequence[float],
    transform: TransformName | None,
    *,
    threshold: float | None = None,
    threshold_upper_limit: float | None = None,
    origin: Sequence[int] | None = None,
    origin_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply one transformation used in the legacy demo pipeline."""
    data = np.asarray(arr)
    if transform is None:
        return data
    if transform not in SUPPORTED_TRANSFORMS:
        raise ValueError(f"Unsupported transform '{transform}'.")

    bw = binarize(
        data,
        threshold=threshold,
        threshold_upper_limit=threshold_upper_limit,
    )
    if transform == "binarisation":
        return bw

    if "distance" in transform:
        distance_transform_edt = _require_distance_transform_edt()
        work = ~bw if transform.endswith("_inv") else bw
        out = distance_transform_edt(work)
        if transform.startswith("signed_"):
            out = out - distance_transform_edt(~work)
        return out

    if transform in {"downward", "upward"}:
        null_idx = bw == 0
        h = _height_axis(bw.shape)
        if transform == "upward":
            h = -h
        out = (bw * h).astype(np.int32)
        out[null_idx] = np.max(out)
        return out

    if "radial" in transform:
        null_idx = bw == 0
        origin_arr = _normalise_origin(origin, bw.ndim)
        coords = np.stack(
            np.meshgrid(*[np.arange(s) for s in bw.shape], indexing="ij"),
            axis=-1,
        )
        radial = np.linalg.norm(coords - origin_arr, axis=-1)
        out = bw * radial
        if transform.endswith("_inv"):
            out *= -1
        else:
            out[null_idx] = np.max(radial)
        return out

    # geodesic / geodesic_inv
    skfmm = _require_skfmm()
    roi = np.ones(bw.shape)
    if origin_mask is not None:
        mask_arr = np.asarray(origin_mask)
        if mask_arr.shape != bw.shape:
            raise ValueError("origin_mask must have the same shape as input.")
        roi[mask_arr > 0] = 0
    else:
        origin_arr = _normalise_origin(origin, bw.ndim)
        roi[tuple(origin_arr.tolist())] = 0

    dist = skfmm.distance(np.ma.MaskedArray(roi, ~bw))
    if transform.endswith("_inv"):
        dist *= -1
    return dist.filled(fill_value=dist.max())


def preprocess_image(
    arr: np.ndarray | Sequence[float],
    *,
    transpose: Sequence[int] | None = None,
    zrange: tuple[int, int] | None = None,
    scaling_factor: float = 1.0,
    tile: int = 1,
    transform: TransformName | None = None,
    threshold: float | None = None,
    threshold_upper_limit: float | None = None,
    origin: Sequence[int] | None = None,
    origin_mask: np.ndarray | None = None,
    shift_value: float = 0.0,
    dtype: str | np.dtype[Any] | None = None,
) -> np.ndarray:
    """Apply the legacy `img2npy.py` preprocessing sequence."""
    out = np.asarray(arr)

    if transpose is not None:
        out = out.transpose(tuple(transpose))

    if zrange is not None:
        out = out[slice(*zrange)]

    if scaling_factor != 1.0:
        rescale = _require_rescale()
        if np.issubdtype(out.dtype, np.integer) and np.ptp(out) < 2:
            out = out.astype(np.bool_)
        out = rescale(out, scaling_factor, mode="reflect", preserve_range=True)

    if tile > 1:
        out = np.tile(out, [tile] * out.ndim)

    if transform is not None:
        out = apply_transform(
            out,
            transform,
            threshold=threshold,
            threshold_upper_limit=threshold_upper_limit,
            origin=origin,
            origin_mask=origin_mask,
        )

    if shift_value:
        out = out + shift_value

    if dtype is not None:
        out = out.astype(dtype)

    return out


__all__ = [
    "SUPPORTED_TRANSFORMS",
    "binarize",
    "apply_transform",
    "preprocess_image",
]
