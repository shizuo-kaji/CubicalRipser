"""Differentiable Wasserstein distance utilities based on POT."""

from __future__ import annotations

from typing import Sequence

import ot
import torch


TensorLike2D = torch.Tensor | Sequence[float] | Sequence[Sequence[float]]


def _as_2d_tensor(data: TensorLike2D, *, name: str) -> torch.Tensor:
    points = data if isinstance(data, torch.Tensor) else torch.as_tensor(data)
    if points.ndim == 1:
        points = points.unsqueeze(-1)
    if points.ndim != 2:
        raise ValueError(f"`{name}` must have shape (n, d) or (n,), got {tuple(points.shape)}.")
    if not points.is_floating_point():
        points = points.to(dtype=torch.float64)
    return points


def _extract_ot_points(
    data: TensorLike2D,
    *,
    name: str,
    dim: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    points = _as_2d_tensor(data, name=name)
    n = points.shape[0]

    # CubicalRipser output format: [dim, birth, death, ...]
    if points.shape[1] in {9, 11}:
        keep = torch.ones(n, dtype=torch.bool, device=points.device)
        if dim is not None:
            keep = keep & (points[:, 0].to(torch.int64) == int(dim))
        pairs = points[:, 1:3]
        finite = torch.isfinite(pairs).all(dim=1)
        keep = keep & finite
        return pairs[keep], keep

    # GUDHI-style diagram: (birth, death)
    if points.shape[1] == 2:
        keep = torch.isfinite(points).all(dim=1)
        return points[keep], keep

    if dim is not None:
        raise ValueError(
            "`dim` is only supported when inputs are CubicalRipser outputs "
            "with shape (n, 9)/(n, 11)."
        )
    if torch.any(~torch.isfinite(points)).item():
        raise ValueError(f"`{name}` contains non-finite coordinates.")
    keep = torch.ones(n, dtype=torch.bool, device=points.device)
    return points, keep


def _distance_to_diagonal(points: torch.Tensor, *, q: float) -> torch.Tensor:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Diagonal projection requires point arrays of shape (n, 2).")
    if points.shape[0] == 0:
        return torch.empty((0,), dtype=points.dtype, device=points.device)
    mid = 0.5 * (points[:, 0] + points[:, 1])
    proj = torch.stack((mid, mid), dim=1)
    return torch.cdist(points, proj, p=float(q)).diagonal()


def wasserstein_distance(
    x: TensorLike2D,
    y: TensorLike2D,
    *,
    dim: int | None = None,
    p: float = 2.0,
    q: float = 2.0,
    num_iter_max: int = 10000,
    return_pth_power: bool = False,
) -> torch.Tensor:
    """Compute a differentiable p-Wasserstein distance with entropic OT.

    Notes
    - If `x`/`y` are shape `(n, 9)`, they are interpreted as
      CubicalRipser rows `[dim, birth, death, ...]` and converted internally
      to GUDHI-style `(birth, death)` pairs.
    - If `dim` is set for CubicalRipser inputs, only that homology dimension
      is used.
    - Non-finite birth/death pairs are dropped.
    - Uses POT's `ot.emd` with torch tensors for autograd compatibility.
    - Ground metric is Minkowski-`q`, and cost is `||x_i - y_j||_q^p`.
    - Returns `W_p` by default (i.e. `(OT_cost)^(1/p)`).
    """

    x_points, x_keep = _extract_ot_points(x, name="x", dim=dim)
    y_points, y_keep = _extract_ot_points(y, name="y", dim=dim)

    if x_points.shape[1] != y_points.shape[1]:
        raise ValueError(
            f"`x` and `y` must have the same point dimension, got "
            f"{x_points.shape[1]} and {y_points.shape[1]}."
        )
    if x_points.device != y_points.device:
        raise ValueError("`x` and `y` must be on the same device.")

    if p <= 0:
        raise ValueError("`p` must be positive.")
    if q <= 0:
        raise ValueError("`q` must be positive.")
    if num_iter_max <= 0:
        raise ValueError("`num_iter_max` must be a positive integer.")

    common_dtype = torch.promote_types(x_points.dtype, y_points.dtype)
    x_points = x_points.to(dtype=common_dtype)
    y_points = y_points.to(dtype=common_dtype)

    n = x_points.shape[0]
    m = y_points.shape[0]
    cost_matrix = torch.zeros(
        (n + 1, m + 1),
        dtype=common_dtype,
        device=x_points.device,
    )
    if n > 0 and m > 0:
        cost_matrix[:n, :m] = torch.cdist(x_points, y_points, p=float(q))
    if n > 0:
        cost_matrix[:n, m] = _distance_to_diagonal(x_points, q=q)
    if m > 0:
        cost_matrix[n, :m] = _distance_to_diagonal(y_points, q=q)
    if p != 1.0:
        cost_matrix = cost_matrix.pow(float(p))

    a = torch.ones(n+1)
    a[-1] = m
    b = torch.ones(m+1)
    b[-1] = n

    distance_p = ot.emd2(
        a,
        b,
        cost_matrix,
        numItermax=int(num_iter_max),
    )
    if not isinstance(distance_p, torch.Tensor):
        raise RuntimeError(
            "POT did not return a torch.Tensor. Ensure POT torch backend is available."
        )

    distance_p = torch.clamp_min(distance_p, 0.0)
    if return_pth_power or p == 1.0:
        return distance_p
    return distance_p.pow(1.0 / float(p))


__all__ = ["wasserstein_distance"]
