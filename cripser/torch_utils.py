"""PyTorch integration for differentiable CubicalRipser computations.

The backward pass treats creator/destroyer assignments as fixed and routes
gradients from birth/death values to their corresponding voxel locations.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch

from .utils import compute_ph

_INF_CUTOFF = np.finfo(np.float64).max / 2.0


def _coord_columns(ndim: int, num_columns: int) -> Tuple[list[int], list[int]]:
    if ndim < 1 or ndim > 4:
        raise ValueError(f"Expected ndim in [1, 4], got {ndim}")
    if num_columns < 9:
        raise ValueError(f"Expected at least 9 columns in PH output, got {num_columns}")

    birth_cols = list(range(3, 3 + ndim))
    if ndim == 4 and num_columns >= 11:
        death_start = 7
    else:
        death_start = 6
    death_cols = list(range(death_start, death_start + ndim))
    return birth_cols, death_cols


def _in_bounds_mask(coords: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    shape_arr = np.asarray(shape, dtype=np.int64)
    return np.all((coords >= 0) & (coords < shape_arr), axis=1)


def _coords_to_linear(coords: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    shape_arr = np.asarray(shape, dtype=np.int64)
    strides = np.ones(len(shape_arr), dtype=np.int64)
    for i in range(len(shape_arr) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape_arr[i + 1]
    return np.sum(coords * strides, axis=1, dtype=np.int64)


def _prepare_indices(ph: np.ndarray, shape: Sequence[int]) -> Tuple[np.ndarray, ...]:
    n_pairs = ph.shape[0]
    ndim = len(shape)
    birth_lin = np.zeros(n_pairs, dtype=np.int64)
    death_lin = np.zeros(n_pairs, dtype=np.int64)
    birth_mask = np.zeros(n_pairs, dtype=bool)
    death_mask = np.zeros(n_pairs, dtype=bool)

    if n_pairs == 0:
        return birth_lin, death_lin, birth_mask, death_mask

    birth_cols, death_cols = _coord_columns(ndim, ph.shape[1])
    birth_coords = np.rint(ph[:, birth_cols]).astype(np.int64, copy=False)
    death_coords = np.rint(ph[:, death_cols]).astype(np.int64, copy=False)

    birth_finite = np.isfinite(ph[:, 1]) & (np.abs(ph[:, 1]) < _INF_CUTOFF)
    death_finite = np.isfinite(ph[:, 2]) & (np.abs(ph[:, 2]) < _INF_CUTOFF)

    birth_mask = birth_finite & _in_bounds_mask(birth_coords, shape)
    death_mask = death_finite & _in_bounds_mask(death_coords, shape)

    if np.any(birth_mask):
        birth_lin[birth_mask] = _coords_to_linear(birth_coords[birth_mask], shape)
    if np.any(death_mask):
        death_lin[death_mask] = _coords_to_linear(death_coords[death_mask], shape)

    return birth_lin, death_lin, birth_mask, death_mask


class _ComputePHTorch(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        arr: torch.Tensor,
        filtration: str,
        maxdim: int,
        top_dim: bool,
        embedded: bool,
        location: str,
    ) -> torch.Tensor:
        if arr.ndim < 1 or arr.ndim > 4:
            raise ValueError(f"Expected 1D-4D tensor, got shape {tuple(arr.shape)}")
        if not arr.is_floating_point():
            raise TypeError(f"Expected floating point tensor, got dtype {arr.dtype}")

        arr_np = arr.detach().to(device="cpu", dtype=torch.float64).numpy()
        ph_np = compute_ph(
            arr_np,
            filtration=filtration,
            maxdim=maxdim,
            top_dim=top_dim,
            embedded=embedded,
            location=location,
        )
        if ph_np.size:
            inf_mask = ph_np[:, 2] >= _INF_CUTOFF
            if np.any(inf_mask):
                ph_np = ph_np.copy()
                ph_np[inf_mask, 2] = np.inf

        if ctx.needs_input_grad[0]:
            birth_lin_np, death_lin_np, birth_mask_np, death_mask_np = _prepare_indices(
                ph_np, tuple(arr.shape)
            )
            birth_lin = torch.from_numpy(birth_lin_np).to(device=arr.device)
            death_lin = torch.from_numpy(death_lin_np).to(device=arr.device)
            birth_mask = torch.from_numpy(birth_mask_np).to(device=arr.device)
            death_mask = torch.from_numpy(death_mask_np).to(device=arr.device)
            ctx.save_for_backward(birth_lin, death_lin, birth_mask, death_mask)

        ctx.input_shape = tuple(arr.shape)
        ctx.input_dtype = arr.dtype
        return torch.from_numpy(ph_np).to(device=arr.device)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor | None, None, None, None, None, None]:
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None, None

        birth_lin, death_lin, birth_mask, death_mask = ctx.saved_tensors
        total_size = int(np.prod(ctx.input_shape))
        grad_input_flat = torch.zeros(
            total_size,
            dtype=ctx.input_dtype,
            device=grad_output.device,
        )

        if grad_output.numel() > 0:
            grad_birth = grad_output[:, 1].to(dtype=ctx.input_dtype)
            grad_death = grad_output[:, 2].to(dtype=ctx.input_dtype)
            if bool(torch.any(birth_mask)):
                grad_input_flat.index_add_(0, birth_lin[birth_mask], grad_birth[birth_mask])
            if bool(torch.any(death_mask)):
                grad_input_flat.index_add_(0, death_lin[death_mask], grad_death[death_mask])

        grad_input = grad_input_flat.reshape(ctx.input_shape)
        return grad_input, None, None, None, None, None


def compute_ph_torch(
    arr: torch.Tensor,
    *,
    filtration: str = "V",
    maxdim: int = 3,
    top_dim: bool = False,
    embedded: bool = False,
    location: str = "yes",
) -> torch.Tensor:
    """Compute PH using CubicalRipser and expose a differentiable torch output.

    Notes
    - Forward uses the CPU NumPy backend internally.
    - Backward propagates gradients only through birth/death columns.
    - Pairing changes are discrete; gradients are piecewise defined.
    """
    filt = filtration.upper()
    if filt not in {"V", "T"}:
        raise ValueError(f"filtration must be 'V' or 'T', got {filtration!r}")
    return _ComputePHTorch.apply(arr, filt, int(maxdim), bool(top_dim), bool(embedded), location)


def finite_lifetimes(ph: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Return differentiable finite lifetimes from a PH tensor."""
    if ph.ndim != 2 or ph.shape[1] < 3:
        raise ValueError("Expected PH tensor of shape (n, >=3)")

    mask = torch.isfinite(ph[:, 2])
    if dim is not None:
        mask = mask & (ph[:, 0].to(torch.int64) == int(dim))
    return ph[mask, 2] - ph[mask, 1]
