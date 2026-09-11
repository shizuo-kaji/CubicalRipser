"""Regression tests for input shapes that used to crash or return wrong values."""

import numpy as np
import pytest

import cripser
import tcripser


# An axis longer than 2^15 overflowed the 15-bit coordinate packing in Cube:
# (40000, 2, 2) segfaulted, and (34000, 2, 2) silently reported a clamped
# coordinate (33000 came back as 32767) instead of failing.
@pytest.mark.parametrize("module", [cripser, tcripser])
@pytest.mark.parametrize(
    "shape, kwargs",
    [
        ((40000, 2, 2), {"maxdim": 2}),
        ((34000, 2, 2), {"maxdim": 2}),
        ((40000, 3), {"maxdim": 1, "top_dim": True}),
        ((40000, 3), {"maxdim": 1, "representatives": True}),
    ],
)
def test_oversized_axis_is_rejected(module, shape, kwargs):
    arr = np.zeros(shape)
    with pytest.raises(ValueError, match="exceeds the maximum"):
        module.computePH(arr, **kwargs)


# The same sizes must keep working on the planar fast path, which uses its own
# wider packing -- a long 1D time series is a documented use case.
@pytest.mark.parametrize("module", [cripser, tcripser])
@pytest.mark.parametrize("shape, maxdim", [((200000,), 0), ((40000, 3), 1)])
def test_large_planar_inputs_still_supported(module, shape, maxdim):
    arr = np.random.default_rng(0).random(shape)
    ph = module.computePH(arr, maxdim=maxdim)
    assert ph.shape[0] > 0


@pytest.mark.parametrize("module", [cripser, tcripser])
@pytest.mark.parametrize("shape", [(0, 5), (5, 0), (0,), (3, 0, 2)])
def test_empty_axis_is_rejected(module, shape):
    with pytest.raises(ValueError, match="length 0"):
        module.computePH(np.zeros(shape))


# A single voxel has no edges, so the essential H0 birth was never initialised
# and came back as DBL_MAX instead of the voxel's own value.
@pytest.mark.parametrize("shape", [(1, 1), (1, 1, 1), (1, 1, 1, 1)])
def test_single_cell_birth_is_the_cell_value(shape):
    ph = cripser.compute_ph(np.full(shape, 3.0))
    assert ph.shape[0] == 1
    assert ph[0, 1] == 3.0
    assert not np.isfinite(ph[0, 2])
