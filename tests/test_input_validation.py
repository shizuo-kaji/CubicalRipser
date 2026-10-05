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


# top_dim computes the top dimension alone.  It used to crash the process (T),
# return an empty table (4D) and, for the V-construction, return pairs that
# differ from the ordinary computation.
@pytest.mark.parametrize("filtration", ["V", "T"])
@pytest.mark.parametrize("embedded", [False, True])
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize(
    "shape", [(9, 10), (2, 6), (5, 6, 7), (4, 3, 5, 4), (7, 1), (6, 1, 7)]
)
def test_top_dim_matches_ordinary_computation(filtration, embedded, ties, shape):
    rng = np.random.default_rng(2)
    arr = rng.integers(0, 3, shape).astype(np.float64) if ties else rng.random(shape)
    d = arr.ndim
    top = cripser.compute_ph(arr, filtration=filtration, embedded=embedded, top_dim=True)
    full = cripser.compute_ph(arr, filtration=filtration, embedded=embedded, maxdim=d - 1)
    full = full[full[:, 0] == d - 1]
    assert np.all(top[:, 0] == d - 1)
    assert sorted(map(tuple, top[:, 1:3])) == sorted(map(tuple, full[:, 1:3]))

    # Creator and destroyer values; a creator outside the input is -1.
    k = 4 if d == 4 else 3
    sign = -1.0 if embedded else 1.0
    creator = top[:, 3 : 3 + d].astype(np.int64)
    destroyer = top[:, 3 + k : 3 + k + d].astype(np.int64)
    inside = np.all(creator >= 0, axis=1)
    assert np.all(creator[~inside] == -1)
    np.testing.assert_array_equal(sign * arr[tuple(creator[inside].T)], top[inside, 1])
    np.testing.assert_array_equal(sign * arr[tuple(destroyer.T)], top[:, 2])


def test_top_dim_of_1d_input_is_the_ordinary_computation():
    arr = np.random.default_rng(1).random(50)
    np.testing.assert_array_equal(
        cripser.compute_ph(arr, top_dim=True), cripser.compute_ph(arr)
    )
