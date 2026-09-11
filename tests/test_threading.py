"""The threaded phases and the released GIL must not change any result."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import cripser
import tcripser


@pytest.mark.parametrize("module", [cripser, tcripser])
@pytest.mark.parametrize(
    "shape, maxdim",
    [((256, 256), 1), ((3000,), 0), ((24, 24, 24), 2)],
)
def test_thread_count_does_not_change_output(module, shape, maxdim):
    arr = np.random.default_rng(1).random(shape)
    expected = module.computePH(arr, maxdim=maxdim, n_threads=1)
    for n_threads in (2, 4, 8, 0):
        got = module.computePH(arr, maxdim=maxdim, n_threads=n_threads)
        assert np.array_equal(got, expected), f"differs at n_threads={n_threads}"


def test_concurrent_calls_match_sequential():
    """The GIL is released during the computation, so calls really do overlap."""
    rng = np.random.default_rng(2)
    images = [rng.random((48, 48, 48)) for _ in range(4)]
    expected = [cripser.compute_ph(a, maxdim=2) for a in images]
    with ThreadPoolExecutor(4) as pool:
        got = list(pool.map(lambda a: cripser.compute_ph(a, maxdim=2), images))
    for want, have in zip(expected, got):
        assert np.array_equal(want, have)
