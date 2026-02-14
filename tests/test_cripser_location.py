import numpy as np
import pytest

import cripser

tcripser = pytest.importorskip("tcripser")


def _finite_rows(info: np.ndarray, value_bound: float = 1e5) -> np.ndarray:
    """Drop essential classes encoded with huge sentinel values."""
    mask = (np.abs(info[:, 1]) <= value_bound) & (np.abs(info[:, 2]) <= value_bound)
    return info[mask]


@pytest.mark.parametrize(
    "module_name, compute_fn",
    [("cripser", cripser.computePH), ("tcripser", tcripser.computePH)],
)
@pytest.mark.parametrize("embedded", [False, True])
def test_birth_death_locations_are_value_consistent(module_name, compute_fn, embedded):
    n = 14
    rng = np.random.default_rng(0)
    arr = rng.random((n, n, n), dtype=np.float64)

    info = compute_fn(arr, embedded=embedded, maxdim=2)
    assert info.ndim == 2
    assert info.shape[1] == 9

    finite = _finite_rows(info)
    assert finite.shape[0] > 0, f"{module_name} returned no finite classes to validate"

    birth_coords = finite[:, 3:6].astype(np.int64)
    death_coords = finite[:, 6:9].astype(np.int64)

    for coords in (birth_coords, death_coords):
        assert np.all(coords >= 0)
        assert np.all(coords < n)

    persistence = finite[:, 2] - finite[:, 1]
    inferred = arr[tuple(death_coords.T)] - arr[tuple(birth_coords.T)]
    if embedded:
        inferred *= -1.0

    np.testing.assert_allclose(persistence, inferred, rtol=0.0, atol=1e-12)
