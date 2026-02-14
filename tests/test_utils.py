import numpy as np

from cripser import (
    compute_ph,
    dual_embedding,
    to_gudhi_diagrams,
    to_gudhi_persistence,
    group_by_dim,
)


def test_utils_converters_basic():
    # Simple 2D constant image ensures an infinite H0 bar
    arr = np.zeros((3, 3), dtype=np.float64)
    ph = compute_ph(arr, maxdim=1)

    # Base shape checks
    assert ph.ndim == 2 and ph.shape[1] == 9

    # Diagrams
    dgms = to_gudhi_diagrams(ph, maxdim=1)
    assert isinstance(dgms, list) and len(dgms) == 2
    assert dgms[0].ndim == 2 and dgms[0].shape[1] == 2
    # Expect at least one infinite death in H0
    assert np.isinf(dgms[0][:, 1]).any()

    # Persistence list
    pers = to_gudhi_persistence(ph)
    assert any(d == 0 and np.isinf(bd[1]) for d, bd in pers)

    # Group by dimension
    groups = group_by_dim(ph)
    assert len(groups) >= 1
    if len(groups[0]) > 0:
        assert np.all(groups[0][:, 0] == 0)


def test_dual_embedding_known_2d_case():
    arr = np.array([[5.0, 2.0], [3.0, 7.0]], dtype=np.float64)
    out = dual_embedding(arr)
    expected = np.array(
        [
            [5.0, 2.0, 2.0],
            [3.0, 2.0, 2.0],
            [3.0, 3.0, 7.0],
        ],
        dtype=np.float64,
    )
    assert out.shape == (3, 3)
    assert np.array_equal(out, expected)


def _dual_embedding_reference(arr: np.ndarray) -> np.ndarray:
    out = np.empty(tuple(s + 1 for s in arr.shape), dtype=np.float64)
    for out_idx in np.ndindex(*out.shape):
        m = np.inf
        for src_idx in np.ndindex(*arr.shape):
            if all((out_idx[d] - src_idx[d]) in (0, 1) for d in range(arr.ndim)):
                m = min(m, arr[src_idx])
        out[out_idx] = m
    return out


def test_dual_embedding_matches_reference_3d():
    rng = np.random.default_rng(0)
    arr = rng.random((4, 3, 2), dtype=np.float64)
    out = dual_embedding(arr)
    ref = _dual_embedding_reference(arr)
    assert out.shape == (5, 4, 3)
    np.testing.assert_allclose(out, ref, rtol=0.0, atol=0.0)
