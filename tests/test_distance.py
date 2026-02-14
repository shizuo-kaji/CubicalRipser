import os

import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
torch = pytest.importorskip("torch")
pytest.importorskip("ot")

from cripser import wasserstein_distance

def test_wasserstein_distance_accepts_compute_ph_and_dim():
    # [dim, birth, death, ...] with 9 columns, matching compute_ph output format.
    x = torch.tensor(
        [
            [0.0, 0.1, 0.8, 0, 0, 0, 0, 0, 0],
            [1.0, 0.5, 1.4, 0, 0, 0, 0, 0, 0],
            [1.0, 0.6, float("inf"), 0, 0, 0, 0, 0, 0],  # dropped as non-finite pair
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    y = torch.tensor(
        [
            [0.0, 0.2, 0.7, 0, 0, 0, 0, 0, 0],
            [1.0, 0.4, 1.2, 0, 0, 0, 0, 0, 0],
            [1.0, 0.9, 1.7, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    dist = wasserstein_distance(x, y, dim=1)
    assert dist.ndim == 0
    assert torch.isfinite(dist)
    dist.backward()

    assert x.grad is not None
    assert y.grad is not None
    # dim=0 row in x is not selected
    assert torch.allclose(x.grad[0, 1:3], torch.zeros_like(x.grad[0, 1:3]))
    # selected finite dim=1 row participates
    assert torch.linalg.vector_norm(x.grad[1, 1:3]).item() > 0.0
    # dropped infinite row does not participate
    assert torch.allclose(x.grad[2, 1:3], torch.zeros_like(x.grad[2, 1:3]))


def test_wasserstein_distance_diagonal_matching_one_empty_diagram():
    x = torch.tensor([[0.0, 2.0]], dtype=torch.float64)
    y = torch.empty((0, 2), dtype=torch.float64)

    # point-to-diagonal distance for (0,2) under q=2 is sqrt(2)
    d = wasserstein_distance(x, y, p=2.0, q=2.0)
    assert torch.isfinite(d)
    assert torch.allclose(d, torch.sqrt(torch.tensor(2.0)), atol=1e-6)
