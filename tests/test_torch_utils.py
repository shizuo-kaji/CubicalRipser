import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cripser import compute_ph_torch, finite_lifetimes


def test_compute_ph_torch_backward_matches_locations_1d():
    arr = torch.tensor([0.0, 2.0, 1.0, 3.0], dtype=torch.float64, requires_grad=True)
    ph = compute_ph_torch(arr, maxdim=0, filtration="V")

    loss = finite_lifetimes(ph, dim=0).sum()
    loss.backward()

    assert arr.grad is not None
    expected = torch.zeros_like(arr)

    ph_np = ph.detach().cpu().numpy()
    finite_mask = (
        (ph_np[:, 0].astype(np.int64) == 0)
        & np.isfinite(ph_np[:, 2])
    )
    for row in ph_np[finite_mask]:
        birth_x = int(round(row[3]))
        death_x = int(round(row[6]))
        expected[birth_x] -= 1.0
        expected[death_x] += 1.0

    assert torch.allclose(arr.grad, expected)


def test_compute_ph_torch_preserves_input_grad_dtype():
    arr = torch.tensor([[0.0, 1.0], [3.0, 2.0]], dtype=torch.float32, requires_grad=True)
    ph = compute_ph_torch(arr, maxdim=0, filtration="V")
    loss = ph[:, 1].sum()
    loss.backward()

    assert arr.grad is not None
    assert arr.grad.dtype == arr.dtype
