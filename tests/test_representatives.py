"""Regression tests for optional homology-cycle representatives."""

from collections import Counter

import numpy as np
import pytest

import cripser

tcripser = pytest.importorskip("tcripser")


def _annulus() -> np.ndarray:
    """A small planar filtration with one finite H_1 interval."""
    arr = np.ones((7, 7), dtype=np.float64)
    arr[1:6, 1:6] = 0.0
    arr[2:5, 2:5] = 1.0
    return arr


def _boundary_mod_2(chain: list[list[int]]) -> Counter[tuple[int, int, int]]:
    """Return the F_2 boundary of a planar 1-chain encoded by the API."""
    boundary: Counter[tuple[int, int, int]] = Counter()
    for x, y, z, cell_type in chain:
        if cell_type == 0:  # x-edge
            endpoints = ((x, y, z), (x + 1, y, z))
        elif cell_type == 1:  # y-edge
            endpoints = ((x, y, z), (x, y + 1, z))
        else:  # A planar representative must not contain an out-of-plane edge.
            raise AssertionError(f"unexpected cell type in planar cycle: {cell_type}")
        for vertex in endpoints:
            boundary[vertex] ^= 1
            if boundary[vertex] == 0:
                del boundary[vertex]
    return boundary


@pytest.mark.parametrize("module", [cripser, tcripser])
def test_representatives_return_closed_homology_cycles(module):
    arr = _annulus()

    default_pairs = module.computePH(arr, maxdim=1)
    disabled_pairs = module.computePH(arr, maxdim=1, representatives=False)
    pairs, cycles = module.computePH(arr, maxdim=1, representatives=True)

    # The default (optimized cohomology) path retains its public result and
    # return type; direct homology reduction is only entered when requested.
    assert isinstance(disabled_pairs, np.ndarray)
    np.testing.assert_array_equal(default_pairs, disabled_pairs)
    np.testing.assert_array_equal(default_pairs, pairs)
    assert len(cycles) == len(pairs)

    h1 = [
        cycle
        for pair, cycle in zip(pairs, cycles)
        if int(pair[0]) == 1 and pair[2] > pair[1]
    ]
    assert h1
    assert _boundary_mod_2(h1[0]) == Counter()


def test_compute_ph_wrapper_preserves_representative_alignment():
    pairs, cycles = cripser.compute_ph(_annulus(), maxdim=1, representatives=True)

    assert pairs.ndim == 2
    assert len(cycles) == len(pairs)


def test_representatives_reject_alexander_top_dim_shortcut():
    with pytest.raises(ValueError, match="top_dim"):
        cripser.computePH(_annulus(), maxdim=1, top_dim=True, representatives=True)


def test_three_dimensional_representative_is_a_closed_h2_cycle():
    n = 5
    coords = np.arange(n, dtype=float)
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    center = (n - 1) / 2.0
    shell = np.abs(np.sqrt((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2) - center) <= 0.85
    arr = np.where(shell, 0.0, 1.0)

    pairs, cycles = cripser.computePH(arr, maxdim=2, representatives=True)
    h2 = next(
        cycle
        for pair, cycle in zip(pairs, cycles)
        if int(pair[0]) == 2 and pair[2] > pair[1]
    )

    # Cubical faces of xy, zx, and yz squares respectively. Coefficients are
    # in F_2, so every boundary edge must occur an even number of times.
    boundary: Counter[tuple[int, int, int, int]] = Counter()
    for x, y, z, cell_type in h2:
        if cell_type == 0:  # xy
            faces = ((x, y, z, 1), (x + 1, y, z, 1), (x, y, z, 0), (x, y + 1, z, 0))
        elif cell_type == 1:  # zx
            faces = ((x, y, z, 2), (x + 1, y, z, 2), (x, y, z, 0), (x, y, z + 1, 0))
        elif cell_type == 2:  # yz
            faces = ((x, y, z, 2), (x, y + 1, z, 2), (x, y, z, 1), (x, y, z + 1, 1))
        else:
            raise AssertionError(f"unexpected square type: {cell_type}")
        for edge in faces:
            boundary[edge] ^= 1
            if boundary[edge] == 0:
                del boundary[edge]

    assert boundary == Counter()
