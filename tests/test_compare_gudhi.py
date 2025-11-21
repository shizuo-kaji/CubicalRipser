import numpy as np
import pytest

import cripser
import tcripser

# Skip these tests gracefully when gudhi is not installed locally.
gd = pytest.importorskip("gudhi", reason="GUDHI is required for cross-library comparisons")

def _against_gudhi(dim=4,filtration="V"):
    arr = np.random.random([10]*dim)
    ph = cripser.compute_ph(arr, maxdim=dim-1, filtration=filtration)
    ph = cripser.to_gudhi_diagrams(ph)
    if filtration=="V":
        cc = gd.CubicalComplex(vertices=arr)
    else:
        cc = gd.CubicalComplex(top_dimensional_cells=arr)
    cc.persistence(homology_coeff_field=2, min_persistence=0)
    for d in range(dim-1):
        cripser_ints =  np.asarray(ph[d])
        gudhi_ints = np.asarray(cc.persistence_intervals_in_dimension(d))
        # Sort intervals lexicographically by (birth, death) for stable comparison
        def sort_ints(a):
            if a.size == 0:
                return a.reshape(0, 2)
            return a[np.lexsort((a[:, 1], a[:, 0]))]
        ci = sort_ints(cripser_ints)
        gi = sort_ints(gudhi_ints)
        print("cripser", d,ci.shape)
        print(gi.shape)
        assert ci.shape == gi.shape, f"Mismatch count in dim {d}: {ci.shape} vs {gi.shape}"
        assert np.allclose(ci, gi), f"Intervals differ in dim {d}"

def test_cripser_vs_gudhi():
    for d in range(1, 4):
        for filtration in ["V", "T"]:
           _against_gudhi(dim=d, filtration=filtration)


_INF_CUTOFF = np.finfo(np.float64).max / 4.0


def _make_unique_grid(shape, seed):
    rng = np.random.default_rng(seed)
    arr = rng.random(shape, dtype=np.float64)
    perturb = np.arange(arr.size, dtype=np.float64).reshape(shape)
    # Keep perturbation tiny relative to the random signal while ensuring uniqueness
    arr += perturb * 1e-9
    return arr


def _sorted_pairs(records):
    return sorted(records, key=lambda r: (r[0], r[1], r[2], r[3], r[4], r[1], r[2]))


def _extract_pairs_from_cripser(arr,filtration):
    ndim = arr.ndim
    ph = cripser.compute_ph(arr, maxdim=ndim - 1, filtration=filtration)
    rows = []
    for row in ph:
        dim = int(row[0])
        birth = float(row[1])
        death = float(row[2])
        birth_coords = tuple(int(row[3 + i]) for i in range(ndim))
        death_coords = tuple(int(row[6 + i]) for i in range(ndim))
        if death >= _INF_CUTOFF:
            death = np.inf
            death_coords = tuple([-1] * ndim)
        rows.append((dim, birth, death, birth_coords, death_coords))
    return _sorted_pairs(rows)


def _extract_pairs_from_gudhi(arr, filtration):
    shape = arr.shape
    flat = np.asfortranarray(arr, dtype=np.float64).ravel(order='F')
    if filtration == "V":
        cc = gd.CubicalComplex(vertices=arr)
        cc.persistence(homology_coeff_field=2, min_persistence=0)
        paired, essential = cc.vertices_of_persistence_pairs()
    else:
        cc = gd.CubicalComplex(top_dimensional_cells=arr)
        cc.persistence(homology_coeff_field=2, min_persistence=0)
        paired, essential = cc.cofaces_of_persistence_pairs()
    #print("gudhi paired:", paired)

    rows = []
    sentinel = tuple([-1] * len(shape))

    def to_coords(index):
        return tuple(int(c) for c in np.unravel_index(int(index), shape, order='F'))

    maxdim = max(len(paired), len(essential), len(shape))
    for dim in range(maxdim):
        arr_pairs = np.asarray(paired[dim] if dim < len(paired) else [])
        if arr_pairs.size:
            arr_pairs = arr_pairs.reshape(-1, 2)
            for b_idx, d_idx in arr_pairs:
                rows.append((
                    dim,
                    float(flat[int(b_idx)]),
                    float(flat[int(d_idx)]),
                    to_coords(b_idx),
                    to_coords(d_idx),
                ))

        arr_ess = np.asarray(essential[dim] if dim < len(essential) else [])
        if arr_ess.size:
            arr_ess = arr_ess.reshape(-1)
            for b_idx in arr_ess:
                rows.append((
                    dim,
                    float(flat[int(b_idx)]),
                    np.inf,
                    to_coords(b_idx),
                    sentinel,
                ))

    return _sorted_pairs(rows)


@pytest.mark.parametrize(
    "shape, seed",
    [((5, 6), 13), ((4, 5, 6), 29)],
)
def test_birth_death_locations_match_gudhi(shape, seed):
    arr = _make_unique_grid(shape, seed)
    for filtration in ["V", "T"]:
        cripser_pairs = _extract_pairs_from_cripser(arr, filtration)
        gudhi_pairs = _extract_pairs_from_gudhi(arr, filtration)
        print("\n\ngudhi_pairs:", gudhi_pairs)
        print("\ncripser_pairs:", cripser_pairs)
        assert len(cripser_pairs) == len(gudhi_pairs)
        for c_pair, g_pair in zip(cripser_pairs, gudhi_pairs):
            assert c_pair[0] == g_pair[0], "Dimension mismatch"
            assert np.isclose(c_pair[1], g_pair[1], atol=1e-9)
            assert c_pair[3] == g_pair[3], "Birth coordinates differ"
            assert c_pair[4] == g_pair[4], "Death coordinates differ"
            if np.isinf(c_pair[2]) and np.isinf(g_pair[2]):
                continue
            assert np.isclose(c_pair[2], g_pair[2], atol=1e-9)
