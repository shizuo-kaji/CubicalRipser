import numpy as np

import cripser

def test_alexander(dim=4):
    rng = np.random.default_rng(0)
    img = rng.random((10, )*dim)

    pdt = cripser.compute_ph(img, filtration="T", maxdim=dim-1)
    pdt_reduced = pdt[np.abs(pdt[:, 1:3]).max(axis=1) < 999999]
    pdt_reduced = pdt_reduced[np.lexsort((pdt_reduced[:, 0], pdt_reduced[:, 2], pdt_reduced[:, 1]))]

    pdt2 = cripser.compute_ph(img, embedded=True, maxdim=dim-1)
    pdt2_reduced = pdt2[np.abs(pdt2[:, 1:3]).max(axis=1) < 999999]
    pdt2_reduced = pdt2_reduced[np.lexsort((pdt2_reduced[:, 0], pdt2_reduced[:, 1], pdt2_reduced[:, 2]))[::-1]]

    counts1 = [np.sum(pdt_reduced[:, 0] == i) for i in range(dim)]
    counts2 = [np.sum(pdt2_reduced[:, 0] == i) for i in range(dim)]
    assert counts1 == counts2[::-1], f"Betti numbers not reversed: {counts1} vs {counts2}"

    p1 = [pdt_reduced[pdt_reduced[:, 0] == i, 1:3] for i in range(dim)]
    p2 = [pdt2_reduced[pdt2_reduced[:, 0] == i, 1:3] for i in range(dim)]

    for i in range(dim):
        assert np.allclose(p1[i], -p2[dim-1-i][:, [1, 0]]), f"PH{i} vs PH{dim-1-i} duality failed"
