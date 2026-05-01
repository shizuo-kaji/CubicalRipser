import numpy as np

import cripser
import tcripser

def create_3d_sphere(n=5):
    """
    Create a 3D array (n x n x n) whose lower-star filtration
    encodes a thin 2-sphere (surface of a 3D ball):
      value 0.0 on the spherical shell,
      value 1.0 elsewhere.
    """
    coords = np.arange(n, dtype=float)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    c = (n - 1) / 2.0
    dist = np.sqrt((X - c)**2 + (Y - c)**2 + (Z - c)**2)

    r = c  # sphere roughly centered, maximal within grid
    thickness = 0.85
    shell = (np.abs(dist - r) <= thickness).astype(np.uint8)

    return np.where(shell == 1, 0.0, 1.0).astype(float)

def test_cripser_module_on_3d_hole():
    arr = create_3d_sphere(n=5)
    ph = cripser.computePH(arr, maxdim=2)
    assert ph.ndim == 2 and ph.shape[1] == 9
    dims = set(ph[:, 0].astype(int))
    # Expect H0 and H2 features on a 3D hole dataset
    assert 0 in dims
    assert 2 in dims


def test_tcripser_module_on_3d_hole():
    arr = create_3d_sphere(n=5)
    ph = tcripser.computePH(arr, maxdim=2)
    assert ph.ndim == 2 and ph.shape[1] == 9
    dims = set(ph[:, 0].astype(int))
    # Expect non-empty result and some higher-dimensional features
    assert 0 in dims
    assert 2 in dims
