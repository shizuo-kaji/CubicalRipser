import numpy as np

import cripser
import tcripser
from cripser import datasets

def create_3d_sphere(n=5):
    """
    Create a 3D array (n x n x n) whose lower-star filtration
    encodes a thin 2-sphere (surface of a 3D ball):
      value 0.0 on the spherical shell,
      value 1.0 elsewhere.
    """
    return np.where(datasets.sphere((n,) * 3) <= 0.85, 0.0, 1.0)

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
