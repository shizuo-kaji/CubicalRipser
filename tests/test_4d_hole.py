import math
import numpy as np
import cripser
import signal


def create_4d_sphere(n=5):
    # --- 1) Build a 4D binary 5x5x5x5 array that approximates a 3-sphere ---
    # Grid coordinates (0..4) with center at 2.0 in each axis
    coords = np.arange(n, dtype=float)
    X, Y, Z, W = np.meshgrid(coords, coords, coords, coords, indexing='ij')
    cx = cy = cz = cw = (n - 1) / 2.0  # center = 2.0
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2 + (W - cw)**2)

    # Choose a radius and thickness so the shell is captured on this tiny grid
    r = 2.0
    thickness = 0.85  # shell half-thickness
    sphere_mask = (np.abs(dist - r) <= thickness).astype(np.uint8)  # 1 on the shell, 0 elsewhere

    # We’ll define a lower-star filtration function:
    # value 0 on the shell (so it appears first), 1 elsewhere.
    return(np.where(sphere_mask == 1, 0.0, 1.0).astype(float))

def test_4d_sphere_ph():
    # Compute up to H3 on the 4D array via Python module
    arr = create_4d_sphere(n=5)
    ph = cripser.compute_ph(arr, maxdim=3, filtration="V")
    check(ph)
    pht = cripser.compute_ph(arr, maxdim=3, filtration="T")
    check(pht)

def check(ph):
    dims = ph[:, 0].astype(int)
    births = ph[:, 1]
    deaths = ph[:, 2]

    # Exactly one H0 infinite bar at birth 0.0
    h0_inf_idx = np.where((dims == 0) & (deaths>1e20))[0]
    assert len(h0_inf_idx) == 1
    assert abs(births[h0_inf_idx[0]] - 0.0) < 1e-9

    # Exactly one H3 bar (0.0, 1.0)
    h3_idx = np.where(dims == 3)[0]
    assert len(h3_idx) == 1
    b = float(births[h3_idx[0]])
    d = float(deaths[h3_idx[0]])
    assert abs(b - 0.0) < 1e-9
    assert abs(d - 1.0) < 1e-9
