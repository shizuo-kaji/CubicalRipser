import numpy as np

from cripser.vectorization import create_PH_histogram_volume


def test_create_ph_histogram_volume_channel_layout():
    # Columns: [dim, birth, death, x1, y1, z1, x2, y2, z2]
    ph = np.array(
        [
            [0, 0.2, 0.7, 0, 1, 0, 1, 1, 0],  # dim 0, life/bin(0), birth/bin(0), at (0,1)
            [1, 0.6, 1.4, 0, 1, 0, 1, 1, 0],  # dim 1, life/bin(1), birth/bin(1), at (0,1)
            [1, 0.1, 0.5, 1, 0, 0, 0, 0, 0],  # dim 1, life/bin(0), birth/bin(0), at (1,0)
            [2, 0.1, 0.5, 0, 0, 0, 0, 0, 0],  # ignored (dim not requested)
        ],
        dtype=np.float64,
    )

    vol = create_PH_histogram_volume(
        ph,
        image_shape=(2, 2),
        homology_dims=(0, 1),
        n_life_bins=2,
        n_birth_bins=2,
        birth_range=(0.0, 1.0),
        life_range=(0.0, 1.0),
        dtype=np.int64,
    )

    assert vol.shape == (8, 2, 2)  # 2 dims * 2 life bins * 2 birth bins
    decoded = np.moveaxis(vol, 0, -1).reshape(2, 2, 2, 2, 2)  # x, y, dim, life, birth

    assert decoded[0, 1, 0, 0, 0] == 1
    assert decoded[0, 1, 1, 1, 1] == 1
    assert decoded[1, 0, 1, 0, 0] == 1
    assert np.sum(decoded) == 3


def test_create_ph_histogram_volume_drops_nonfinite_by_default():
    ph = np.array(
        [
            [0, 0.2, 0.8, 0, 0, 0, 0, 0, 0],
            [0, 0.1, np.inf, 0, 0, 0, 0, 0, 0],
            [0, np.nan, 0.4, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )

    vol, meta = create_PH_histogram_volume(
        ph,
        image_shape=(1, 1),
        homology_dims=(0,),
        n_life_bins=2,
        n_birth_bins=2,
        birth_range=(0.0, 1.0),
        life_range=(0.0, 1.0),
        return_metadata=True,
        dtype=np.int64,
    )

    assert vol.shape == (4, 1, 1)
    assert int(np.sum(vol)) == 1
    assert meta["n_input_pairs"] == 3
    assert meta["n_finite_pairs"] == 1
    assert meta["n_counted_pairs"] == 1


def test_create_ph_histogram_volume_death_location_and_bounds_filter():
    ph = np.array(
        [
            [0, 0.1, 0.7, 0, 0, 0, 1, 1, 0],  # death location in bounds
            [0, 0.2, 0.8, 0, 0, 0, 2, 2, 0],  # death location out of bounds
        ],
        dtype=np.float64,
    )

    ref = np.zeros((2, 2), dtype=np.float64)
    vol = create_PH_histogram_volume(
        ph,
        reference_volume=ref,
        homology_dims=(0,),
        n_life_bins=1,
        n_birth_bins=1,
        birth_range=(0.0, 1.0),
        life_range=(0.0, 1.0),
        location="death",
        dtype=np.int64,
    )

    assert vol.shape == (1, 2, 2)
    assert vol[0, 1, 1] == 1
    assert int(np.sum(vol)) == 1


def test_create_ph_histogram_volume_all_nonfinite_with_explicit_ranges():
    ph = np.array(
        [
            [0, 0.1, np.inf, 0, 0, 0, 0, 0, 0],
            [0, np.nan, 0.5, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )

    vol, meta = create_PH_histogram_volume(
        ph,
        image_shape=(1, 1),
        homology_dims=(0,),
        n_life_bins=2,
        n_birth_bins=2,
        birth_range=(0.0, 1.0),
        life_range=(0.0, 1.0),
        drop_nonfinite=True,
        return_metadata=True,
        dtype=np.int64,
    )

    assert vol.shape == (4, 1, 1)
    assert int(np.sum(vol)) == 0
    assert meta["n_finite_pairs"] == 0
    assert meta["n_counted_pairs"] == 0
