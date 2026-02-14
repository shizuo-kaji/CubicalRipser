import numpy as np
import pytest

from cripser.vectorization import create_PH_histogram_volume, persistence_image


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


def test_persistence_image_matches_persim_on_fixed_pairs(tmp_path, monkeypatch):
    mplconfig = tmp_path / "mplconfig"
    mplconfig.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MPLCONFIGDIR", str(mplconfig))

    persim = pytest.importorskip("persim")
    PersistenceImager = persim.PersistenceImager

    ph = np.array(
        [
            [0, 0.10, 0.45],
            [0, 0.22, 0.62],
            [0, 0.34, 0.70],
            [1, 0.18, 0.58],
            [1, 0.40, 0.92],
            [1, 0.55, 0.95],
        ],
        dtype=np.float64,
    )

    homology_dims = (0, 1)
    n_bins = 48
    birth_range = (0.0, 1.2)
    life_range = (0.0, 1.2)
    sigma = 0.09
    weight_power = 1.0

    ours = persistence_image(
        ph,
        homology_dims=homology_dims,
        n_birth_bins=n_bins,
        n_life_bins=n_bins,
        birth_range=birth_range,
        life_range=life_range,
        sigma=sigma,
        weight_power=weight_power,
        dtype=np.float64,
    )

    pixel_size = (birth_range[1] - birth_range[0]) / n_bins
    pimgr = PersistenceImager(
        birth_range=birth_range,
        pers_range=life_range,
        pixel_size=pixel_size,
        weight_params={"n": weight_power},
        # persim scalar sigma is covariance; our sigma is standard deviation.
        kernel_params={"sigma": float(sigma**2)},
    )

    for channel, dim in enumerate(homology_dims):
        diagram_bd = ph[ph[:, 0].astype(np.int64) == dim][:, 1:3]
        persim_img = np.asarray(pimgr.transform(diagram_bd, skew=True), dtype=np.float64).T

        ours_img = ours[channel]
        ours_norm = ours_img / float(np.sum(ours_img))
        persim_norm = persim_img / float(np.sum(persim_img))

        assert ours_norm.shape == persim_norm.shape

        diff = ours_norm - persim_norm
        mean_abs = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff**2)))
        corr = float(np.corrcoef(ours_norm.ravel(), persim_norm.ravel())[0, 1])

        assert mean_abs < 5e-4
        assert rmse < 1e-3
        assert corr > 0.99
