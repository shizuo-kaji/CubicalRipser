import gzip
import hashlib
import os
import struct

import numpy as np
import pytest

import cripser
from cripser import datasets


def _betti_of_binary(mask, filtration):
    """Betti numbers of the cells where ``mask`` is true."""
    arr = np.where(mask, 0.0, 1.0)
    ph = cripser.compute_ph(arr, filtration=filtration, maxdim=arr.ndim - 1)
    born = (ph[:, 1] == 0) & (ph[:, 2] > 0)
    return [int(np.sum(born & (ph[:, 0] == k))) for k in range(arr.ndim)]


@pytest.mark.parametrize("shape", [7, (7,), (6, 5), (5, 4, 3), (4, 3, 3, 2)])
@pytest.mark.parametrize(
    "make",
    [
        datasets.uniform_noise,
        lambda s, seed: datasets.uniform_noise(s, levels=3, seed=seed),
        datasets.gaussian_random_field,
        lambda s, seed: datasets.gaussian_random_field(s, sigma=0, seed=seed),
        lambda s, seed: datasets.sphere(s),
        datasets.distance_to_points,
    ],
)
def test_shape_dtype_layout(make, shape):
    arr = make(shape, seed=1)
    expected = (shape,) if isinstance(shape, int) else shape
    assert arr.shape == expected
    assert arr.dtype == np.float64
    assert arr.flags.c_contiguous


@pytest.mark.parametrize(
    "make",
    [datasets.uniform_noise, datasets.gaussian_random_field, datasets.distance_to_points],
)
def test_seed_reproducible(make):
    a = make((9, 8), seed=3)
    assert np.array_equal(a, make((9, 8), seed=3))
    assert not np.array_equal(a, make((9, 8), seed=4))
    assert np.array_equal(a, make((9, 8), seed=np.random.default_rng(3)))


def test_uniform_noise_levels():
    arr = datasets.uniform_noise((20, 20), levels=4, seed=0)
    assert set(np.unique(arr)) == {0.0, 1.0, 2.0, 3.0}
    arr = datasets.uniform_noise((20, 20), seed=0)
    assert arr.min() >= 0.0 and arr.max() < 1.0
    assert len(np.unique(arr)) == arr.size


def test_gaussian_random_field_smoothness():
    smooth = datasets.gaussian_random_field((64, 64), sigma=4.0, seed=0)
    white = datasets.gaussian_random_field((64, 64), sigma=0, seed=0)
    for arr in (smooth, white):
        assert abs(arr.mean()) < 1e-12
        assert abs(arr.std() - 1.0) < 1e-12

    def neighbour_corr(a):
        return np.corrcoef(a[:, :-1].ravel(), a[:, 1:].ravel())[0, 1]

    assert neighbour_corr(smooth) > 0.9
    assert abs(neighbour_corr(white)) < 0.1


@pytest.mark.parametrize("filtration", ["V", "T"])
@pytest.mark.parametrize("ndim,n", [(2, 9), (3, 9), (4, 9)])
def test_sphere(filtration, ndim, n):
    arr = datasets.sphere((n,) * ndim)
    betti = [0] * ndim
    betti[0] = betti[-1] = 1
    assert _betti_of_binary(arr <= 0.85, filtration) == betti

    ph = cripser.compute_ph(arr, filtration=filtration, maxdim=ndim - 1)
    top = ph[ph[:, 0] == ndim - 1]
    assert len(top) == 1
    radius = (n - 1) / 2
    assert top[0, 1] < 1.0
    assert top[0, 2] == radius  # the center voxel lies on the grid


def test_sphere_center_and_radius():
    arr = datasets.sphere((11, 13), radius=3.0, center=(4, 5))
    assert arr[4, 5] == 3.0
    assert arr[4, 8] == 0.0
    assert arr[1, 5] == 0.0


@pytest.mark.parametrize("filtration", ["V", "T"])
@pytest.mark.parametrize("shape", [(16, 16, 8), (32, 32, 16)])
def test_torus(filtration, shape):
    arr = datasets.torus(shape)
    assert _betti_of_binary(arr <= 0.85, filtration) == [1, 2, 1]


def test_torus_requires_3d():
    with pytest.raises(ValueError):
        datasets.torus((16, 16))


@pytest.mark.parametrize("filtration", ["V", "T"])
def test_distance_to_points(filtration):
    points = np.array([[2, 2], [2, 10], [10, 6]])
    arr = datasets.distance_to_points((13, 13), points)
    assert arr[2, 2] == 0.0 and arr[2, 5] == 3.0
    ph = cripser.compute_ph(arr, filtration=filtration, maxdim=1)
    h0 = ph[ph[:, 0] == 0]
    assert len(h0) == 3
    assert np.all(h0[:, 1] == 0.0)


def test_distance_to_points_random():
    arr = datasets.distance_to_points((20, 30), 5, seed=0)
    ph = cripser.compute_ph(arr, maxdim=0)
    assert len(ph) == 5
    with pytest.raises(ValueError):
        datasets.distance_to_points((20, 30), np.zeros((3, 3)))


def _register(monkeypatch, path, shape, dtype):
    digest = hashlib.sha512(path.read_bytes()).hexdigest()
    volume = datasets.Volume(shape, dtype, "test", "test", path.as_uri(), digest)
    monkeypatch.setitem(datasets.VOLUMES, "test_volume", volume)
    return volume


def test_fetch_raw_downloads_once(tmp_path, monkeypatch, capsys):
    arr = np.arange(60, dtype=np.uint16).reshape(3, 4, 5) * 997
    source = tmp_path / "src" / "vol_5x4x3_uint16.raw"
    source.parent.mkdir()
    source.write_bytes(arr.astype("<u2").tobytes())
    _register(monkeypatch, source, (3, 4, 5), "uint16")
    cache = tmp_path / "cache"

    out = datasets.fetch("test_volume", data_home=cache)
    assert "Downloading test_volume" in capsys.readouterr().err
    assert out.dtype == np.float64 and out.flags.c_contiguous
    assert np.array_equal(out, arr)
    assert [p.name for p in cache.iterdir()] == [source.name]

    source.unlink()  # later calls read the saved file
    assert np.array_equal(datasets.fetch("test_volume", data_home=cache), arr)
    assert capsys.readouterr().err == ""


def test_fetch_uses_environment_data_dir(tmp_path, monkeypatch):
    source = tmp_path / "v.raw"
    source.write_bytes(bytes(range(8)))
    _register(monkeypatch, source, (2, 2, 2), "uint8")
    monkeypatch.setenv("CRIPSER_DATA_DIR", str(tmp_path / "env"))
    datasets.fetch("test_volume")
    assert (tmp_path / "env" / "v.raw").exists()


def test_fetch_rejects_digest_mismatch(tmp_path, monkeypatch):
    source = tmp_path / "v.raw"
    source.write_bytes(bytes(8))
    volume = _register(monkeypatch, source, (2, 2, 2), "uint8")
    monkeypatch.setitem(
        datasets.VOLUMES, "test_volume", datasets.Volume(**{**volume.__dict__, "sha512": "0" * 128})
    )
    cache = tmp_path / "cache"
    with pytest.raises(OSError, match="SHA-512"):
        datasets.fetch("test_volume", data_home=cache)
    assert list(cache.iterdir()) == []


def _write_nifti_gz(path, arr, code, slope=1.0, inter=0.0):
    header = bytearray(352)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, arr.ndim, *arr.shape, *[1] * (7 - arr.ndim))
    struct.pack_into("<2h", header, 70, code, arr.dtype.itemsize * 8)
    struct.pack_into("<f", header, 108, 352.0)
    struct.pack_into("<2f", header, 112, slope, inter)
    header[344:348] = b"n+1\0"
    with gzip.open(path, "wb") as f:
        f.write(bytes(header) + arr.tobytes(order="F"))


@pytest.mark.parametrize("slope,inter", [(1.0, 0.0), (0.0, 0.0), (2.0, -1.0)])
def test_fetch_nifti(tmp_path, monkeypatch, slope, inter):
    arr = (np.arange(2 * 3 * 4 * 5, dtype=np.int16) - 50).reshape(2, 3, 4, 5)  # [x, y, z, t]
    source = tmp_path / "bold.nii.gz"
    _write_nifti_gz(source, arr, 4, slope, inter)
    _register(monkeypatch, source, arr.shape, None)

    out = datasets.fetch("test_volume", data_home=tmp_path / "cache")
    expected = arr * slope + inter if slope != 0 else arr
    assert out.dtype == np.float64 and out.flags.c_contiguous
    assert np.array_equal(out, expected)


def test_fetch_unknown_name():
    with pytest.raises(ValueError, match="bonsai"):
        datasets.fetch("no_such_volume")


def test_volume_registry():
    for name, volume in datasets.VOLUMES.items():
        assert len(volume.shape) in (3, 4), name
        assert len(volume.sha512) == 128 and int(volume.sha512, 16) >= 0, name
        if volume.dtype is None:
            assert ".nii.gz" in volume.url, name
        else:
            assert volume.url.endswith(f"_{volume.dtype}.raw"), name
            assert "x".join(map(str, volume.shape[::-1])) in volume.url, name


@pytest.mark.skipif(
    not os.environ.get("CRIPSER_TEST_DOWNLOADS"),
    reason="set CRIPSER_TEST_DOWNLOADS=1 to download the volumes",
)
@pytest.mark.parametrize("name", sorted(datasets.VOLUMES))
def test_fetch_downloads(name):
    arr = datasets.fetch(name)
    assert arr.shape == datasets.VOLUMES[name].shape
    assert arr.dtype == np.float64
