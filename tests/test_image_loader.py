import numpy as np

from cripser.image_loader import (
    detect_format,
    load_dipha_complex,
    save_dipha_complex,
    load_perseus,
    save_perseus,
    load_image,
    load_series,
    save_image,
)


def test_detect_format():
    assert detect_format("a.npy") == "numpy"
    assert detect_format("a.complex") == "dipha"
    assert detect_format("a.txt") == "perseus"
    assert detect_format("a.dcm") == "dicom"
    assert detect_format("a.nrrd") == "nrrd"
    assert detect_format("a.jpg") == "image"


def test_dipha_roundtrip_no_transpose(tmp_path):
    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    fn = tmp_path / "arr.complex"
    save_dipha_complex(arr, fn, transpose_for_dipha=False)
    out = load_dipha_complex(fn)
    np.testing.assert_array_equal(out, arr)


def test_dipha_roundtrip_with_dipha_transpose(tmp_path):
    arr = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    fn = tmp_path / "arr_dipha.complex"
    save_dipha_complex(arr, fn, transpose_for_dipha=True)
    out = load_dipha_complex(fn, transpose_to_numpy=True)
    np.testing.assert_array_equal(out, arr)


def test_perseus_roundtrip(tmp_path):
    arr = np.array([[0.0, 1.5, 2.0], [3.0, 4.0, 5.0]])
    fn = tmp_path / "arr.txt"
    save_perseus(arr, fn)
    out = load_perseus(fn)
    np.testing.assert_allclose(out, arr)


def test_load_series_numeric_sort(tmp_path):
    a = np.full((2, 2), 2, dtype=np.float64)
    b = np.full((2, 2), 10, dtype=np.float64)
    np.save(tmp_path / "slice_10.npy", b)
    np.save(tmp_path / "slice_2.npy", a)

    out = load_series(tmp_path, input_extension=".npy", numeric_sort=True)
    assert out.shape == (2, 2, 2)
    np.testing.assert_array_equal(out[0], a)
    np.testing.assert_array_equal(out[1], b)


def test_save_image_and_load_image_perseus(tmp_path):
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    fn = tmp_path / "saved.txt"
    save_image(arr, fn)
    out = load_image(fn)
    np.testing.assert_allclose(out, arr)
