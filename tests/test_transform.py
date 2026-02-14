import numpy as np
import pytest

from cripser.transform import binarize, apply_transform, preprocess_image


def test_binarize_with_threshold_range():
    arr = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
    out = binarize(arr, threshold=0.5, threshold_upper_limit=1.0)
    np.testing.assert_array_equal(out, np.array([False, True, True, False]))


def test_apply_upward_transform():
    arr = np.array(
        [
            [0, 1, 1],
            [1, 1, 0],
        ],
        dtype=np.float64,
    )
    out = apply_transform(arr, "upward", threshold=0.5)
    expected = np.array(
        [
            [0, 0, 0],
            [-1, -1, 0],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(out, expected)


def test_apply_radial_transform():
    arr = np.array(
        [
            [1, 1],
            [1, 0],
        ],
        dtype=np.float64,
    )
    out = apply_transform(arr, "radial", threshold=0.5, origin=(0, 0))
    expected = np.array(
        [
            [0.0, 1.0],
            [1.0, np.sqrt(2.0)],
        ]
    )
    np.testing.assert_allclose(out, expected)


def test_apply_distance_transform_matches_reference():
    scipy = pytest.importorskip("scipy.ndimage")
    distance_transform_edt = scipy.distance_transform_edt
    arr = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    bw = arr >= 0.5
    expected = distance_transform_edt(bw)
    out = apply_transform(arr, "distance", threshold=0.5)
    np.testing.assert_allclose(out, expected)


def test_preprocess_image_pipeline_without_optional_deps():
    arr = np.arange(6).reshape(2, 3)
    out = preprocess_image(
        arr,
        transpose=(1, 0),
        zrange=(0, 2),
        tile=2,
        shift_value=5,
        dtype=np.int32,
    )
    expected = np.array(
        [
            [5, 8, 5, 8],
            [6, 9, 6, 9],
            [5, 8, 5, 8],
            [6, 9, 6, 9],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(out, expected)
