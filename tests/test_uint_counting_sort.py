import numpy as np

import cripser
import tcripser


def _sample_grid(dtype):
    return np.array(
        [
            [0, 7, 7, 1, 1],
            [2, 7, 3, 3, 1],
            [2, 5, 5, 3, 4],
            [6, 6, 5, 4, 4],
        ],
        dtype=dtype,
    )


def _assert_same_output(module, arr):
    out_int = module.computePH(arr, maxdim=1)
    out_f64 = module.computePH(arr.astype(np.float64), maxdim=1)
    assert out_int.shape == out_f64.shape
    np.testing.assert_allclose(out_int, out_f64, rtol=0.0, atol=0.0)


def _assert_same_output_wrapper(arr, filtration):
    out_int = cripser.compute_ph(arr, maxdim=1, filtration=filtration)
    out_f64 = cripser.compute_ph(arr.astype(np.float64), maxdim=1, filtration=filtration)
    assert out_int.shape == out_f64.shape
    np.testing.assert_allclose(out_int, out_f64, rtol=0.0, atol=0.0)


def test_cripser_uint8_matches_float64():
    _assert_same_output(cripser, _sample_grid(np.uint8))


def test_cripser_uint16_matches_float64():
    _assert_same_output(cripser, _sample_grid(np.uint16) * np.uint16(257))


def test_tcripser_uint8_matches_float64():
    _assert_same_output(tcripser, _sample_grid(np.uint8))


def test_tcripser_uint16_matches_float64():
    _assert_same_output(tcripser, _sample_grid(np.uint16) * np.uint16(257))


def test_compute_ph_wrapper_uint8_matches_float64_v():
    _assert_same_output_wrapper(_sample_grid(np.uint8), "V")


def test_compute_ph_wrapper_uint16_matches_float64_t():
    _assert_same_output_wrapper(_sample_grid(np.uint16) * np.uint16(257), "T")
