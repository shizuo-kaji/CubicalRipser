from collections import Counter

import numpy as np
import pytest

import cripser
from zigzag_reference import explicit_fzz_zigzag, spacetime_cone_zigzag


def barcode(masks, **kwargs):
    """Counter of (dim, 2*birth, 2*death) from compute_zigzag."""
    out = cripser.compute_zigzag(masks, **kwargs)
    assert out.shape[1] == 9
    return Counter(
        (int(r[0]), int(round(2 * r[1])), int(round(2 * r[2]))) for r in out
    )


def random_masks(rng, frames, shape, density=None):
    if density is None:
        density = rng.uniform(0.3, 0.9)
    return rng.random((frames,) + tuple(shape)) < density


def ring(n=5):
    r = np.zeros((n, n), dtype=bool)
    r[1:-1, 1:-1] = True
    r[n // 2, n // 2] = False
    return r


def test_blob_lifetime_and_location():
    masks = np.zeros((4, 5, 5), dtype=bool)
    masks[1:3, 2, 3] = True
    out = cripser.compute_zigzag(masks)
    np.testing.assert_array_equal(out, [[0, 1.0, 2.0, 2, 3, 0, 2, 3, 0]])


def test_class_alive_in_last_frame_has_no_destroyer():
    masks = np.zeros((3, 5, 5), dtype=bool)
    masks[1:, 2, 3] = True
    out = cripser.compute_zigzag(masks)
    np.testing.assert_array_equal(out, [[0, 1.0, 2.0, 2, 3, 0, -1, -1, -1]])
    for filtration in ("V", "T"):
        for connect in ("intersection", "union"):
            out = cripser.compute_zigzag(
                np.stack([ring()] * 3), filtration=filtration, connect=connect
            )
            assert np.all(out[:, 6:9] == -1)


def test_union_connect_extends_into_connecting_complexes():
    masks = np.zeros((4, 5, 5), dtype=bool)
    masks[1:3, 2, 3] = True
    assert barcode(masks, connect="union") == Counter({(0, 1, 5): 1})


def test_static_ring_lives_throughout():
    masks = np.stack([ring()] * 3)
    for filtration in ("V", "T"):
        for connect in ("intersection", "union"):
            assert barcode(masks, filtration=filtration, connect=connect) == Counter(
                {(0, 0, 4): 1, (1, 0, 4): 1}
            )


def test_open_open_component_between_frames():
    # Both frames are connected, but their intersection is two bars.
    u = np.zeros((4, 3), dtype=bool)
    u[:, 0] = u[:, 2] = True
    cap = u.copy()
    u[3, :] = True
    cap[0, :] = True
    assert barcode(np.stack([u, cap])) == Counter({(0, 0, 2): 1, (0, 1, 1): 1})


def test_loop_present_in_a_single_frame():
    closed = ring()
    opened = closed.copy()
    opened[1, 2] = False
    assert barcode(np.stack([opened, closed, opened])) == Counter(
        {(0, 0, 4): 1, (1, 2, 2): 1}
    )


def test_t_construction_connects_diagonal_voxels():
    masks = np.zeros((1, 2, 2), dtype=bool)
    masks[0, 0, 0] = masks[0, 1, 1] = True
    assert barcode(masks, filtration="V") == Counter({(0, 0, 0): 2})
    assert barcode(masks, filtration="T") == Counter({(0, 0, 0): 1})


@pytest.mark.parametrize("seed", range(4))
def test_matches_spacetime_cone_reference(seed):
    rng = np.random.default_rng(seed)
    for _ in range(60):
        frames = int(rng.integers(1, 6))
        shape = rng.integers(2, 6, size=int(rng.integers(1, 3)))
        masks = random_masks(rng, frames, shape)
        assert barcode(masks) == spacetime_cone_zigzag(masks)


@pytest.mark.parametrize("filtration", ["V", "T"])
@pytest.mark.parametrize("connect", ["intersection", "union"])
def test_matches_explicit_fastzigzag_reference(filtration, connect):
    rng = np.random.default_rng(7)
    for spatial_dim, cases in ((1, 30), (2, 30), (3, 6)):
        for _ in range(cases):
            frames = int(rng.integers(1, 5))
            shape = rng.integers(2, 4 if spatial_dim == 3 else 5, size=spatial_dim)
            masks = random_masks(rng, frames, shape)
            expected = explicit_fzz_zigzag(masks, filtration, connect)
            assert barcode(masks, filtration=filtration, connect=connect) == expected


@pytest.mark.parametrize("filtration", ["V", "T"])
@pytest.mark.parametrize("connect", ["intersection", "union"])
def test_matches_explicit_fastzigzag_reference_with_cycles(filtration, connect):
    # Denser masks with more frames, so that H_1 and H_2 intervals of all
    # endpoint types occur.
    rng = np.random.default_rng(13)
    for shape, cases in (((7, 7), 8), ((4, 4, 4), 4)):
        for _ in range(cases):
            masks = random_masks(
                rng, int(rng.integers(3, 6)), shape, rng.uniform(0.55, 0.85)
            )
            expected = explicit_fzz_zigzag(masks, filtration, connect)
            assert barcode(masks, filtration=filtration, connect=connect) == expected


@pytest.mark.parametrize("filtration", ["V", "T"])
def test_single_frame_gives_homology(filtration):
    rng = np.random.default_rng(3)
    for shape in ((7,), (6, 5), (4, 4, 3)):
        masks = random_masks(rng, 1, shape, density=0.6)
        ph = cripser.compute_ph(
            np.where(masks[0], 0.0, 1.0), filtration=filtration, maxdim=len(shape) - 1
        )
        born_alive = ph[(ph[:, 1] == 0.0) & (ph[:, 2] > 0.0)]
        expected = Counter((int(k), 0, 0) for k in born_alive[:, 0])
        assert barcode(masks, filtration=filtration) == expected


@pytest.mark.parametrize("filtration", ["V", "T"])
@pytest.mark.parametrize("connect", ["intersection", "union"])
def test_growing_masks_match_sublevel_persistence(filtration, connect):
    rng = np.random.default_rng(11)
    for shape in ((9,), (7, 6), (4, 5, 4)):
        frames = 5
        first = rng.integers(0, frames + 1, size=shape)  # frames = never
        masks = np.stack([first <= t for t in range(frames)])
        ph = cripser.compute_ph(
            first.astype(np.float64), filtration=filtration, maxdim=len(shape) - 1
        )
        expected = Counter()
        for k, b, d in ph[:, :3]:
            if b >= frames or d == b:
                continue
            last = frames - 1 if d >= frames else d - 1  # last frame alive
            if connect == "intersection":
                key = (int(k), int(2 * b), int(2 * last + 1 if d < frames else 2 * last))
            else:
                key = (int(k), int(max(2 * b - 1, 0)), int(2 * last))
            expected[key] += 1
        assert barcode(masks, filtration=filtration, connect=connect) == expected


@pytest.mark.parametrize("filtration", ["V", "T"])
@pytest.mark.parametrize("connect", ["intersection", "union"])
def test_time_reversal_mirrors_intervals(filtration, connect):
    rng = np.random.default_rng(5)
    for shape in ((8,), (6, 6), (4, 3, 4)):
        masks = random_masks(rng, 5, shape)
        last = 2 * (masks.shape[0] - 1)
        forward = barcode(masks, filtration=filtration, connect=connect)
        backward = barcode(masks[::-1], filtration=filtration, connect=connect)
        assert backward == Counter(
            {(k, last - d, last - b): m for (k, b, d), m in forward.items()}
        )


@pytest.mark.parametrize("filtration", ["V", "T"])
def test_memory_layout_and_dtype_do_not_change_result(filtration):
    rng = np.random.default_rng(9)
    masks = random_masks(rng, 4, (6, 5, 3), density=0.6)
    expected = cripser.compute_zigzag(masks, filtration=filtration)
    for variant in (
        np.asfortranarray(masks),
        masks.astype(np.uint8),
        np.asfortranarray(masks.astype(np.int32)),
    ):
        np.testing.assert_array_equal(
            cripser.compute_zigzag(variant, filtration=filtration), expected
        )
    strided = np.repeat(masks, 2, axis=2)[:, :, ::2]
    assert not (strided.flags.c_contiguous or strided.flags.f_contiguous)
    np.testing.assert_array_equal(
        cripser.compute_zigzag(strided, filtration=filtration), expected
    )


@pytest.mark.parametrize("filtration", ["V", "T"])
@pytest.mark.parametrize("connect", ["intersection", "union"])
def test_exhaustive_reduction_does_not_change_result(filtration, connect):
    rng = np.random.default_rng(17)
    for shape in ((9,), (8, 8), (7, 9), (4, 4, 4)):
        masks = random_masks(rng, 6, shape, rng.uniform(0.4, 0.8))
        kwargs = dict(filtration=filtration, connect=connect)
        np.testing.assert_array_equal(
            cripser.compute_zigzag(masks, exhaustive=False, **kwargs),
            cripser.compute_zigzag(masks, **kwargs),
        )


def test_maxdim_truncates_dimensions():
    rng = np.random.default_rng(2)
    masks = random_masks(rng, 4, (5, 5, 4))
    full = cripser.compute_zigzag(masks)
    for maxdim in range(3):
        np.testing.assert_array_equal(
            cripser.compute_zigzag(masks, maxdim=maxdim), full[full[:, 0] <= maxdim]
        )


def test_empty_frames():
    assert cripser.compute_zigzag(np.zeros((3, 4, 4), dtype=bool)).shape == (0, 9)
    masks = np.zeros((3, 4, 4), dtype=bool)
    masks[1, 1:3, 1:3] = True
    assert barcode(masks) == Counter({(0, 2, 2): 1})


@pytest.mark.parametrize(
    "masks, kwargs, error",
    [
        (np.zeros((3, 3, 3)), {}, TypeError),
        (np.zeros(4, dtype=bool), {}, ValueError),
        (np.zeros((2, 2, 2, 2, 2), dtype=bool), {}, ValueError),
        (np.zeros((2, 0, 3), dtype=bool), {}, ValueError),
        (np.zeros((2, 3), dtype=bool), {"filtration": "X"}, ValueError),
        (np.zeros((2, 3), dtype=bool), {"connect": "both"}, ValueError),
        (np.zeros((2, 3), dtype=bool), {"maxdim": -1}, ValueError),
    ],
)
def test_input_validation(masks, kwargs, error):
    with pytest.raises(error):
        cripser.compute_zigzag(masks, **kwargs)
