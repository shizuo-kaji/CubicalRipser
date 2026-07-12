import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
import matplotlib.pyplot as plt

import cripser


def test_plot_diagrams_smoke_returns_axis():
    diagrams = [
        np.array([[0.0, 1.0], [0.2, np.inf]], dtype=np.float64),
        np.array([[0.4, 0.9]], dtype=np.float64),
    ]

    fig, ax = plt.subplots()
    cripser.plot_diagrams(diagrams, labels=["H0", "H1"], ax=ax, show=False)
    plt.close(fig)


def test_plot_diagrams_rejects_bad_shapes():
    with pytest.raises(ValueError):
        cripser.plot_diagrams(np.array([1.0, 2.0, 3.0]), show=False)


def test_plot_diagrams_accepts_cripser_nine_column_output():
    ph = np.array(
        [
            [0.0, 0.1, 0.7, 0, 0, 0, 0, 0, 0],
            [0.0, 0.4, np.inf, 0, 0, 0, -1, -1, -1],
            [1.0, 0.6, 0.9, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )

    fig, ax = plt.subplots()
    cripser.plot_diagrams(ph, labels=["H0", "H1"], ax=ax, show=False)
    plt.close(fig)


def test_plot_cycle_overlays_a_planar_cycle_on_an_image():
    image = np.zeros((5, 5), dtype=np.float64)
    cycle = np.array(
        [[1, 1, 0, 0], [2, 1, 0, 1], [1, 2, 0, 0], [1, 1, 0, 1]],
        dtype=np.int64,
    )

    fig, ax = plt.subplots()
    result = cripser.plot_cycle(cycle, image=image, ax=ax, label="cycle", show=False)

    assert result is ax
    assert len(ax.images) == 1
    assert len(ax.collections) == 1
    assert ax.get_legend() is not None
    plt.close(fig)


def test_plot_cycle_can_skip_the_image_overlay():
    cycle = np.array([[1, 1, 0, 0], [2, 1, 0, 1]], dtype=np.int64)

    fig, ax = plt.subplots()
    cripser.plot_cycle(cycle, image=np.zeros((5, 5)), overlay=False, ax=ax, show=False)

    assert len(ax.images) == 0
    assert len(ax.collections) == 1
    plt.close(fig)


def test_plot_cycles_draws_multiple_cycles_without_repeating_the_image():
    cycle_a = np.array([[0, 0, 0, 0], [1, 0, 0, 1]], dtype=np.int64)
    cycle_b = np.array([[2, 2, 0, 0], [3, 2, 0, 1]], dtype=np.int64)

    fig, ax = plt.subplots()
    cripser.plot_cycles(
        [cycle_a, cycle_b],
        image=np.zeros((5, 5)),
        labels=["first", "second"],
        ax=ax,
        show=False,
    )

    assert len(ax.images) == 1
    assert len(ax.collections) == 2
    plt.close(fig)


def test_plot_cycle_rejects_non_planar_or_non_h1_cells():
    with pytest.raises(ValueError, match="Planar H_1"):
        cripser.plot_cycle(np.array([[0, 0, 0, 2]], dtype=np.int64), show=False)
