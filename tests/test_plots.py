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
