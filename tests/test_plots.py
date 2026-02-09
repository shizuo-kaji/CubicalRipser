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
    out_ax = cripser.plot_diagrams(diagrams, labels=["H0", "H1"], ax=ax, show=False)
    assert out_ax is ax
    assert out_ax.get_xlabel() == "Birth"
    assert out_ax.get_ylabel() == "Death"
    plt.close(fig)


def test_plot_diagrams_rejects_bad_shapes():
    with pytest.raises(ValueError):
        cripser.plot_diagrams(np.array([1.0, 2.0, 3.0]), show=False)
