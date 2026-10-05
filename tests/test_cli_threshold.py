"""The CLI threshold truncates the filtration.

Computing with --threshold t must give the intervals of the unthresholded
computation that are born below t, with deaths at or above t replaced by t and
no destroyer.  The tests run the CLI programs from $CRIPSER_CLI_DIR (default
build/, as in the development instructions) and are skipped without them.
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

CLI_DIR = Path(os.environ.get("CRIPSER_CLI_DIR", Path(__file__).parent.parent / "build"))
PROGRAMS = {"V": CLI_DIR / "cubicalripser", "T": CLI_DIR / "tcubicalripser"}

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in PROGRAMS.values()),
    reason=f"CLI programs not found in {CLI_DIR}",
)


def _run(tmp_path, arr, filtration, *options):
    src = tmp_path / "input.npy"
    out = tmp_path / "output.csv"
    np.save(src, arr)
    subprocess.run(
        [str(PROGRAMS[filtration]), *options, "--output", str(out), str(src)],
        check=True,
        capture_output=True,
    )
    if out.stat().st_size == 0:
        return np.zeros((0, 9))
    return np.loadtxt(out, delimiter=",", ndmin=2)


def _pairs(table):
    return sorted(map(tuple, table[:, :3]))


@pytest.mark.parametrize("filtration", ["V", "T"])
@pytest.mark.parametrize("shape", [(3, 2), (3, 2, 2)])
def test_every_component_below_the_threshold_is_reported(tmp_path, filtration, shape):
    # Two components, born at 1 and 2, separated by values above the threshold.
    arr = np.full(shape, 9.0)
    arr[0] = 1.0
    arr[2] = 2.0
    table = _run(tmp_path, arr, filtration, "--threshold", "5", "--maxdim", "0")
    assert _pairs(table) == [(0, 1, 5), (0, 2, 5)]
    assert np.all(table[:, 6:9] == -1)  # neither dies


@pytest.mark.parametrize("filtration", ["V", "T"])
@pytest.mark.parametrize("embedded", [False, True])
@pytest.mark.parametrize("shape", [(12, 13), (6, 7, 5)])
@pytest.mark.parametrize("top_dim", [False, True])
def test_threshold_truncates_the_unthresholded_output(
    tmp_path, filtration, embedded, shape, top_dim
):
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 6, shape).astype(np.float64)
    d = arr.ndim
    options = ["--top_dim"] if top_dim else ["--maxdim", str(d - 1)]
    if embedded:
        options.append("--embedded")
    t = -2.5 if embedded else 2.5

    full = _run(tmp_path, arr, filtration, *options)
    expected = full[full[:, 1] < t].copy()
    expected[expected[:, 2] >= t, 2] = t
    table = _run(tmp_path, arr, filtration, *options, "--threshold", str(t))
    assert _pairs(table) == _pairs(expected)

    # Classes alive at the threshold have no destroyer; the others die below it.
    destroyer = table[:, 6:9]
    alive = table[:, 2] == t
    assert np.all(destroyer[alive] == -1)
    assert np.all(destroyer[~alive] >= 0)
    assert np.all(table[:, 2] <= t)
