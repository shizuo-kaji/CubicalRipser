import gc
import os
import platform
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pytest
import cripser


@dataclass
class MemoryCheckResult:
    """RSS snapshots and growth values in bytes."""

    rss_before: int
    rss_after_phase1: int
    rss_after_phase2: int
    growth_phase1: int
    growth_phase2: int


def _linux_rss_bytes() -> int:
    """Return current RSS in bytes using ``/proc/self/statm`` on Linux."""
    with open("/proc/self/statm", "r", encoding="utf-8") as f:
        fields = f.read().split()
    rss_pages = int(fields[1])
    return rss_pages * os.sysconf("SC_PAGE_SIZE")


def _rss_bytes() -> int:
    """Return current process RSS in bytes (Linux only)."""
    if platform.system() != "Linux":
        raise RuntimeError("RSS check is implemented only for Linux (/proc).")
    return _linux_rss_bytes()


def _run_compute_ph_repeatedly(arr: np.ndarray, repeats: int) -> None:
    """Run ``cripser.computePH`` repeatedly and validate basic output shape."""
    out = None
    for _ in range(repeats):
        out = cripser.computePH(arr, maxdim=2)
    assert out is not None
    assert out.ndim == 2 and out.shape[1] == 9


def _collect_and_measure_rss() -> int:
    gc.collect()
    return _rss_bytes()


def check_computeph_memory_stability(
    phases: Iterable[int] = (10, 20),
    shape: tuple[int, int, int] = (50, 50, 50),
    seed: int = 0,
    max_phase2_growth_mb: float = 4.0,
) -> MemoryCheckResult:
    """Run a two-phase memory check and assert no strong linear RSS growth."""
    p1, p2 = phases
    rng = np.random.default_rng(seed)
    arr = rng.random(shape, dtype=np.float64)

    # Warmup to avoid one-time initialization noise.
    _run_compute_ph_repeatedly(arr, repeats=3)

    rss_before = _collect_and_measure_rss()
    _run_compute_ph_repeatedly(arr, repeats=p1)
    rss_after_phase1 = _collect_and_measure_rss()
    _run_compute_ph_repeatedly(arr, repeats=p2)
    rss_after_phase2 = _collect_and_measure_rss()

    growth_phase1 = rss_after_phase1 - rss_before
    growth_phase2 = rss_after_phase2 - rss_after_phase1

    max_phase2_growth_bytes = int(max_phase2_growth_mb * 1024 * 1024)
    assert growth_phase2 <= max_phase2_growth_bytes, (
        f"Possible memory leak: phase2 RSS growth is {growth_phase2} bytes "
        f"(allowed <= {max_phase2_growth_bytes} bytes). "
        f"phase1={growth_phase1} bytes"
    )

    return MemoryCheckResult(
        rss_before=rss_before,
        rss_after_phase1=rss_after_phase1,
        rss_after_phase2=rss_after_phase2,
        growth_phase1=growth_phase1,
        growth_phase2=growth_phase2,
    )


@pytest.mark.skipif(
    platform.system() != "Linux",
    reason="memory leak RSS check uses /proc and is Linux-only",
)
def test_cripser_computeph_memory_stability():
    result = check_computeph_memory_stability()
    # Basic sanity to make debugging easier if this fails.
    assert result.rss_before > 0
