#!/usr/bin/env python3
"""Compare 1D/2D/3D/4D V/T paths against GUDHI for correctness, speed, and memory."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import functools
import importlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class CaseSpec:
    name: str
    filtration: str
    benchmark_shape: tuple[int, ...]
    benchmark_seed: int
    maxdim: int


@dataclass
class BenchmarkResult:
    case: str
    filtration: str
    shape: tuple[int, ...]
    runs: int
    ours_times_ms: list[float]
    gudhi_times_ms: list[float]
    gudhi_sklearn_times_ms: list[float]
    ours_peak_rss_mib: list[float]
    gudhi_peak_rss_mib: list[float]
    gudhi_sklearn_peak_rss_mib: list[float]
    ours_median_ms: float
    gudhi_median_ms: float
    gudhi_sklearn_median_ms: float
    ours_median_peak_rss_mib: float
    gudhi_median_peak_rss_mib: float
    gudhi_sklearn_median_peak_rss_mib: float
    speed_ratio_ours_over_gudhi: float
    speed_ratio_ours_over_gudhi_sklearn: float
    memory_ratio_ours_over_gudhi: float
    memory_ratio_ours_over_gudhi_sklearn: float
    pair_count_ours: int | None
    pair_count_gudhi: int | None
    pair_count_gudhi_sklearn: int | None
    match_ours_vs_gudhi: bool | None
    match_ours_vs_gudhi_sklearn: bool | None
    location_match_ours_vs_gudhi: bool | None
    location_match_ours_vs_gudhi_sklearn: bool | None
    location_note_ours_vs_gudhi: str | None
    location_note_ours_vs_gudhi_sklearn: str | None


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    make_uint: Callable[[tuple[int, int]], np.ndarray]


@dataclass
class DatasetBenchmarkResult:
    dataset: str
    dataset_path: str | None
    filtration: str
    input_mode: str
    cli_output_mode: str | None
    raw_dtype: str
    shape: tuple[int, ...]
    runs: int
    maxdim: int
    pair_count_ours: int | None
    pair_count_gudhi: int | None
    pair_count_gudhi_sklearn: int | None
    pair_count_cli: int | None
    match_ours_vs_gudhi: bool | None
    match_ours_vs_gudhi_sklearn: bool | None
    match_ours_vs_cli: bool | None
    location_match_ours_vs_gudhi: bool | None
    location_match_ours_vs_gudhi_sklearn: bool | None
    location_match_ours_vs_cli: bool | None
    location_note_ours_vs_gudhi: str | None
    location_note_ours_vs_gudhi_sklearn: str | None
    location_note_ours_vs_cli: str | None
    ours_times_ms: list[float]
    gudhi_times_ms: list[float]
    gudhi_sklearn_times_ms: list[float]
    cli_times_ms: list[float]
    ours_median_ms: float
    gudhi_median_ms: float
    gudhi_sklearn_median_ms: float
    cli_median_ms: float
    speed_ratio_ours_over_gudhi: float
    speed_ratio_ours_over_gudhi_sklearn: float
    speed_ratio_ours_over_cli: float


@dataclass
class ReferenceTimingRow:
    binary: str
    dataset: str
    row_type: str
    run_index: str
    elapsed_seconds: float | None
    mean_seconds: float | None
    std_seconds: float | None
    min_seconds: float | None
    max_seconds: float | None
    reference_mean_seconds: float | None
    slowdown_ratio: float | None
    status: str
    runs: int
    warmup: int
    maxdim: int | None
    binary_path: str
    input_path: str
    output_mode: str


PAIR_ATOL = 1e-6
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "logs"
CRIPSER_OPTIONS_ENV = "CRIPSER_BENCH_OPTIONS"


def _cripser_options_from_env() -> dict[str, object]:
    raw = os.environ.get(CRIPSER_OPTIONS_ENV)
    if not raw:
        return {}
    return json.loads(raw)


def _cli_option_args_from_env() -> list[str]:
    options = _cripser_options_from_env()
    args: list[str] = []
    if options.get("vector_working_column") is True:
        args.append("--vector-working-column")
    if options.get("explicit_clearing") is False:
        args.append("--no-explicit-clearing")
    return args


CASES = [
    CaseSpec(
        name="1D-V",
        filtration="V",
        benchmark_shape=(262_144,),
        benchmark_seed=101,
        maxdim=0,
    ),
    CaseSpec(
        name="1D-T",
        filtration="T",
        benchmark_shape=(262_144,),
        benchmark_seed=103,
        maxdim=0,
    ),
    CaseSpec(
        name="2D-V",
        filtration="V",
        benchmark_shape=(1024, 1024),
        benchmark_seed=107,
        maxdim=1,
    ),
    CaseSpec(
        name="2D-T",
        filtration="T",
        benchmark_shape=(1024, 1024),
        benchmark_seed=109,
        maxdim=1,
    ),
    CaseSpec(
        name="3D-V",
        filtration="V",
        benchmark_shape=(64, 64, 64),
        benchmark_seed=113,
        maxdim=2,
    ),
    CaseSpec(
        name="3D-T",
        filtration="T",
        benchmark_shape=(64, 64, 64),
        benchmark_seed=127,
        maxdim=2,
    ),
    CaseSpec(
        name="4D-V",
        filtration="V",
        benchmark_shape=(20, 20, 20, 20),
        benchmark_seed=131,
        maxdim=3,
    ),
    CaseSpec(
        name="4D-T",
        filtration="T",
        benchmark_shape=(20, 20, 20, 20),
        benchmark_seed=137,
        maxdim=3,
    ),
]


def _make_random_uint8(shape: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def _make_gaussian_rf_uint8(shape: tuple[int, int], seed: int, sigma: float = 3.0) -> np.ndarray:
    import torch

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    h, w = shape
    noise = torch.randn((1, 1, h, w), generator=gen, dtype=torch.float32)
    k = 15
    t = torch.arange(k, dtype=torch.float32) - k // 2
    g = torch.exp(-0.5 * (t / sigma) ** 2)
    g2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
    g2d /= g2d.sum()
    out = torch.nn.functional.conv2d(noise, g2d, padding=k // 2).squeeze().numpy()
    lo = float(out.min())
    hi = float(out.max())
    scaled = (out - lo) / (hi - lo + 1e-8)
    return np.rint(255.0 * scaled).astype(np.uint8)


def _make_checkerboard_uint8(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    xs = np.arange(h)
    ys = np.arange(w)
    xe = (xs % 2 == 0)[:, None]
    ye = (ys % 2 == 0)[None, :]
    out = np.full((h, w), 128, dtype=np.uint8)
    out[xe & ye] = 0
    out[~xe & ~ye] = 255
    return out


def _make_skimage_uint8(name: str, shape: tuple[int, int]) -> np.ndarray:
    import skimage.color
    import skimage.data

    img = getattr(skimage.data, name)()
    if img.ndim == 3:
        img = skimage.color.rgb2gray(img)
        img = np.rint(255.0 * img).astype(np.uint8)
    elif img.dtype != np.uint8:
        lo = float(img.min())
        hi = float(img.max())
        img = np.rint(255.0 * ((img.astype(np.float64) - lo) / (hi - lo + 1e-8))).astype(np.uint8)
    h, w = shape
    rh = (h + img.shape[0] - 1) // img.shape[0]
    rw = (w + img.shape[1] - 1) // img.shape[1]
    return np.tile(img, (rh, rw))[:h, :w].copy()


DATASET_SPECS = [
    DatasetSpec("random", lambda shape: _make_random_uint8(shape, 2001)),
    DatasetSpec("gaussian_rf", lambda shape: _make_gaussian_rf_uint8(shape, 2003)),
    DatasetSpec("checkerboard", _make_checkerboard_uint8),
    DatasetSpec("camera", lambda shape: _make_skimage_uint8("camera", shape)),
    DatasetSpec("moon", lambda shape: _make_skimage_uint8("moon", shape)),
    DatasetSpec("coins", lambda shape: _make_skimage_uint8("coins", shape)),
    DatasetSpec("eagle", lambda shape: _make_skimage_uint8("eagle", shape)),
]


def _make_unique_grid(shape: tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.random(shape, dtype=np.float64)
    perturb = np.arange(arr.size, dtype=np.float64).reshape(shape)
    arr += perturb * 1e-12
    return arr


def _compute_ours(arr: np.ndarray, filtration: str, maxdim: int) -> np.ndarray:
    options = _cripser_options_from_env()
    if filtration == "V":
        import cripser

        return cripser.computePH(arr, maxdim=maxdim, **options)
    import tcripser

    return tcripser.computePH(arr, maxdim=maxdim, **options)


def _compute_gudhi(arr: np.ndarray, filtration: str):
    import gudhi as gd

    if filtration == "V":
        cc = gd.CubicalComplex(vertices=arr)
    else:
        cc = gd.CubicalComplex(top_dimensional_cells=arr)
    cc.persistence(homology_coeff_field=2, min_persistence=0)
    return cc


def _compute_gudhi_sklearn(arr: np.ndarray, filtration: str, maxdim: int):
    from gudhi.sklearn.cubical_persistence import CubicalPersistence

    input_type = "vertices" if filtration == "V" else "top_dimensional_cells"
    return CubicalPersistence(
        list(range(maxdim + 1)),
        input_type=input_type,
        homology_coeff_field=2,
        min_persistence=0.0,
    ).fit_transform([arr])[0]


def _sort_diagram(diagram: np.ndarray) -> np.ndarray:
    if diagram.size == 0:
        return diagram.reshape(0, 2)
    return diagram[np.lexsort((diagram[:, 1], diagram[:, 0]))]


def _ours_diagrams(arr: np.ndarray, filtration: str, maxdim: int) -> list[np.ndarray]:
    import cripser

    ph = _compute_ours(arr, filtration, maxdim)
    return [_sort_diagram(dgm) for dgm in cripser.to_gudhi_diagrams(ph, maxdim=maxdim)]


def _gudhi_diagrams(arr: np.ndarray, filtration: str, maxdim: int) -> list[np.ndarray]:
    cc = _compute_gudhi(arr, filtration)
    diagrams = []
    for dim in range(maxdim + 1):
        diagrams.append(_sort_diagram(np.asarray(cc.persistence_intervals_in_dimension(dim), dtype=np.float64)))
    return diagrams


def _gudhi_sklearn_diagrams(arr: np.ndarray, filtration: str, maxdim: int) -> list[np.ndarray]:
    diagrams = []
    raw = _compute_gudhi_sklearn(arr, filtration, maxdim)
    for dim in range(maxdim + 1):
        dgm = raw[dim] if dim < len(raw) else np.empty((0, 2), dtype=np.float64)
        diagrams.append(_sort_diagram(np.asarray(dgm, dtype=np.float64)))
    return diagrams


def _normalize_deaths(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    huge = np.finfo(np.float64).max * 0.5
    finite = np.isfinite(out)
    out[finite & (out >= huge)] = np.inf
    return out


def _coord_dims_for_ndim(ndim: int) -> int:
    return 4 if ndim >= 4 else 3


def _pad_location(coords: tuple[int, ...], coord_dims: int) -> list[float]:
    padded = list(coords[:coord_dims])
    if len(padded) < coord_dims:
        padded.extend([0] * (coord_dims - len(padded)))
    return [float(x) for x in padded]


def _location_from_linear_index(index: int, shape: tuple[int, ...], coord_dims: int) -> list[float]:
    coords = tuple(int(x) for x in np.unravel_index(int(index), shape, order="F"))
    return _pad_location(coords, coord_dims)


def _sort_pair_rows(rows: np.ndarray) -> np.ndarray:
    if rows.size == 0:
        return rows.reshape(0, rows.shape[1] if rows.ndim == 2 else 0)
    keys = tuple(rows[:, col] for col in range(rows.shape[1] - 1, -1, -1))
    return rows[np.lexsort(keys)]


def _diagram_rows(diagrams: list[np.ndarray]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for dim, dgm in enumerate(diagrams):
        if dgm.size == 0:
            continue
        dim_col = np.full((dgm.shape[0], 1), float(dim), dtype=np.float64)
        rows.append(np.hstack([dim_col, np.asarray(dgm, dtype=np.float64)]))
    if not rows:
        return np.empty((0, 3), dtype=np.float64)
    out = np.vstack(rows)
    out[:, 2] = _normalize_deaths(out[:, 2])
    return _sort_pair_rows(out)


def _ours_pair_rows(arr: np.ndarray, filtration: str, maxdim: int) -> np.ndarray:
    rows = np.asarray(_compute_ours(arr, filtration, maxdim), dtype=np.float64)
    if rows.size == 0:
        coord_dims = _coord_dims_for_ndim(arr.ndim)
        return np.empty((0, 3 + 2 * coord_dims), dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    rows = rows.copy()
    rows[:, 2] = _normalize_deaths(rows[:, 2])
    return _sort_pair_rows(rows)


def _gudhi_pair_rows(arr: np.ndarray, filtration: str, maxdim: int) -> np.ndarray:
    cc = _compute_gudhi(arr, filtration)
    if filtration == "V":
        regular_pairs, essential_pairs = cc.vertices_of_persistence_pairs()
    else:
        regular_pairs, essential_pairs = cc.cofaces_of_persistence_pairs()

    coord_dims = _coord_dims_for_ndim(arr.ndim)
    rows: list[list[float]] = []
    zero_loc = [0.0] * coord_dims
    for dim in range(maxdim + 1):
        intervals = np.asarray(cc.persistence_intervals_in_dimension(dim), dtype=np.float64)
        if intervals.size == 0:
            intervals = np.empty((0, 2), dtype=np.float64)
        elif intervals.ndim == 1:
            intervals = intervals.reshape(1, -1)
        reg_dim = (
            np.asarray(regular_pairs[dim], dtype=np.int64).reshape(-1, 2)
            if dim < len(regular_pairs)
            else np.empty((0, 2), dtype=np.int64)
        )
        ess_dim = (
            np.asarray(essential_pairs[dim], dtype=np.int64).reshape(-1)
            if dim < len(essential_pairs)
            else np.empty((0,), dtype=np.int64)
        )
        finite_mask = np.isfinite(intervals[:, 1]) if intervals.size else np.empty((0,), dtype=bool)
        finite_intervals = intervals[finite_mask]
        essential_intervals = intervals[~finite_mask]
        if finite_intervals.shape[0] != reg_dim.shape[0]:
            raise ValueError(
                f"GUDHI regular pair count mismatch: dim={dim}, intervals={finite_intervals.shape[0]}, pairs={reg_dim.shape[0]}"
            )
        if essential_intervals.shape[0] != ess_dim.shape[0]:
            raise ValueError(
                f"GUDHI essential pair count mismatch: dim={dim}, intervals={essential_intervals.shape[0]}, pairs={ess_dim.shape[0]}"
            )
        for interval, pair in zip(finite_intervals, reg_dim, strict=True):
            rows.append(
                [
                    float(dim),
                    float(interval[0]),
                    float(interval[1]),
                    *_location_from_linear_index(int(pair[0]), arr.shape, coord_dims),
                    *_location_from_linear_index(int(pair[1]), arr.shape, coord_dims),
                ]
            )
        for interval, birth_idx in zip(essential_intervals, ess_dim, strict=True):
            rows.append(
                [
                    float(dim),
                    float(interval[0]),
                    np.inf,
                    *_location_from_linear_index(int(birth_idx), arr.shape, coord_dims),
                    *zero_loc,
                ]
            )
    if not rows:
        return np.empty((0, 3 + 2 * coord_dims), dtype=np.float64)
    return _sort_pair_rows(np.asarray(rows, dtype=np.float64))


def _compare_pair_rows(a: np.ndarray, b: np.ndarray, atol: float = PAIR_ATOL) -> tuple[bool, str]:
    if a.shape != b.shape:
        return False, f"shape mismatch ({a.shape} vs {b.shape})"
    if a.size == 0:
        return True, ""
    int_cols = [0] + list(range(3, a.shape[1]))
    a_int = np.rint(a[:, int_cols]).astype(np.int64)
    b_int = np.rint(b[:, int_cols]).astype(np.int64)
    int_diff = np.argwhere(a_int != b_int)
    if int_diff.size:
        row = int(int_diff[0, 0])
        col = int(int_diff[0, 1])
        actual_col = int_cols[col]
        return False, (
            f"integer mismatch at row {row}, col {actual_col} "
            f"({a[row, actual_col]:.17g} vs {b[row, actual_col]:.17g})"
        )
    a_float = a[:, 1:3]
    b_float = b[:, 1:3]
    same_inf = np.isinf(a_float) & np.isinf(b_float)
    a_cmp = np.where(same_inf, 0.0, a_float)
    b_cmp = np.where(same_inf, 0.0, b_float)
    close = np.isclose(a_cmp, b_cmp, rtol=0.0, atol=atol)
    if not np.all(close):
        bad = np.argwhere(~close)[0]
        row = int(bad[0])
        col = int(bad[1]) + 1
        return False, (
            f"float mismatch at row {row}, col {col} "
            f"({a[row, col]:.17g} vs {b[row, col]:.17g}, atol={atol})"
        )
    return True, ""


def _compare_location_rows(a: np.ndarray, b: np.ndarray) -> tuple[bool | None, str | None]:
    if a.shape != b.shape:
        return False, f"shape mismatch ({a.shape} vs {b.shape})"
    if a.ndim != 2 or a.shape[1] <= 3:
        return None, None
    if a.size == 0:
        return True, None
    int_cols = list(range(3, a.shape[1]))
    a_int = np.rint(a[:, int_cols]).astype(np.int64)
    b_int = np.rint(b[:, int_cols]).astype(np.int64)
    int_diff = np.argwhere(a_int != b_int)
    if int_diff.size:
        row = int(int_diff[0, 0])
        col = int(int_diff[0, 1])
        actual_col = int_cols[col]
        return False, (
            f"integer mismatch at row {row}, col {actual_col} "
            f"({a[row, actual_col]:.17g} vs {b[row, actual_col]:.17g})"
        )
    return True, None


def _compare_impl_outputs(
    ref_impl: str,
    ref_rows: np.ndarray,
    cand_impl: str,
    cand_rows: np.ndarray,
    *,
    atol: float = PAIR_ATOL,
) -> dict[str, object]:
    barcode_ok, barcode_note = _compare_pair_rows(ref_rows[:, :3], cand_rows[:, :3], atol=atol)
    location_ok, location_note = _compare_location_rows(ref_rows, cand_rows)
    return {
        "barcode_ok": barcode_ok,
        "barcode_note": barcode_note,
        "location_ok": location_ok,
        "location_note": location_note,
    }


def _location_key(ref_impl: str, cand_impl: str) -> str | None:
    pair = {ref_impl, cand_impl}
    if pair == {"cripser", "gudhi"}:
        return "ours_vs_gudhi"
    if pair == {"cripser", "gudhi_sk"}:
        return "ours_vs_gudhi_sklearn"
    if pair == {"cripser", "cli"}:
        return "ours_vs_cli"
    return None


def _verification_array(arr: np.ndarray) -> np.ndarray:
    arr64 = np.asarray(arr, dtype=np.float64)
    rounded = np.rint(arr64)
    if np.array_equal(arr64, rounded):
        # Integer-valued datasets contain many ties, and GUDHI documents that
        # generator locations may be chosen arbitrarily on equal-valued adjacent
        # cells. Break ties deterministically for the pre-benchmark correctness
        # check only, while keeping the timed runs on the original array.
        step = 1e-7 / max(arr64.size, 1)
        perturb = np.arange(arr64.size, dtype=np.float64).reshape(arr64.shape)
        return arr64 + perturb * step
    return arr64


def _verify_benchmark_outputs(
    arr: np.ndarray,
    filtration: str,
    maxdim: int,
    methods: frozenset[str],
) -> dict[str, object]:
    arr = _verification_array(arr)
    pair_rows: dict[str, np.ndarray] = {}
    if "cripser" in methods:
        pair_rows["cripser"] = _ours_pair_rows(arr, filtration, maxdim)

    gudhi_rows: np.ndarray | None = None
    if "gudhi" in methods or "gudhi_sk" in methods:
        gudhi_rows = _gudhi_pair_rows(arr, filtration, maxdim)
    if "gudhi" in methods and gudhi_rows is not None:
        pair_rows["gudhi"] = gudhi_rows
    if "gudhi_sk" in methods and gudhi_rows is not None:
        skl_diagram_rows = _diagram_rows(_gudhi_sklearn_diagrams(arr, filtration, maxdim))
        gudhi_diagram_rows = _diagram_rows(_gudhi_diagrams(arr, filtration, maxdim))
        ok, note = _compare_pair_rows(skl_diagram_rows, gudhi_diagram_rows, atol=PAIR_ATOL)
        if not ok:
            raise ValueError(
                "gudhi_sklearn diagram mismatch before benchmarking "
                f"(filtration={filtration}, shape={_fmt_shape(arr.shape)}): {note}"
            )
        # CubicalPersistence exposes only diagrams, so reuse GUDHI's generator locations
        # once the diagrams are confirmed to match exactly.
        pair_rows["gudhi_sk"] = gudhi_rows.copy()

    location_results: dict[str, object] = {
        "location_match_ours_vs_gudhi": None,
        "location_match_ours_vs_gudhi_sklearn": None,
        "location_note_ours_vs_gudhi": None,
        "location_note_ours_vs_gudhi_sklearn": None,
        "location_warnings": [],
    }
    impls = list(pair_rows)
    for i, ref_impl in enumerate(impls):
        for cand_impl in impls[i + 1:]:
            result = _compare_impl_outputs(
                ref_impl,
                pair_rows[ref_impl],
                cand_impl,
                pair_rows[cand_impl],
                atol=PAIR_ATOL,
            )
            if not result["barcode_ok"]:
                raise ValueError(
                    f"output mismatch before benchmarking ({ref_impl} vs {cand_impl}, "
                    f"filtration={filtration}, shape={_fmt_shape(arr.shape)}): {result['barcode_note']}"
                )
            key = _location_key(ref_impl, cand_impl)
            if key is not None:
                location_results[f"location_match_{key}"] = result["location_ok"]
                location_results[f"location_note_{key}"] = result["location_note"]
                if result["location_ok"] is False:
                    location_results["location_warnings"].append(
                        f"location mismatch ({ref_impl} vs {cand_impl}, filtration={filtration}, "
                        f"shape={_fmt_shape(arr.shape)}): {result['location_note']}"
                    )

    return {
        "pair_count_ours": len(pair_rows["cripser"]) if "cripser" in pair_rows else None,
        "pair_count_gudhi": len(pair_rows["gudhi"]) if "gudhi" in pair_rows else None,
        "pair_count_gudhi_sklearn": len(pair_rows["gudhi_sk"]) if "gudhi_sk" in pair_rows else None,
        "match_ours_vs_gudhi": (("cripser" in pair_rows) and ("gudhi" in pair_rows)) or None,
        "match_ours_vs_gudhi_sklearn": (("cripser" in pair_rows) and ("gudhi_sk" in pair_rows)) or None,
        **location_results,
    }


def _verify_sample_benchmark_outputs(
    arr: np.ndarray,
    dataset_path: Path,
    filtration: str,
    maxdim: int,
    methods: frozenset[str],
    binary_paths: dict[str, Path],
) -> dict[str, object]:
    arr_ver = _verification_array(arr)
    pair_rows: dict[str, np.ndarray] = {}
    if "cripser" in methods:
        pair_rows["cripser"] = _ours_pair_rows(arr_ver, filtration, maxdim)

    gudhi_rows: np.ndarray | None = None
    if "gudhi" in methods or "gudhi_sk" in methods:
        gudhi_rows = _gudhi_pair_rows(arr_ver, filtration, maxdim)
    if "gudhi" in methods and gudhi_rows is not None:
        pair_rows["gudhi"] = gudhi_rows
    if "gudhi_sk" in methods and gudhi_rows is not None:
        skl_diagram_rows = _diagram_rows(_gudhi_sklearn_diagrams(arr_ver, filtration, maxdim))
        gudhi_diagram_rows = _diagram_rows(_gudhi_diagrams(arr_ver, filtration, maxdim))
        ok, note = _compare_pair_rows(skl_diagram_rows, gudhi_diagram_rows, atol=PAIR_ATOL)
        if not ok:
            raise ValueError(
                "gudhi_sklearn diagram mismatch before benchmarking "
                f"(dataset={dataset_path.name}, filtration={filtration}, shape={_fmt_shape(arr.shape)}): {note}"
            )
        pair_rows["gudhi_sk"] = gudhi_rows.copy()
    if "cli" in methods:
        with tempfile.TemporaryDirectory(prefix="compare_gudhi_cli_verify_") as tmp:
            verify_path = Path(tmp) / dataset_path.name
            np.save(verify_path, arr_ver)
            pair_rows["cli"] = _run_cli_and_load_pairs(
                verify_path,
                filtration,
                maxdim,
                binary_paths,
                Path(tmp),
                ndim=arr.ndim,
            )

    location_results: dict[str, object] = {
        "location_match_ours_vs_gudhi": None,
        "location_match_ours_vs_gudhi_sklearn": None,
        "location_match_ours_vs_cli": None,
        "location_note_ours_vs_gudhi": None,
        "location_note_ours_vs_gudhi_sklearn": None,
        "location_note_ours_vs_cli": None,
        "location_warnings": [],
    }
    impls = list(pair_rows)
    for i, ref_impl in enumerate(impls):
        for cand_impl in impls[i + 1:]:
            result = _compare_impl_outputs(
                ref_impl,
                pair_rows[ref_impl],
                cand_impl,
                pair_rows[cand_impl],
                atol=PAIR_ATOL,
            )
            if not result["barcode_ok"]:
                raise ValueError(
                    f"output mismatch before benchmarking ({ref_impl} vs {cand_impl}, "
                    f"dataset={dataset_path.name}, filtration={filtration}, shape={_fmt_shape(arr.shape)}): {result['barcode_note']}"
                )
            key = _location_key(ref_impl, cand_impl)
            if key is not None:
                location_results[f"location_match_{key}"] = result["location_ok"]
                location_results[f"location_note_{key}"] = result["location_note"]
                if result["location_ok"] is False:
                    location_results["location_warnings"].append(
                        f"location mismatch ({ref_impl} vs {cand_impl}, dataset={dataset_path.name}, "
                        f"filtration={filtration}, shape={_fmt_shape(arr.shape)}): {result['location_note']}"
                    )

    return {
        "pair_count_ours": len(pair_rows["cripser"]) if "cripser" in pair_rows else None,
        "pair_count_gudhi": len(pair_rows["gudhi"]) if "gudhi" in pair_rows else None,
        "pair_count_gudhi_sklearn": len(pair_rows["gudhi_sk"]) if "gudhi_sk" in pair_rows else None,
        "pair_count_cli": len(pair_rows["cli"]) if "cli" in pair_rows else None,
        "match_ours_vs_gudhi": (("cripser" in pair_rows) and ("gudhi" in pair_rows)) or None,
        "match_ours_vs_gudhi_sklearn": (("cripser" in pair_rows) and ("gudhi_sk" in pair_rows)) or None,
        "match_ours_vs_cli": (("cripser" in pair_rows) and ("cli" in pair_rows)) or None,
        **location_results,
    }


def _time_call(fn, arr: np.ndarray, runs: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn(arr)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(arr)
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def _time_call0(fn, runs: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def _peak_rss_mib() -> float:
    if sys.platform == "win32":
        import psutil
        return psutil.Process().memory_info().peak_wset / (1024.0 * 1024.0)
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return peak / (1024.0 * 1024.0)
    return peak / 1024.0


def _lookup_case(name: str) -> CaseSpec:
    for case in CASES:
        if case.name == name:
            return case
    raise KeyError(f"Unknown case: {name}")


def _cases_for_dim(ndim: int) -> list[CaseSpec]:
    prefix = f"{ndim}D-"
    return [c for c in CASES if c.name.startswith(prefix)]


def _worker_payload(case_name: str, impl: str, kind: str, shape_str: str | None = None) -> dict[str, float]:
    case = _lookup_case(case_name)
    if shape_str is not None:
        actual_shape: tuple[int, ...] = tuple(int(x) for x in shape_str.split(","))
    else:
        actual_shape = case.benchmark_shape
    rng = np.random.default_rng(case.benchmark_seed)
    arr = rng.random(actual_shape, dtype=np.float64)

    if impl == "ours":
        fn = lambda: _compute_ours(arr, case.filtration, case.maxdim)
    elif impl == "gudhi":
        fn = lambda: _gudhi_diagrams(arr, case.filtration, case.maxdim)
    elif impl == "gudhi_sklearn":
        fn = lambda: _gudhi_sklearn_diagrams(arr, case.filtration, case.maxdim)
    else:
        raise ValueError(f"Unknown implementation: {impl}")

    t0 = time.perf_counter()
    fn()
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    return {"elapsed_ms": elapsed_ms, "peak_rss_mib": _peak_rss_mib()}


def _run_worker(case: CaseSpec, impl: str, kind: str, shape: tuple[int, ...] | None = None) -> dict[str, float]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-case",
        case.name,
        "--worker-impl", impl,
    ]
    if shape is not None:
        cmd.extend(["--worker-shape", ",".join(str(x) for x in shape)])
    env = os.environ.copy()
    proc = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(proc.stdout)


def _load_cli_pair_rows(csv_path: Path, ndim: int) -> np.ndarray:
    text = csv_path.read_text(encoding="utf-8").strip()
    coord_dims = _coord_dims_for_ndim(ndim)
    if not text:
        return np.empty((0, 3 + 2 * coord_dims), dtype=np.float64)
    rows = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    expected_cols = 3 + 2 * coord_dims
    if rows.shape[1] != expected_cols:
        raise ValueError(
            f"Unexpected CLI output column count in {csv_path}: got {rows.shape[1]}, expected {expected_cols}"
        )
    rows = rows.copy()
    rows[:, 2] = _normalize_deaths(rows[:, 2])
    return _sort_pair_rows(rows)


def _cli_binary_path_for_filtration(filtration: str, binary_paths: dict[str, Path]) -> Path:
    return binary_paths["cubicalripser" if filtration == "V" else "tcubicalripser"]


def _run_cli_and_load_pairs(
    input_path: Path,
    filtration: str,
    maxdim: int,
    binary_paths: dict[str, Path],
    temp_dir: Path,
    ndim: int,
) -> np.ndarray:
    binary_path = _cli_binary_path_for_filtration(filtration, binary_paths)
    out_path = temp_dir / f"{binary_path.name}_{input_path.stem}_{time.time_ns()}.csv"
    cmd = [str(binary_path), "--output", str(out_path), "--location", "yes", "--maxdim", str(maxdim)]
    cmd.extend(_cli_option_args_from_env())
    cmd.append(str(input_path))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"CLI command failed ({proc.returncode}): {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return _load_cli_pair_rows(out_path, ndim=ndim)


def _run_cli_once(
    input_path: Path,
    filtration: str,
    maxdim: int,
    binary_paths: dict[str, Path],
    *,
    output_mode: str,
    temp_dir: Path | None,
) -> None:
    binary_path = _cli_binary_path_for_filtration(filtration, binary_paths)
    if output_mode == "none":
        output_arg = "none"
    else:
        if temp_dir is None:
            raise ValueError("temp_dir is required when output_mode=tmpcsv")
        output_arg = str(temp_dir / f"{binary_path.name}_{input_path.stem}_{time.time_ns()}.csv")
    cmd = [str(binary_path), "--output", output_arg, "--maxdim", str(maxdim)]
    cmd.extend(_cli_option_args_from_env())
    cmd.append(str(input_path))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"CLI command failed ({proc.returncode}): {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


def benchmark(
    case: CaseSpec,
    runs: int,
    warmup: int,
    shape: tuple[int, ...] | None = None,
    methods: frozenset[str] | None = None,
) -> BenchmarkResult:
    if methods is None:
        methods = frozenset({"cripser", "gudhi", "gudhi_sk"})
    effective_methods = methods
    actual_shape = shape if shape is not None else case.benchmark_shape
    rng = np.random.default_rng(case.benchmark_seed)
    arr = rng.random(actual_shape, dtype=np.float64)
    verification = _verify_benchmark_outputs(arr, case.filtration, case.maxdim, effective_methods)

    run_cripser = "cripser" in effective_methods
    run_gudhi = "gudhi" in effective_methods
    run_sk = "gudhi_sk" in effective_methods

    ours_times: list[float] = (
        _time_call(lambda a: _compute_ours(a, case.filtration, case.maxdim), arr, runs, warmup)
        if run_cripser else []
    )
    gudhi_times: list[float] = (
        _time_call(lambda a: _gudhi_diagrams(a, case.filtration, case.maxdim), arr, runs, warmup)
        if run_gudhi else []
    )
    gudhi_sklearn_times: list[float] = (
        _time_call(lambda a: _gudhi_sklearn_diagrams(a, case.filtration, case.maxdim), arr, runs, warmup)
        if run_sk else []
    )

    ours_mem: list[float] = []
    gudhi_mem: list[float] = []
    gudhi_sklearn_mem: list[float] = []
    for _ in range(runs):
        if run_cripser:
            ours_mem.append(_run_worker(case, "ours", "benchmark", shape=actual_shape)["peak_rss_mib"])
        if run_gudhi:
            gudhi_mem.append(_run_worker(case, "gudhi", "benchmark", shape=actual_shape)["peak_rss_mib"])
        if run_sk:
            gudhi_sklearn_mem.append(_run_worker(case, "gudhi_sklearn", "benchmark", shape=actual_shape)["peak_rss_mib"])

    ours_median = _med(ours_times)
    gudhi_median = _med(gudhi_times)
    gudhi_sklearn_median = _med(gudhi_sklearn_times)
    ours_mem_median = _med(ours_mem)
    gudhi_mem_median = _med(gudhi_mem)
    gudhi_sklearn_mem_median = _med(gudhi_sklearn_mem)

    return BenchmarkResult(
        case=case.name,
        filtration=case.filtration,
        shape=actual_shape,
        runs=runs,
        ours_times_ms=ours_times,
        gudhi_times_ms=gudhi_times,
        gudhi_sklearn_times_ms=gudhi_sklearn_times,
        ours_peak_rss_mib=ours_mem,
        gudhi_peak_rss_mib=gudhi_mem,
        gudhi_sklearn_peak_rss_mib=gudhi_sklearn_mem,
        ours_median_ms=ours_median,
        gudhi_median_ms=gudhi_median,
        gudhi_sklearn_median_ms=gudhi_sklearn_median,
        ours_median_peak_rss_mib=ours_mem_median,
        gudhi_median_peak_rss_mib=gudhi_mem_median,
        gudhi_sklearn_median_peak_rss_mib=gudhi_sklearn_mem_median,
        speed_ratio_ours_over_gudhi=_ratio(ours_median, gudhi_median),
        speed_ratio_ours_over_gudhi_sklearn=_ratio(ours_median, gudhi_sklearn_median),
        memory_ratio_ours_over_gudhi=_ratio(ours_mem_median, gudhi_mem_median),
        memory_ratio_ours_over_gudhi_sklearn=_ratio(ours_mem_median, gudhi_sklearn_mem_median),
        pair_count_ours=verification["pair_count_ours"],
        pair_count_gudhi=verification["pair_count_gudhi"],
        pair_count_gudhi_sklearn=verification["pair_count_gudhi_sklearn"],
        match_ours_vs_gudhi=verification["match_ours_vs_gudhi"],
        match_ours_vs_gudhi_sklearn=verification["match_ours_vs_gudhi_sklearn"],
        location_match_ours_vs_gudhi=verification["location_match_ours_vs_gudhi"],
        location_match_ours_vs_gudhi_sklearn=verification["location_match_ours_vs_gudhi_sklearn"],
        location_note_ours_vs_gudhi=verification["location_note_ours_vs_gudhi"],
        location_note_ours_vs_gudhi_sklearn=verification["location_note_ours_vs_gudhi_sklearn"],
    )


def _make_dataset_input(spec: DatasetSpec, shape: tuple[int, int], input_mode: str) -> tuple[np.ndarray, str]:
    raw = spec.make_uint(shape)
    if input_mode == "uint":
        return raw, str(raw.dtype)
    if input_mode == "float64":
        return raw.astype(np.float64), str(raw.dtype)
    raise ValueError(f"Unknown input mode: {input_mode}")


def _med(lst: list[float]) -> float:
    return statistics.median(lst) if lst else float("nan")


def _ratio(a: float, b: float) -> float:
    return a / b if (not math.isnan(a) and not math.isnan(b) and b != 0.0) else float("nan")


def benchmark_dataset(
    spec: DatasetSpec,
    filtration: str,
    input_mode: str,
    shape: tuple[int, int],
    runs: int,
    warmup: int,
    methods: frozenset[str] | None = None,
) -> DatasetBenchmarkResult:
    if methods is None:
        methods = frozenset({"cripser", "gudhi", "gudhi_sk"})
    arr, raw_dtype = _make_dataset_input(spec, shape, input_mode)
    maxdim = 1
    verification = _verify_benchmark_outputs(arr, filtration, maxdim, methods)

    run_cripser = "cripser" in methods
    run_gudhi = "gudhi" in methods
    run_sk = "gudhi_sk" in methods

    ours_times: list[float] = (
        _time_call(lambda a: _compute_ours(a, filtration, maxdim), arr, runs, warmup)
        if run_cripser else []
    )
    gudhi_times: list[float] = (
        _time_call(lambda a: _gudhi_diagrams(a, filtration, maxdim), arr, runs, warmup)
        if run_gudhi else []
    )
    gudhi_sklearn_times: list[float] = (
        _time_call(lambda a: _gudhi_sklearn_diagrams(a, filtration, maxdim), arr, runs, warmup)
        if run_sk else []
    )
    ours_median = _med(ours_times)
    gudhi_median = _med(gudhi_times)
    gudhi_sklearn_median = _med(gudhi_sklearn_times)

    return DatasetBenchmarkResult(
        dataset=spec.name,
        dataset_path=None,
        filtration=filtration,
        input_mode=input_mode,
        cli_output_mode=None,
        raw_dtype=raw_dtype,
        shape=shape,
        runs=runs,
        maxdim=maxdim,
        pair_count_ours=verification["pair_count_ours"],
        pair_count_gudhi=verification["pair_count_gudhi"],
        pair_count_gudhi_sklearn=verification["pair_count_gudhi_sklearn"],
        pair_count_cli=None,
        match_ours_vs_gudhi=verification["match_ours_vs_gudhi"],
        match_ours_vs_gudhi_sklearn=verification["match_ours_vs_gudhi_sklearn"],
        match_ours_vs_cli=None,
        location_match_ours_vs_gudhi=verification["location_match_ours_vs_gudhi"],
        location_match_ours_vs_gudhi_sklearn=verification["location_match_ours_vs_gudhi_sklearn"],
        location_match_ours_vs_cli=None,
        location_note_ours_vs_gudhi=verification["location_note_ours_vs_gudhi"],
        location_note_ours_vs_gudhi_sklearn=verification["location_note_ours_vs_gudhi_sklearn"],
        location_note_ours_vs_cli=None,
        ours_times_ms=ours_times,
        gudhi_times_ms=gudhi_times,
        gudhi_sklearn_times_ms=gudhi_sklearn_times,
        cli_times_ms=[],
        ours_median_ms=ours_median,
        gudhi_median_ms=gudhi_median,
        gudhi_sklearn_median_ms=gudhi_sklearn_median,
        cli_median_ms=float("nan"),
        speed_ratio_ours_over_gudhi=_ratio(ours_median, gudhi_median),
        speed_ratio_ours_over_gudhi_sklearn=_ratio(ours_median, gudhi_sklearn_median),
        speed_ratio_ours_over_cli=float("nan"),
    )


def benchmark_sample_dataset(
    dataset_name: str,
    dataset_path: Path,
    arr: np.ndarray,
    filtration: str,
    maxdim: int,
    runs: int,
    warmup: int,
    methods: frozenset[str],
    binary_paths: dict[str, Path],
    repo_root: Path,
    output_mode: str,
) -> DatasetBenchmarkResult:
    effective_methods = methods
    verification = _verify_sample_benchmark_outputs(arr, dataset_path, filtration, maxdim, effective_methods, binary_paths)

    run_cripser = "cripser" in effective_methods
    run_gudhi = "gudhi" in effective_methods
    run_sk = "gudhi_sk" in effective_methods
    run_cli = "cli" in effective_methods

    ours_times: list[float] = (
        _time_call(lambda a: _compute_ours(a, filtration, maxdim), arr, runs, warmup)
        if run_cripser else []
    )
    gudhi_times: list[float] = (
        _time_call(lambda a: _gudhi_diagrams(a, filtration, maxdim), arr, runs, warmup)
        if run_gudhi else []
    )
    gudhi_sklearn_times: list[float] = (
        _time_call(lambda a: _gudhi_sklearn_diagrams(a, filtration, maxdim), arr, runs, warmup)
        if run_sk else []
    )
    cli_times: list[float] = []
    if run_cli:
        with tempfile.TemporaryDirectory(prefix="compare_gudhi_cli_timing_") as tmp:
            cli_temp_dir = Path(tmp)
            cli_times = _time_call0(
                lambda: _run_cli_once(
                    dataset_path,
                    filtration,
                    maxdim,
                    binary_paths,
                    output_mode=output_mode,
                    temp_dir=cli_temp_dir,
                ),
                runs,
                warmup,
            )

    ours_median = _med(ours_times)
    gudhi_median = _med(gudhi_times)
    gudhi_sklearn_median = _med(gudhi_sklearn_times)
    cli_median = _med(cli_times)

    return DatasetBenchmarkResult(
        dataset=dataset_name,
        dataset_path=_display_path(dataset_path, repo_root),
        filtration=filtration,
        input_mode="file",
        cli_output_mode=output_mode if run_cli else None,
        raw_dtype=str(arr.dtype),
        shape=tuple(int(x) for x in arr.shape),
        runs=runs,
        maxdim=maxdim,
        pair_count_ours=verification["pair_count_ours"],
        pair_count_gudhi=verification["pair_count_gudhi"],
        pair_count_gudhi_sklearn=verification["pair_count_gudhi_sklearn"],
        pair_count_cli=verification["pair_count_cli"],
        match_ours_vs_gudhi=verification["match_ours_vs_gudhi"],
        match_ours_vs_gudhi_sklearn=verification["match_ours_vs_gudhi_sklearn"],
        match_ours_vs_cli=verification["match_ours_vs_cli"],
        location_match_ours_vs_gudhi=verification["location_match_ours_vs_gudhi"],
        location_match_ours_vs_gudhi_sklearn=verification["location_match_ours_vs_gudhi_sklearn"],
        location_match_ours_vs_cli=verification["location_match_ours_vs_cli"],
        location_note_ours_vs_gudhi=verification["location_note_ours_vs_gudhi"],
        location_note_ours_vs_gudhi_sklearn=verification["location_note_ours_vs_gudhi_sklearn"],
        location_note_ours_vs_cli=verification["location_note_ours_vs_cli"],
        ours_times_ms=ours_times,
        gudhi_times_ms=gudhi_times,
        gudhi_sklearn_times_ms=gudhi_sklearn_times,
        cli_times_ms=cli_times,
        ours_median_ms=ours_median,
        gudhi_median_ms=gudhi_median,
        gudhi_sklearn_median_ms=gudhi_sklearn_median,
        cli_median_ms=cli_median,
        speed_ratio_ours_over_gudhi=_ratio(ours_median, gudhi_median),
        speed_ratio_ours_over_gudhi_sklearn=_ratio(ours_median, gudhi_sklearn_median),
        speed_ratio_ours_over_cli=_ratio(ours_median, cli_median),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5, help="Timed runs per case (default: 5)")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs per case (default: 2)")
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write machine-readable JSON results",
    )
    parser.add_argument(
        "--datasize-1d",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="1D benchmark sizes (shape: N). Multiple values run separate experiments. Default: %(default)s (from CASES)",
    )
    parser.add_argument(
        "--datasize-2d",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="2D benchmark sizes (shape: NxN). Also drives dataset benchmarks. Default: %(default)s (from CASES)",
    )
    parser.add_argument(
        "--datasize-3d",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="3D benchmark sizes (shape: NxNxN). Default: %(default)s (from CASES)",
    )
    parser.add_argument(
        "--datasize-4d",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="4D benchmark sizes (shape: NxNxNxN). Default: %(default)s (from CASES)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["cripser", "gudhi", "gudhi_sk", "cli"],
        default=["cripser", "gudhi", "gudhi_sk"],
        metavar="METHOD",
        help="Which implementations to benchmark: cripser gudhi gudhi_sk cli (cli applies to --sample-datasets)",
    )
    parser.add_argument(
        "--vector-working-column",
        action="store_true",
        help="Pass vector_working_column=True to cripser/tcripser",
    )
    parser.add_argument(
        "--no-explicit-clearing",
        action="store_true",
        help="Pass explicit_clearing=False to cripser/tcripser",
    )
    parser.add_argument(
        "--sample-dataset",
        "--sample-datasets",
        dest="sample_datasets",
        nargs="+",
        default=[],
        metavar="DATASET",
        help=(
            "Additional .npy datasets to benchmark as file-backed inputs. "
            "Accepts stems under sample/, file paths, or directories containing .npy files. "
            "A missing sample/bonsai128, sample/bonsai256 (bonsai at stride 2 / 1) or "
            "sample/<name> for a name in cripser.datasets.VOLUMES is downloaded and created."
        ),
    )
    parser.add_argument(
        "--sample-dir",
        default="sample",
        help="Directory used to resolve --sample-datasets stems (default: sample)",
    )
    parser.add_argument(
        "--sample-filtrations",
        nargs="+",
        choices=["V", "T"],
        default=["V", "T"],
        help="Filtrations to run for --sample-datasets (default: V T)",
    )
    parser.add_argument(
        "--maxdim",
        type=int,
        default=None,
        help="Optional maxdim override for --sample-datasets timing/correctness runs",
    )
    parser.add_argument(
        "--cubicalripser-bin",
        default="src/cubicalripser",
        help="Path to V-construction CLI binary used when --methods includes cli (default: src/cubicalripser)",
    )
    parser.add_argument(
        "--tcubicalripser-bin",
        default="src/tcubicalripser",
        help="Path to T-construction CLI binary used when --methods includes cli (default: src/tcubicalripser)",
    )
    parser.add_argument(
        "--output-mode",
        choices=["none", "tmpcsv"],
        default="none",
        help="CLI timing mode for --methods cli on sample datasets (default: none)",
    )
    parser.add_argument(
        "--csv-out",
        default=None,
        help="Optional CSV path for timing summary rows compatible with timing_cr.py",
    )
    parser.add_argument(
        "--reference-csv",
        default=None,
        help="Optional timing reference CSV for slowdown comparison on sample datasets",
    )
    parser.add_argument(
        "--max-slowdown",
        type=float,
        default=1.10,
        help="Allowed slowdown ratio vs --reference-csv mean (default: 1.10)",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero if any sample-dataset timing row exceeds --max-slowdown",
    )
    parser.add_argument(
        "--fail-on-missing-reference",
        action="store_true",
        help="Exit non-zero if a sample-dataset timing row has no matching reference row",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-case", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-impl", choices=("ours", "gudhi", "gudhi_sklearn"), default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-shape", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def _cripser_options_from_args(args: argparse.Namespace) -> dict[str, object]:
    options: dict[str, object] = {}
    if args.vector_working_column:
        options["vector_working_column"] = True
    if args.no_explicit_clearing:
        options["explicit_clearing"] = False
    return options


def _fmt_shape(shape: tuple[int, ...]) -> str:
    return "x".join(str(x) for x in shape)


def _fmt_match(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _fmt_loc_match(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "ok" if value else "mismatch"


def _fmt_bytes_mib(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown"
    return f"{num_bytes / (1024.0 * 1024.0):.1f} MiB"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _run_text_command(args: list[str]) -> str | None:
    try:
        proc = subprocess.run(args, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    text = proc.stdout.strip()
    return text or None


def _resolve_dataset_path(dataset: str, sample_dir: Path) -> Path:
    candidate = Path(dataset)
    if candidate.is_absolute():
        return candidate.resolve()

    repo_root = sample_dir.parent
    candidates: list[Path] = []
    if candidate.suffix == ".npy":
        candidates.append((repo_root / candidate).resolve())
    else:
        candidates.append((sample_dir / f"{dataset}.npy").resolve())
    candidates.append((repo_root / candidate).resolve())

    for resolved in candidates:
        if resolved.exists():
            return resolved
    return candidates[0]


def _dataset_id_from_path(path: Path, repo_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return path.stem
    if rel.parent == Path("sample"):
        return path.stem
    return rel.with_suffix("").as_posix()


# Benchmark volumes created from cripser.datasets when missing: stem -> (volume, stride).
# Any other name in cripser.datasets.VOLUMES is created at full resolution.
_FETCHED_SAMPLES = {
    "bonsai256": ("bonsai", 1),
    "bonsai128": ("bonsai", 2),
}


def _materialize_sample(path: Path, sample_dir: Path) -> bool:
    """Create a missing ``sample/<stem>.npy`` from cripser.datasets, if it is known."""
    from cripser import datasets

    if path.suffix != ".npy" or path.parent != sample_dir.resolve():
        return False
    name, step = _FETCHED_SAMPLES.get(path.stem, (path.stem, 1))
    if name not in datasets.VOLUMES:
        return False
    arr = datasets.fetch(name)
    arr = arr[(slice(None, None, step),) * arr.ndim]
    sample_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    print(f"Created {path} from cripser.datasets.fetch({name!r})", file=sys.stderr)
    return True


def _collect_sample_dataset_paths(
    datasets: list[str],
    sample_dir: Path,
    repo_root: Path,
) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    seen_paths: set[Path] = set()
    used_names: set[str] = set()

    def add_path(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen_paths:
            return
        seen_paths.add(resolved)
        dataset_id = _dataset_id_from_path(resolved, repo_root)
        if dataset_id in used_names:
            dataset_id = resolved.with_suffix("").name
            if dataset_id in used_names:
                dataset_id = resolved.with_suffix("").as_posix()
        used_names.add(dataset_id)
        entries.append((dataset_id, resolved))

    for dataset in datasets:
        dataset_path = _resolve_dataset_path(dataset, sample_dir)
        if not dataset_path.exists() and not _materialize_sample(dataset_path, sample_dir):
            raise FileNotFoundError(f"Sample dataset not found: {_display_path(dataset_path, repo_root)}")
        if dataset_path.is_dir():
            npy_paths = sorted(p for p in dataset_path.rglob("*.npy") if p.is_file())
            if not npy_paths:
                raise FileNotFoundError(
                    f"No .npy datasets found under directory: {_display_path(dataset_path, repo_root)}"
                )
            for npy_path in npy_paths:
                add_path(npy_path)
            continue
        add_path(dataset_path)
    return entries


def _display_path(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved)


def _latest_cli_source_mtime(src_dir: Path) -> float:
    latest = 0.0
    for pattern in ("*.cpp", "*.h"):
        for path in src_dir.glob(pattern):
            latest = max(latest, path.stat().st_mtime)
    return latest


def _ensure_default_cli_binaries_current(
    binary_paths: dict[str, Path],
    repo_root: Path,
) -> None:
    if not binary_paths:
        return
    src_dir = (repo_root / "src").resolve()
    latest_src_mtime = _latest_cli_source_mtime(src_dir)
    default_targets = {
        "cubicalripser": (src_dir / "cubicalripser").resolve(),
        "tcubicalripser": (src_dir / "tcubicalripser").resolve(),
    }
    rebuild_targets: list[str] = []
    for name, binary_path in binary_paths.items():
        default_path = default_targets.get(name)
        if default_path is None or binary_path != default_path:
            continue
        if binary_path.stat().st_mtime < latest_src_mtime:
            rebuild_targets.append(name)

    if not rebuild_targets:
        return

    print(
        "\nCLI binaries are older than src/*.cpp or src/*.h; rebuilding "
        f"{', '.join(rebuild_targets)} via `make -C src ...`.",
        flush=True,
    )
    proc = subprocess.run(
        ["make", "-C", str(src_dir), *rebuild_targets],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to rebuild stale CLI binaries before sample verification.\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    for name in rebuild_targets:
        rebuilt_path = default_targets[name]
        if not rebuilt_path.exists():
            raise FileNotFoundError(
                f"Expected rebuilt CLI binary was not produced: {_display_path(rebuilt_path, repo_root)}"
            )


@functools.lru_cache(maxsize=1)
def _darwin_hardware_overview() -> dict[str, str]:
    text = _run_text_command(["hostinfo"])
    if text is None:
        return {}
    info: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key and value:
                info[key] = value
            continue
        if line.endswith("processors are physically available."):
            info["processors are physically available."] = line
        elif line.endswith("processors are logically available."):
            info["processors are logically available."] = line
    return info


def _parse_memory_string(text: str | None) -> int | None:
    if text is None:
        return None
    parts = text.split()
    if len(parts) < 2:
        return None
    try:
        value = float(parts[0])
    except ValueError:
        return None
    unit = parts[1].upper().rstrip(".")
    scale = {
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
        "KILOBYTES": 1024,
        "MEGABYTES": 1024**2,
        "GIGABYTES": 1024**3,
        "TERABYTES": 1024**4,
    }.get(unit)
    if scale is None:
        return None
    return int(value * scale)


@functools.lru_cache(maxsize=1)
def _physical_memory_bytes() -> int | None:
    if sys.platform == "win32":
        try:
            import psutil
            return psutil.virtual_memory().total
        except ImportError:
            return None
    if sys.platform == "darwin":
        raw = _run_text_command(["sysctl", "-n", "hw.memsize"])
        if raw is not None:
            return int(raw)
        return _parse_memory_string(_darwin_hardware_overview().get("Primary memory available"))
    if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    return None


@functools.lru_cache(maxsize=1)
def _cpu_info() -> dict[str, object]:
    info: dict[str, object] = {
        "model": platform.processor() or platform.machine(),
        "architecture": platform.machine(),
        "logical_cores": os.cpu_count(),
        "physical_cores": None,
    }
    if sys.platform == "darwin":
        hw = _darwin_hardware_overview()
        brand = _run_text_command(["sysctl", "-n", "machdep.cpu.brand_string"])
        physical = _run_text_command(["sysctl", "-n", "hw.physicalcpu"])
        logical = _run_text_command(["sysctl", "-n", "hw.logicalcpu"])
        if brand is not None:
            info["model"] = brand
        elif hw.get("Processor type"):
            info["model"] = hw["Processor type"]
        if physical is not None:
            info["physical_cores"] = int(physical)
        if logical is not None:
            info["logical_cores"] = int(logical)
        physical_hint = hw.get("processors are physically available.")
        if physical_hint and info["physical_cores"] is None:
            try:
                info["physical_cores"] = int(physical_hint.split()[0])
            except ValueError:
                pass
        logical_hint = hw.get("processors are logically available.")
        if logical_hint and info["logical_cores"] is None:
            try:
                info["logical_cores"] = int(logical_hint.split()[0])
            except ValueError:
                pass
    return info


def _module_version(module_name: str, dist_name: str | None = None) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        module = None
    if module is not None:
        version = getattr(module, "__version__", None)
        if version is not None:
            return str(version)
    try:
        return importlib.metadata.version(dist_name or module_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    return _run_text_command(["git", "rev-parse", "HEAD"])


def _default_json_out_path(started_at: dt.datetime) -> Path:
    stamp = started_at.strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"compare_gudhi_{stamp}.json"


def _collect_run_metadata(started_at: dt.datetime, args: argparse.Namespace) -> dict[str, object]:
    ended_at = dt.datetime.now().astimezone()
    python_impl = platform.python_implementation()
    python_build = " ".join(platform.python_build())
    mem_bytes = _physical_memory_bytes()
    return {
        "timestamp": {
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "timezone": started_at.tzname(),
        },
        "system": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "hostname": platform.node(),
            "cpu": _cpu_info(),
            "memory_bytes": mem_bytes,
            "memory_human": _fmt_bytes_mib(mem_bytes),
        },
        "versions": {
            "python": platform.python_version(),
            "python_implementation": python_impl,
            "python_build": python_build,
            "numpy": np.__version__,
            "cripser": _module_version("cripser"),
            "tcripser": _module_version("tcripser", dist_name="cripser"),
            "gudhi": _module_version("gudhi"),
            "scikit_image": _module_version("skimage", dist_name="scikit-image"),
            "torch": _module_version("torch"),
        },
        "git": {
            "commit": _git_commit(),
        },
        "command": {
            "argv": sys.argv,
            "runs": args.runs,
            "warmup": args.warmup,
            "methods": list(args.methods),
            "cripser_options": _cripser_options_from_args(args),
            "sample_datasets": list(args.sample_datasets),
            "sample_filtrations": list(args.sample_filtrations),
            "sample_dir": args.sample_dir,
            "maxdim": args.maxdim,
            "cubicalripser_bin": args.cubicalripser_bin,
            "tcubicalripser_bin": args.tcubicalripser_bin,
            "output_mode": args.output_mode,
            "csv_out": args.csv_out,
            "reference_csv": args.reference_csv,
            "max_slowdown": args.max_slowdown,
        },
    }


def _print_run_metadata(metadata: dict[str, object], out_path: Path) -> None:
    system = metadata["system"]
    cpu = system["cpu"]
    versions = metadata["versions"]
    timestamp = metadata["timestamp"]
    git_info = metadata["git"]
    print("\nRun Metadata")
    print(
        f"started_at={timestamp['started_at']}  timezone={timestamp['timezone']}  "
        f"git_commit={git_info['commit'] or 'unknown'}"
    )
    print(
        f"system={system['system']} {system['release']}  machine={system['machine']}  "
        f"cpu={cpu['model']}  cores={cpu['physical_cores']}/{cpu['logical_cores']}  "
        f"memory={system['memory_human']}"
    )
    print(
        f"versions: python={versions['python']} numpy={versions['numpy']} "
        f"cripser={versions['cripser'] or 'n/a'} tcripser={versions['tcripser'] or 'n/a'} "
        f"gudhi={versions['gudhi'] or 'n/a'} skimage={versions['scikit_image'] or 'n/a'} "
        f"torch={versions['torch'] or 'n/a'}"
    )
    print(f"cripser_options={metadata['command']['cripser_options']}")
    print(f"json_out={out_path}", flush=True)


def _format_float(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.6f}"


def _read_reference_csv(reference_csv: Path) -> dict[tuple[str, str], float]:
    reference_map: dict[tuple[str, str], float] = {}
    with reference_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("row_type") != "summary":
                continue
            binary = (row.get("binary") or "").strip()
            dataset = (row.get("dataset") or "").strip()
            mean_str = (row.get("mean_seconds") or "").strip()
            if not binary or not dataset or not mean_str:
                continue
            reference_map[(binary, dataset)] = float(mean_str)
    return reference_map


def _reference_binary_specs(row: DatasetBenchmarkResult, methods: frozenset[str]) -> list[tuple[str, list[float], str, str]]:
    specs: list[tuple[str, list[float], str, str]] = []
    if "cripser" in methods and row.ours_times_ms:
        specs.append((f"py_compute_ph_{row.filtration}", row.ours_times_ms, f"python:cripser.computePH(filtration={row.filtration})", "python"))
    if "cli" in methods and row.cli_times_ms:
        binary_name = "cubicalripser" if row.filtration == "V" else "tcubicalripser"
        specs.append((binary_name, row.cli_times_ms, binary_name, row.cli_output_mode or "none"))
    return specs


def _make_reference_timing_rows(
    sample_rows: list[DatasetBenchmarkResult],
    *,
    methods: frozenset[str],
    runs: int,
    warmup: int,
    max_slowdown: float,
    reference_map: dict[tuple[str, str], float],
) -> list[ReferenceTimingRow]:
    rows: list[ReferenceTimingRow] = []
    for sample_row in sample_rows:
        for binary, times_ms, binary_path, output_mode in _reference_binary_specs(sample_row, methods):
            ref_mean = reference_map.get((binary, sample_row.dataset))
            for run_index, elapsed_ms in enumerate(times_ms, start=1):
                rows.append(
                    ReferenceTimingRow(
                        binary=binary,
                        dataset=sample_row.dataset,
                        row_type="run",
                        run_index=str(run_index),
                        elapsed_seconds=elapsed_ms / 1e3,
                        mean_seconds=None,
                        std_seconds=None,
                        min_seconds=None,
                        max_seconds=None,
                        reference_mean_seconds=None,
                        slowdown_ratio=None,
                        status="",
                        runs=runs,
                        warmup=warmup,
                        maxdim=sample_row.maxdim,
                        binary_path=binary_path,
                        input_path=sample_row.dataset_path or sample_row.dataset,
                        output_mode=output_mode,
                    )
                )
            mean_seconds = statistics.mean(times_ms) / 1e3
            std_seconds = (statistics.pstdev(times_ms) / 1e3) if len(times_ms) > 1 else 0.0
            min_seconds = min(times_ms) / 1e3
            max_seconds = max(times_ms) / 1e3
            ratio = (mean_seconds / ref_mean) if ref_mean not in (None, 0.0) else None
            status = "NO_REF"
            if ratio is not None:
                status = "PASS" if ratio <= max_slowdown else "FAIL"
            rows.append(
                ReferenceTimingRow(
                    binary=binary,
                    dataset=sample_row.dataset,
                    row_type="summary",
                    run_index="",
                    elapsed_seconds=None,
                    mean_seconds=mean_seconds,
                    std_seconds=std_seconds,
                    min_seconds=min_seconds,
                    max_seconds=max_seconds,
                    reference_mean_seconds=ref_mean,
                    slowdown_ratio=ratio,
                    status=status,
                    runs=runs,
                    warmup=warmup,
                    maxdim=sample_row.maxdim,
                    binary_path=binary_path,
                    input_path=sample_row.dataset_path or sample_row.dataset,
                    output_mode=output_mode,
                )
            )
    return rows


def _write_reference_timing_csv(
    rows: list[ReferenceTimingRow],
    metadata: dict[str, object],
    out_path: Path,
    reference_csv: Path | None,
) -> None:
    fieldnames = [
        "timestamp_utc",
        "git_commit",
        "binary",
        "dataset",
        "row_type",
        "run_index",
        "elapsed_seconds",
        "mean_seconds",
        "std_seconds",
        "min_seconds",
        "max_seconds",
        "reference_mean_seconds",
        "slowdown_ratio",
        "status",
        "runs",
        "warmup",
        "maxdim",
        "binary_path",
        "input_path",
        "output_mode",
        "reference_csv",
    ]
    timestamp_utc = dt.datetime.fromisoformat(str(metadata["timestamp"]["started_at"])).astimezone(dt.timezone.utc)
    git_commit = metadata["git"]["commit"] or "unknown"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "timestamp_utc": timestamp_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "git_commit": git_commit,
                    "binary": row.binary,
                    "dataset": row.dataset,
                    "row_type": row.row_type,
                    "run_index": row.run_index,
                    "elapsed_seconds": _format_float(row.elapsed_seconds),
                    "mean_seconds": _format_float(row.mean_seconds),
                    "std_seconds": _format_float(row.std_seconds),
                    "min_seconds": _format_float(row.min_seconds),
                    "max_seconds": _format_float(row.max_seconds),
                    "reference_mean_seconds": _format_float(row.reference_mean_seconds),
                    "slowdown_ratio": _format_float(row.slowdown_ratio),
                    "status": row.status,
                    "runs": str(row.runs),
                    "warmup": str(row.warmup),
                    "maxdim": "" if row.maxdim is None else str(row.maxdim),
                    "binary_path": row.binary_path,
                    "input_path": row.input_path,
                    "output_mode": row.output_mode,
                    "reference_csv": "" if reference_csv is None else str(reference_csv),
                }
            )


def _print_reference_summary(rows: list[ReferenceTimingRow], csv_out: Path | None) -> None:
    summaries = [row for row in rows if row.row_type == "summary"]
    if not summaries:
        return
    print("\nReference Timing Summary")
    print("binary,dataset,mean_seconds,std_seconds,min_seconds,max_seconds,ref_mean,ratio,status")
    for row in summaries:
        print(
            f"{row.binary},{row.dataset},{_format_float(row.mean_seconds)},{_format_float(row.std_seconds)},"
            f"{_format_float(row.min_seconds)},{_format_float(row.max_seconds)},"
            f"{_format_float(row.reference_mean_seconds)},{_format_float(row.slowdown_ratio)},{row.status}"
        )
    if csv_out is not None:
        print(f"csv_out={csv_out}")


def _build_dim_shapes(args: argparse.Namespace) -> dict[int, list[tuple[int, ...]]]:
    """Return {ndim: [shape, ...]} using CLI overrides or CASES defaults."""
    result: dict[int, list[tuple[int, ...]]] = {}
    for ndim, attr in [(1, "datasize_1d"), (2, "datasize_2d"), (3, "datasize_3d"), (4, "datasize_4d")]:
        sizes = getattr(args, attr, None)
        if sizes:
            result[ndim] = [tuple([s] * ndim) for s in sizes]
        else:
            cases = _cases_for_dim(ndim)
            if cases:
                result[ndim] = [cases[0].benchmark_shape]
    return result


def _print_benchmark_progress(row: BenchmarkResult) -> None:
    print(
        f"[done] benchmark   {row.case} filt={row.filtration} shape={_fmt_shape(row.shape)} "
        f"ours_ms={row.ours_median_ms:.3f} gudhi_ms={row.gudhi_median_ms:.3f} "
        f"gudhi_skl_ms={row.gudhi_sklearn_median_ms:.3f} "
        f"ours_mem={row.ours_median_peak_rss_mib:.1f}MiB "
        f"gudhi_mem={row.gudhi_median_peak_rss_mib:.1f}MiB "
        f"gudhi_skl_mem={row.gudhi_sklearn_median_peak_rss_mib:.1f}MiB "
        f"match={_fmt_match(row.match_ours_vs_gudhi)}/{_fmt_match(row.match_ours_vs_gudhi_sklearn)} "
        f"loc={_fmt_loc_match(row.location_match_ours_vs_gudhi)}/{_fmt_loc_match(row.location_match_ours_vs_gudhi_sklearn)}",
        flush=True,
    )
    for note in [row.location_note_ours_vs_gudhi, row.location_note_ours_vs_gudhi_sklearn]:
        if note:
            print(f"[note] location {note}", flush=True)


def _print_dataset_benchmark_progress(row: DatasetBenchmarkResult) -> None:
    print(
        f"[done] dataset     {row.dataset} filt={row.filtration} mode={row.input_mode} "
        f"shape={_fmt_shape(row.shape)} ours_ms={row.ours_median_ms:.3f} "
        f"gudhi_ms={row.gudhi_median_ms:.3f} gudhi_skl_ms={row.gudhi_sklearn_median_ms:.3f} "
        f"cli_ms={row.cli_median_ms:.3f} "
        f"match={_fmt_match(row.match_ours_vs_gudhi)}/{_fmt_match(row.match_ours_vs_gudhi_sklearn)}/{_fmt_match(row.match_ours_vs_cli)} "
        f"loc={_fmt_loc_match(row.location_match_ours_vs_gudhi)}/{_fmt_loc_match(row.location_match_ours_vs_gudhi_sklearn)}/{_fmt_loc_match(row.location_match_ours_vs_cli)}",
        flush=True,
    )
    for note in [row.location_note_ours_vs_gudhi, row.location_note_ours_vs_gudhi_sklearn, row.location_note_ours_vs_cli]:
        if note:
            print(f"[note] location {note}", flush=True)


def main() -> int:
    args = parse_args()

    if args.runs <= 0:
        raise ValueError("--runs must be >= 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.max_slowdown <= 0:
        raise ValueError("--max-slowdown must be > 0")

    if args.worker:
        payload = _worker_payload(args.worker_case, args.worker_impl, "benchmark", shape_str=args.worker_shape)
        print(json.dumps(payload))
        return 0

    cripser_options = _cripser_options_from_args(args)
    if cripser_options:
        os.environ[CRIPSER_OPTIONS_ENV] = json.dumps(cripser_options, sort_keys=True)
    else:
        os.environ.pop(CRIPSER_OPTIONS_ENV, None)

    started_at = dt.datetime.now().astimezone()
    out_path = Path(args.json_out).resolve() if args.json_out else _default_json_out_path(started_at)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = _collect_run_metadata(started_at, args)

    benchmark_results: list[BenchmarkResult] = []
    dataset_benchmark_results: list[DatasetBenchmarkResult] = []
    sample_dataset_benchmark_results: list[DatasetBenchmarkResult] = []

    methods = frozenset(args.methods)
    repo_root = _repo_root()
    sample_dir = (repo_root / args.sample_dir).resolve()
    sample_dataset_paths = _collect_sample_dataset_paths(args.sample_datasets, sample_dir, repo_root)
    cli_binary_paths: dict[str, Path] = {}
    if "cli" in methods and sample_dataset_paths:
        for name, arg_name in [("cubicalripser", "cubicalripser_bin"), ("tcubicalripser", "tcubicalripser_bin")]:
            binary_path = Path(getattr(args, arg_name)).expanduser().resolve()
            if not binary_path.exists():
                raise FileNotFoundError(f"CLI binary not found: {_display_path(binary_path, repo_root)}")
            cli_binary_paths[name] = binary_path
        _ensure_default_cli_binaries_current(cli_binary_paths, repo_root)
    dim_shapes = _build_dim_shapes(args)
    _print_run_metadata(metadata, out_path)
    total_bench = sum(len(shapes) * len(_cases_for_dim(ndim)) for ndim, shapes in dim_shapes.items())
    print(
        f"\nStarting random benchmarks ({total_bench} case×shape combinations, methods={sorted(methods)})...",
        flush=True,
    )
    for ndim in sorted(dim_shapes):
        for shape in dim_shapes[ndim]:
            for case in _cases_for_dim(ndim):
                print(f"  {case.name} shape={_fmt_shape(shape)} ...", flush=True)
                row = benchmark(case, args.runs, args.warmup, shape=shape, methods=methods)
                benchmark_results.append(row)
                _print_benchmark_progress(row)

    dataset_shapes = dim_shapes.get(2, [])
    if dataset_shapes:
        total_dataset_rows = len(DATASET_SPECS) * 2 * len(dataset_shapes)
        print(
            f"\nStarting 2D dataset benchmarks ({total_dataset_rows} rows, "
            f"{len(dataset_shapes)} shape(s), T-construction, uint+float64, methods={sorted(methods)})...",
            flush=True,
        )
        for dataset_shape in dataset_shapes:
            print(f"  shape={_fmt_shape(dataset_shape)}", flush=True)
            for spec in DATASET_SPECS:
                for filtration in ("T",):
                    for input_mode in ("uint", "float64"):
                        row = benchmark_dataset(
                            spec, filtration, input_mode, dataset_shape,
                            args.runs, args.warmup, methods=methods,
                        )
                        dataset_benchmark_results.append(row)
                        _print_dataset_benchmark_progress(row)

    if sample_dataset_paths:
        total_sample_rows = len(sample_dataset_paths) * len(args.sample_filtrations)
        print(
            f"\nStarting sample dataset benchmarks ({total_sample_rows} rows, methods={sorted(methods)})...",
            flush=True,
        )
        for dataset_name, dataset_path in sample_dataset_paths:
            arr = np.load(dataset_path)
            maxdim = args.maxdim if args.maxdim is not None else (arr.ndim - 1)
            print(
                f"  {dataset_name} path={_display_path(dataset_path, repo_root)} "
                f"shape={_fmt_shape(tuple(int(x) for x in arr.shape))} ...",
                flush=True,
            )
            for filtration in args.sample_filtrations:
                row = benchmark_sample_dataset(
                    dataset_name,
                    dataset_path,
                    arr,
                    filtration,
                    maxdim,
                    args.runs,
                    args.warmup,
                    methods,
                    cli_binary_paths,
                    repo_root,
                    args.output_mode,
                )
                sample_dataset_benchmark_results.append(row)
                _print_dataset_benchmark_progress(row)

    print("\nBenchmark (median ms / peak RSS MiB)")
    print("case   filt  shape                ours_ms   gudhi_ms   gudhi_skl_ms   ours/gudhi   ours/g_skl   ours_mem   gudhi_mem   gudhi_skl_mem  match_cc  match_skl  loc_cc   loc_skl")
    for row in benchmark_results:
        print(
            f"{row.case:5}  {row.filtration:4}  {_fmt_shape(row.shape):18}  "
            f"{row.ours_median_ms:8.3f}  {row.gudhi_median_ms:9.3f}  {row.gudhi_sklearn_median_ms:13.3f}  "
            f"{row.speed_ratio_ours_over_gudhi:10.3f}  {row.speed_ratio_ours_over_gudhi_sklearn:10.3f}  "
            f"{row.ours_median_peak_rss_mib:9.1f}  {row.gudhi_median_peak_rss_mib:10.1f}  {row.gudhi_sklearn_median_peak_rss_mib:14.1f}  "
            f"{_fmt_match(row.match_ours_vs_gudhi):8}  {_fmt_match(row.match_ours_vs_gudhi_sklearn):9}  "
            f"{_fmt_loc_match(row.location_match_ours_vs_gudhi):7}  {_fmt_loc_match(row.location_match_ours_vs_gudhi_sklearn):7}"
        )

    if dataset_benchmark_results:
        print("\n2D Dataset Benchmark (median ms, T-construction datasets from compare_cucube.py)")
        print("dataset         mode     raw_dtype  shape        ours_ms   gudhi_ms   gudhi_skl_ms   cli_ms    ours/gudhi   ours/g_skl   ours/cli   pairs   match_cc  match_skl  match_cli  loc_cc   loc_skl  loc_cli")
        for row in dataset_benchmark_results:
            print(
                f"{row.dataset:14}  {row.input_mode:7}  {row.raw_dtype:9}  {_fmt_shape(row.shape):10}  "
                f"{row.ours_median_ms:8.3f}  {row.gudhi_median_ms:9.3f}  {row.gudhi_sklearn_median_ms:13.3f}  "
                f"{row.cli_median_ms:8.3f}  {row.speed_ratio_ours_over_gudhi:10.3f}  {row.speed_ratio_ours_over_gudhi_sklearn:10.3f}  "
                f"{row.speed_ratio_ours_over_cli:8.3f}  {int(row.pair_count_ours or 0):6d}  {_fmt_match(row.match_ours_vs_gudhi):8}  {_fmt_match(row.match_ours_vs_gudhi_sklearn):9}  {_fmt_match(row.match_ours_vs_cli):9}  "
                f"{_fmt_loc_match(row.location_match_ours_vs_gudhi):7}  {_fmt_loc_match(row.location_match_ours_vs_gudhi_sklearn):7}  {_fmt_loc_match(row.location_match_ours_vs_cli):7}"
            )

    if sample_dataset_benchmark_results:
        print("\nSample Dataset Benchmark (median ms, file-backed inputs under sample/)")
        print("dataset         filt  path                           shape        ours_ms   gudhi_ms   gudhi_skl_ms   cli_ms    ours/gudhi   ours/g_skl   ours/cli   pairs   match_cc  match_skl  match_cli  loc_cc   loc_skl  loc_cli")
        for row in sample_dataset_benchmark_results:
            print(
                f"{row.dataset:14}  {row.filtration:4}  {(row.dataset_path or ''):29.29}  {_fmt_shape(row.shape):10}  "
                f"{row.ours_median_ms:8.3f}  {row.gudhi_median_ms:9.3f}  {row.gudhi_sklearn_median_ms:13.3f}  "
                f"{row.cli_median_ms:8.3f}  {row.speed_ratio_ours_over_gudhi:10.3f}  {row.speed_ratio_ours_over_gudhi_sklearn:10.3f}  "
                f"{row.speed_ratio_ours_over_cli:8.3f}  {int(row.pair_count_ours or 0):6d}  {_fmt_match(row.match_ours_vs_gudhi):8}  {_fmt_match(row.match_ours_vs_gudhi_sklearn):9}  {_fmt_match(row.match_ours_vs_cli):9}  "
                f"{_fmt_loc_match(row.location_match_ours_vs_gudhi):7}  {_fmt_loc_match(row.location_match_ours_vs_gudhi_sklearn):7}  {_fmt_loc_match(row.location_match_ours_vs_cli):7}"
            )

    payload = {
        "metadata": metadata,
        "benchmark": [asdict(row) for row in benchmark_results],
        "dataset_benchmark": [asdict(row) for row in dataset_benchmark_results],
        "sample_dataset_benchmark": [asdict(row) for row in sample_dataset_benchmark_results],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote JSON results to {out_path}")
    reference_map: dict[tuple[str, str], float] = {}
    reference_csv_path: Path | None = None
    if args.reference_csv:
        reference_csv_path = Path(args.reference_csv).expanduser().resolve()
        if not reference_csv_path.exists():
            raise FileNotFoundError(f"Reference CSV not found: {reference_csv_path}")
        reference_map = _read_reference_csv(reference_csv_path)

    reference_rows: list[ReferenceTimingRow] = []
    if sample_dataset_benchmark_results and (args.csv_out or args.reference_csv):
        reference_rows = _make_reference_timing_rows(
            sample_dataset_benchmark_results,
            methods=methods,
            runs=args.runs,
            warmup=args.warmup,
            max_slowdown=args.max_slowdown,
            reference_map=reference_map,
        )
        csv_out_path = Path(args.csv_out).expanduser().resolve() if args.csv_out else None
        if csv_out_path is not None:
            _write_reference_timing_csv(reference_rows, metadata, csv_out_path, reference_csv_path)
        _print_reference_summary(reference_rows, csv_out_path)

    summary_rows = [row for row in reference_rows if row.row_type == "summary"]
    failures = [row for row in summary_rows if row.status == "FAIL"]
    missing = [row for row in summary_rows if row.status == "NO_REF"]
    if args.fail_on_regression and failures:
        print(f"FAILED: {len(failures)} regression(s) exceeded max slowdown {args.max_slowdown:.3f}")
        return 1
    if args.fail_on_missing_reference and missing:
        print(f"FAILED: {len(missing)} case(s) had no matching reference row")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
