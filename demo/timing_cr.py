#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark ``demo/cr.py`` through direct Python calls (no CLI subprocess).

This script is intended for regression tracking. It runs persistent homology
computation multiple times, then writes a human-readable CSV containing:
  - Per-run timings
  - One summary row (mean/std/min/max)
  - Metadata (timestamp, git commit, settings, input shape)
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

# ``cr.py`` imports matplotlib at module import time; use a writable cache dir.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import cr


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _resolve_inputs(args: argparse.Namespace) -> tuple[list[str], str]:
    first = args.input[0]
    if os.path.isdir(first):
        if args.imgtype is not None:
            ext = args.imgtype.lstrip(".")
            pattern = os.path.join(first, f"**/*.{ext}")
            files = glob.glob(pattern, recursive=True)
        else:
            files = [os.path.join(first, f) for f in os.listdir(first)]
        input_mode = "directory"
    else:
        files = list(args.input)
        input_mode = "files"

    if not files:
        raise FileNotFoundError("No input files found.")
    return files, input_mode


def _prepare_loader_dependencies(first_file: str) -> None:
    _, ext = os.path.splitext(first_file)
    ext = ext.lower()

    if ext == ".dcm":
        try:
            import pydicom as dicom  # type: ignore
        except ImportError as exc:
            raise ImportError("Install pydicom first: pip install pydicom") from exc
        cr.dicom = dicom

    if ext == ".nrrd":
        try:
            import nrrd  # type: ignore
        except ImportError as exc:
            raise ImportError("Install pynrrd first: pip install pynrrd") from exc
        cr.nrrd = nrrd


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _compute_ph(
    img_arr: np.ndarray,
    args: argparse.Namespace,
    software: str,
    filtration: str,
) -> tuple[np.ndarray, list[int]]:
    if software == "gudhi":
        try:
            import gudhi  # type: ignore
        except ImportError as exc:
            raise ImportError("Install gudhi first: conda install -c conda-forge gudhi") from exc

        if filtration == "V":
            gd = gudhi.CubicalComplex(vertices=img_arr)
        else:
            gd = gudhi.CubicalComplex(top_dimensional_cells=img_arr)

        pers = gd.persistence(2, 0)
        res = np.array([[d, b, de] for d, (b, de) in pers], dtype=float)
        betti = [int(np.count_nonzero(res[:, 0] == i)) for i in range(args.maxdim + 1)]
        return res, betti

    if filtration == "V":
        res = cr.cripser.computePH(
            img_arr,
            maxdim=args.maxdim,
            top_dim=args.top_dim,
            embedded=args.embedded,
        )
    else:
        res = cr.tcripser.computePH(
            img_arr,
            maxdim=args.maxdim,
            top_dim=args.top_dim,
            embedded=args.embedded,
        )

    betti = [int(np.count_nonzero(res[:, 0] == i)) for i in range(args.maxdim + 1)]
    return res, betti


def _write_csv(
    output_csv: Path,
    metadata: dict[str, str],
    combo_results: list[dict[str, object]],
    load_seconds: float,
) -> None:
    fieldnames = [
        "row_type",
        "timestamp_utc",
        "git_commit",
        "input_mode",
        "input_spec",
        "input_shape",
        "software_choices",
        "filtration_choices",
        "software",
        "filtration",
        "maxdim",
        "top_dim",
        "embedded",
        "transform",
        "negative",
        "sort",
        "threshold",
        "threshold_upper_limit",
        "scaling_factor",
        "shift_value",
        "warmup_runs",
        "timed_runs",
        "run_index",
        "load_seconds",
        "elapsed_seconds",
        "mean_seconds",
        "std_seconds",
        "min_seconds",
        "max_seconds",
        "total_pairs",
        "betti_numbers",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for combo in combo_results:
            software = str(combo["software"])
            filtration = str(combo["filtration"])
            run_times = combo["run_times"]
            run_pair_counts = combo["run_pair_counts"]
            run_betti = combo["run_betti"]

            assert isinstance(run_times, list)
            assert isinstance(run_pair_counts, list)
            assert isinstance(run_betti, list)

            avg = mean(run_times)
            stdev = pstdev(run_times) if len(run_times) > 1 else 0.0
            min_t = min(run_times)
            max_t = max(run_times)

            for idx, elapsed in enumerate(run_times, start=1):
                writer.writerow(
                    {
                        **metadata,
                        "software": software,
                        "filtration": filtration,
                        "row_type": "run",
                        "run_index": str(idx),
                        "load_seconds": _format_float(load_seconds),
                        "elapsed_seconds": _format_float(elapsed),
                        "mean_seconds": "",
                        "std_seconds": "",
                        "min_seconds": "",
                        "max_seconds": "",
                        "total_pairs": str(run_pair_counts[idx - 1]),
                        "betti_numbers": "|".join(map(str, run_betti[idx - 1])),
                    }
                )

            writer.writerow(
                {
                    **metadata,
                    "software": software,
                    "filtration": filtration,
                    "row_type": "summary",
                    "run_index": "",
                    "load_seconds": _format_float(load_seconds),
                    "elapsed_seconds": "",
                    "mean_seconds": _format_float(avg),
                    "std_seconds": _format_float(stdev),
                    "min_seconds": _format_float(min_t),
                    "max_seconds": _format_float(max_t),
                    "total_pairs": str(run_pair_counts[-1]),
                    "betti_numbers": "|".join(map(str, run_betti[-1])),
                }
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Timing test for demo/cr.py through direct Python calls.",
    )
    parser.add_argument(
        "input",
        nargs="+",
        help="Input file(s) or one directory, same semantics as demo/cr.py",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="timing_cr.csv",
        help="Output CSV path (default: timing_cr.csv)",
    )
    parser.add_argument("--runs", type=int, default=5, help="Number of timed runs (default: 5)")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs (default: 1)")

    parser.add_argument(
        "--software",
        nargs="+",
        choices=["cubicalripser", "gudhi"],
        default=["cubicalripser", "gudhi"],
        help="One or more backends (default: cubicalripser gudhi)",
    )
    parser.add_argument(
        "--filtration",
        "-f",
        nargs="+",
        choices=["V", "T"],
        default=["V", "T"],
        help="One or more filtrations (default: V T)",
    )
    parser.add_argument("--maxdim", "-m", type=int, default=3)
    parser.add_argument("--top_dim", action="store_true")
    parser.add_argument("--embedded", "-e", action="store_true")

    parser.add_argument("--negative", "-n", action="store_true")
    parser.add_argument("--sort", "-s", action="store_true")
    parser.add_argument(
        "--transform",
        "-tr",
        choices=[
            None,
            "distance",
            "signed_distance",
            "distance_inv",
            "signed_distance_inv",
            "radial",
            "radial_inv",
            "geodesic",
            "geodesic_inv",
            "upward",
            "downward",
        ],
        default=None,
    )
    parser.add_argument("--shift_value", "-sv", type=float, default=None)
    parser.add_argument("--scaling_factor", "-sf", type=float, default=1.0)
    parser.add_argument("--threshold", "-th", type=float, default=None)
    parser.add_argument("--threshold_upper_limit", "-thu", type=float, default=None)
    parser.add_argument("--origin", type=int, nargs="*", default=(0, 0, 0))
    parser.add_argument("--origin_mask", "-om", type=str, default=None)
    parser.add_argument("--imgtype", "-it", type=str, default=None)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.runs <= 0:
        parser.error("--runs must be >= 1")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")

    files, input_mode = _resolve_inputs(args)
    _prepare_loader_dependencies(files[0])

    # ``cr.load_vol`` references ``cr.args`` for geodesic options.
    cr.args = argparse.Namespace(origin=args.origin, origin_mask=args.origin_mask)

    load_start = time.perf_counter()
    img_arr, _ = cr.load_vol(
        files,
        transform=args.transform,
        shift_value=args.shift_value,
        threshold=args.threshold,
        threshold_upper_limit=args.threshold_upper_limit,
        scaling_factor=args.scaling_factor,
        negative=args.negative,
        sort=args.sort,
        origin=tuple(args.origin),
    )
    load_seconds = time.perf_counter() - load_start

    software_choices = _unique(args.software)
    filtration_choices = _unique(args.filtration)
    combos: list[tuple[str, str]] = [
        (software, filtration)
        for software in software_choices
        for filtration in filtration_choices
    ]

    combo_results: list[dict[str, object]] = []
    for software, filtration in combos:
        for _ in range(args.warmup):
            _compute_ph(img_arr, args, software=software, filtration=filtration)

        run_times: list[float] = []
        run_pair_counts: list[int] = []
        run_betti: list[list[int]] = []
        for _ in range(args.runs):
            start = time.perf_counter()
            res, betti = _compute_ph(img_arr, args, software=software, filtration=filtration)
            elapsed = time.perf_counter() - start

            run_times.append(elapsed)
            run_pair_counts.append(int(res.shape[0]))
            run_betti.append(betti)

        combo_results.append(
            {
                "software": software,
                "filtration": filtration,
                "run_times": run_times,
                "run_pair_counts": run_pair_counts,
                "run_betti": run_betti,
            }
        )

    repo_root = Path(__file__).resolve().parent.parent
    output_csv = Path(args.output).expanduser().resolve()
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_commit(repo_root),
        "input_mode": input_mode,
        "input_spec": ";".join(args.input),
        "input_shape": "x".join(map(str, img_arr.shape)),
        "software_choices": "|".join(software_choices),
        "filtration_choices": "|".join(filtration_choices),
        "software": "",
        "filtration": "",
        "maxdim": str(args.maxdim),
        "top_dim": str(bool(args.top_dim)),
        "embedded": str(bool(args.embedded)),
        "transform": str(args.transform),
        "negative": str(bool(args.negative)),
        "sort": str(bool(args.sort)),
        "threshold": "" if args.threshold is None else str(args.threshold),
        "threshold_upper_limit": "" if args.threshold_upper_limit is None else str(args.threshold_upper_limit),
        "scaling_factor": str(args.scaling_factor),
        "shift_value": "" if args.shift_value is None else str(args.shift_value),
        "warmup_runs": str(args.warmup),
        "timed_runs": str(args.runs),
    }
    _write_csv(output_csv, metadata, combo_results, load_seconds)

    print(f"input shape: {img_arr.shape}")
    print(f"load time: {_format_float(load_seconds)} sec")
    for combo in combo_results:
        software = str(combo["software"])
        filtration = str(combo["filtration"])
        run_times = combo["run_times"]
        assert isinstance(run_times, list)
        print(
            f"[{software}/{filtration}] compute time (sec): "
            f"mean={_format_float(mean(run_times))}, "
            f"std={_format_float(pstdev(run_times) if len(run_times) > 1 else 0.0)}, "
            f"min={_format_float(min(run_times))}, "
            f"max={_format_float(max(run_times))}"
        )
    total_rows = len(combo_results) * (args.runs + 1)
    print(
        f"rows written: {total_rows} "
        f"({len(combo_results)} combos x ({args.runs} runs + 1 summary))"
    )
    print(f"output csv: {output_csv}")


if __name__ == "__main__":
    main()
