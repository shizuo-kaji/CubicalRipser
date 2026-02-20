#!/usr/bin/env python3
"""Benchmark CLI binaries and/or Python module compute_ph with reference comparison."""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_DATASETS = ["bonsai128", "bonsai256", "rand4d"]
DEFAULT_BINARIES = ["cubicalripser", "tcubicalripser"]
DEFAULT_FILTRATIONS = ["V", "T"]


@dataclass
class SummaryRow:
    binary: str
    dataset: str
    mean_seconds: float
    std_seconds: float
    min_seconds: float
    max_seconds: float
    reference_mean_seconds: float | None
    slowdown_ratio: float | None
    status: str


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


def _resolve_binary_path(name: str, args: argparse.Namespace) -> Path:
    if name == "cubicalripser":
        return Path(args.cubicalripser_bin).expanduser().resolve()
    if name == "tcubicalripser":
        return Path(args.tcubicalripser_bin).expanduser().resolve()
    raise ValueError(f"Unknown binary name: {name}")


def _resolve_dataset_path(dataset: str, sample_dir: Path) -> Path:
    candidate = Path(dataset)
    if candidate.suffix == ".npy":
        if not candidate.is_absolute():
            candidate = (sample_dir.parent / candidate).resolve()
        else:
            candidate = candidate.resolve()
        return candidate

    if "/" in dataset:
        return (sample_dir.parent / dataset).resolve()

    return (sample_dir / f"{dataset}.npy").resolve()


def _display_path(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return resolved.name


def _read_reference(reference_csv: Path) -> dict[tuple[str, str], float]:
    ref: dict[tuple[str, str], float] = {}
    with reference_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("row_type") != "summary":
                continue
            binary = (row.get("binary") or "").strip()
            dataset = (row.get("dataset") or "").strip()
            mean_str = (row.get("mean_seconds") or "").strip()
            if not binary or not dataset or not mean_str:
                continue
            ref[(binary, dataset)] = float(mean_str)
    return ref


def _run_once_cli(
    binary_path: Path,
    input_path: Path,
    output_mode: str,
    temp_dir: Path,
    maxdim: int | None,
) -> float:
    if output_mode == "none":
        output_arg = "none"
    else:
        output_arg = str(temp_dir / f"{binary_path.name}_{input_path.stem}_timing.csv")

    cmd = [str(binary_path), "--output", output_arg]
    if maxdim is not None:
        cmd.extend(["--maxdim", str(maxdim)])
    cmd.append(str(input_path))

    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start

    if proc.returncode != 0:
        display_cmd = [binary_path.name, "--output", output_arg]
        if maxdim is not None:
            display_cmd.extend(["--maxdim", str(maxdim)])
        display_cmd.append(input_path.name)
        message = (
            f"Command failed ({proc.returncode}): {' '.join(display_cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
        raise RuntimeError(message)

    return elapsed


def _run_once_python(
    cripser_module,
    arr,
    filtration: str,
    maxdim: int | None,
    location: str,
) -> float:
    kwargs = {"filtration": filtration, "location": location}
    if maxdim is not None:
        kwargs["maxdim"] = maxdim

    start = time.perf_counter()
    _ = cripser_module.compute_ph(arr, **kwargs)
    elapsed = time.perf_counter() - start
    return elapsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark cubicalripser/tcubicalripser and/or cripser.compute_ph, "
            "then optionally compare against a reference timing CSV."
        ),
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
        help=(
            "Dataset stems (e.g., bonsai128) or .npy paths. "
            "Default: bonsai128 bonsai256 rand4d"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["cli", "python", "all"],
        default="cli",
        help="Benchmark CLI binaries, python module, or both (default: cli)",
    )
    parser.add_argument(
        "--binaries",
        nargs="+",
        choices=DEFAULT_BINARIES,
        default=DEFAULT_BINARIES,
        help="CLI binaries to benchmark (default: cubicalripser tcubicalripser)",
    )
    parser.add_argument(
        "--python-filtrations",
        nargs="+",
        choices=DEFAULT_FILTRATIONS,
        default=DEFAULT_FILTRATIONS,
        help="Filtrations for cripser.compute_ph in python mode (default: V T)",
    )
    parser.add_argument(
        "--python-location",
        default="yes",
        help="location argument for cripser.compute_ph in python mode (default: yes)",
    )
    parser.add_argument(
        "--sample-dir",
        default="sample",
        help="Directory used to resolve dataset stems (default: sample)",
    )
    parser.add_argument(
        "--cubicalripser-bin",
        default="build/cubicalripser",
        help="Path to cubicalripser binary (default: build/cubicalripser)",
    )
    parser.add_argument(
        "--tcubicalripser-bin",
        default="build/tcubicalripser",
        help="Path to tcubicalripser binary (default: build/tcubicalripser)",
    )
    parser.add_argument("--runs", type=int, default=3, help="Timed runs per case (default: 3)")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per case (default: 1)")
    parser.add_argument(
        "--output",
        "-o",
        default="timing_cr.csv",
        help="Output CSV path (default: timing_cr.csv)",
    )
    parser.add_argument(
        "--output-mode",
        choices=["none", "tmpcsv"],
        default="none",
        help="In CLI mode, benchmark with --output none (default) or temporary CSV files",
    )
    parser.add_argument(
        "--maxdim",
        type=int,
        default=None,
        help="Optional maxdim for both CLI and python compute_ph",
    )

    parser.add_argument(
        "--reference-csv",
        default=None,
        help="Reference timing CSV (produced by this script) for slowdown comparison",
    )
    parser.add_argument(
        "--max-slowdown",
        type=float,
        default=1.10,
        help="Allowed slowdown ratio vs reference mean (default: 1.10)",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero if any case exceeds --max-slowdown",
    )
    parser.add_argument(
        "--fail-on-missing-reference",
        action="store_true",
        help="Exit non-zero if a case is missing in --reference-csv",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.runs <= 0:
        parser.error("--runs must be >= 1")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.max_slowdown <= 0:
        parser.error("--max-slowdown must be > 0")

    repo_root = Path(__file__).resolve().parent.parent.parent
    sample_dir = (repo_root / args.sample_dir).resolve()

    reference_map: dict[tuple[str, str], float] = {}
    reference_csv_path: Path | None = None
    if args.reference_csv is not None:
        reference_csv_path = Path(args.reference_csv).expanduser().resolve()
        if not reference_csv_path.exists():
            raise FileNotFoundError(f"Reference CSV not found: {reference_csv_path}")
        reference_map = _read_reference(reference_csv_path)

    dataset_paths: list[tuple[str, Path]] = []
    for dataset in args.datasets:
        path = _resolve_dataset_path(dataset, sample_dir)
        if not path.exists():
            raise FileNotFoundError(f"Input not found: {_display_path(path, repo_root)}")
        dataset_paths.append((path.stem, path))

    binary_paths: dict[str, Path] = {}
    binary_display_paths: dict[str, str] = {}
    if args.mode in {"cli", "all"}:
        for name in args.binaries:
            binary_path = _resolve_binary_path(name, args)
            if not binary_path.exists():
                raise FileNotFoundError(f"Binary not found: {_display_path(binary_path, repo_root)}")
            binary_paths[name] = binary_path
            binary_display_paths[name] = _display_path(binary_path, repo_root)

    np = None
    cripser_module = None
    dataset_arrays: dict[str, object] = {}
    if args.mode in {"python", "all"}:
        import numpy as np_import
        import cripser as cripser_import

        np = np_import
        cripser_module = cripser_import
        for dataset_name, input_path in dataset_paths:
            dataset_arrays[dataset_name] = np.load(input_path)

    rows: list[dict[str, str]] = []
    summary_rows: list[SummaryRow] = []

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    git_commit = _git_commit(repo_root)

    with tempfile.TemporaryDirectory(prefix="timing_cr_") as tmp:
        temp_dir = Path(tmp)

        if args.mode in {"cli", "all"}:
            for binary_name in args.binaries:
                binary_path = binary_paths[binary_name]
                binary_display_path = binary_display_paths[binary_name]
                for dataset_name, input_path in dataset_paths:
                    input_display_path = _display_path(input_path, repo_root)
                    for _ in range(args.warmup):
                        _run_once_cli(binary_path, input_path, args.output_mode, temp_dir, args.maxdim)

                    run_times: list[float] = []
                    for run_idx in range(1, args.runs + 1):
                        elapsed = _run_once_cli(binary_path, input_path, args.output_mode, temp_dir, args.maxdim)
                        run_times.append(elapsed)
                        rows.append(
                            {
                                "timestamp_utc": timestamp,
                                "git_commit": git_commit,
                                "binary": binary_name,
                                "dataset": dataset_name,
                                "row_type": "run",
                                "run_index": str(run_idx),
                                "elapsed_seconds": _format_float(elapsed),
                                "mean_seconds": "",
                                "std_seconds": "",
                                "min_seconds": "",
                                "max_seconds": "",
                                "reference_mean_seconds": "",
                                "slowdown_ratio": "",
                                "status": "",
                                "runs": str(args.runs),
                                "warmup": str(args.warmup),
                                "maxdim": "" if args.maxdim is None else str(args.maxdim),
                                "binary_path": binary_display_path,
                                "input_path": input_display_path,
                                "output_mode": args.output_mode,
                                "reference_csv": "" if reference_csv_path is None else str(reference_csv_path),
                            }
                        )

                    mean_seconds = statistics.mean(run_times)
                    std_seconds = statistics.pstdev(run_times) if len(run_times) > 1 else 0.0
                    min_seconds = min(run_times)
                    max_seconds = max(run_times)

                    ref_mean = reference_map.get((binary_name, dataset_name))
                    ratio: float | None = None
                    status = "NO_REF"
                    if ref_mean is not None:
                        ratio = mean_seconds / ref_mean if ref_mean > 0 else None
                        if ratio is not None:
                            status = "PASS" if ratio <= args.max_slowdown else "FAIL"

                    summary_rows.append(
                        SummaryRow(
                            binary=binary_name,
                            dataset=dataset_name,
                            mean_seconds=mean_seconds,
                            std_seconds=std_seconds,
                            min_seconds=min_seconds,
                            max_seconds=max_seconds,
                            reference_mean_seconds=ref_mean,
                            slowdown_ratio=ratio,
                            status=status,
                        )
                    )

                    rows.append(
                        {
                            "timestamp_utc": timestamp,
                            "git_commit": git_commit,
                            "binary": binary_name,
                            "dataset": dataset_name,
                            "row_type": "summary",
                            "run_index": "",
                            "elapsed_seconds": "",
                            "mean_seconds": _format_float(mean_seconds),
                            "std_seconds": _format_float(std_seconds),
                            "min_seconds": _format_float(min_seconds),
                            "max_seconds": _format_float(max_seconds),
                            "reference_mean_seconds": _format_float(ref_mean),
                            "slowdown_ratio": _format_float(ratio),
                            "status": status,
                            "runs": str(args.runs),
                            "warmup": str(args.warmup),
                            "maxdim": "" if args.maxdim is None else str(args.maxdim),
                            "binary_path": binary_display_path,
                            "input_path": input_display_path,
                            "output_mode": args.output_mode,
                            "reference_csv": "" if reference_csv_path is None else str(reference_csv_path),
                        }
                    )

        if args.mode in {"python", "all"}:
            assert cripser_module is not None
            for filtration in args.python_filtrations:
                runner_name = f"py_compute_ph_{filtration}"
                runner_desc = f"python:cripser.compute_ph(filtration={filtration})"
                for dataset_name, input_path in dataset_paths:
                    input_display_path = _display_path(input_path, repo_root)
                    arr = dataset_arrays[dataset_name]

                    for _ in range(args.warmup):
                        _run_once_python(
                            cripser_module,
                            arr,
                            filtration=filtration,
                            maxdim=args.maxdim,
                            location=args.python_location,
                        )

                    run_times = []
                    for run_idx in range(1, args.runs + 1):
                        elapsed = _run_once_python(
                            cripser_module,
                            arr,
                            filtration=filtration,
                            maxdim=args.maxdim,
                            location=args.python_location,
                        )
                        run_times.append(elapsed)
                        rows.append(
                            {
                                "timestamp_utc": timestamp,
                                "git_commit": git_commit,
                                "binary": runner_name,
                                "dataset": dataset_name,
                                "row_type": "run",
                                "run_index": str(run_idx),
                                "elapsed_seconds": _format_float(elapsed),
                                "mean_seconds": "",
                                "std_seconds": "",
                                "min_seconds": "",
                                "max_seconds": "",
                                "reference_mean_seconds": "",
                                "slowdown_ratio": "",
                                "status": "",
                                "runs": str(args.runs),
                                "warmup": str(args.warmup),
                                "maxdim": "" if args.maxdim is None else str(args.maxdim),
                                "binary_path": runner_desc,
                                "input_path": input_display_path,
                                "output_mode": "python",
                                "reference_csv": "" if reference_csv_path is None else str(reference_csv_path),
                            }
                        )

                    mean_seconds = statistics.mean(run_times)
                    std_seconds = statistics.pstdev(run_times) if len(run_times) > 1 else 0.0
                    min_seconds = min(run_times)
                    max_seconds = max(run_times)

                    ref_mean = reference_map.get((runner_name, dataset_name))
                    ratio: float | None = None
                    status = "NO_REF"
                    if ref_mean is not None:
                        ratio = mean_seconds / ref_mean if ref_mean > 0 else None
                        if ratio is not None:
                            status = "PASS" if ratio <= args.max_slowdown else "FAIL"

                    summary_rows.append(
                        SummaryRow(
                            binary=runner_name,
                            dataset=dataset_name,
                            mean_seconds=mean_seconds,
                            std_seconds=std_seconds,
                            min_seconds=min_seconds,
                            max_seconds=max_seconds,
                            reference_mean_seconds=ref_mean,
                            slowdown_ratio=ratio,
                            status=status,
                        )
                    )

                    rows.append(
                        {
                            "timestamp_utc": timestamp,
                            "git_commit": git_commit,
                            "binary": runner_name,
                            "dataset": dataset_name,
                            "row_type": "summary",
                            "run_index": "",
                            "elapsed_seconds": "",
                            "mean_seconds": _format_float(mean_seconds),
                            "std_seconds": _format_float(std_seconds),
                            "min_seconds": _format_float(min_seconds),
                            "max_seconds": _format_float(max_seconds),
                            "reference_mean_seconds": _format_float(ref_mean),
                            "slowdown_ratio": _format_float(ratio),
                            "status": status,
                            "runs": str(args.runs),
                            "warmup": str(args.warmup),
                            "maxdim": "" if args.maxdim is None else str(args.maxdim),
                            "binary_path": runner_desc,
                            "input_path": input_display_path,
                            "output_mode": "python",
                            "reference_csv": "" if reference_csv_path is None else str(reference_csv_path),
                        }
                    )

    output_csv = Path(args.output).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
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

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("binary,dataset,mean_seconds,std_seconds,min_seconds,max_seconds,ref_mean,ratio,status")
    for s in summary_rows:
        print(
            f"{s.binary},{s.dataset},{_format_float(s.mean_seconds)},{_format_float(s.std_seconds)},"
            f"{_format_float(s.min_seconds)},{_format_float(s.max_seconds)},"
            f"{_format_float(s.reference_mean_seconds)},{_format_float(s.slowdown_ratio)},{s.status}"
        )
    print(f"output csv: {output_csv}")

    failures = [s for s in summary_rows if s.status == "FAIL"]
    missing = [s for s in summary_rows if s.status == "NO_REF"]

    if args.fail_on_regression and failures:
        print(f"FAILED: {len(failures)} regression(s) exceeded max slowdown {args.max_slowdown:.3f}")
        return 1

    if args.fail_on_missing_reference and missing:
        print(f"FAILED: {len(missing)} case(s) had no matching reference row")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
