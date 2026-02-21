#!/usr/bin/env python3
"""Compare CLI and/or Python compute outputs with ground truth snapshots."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATASETS = ["bonsai128", "bonsai256", "rand4d"]
DEFAULT_BINARIES = ["cubicalripser", "tcubicalripser"]
DEFAULT_FILTRATIONS = ["V", "T"]
PYTHON_FILTRATION_TO_REFERENCE = {"V": "cubicalripser", "T": "tcubicalripser"}


@dataclass
class CheckResult:
    binary: str
    dataset: str
    ok: bool
    ref_lines: int
    new_lines: int
    ref_sha1: str
    new_sha1: str
    note: str = ""


def sha1sum(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def line_count(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def run_one(binary_path: Path, input_path: Path, output_path: Path) -> None:
    cmd = [str(binary_path), "--output", str(output_path), str(input_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        message = (
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
        raise RuntimeError(message)


def load_csv_array(path: Path):
    import numpy as np

    arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D CSV array in {path}, got ndim={arr.ndim}")
    return arr


def normalize_death_values(values):
    import numpy as np

    out = np.asarray(values, dtype=np.float64).copy()
    huge = np.finfo(np.float64).max * 0.5
    finite = np.isfinite(out)
    out[finite & (out >= huge)] = np.inf
    return out


def compare_python_against_cli_reference(ref_csv: Path, new_csv: Path, float_atol: float) -> tuple[bool, str]:
    import numpy as np

    ref = load_csv_array(ref_csv)
    new = load_csv_array(new_csv)

    if ref.shape[0] != new.shape[0]:
        return False, f"row mismatch ({ref.shape[0]} vs {new.shape[0]})"
    if ref.shape[1] != new.shape[1]:
        return False, f"column mismatch ({ref.shape[1]} vs {new.shape[1]})"
    if ref.shape[1] < 3:
        return False, f"column mismatch (<3 columns): {ref.shape[1]}"

    ref3 = ref[:, :3]
    new3 = new[:, :3]

    ref_dim = np.rint(ref3[:, 0]).astype(np.int64)
    new_dim = np.rint(new3[:, 0]).astype(np.int64)
    if not np.array_equal(ref_dim, new_dim):
        idx = int(np.flatnonzero(ref_dim != new_dim)[0])
        return False, f"dim mismatch at row {idx} ({ref_dim[idx]} vs {new_dim[idx]})"

    birth_ok = np.isclose(ref3[:, 1], new3[:, 1], rtol=0.0, atol=float_atol)
    if not np.all(birth_ok):
        idx = int(np.flatnonzero(~birth_ok)[0])
        return False, (
            f"birth mismatch at row {idx} "
            f"({ref3[idx, 1]:.17g} vs {new3[idx, 1]:.17g}, atol={float_atol})"
        )

    ref_death = normalize_death_values(ref3[:, 2])
    new_death = normalize_death_values(new3[:, 2])
    ref_inf = np.isinf(ref_death)
    new_inf = np.isinf(new_death)
    if not np.array_equal(ref_inf, new_inf):
        idx = int(np.flatnonzero(ref_inf != new_inf)[0])
        return False, (
            f"infinite-cycle mismatch at row {idx} "
            f"({ref3[idx, 2]:.17g} vs {new3[idx, 2]:.17g})"
        )

    finite = ~ref_inf
    death_ok = np.isclose(ref_death[finite], new_death[finite], rtol=0.0, atol=float_atol)
    if not np.all(death_ok):
        finite_indices = np.flatnonzero(finite)
        bad_local = int(np.flatnonzero(~death_ok)[0])
        idx = int(finite_indices[bad_local])
        return False, (
            f"death mismatch at row {idx} "
            f"({ref3[idx, 2]:.17g} vs {new3[idx, 2]:.17g}, atol={float_atol})"
        )

    if ref.shape[1] > 3:
        ref_coord = ref[:, 3:]
        new_coord = new[:, 3:]

        ref_rounded = np.rint(ref_coord)
        new_rounded = np.rint(new_coord)
        ref_integer_like = np.isfinite(ref_coord) & (np.abs(ref_coord - ref_rounded) <= 1e-9)
        new_integer_like = np.isfinite(new_coord) & (np.abs(new_coord - new_rounded) <= 1e-9)
        if not np.all(ref_integer_like):
            bad = np.argwhere(~ref_integer_like)[0]
            return False, f"non-integer coordinate in reference at row {int(bad[0])}, col {int(bad[1]) + 3}"
        if not np.all(new_integer_like):
            bad = np.argwhere(~new_integer_like)[0]
            return False, f"non-integer coordinate in python output at row {int(bad[0])}, col {int(bad[1]) + 3}"

        ref_coord_int = ref_rounded.astype(np.int64)
        new_coord_int = new_rounded.astype(np.int64)
        if not np.array_equal(ref_coord_int, new_coord_int):
            bad = np.argwhere(ref_coord_int != new_coord_int)[0]
            row = int(bad[0])
            col = int(bad[1]) + 3
            return False, (
                f"coordinate mismatch at row {row}, col {col} "
                f"({ref_coord_int[bad[0], bad[1]]} vs {new_coord_int[bad[0], bad[1]]})"
            )

    return True, ""


def resolve_binary(repo_root: Path, name: str, args: argparse.Namespace) -> Path:
    if name == "cubicalripser":
        return Path(args.cubicalripser_bin).expanduser().resolve()
    if name == "tcubicalripser":
        return Path(args.tcubicalripser_bin).expanduser().resolve()
    raise ValueError(f"Unknown binary name: {name}")


def resolve_dataset(sample_dir: Path, dataset: str) -> tuple[str, Path]:
    candidate = Path(dataset)
    if candidate.suffix == ".npy":
        if not candidate.is_absolute():
            candidate = (sample_dir.parent / candidate).resolve()
        else:
            candidate = candidate.resolve()
        return candidate.stem, candidate

    if "/" in dataset:
        candidate = (sample_dir.parent / dataset).resolve()
        return candidate.stem, candidate

    candidate = (sample_dir / f"{dataset}.npy").resolve()
    return dataset, candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute CSV outputs and compare them to ground-truth CSV files "
            "for future regression checks."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset stems under sample/ (default: bonsai128 bonsai256 rand4d)",
    )
    parser.add_argument(
        "--binaries",
        nargs="+",
        choices=DEFAULT_BINARIES,
        default=DEFAULT_BINARIES,
        help="Binaries to test (default: cubicalripser tcubicalripser)",
    )
    parser.add_argument(
        "--mode",
        choices=["cli", "python", "all"],
        default="cli",
        help="Check CLI binaries, Python module, or both (default: cli)",
    )
    parser.add_argument(
        "--sample-dir",
        default="sample",
        help="Directory containing input .npy files (default: sample)",
    )
    parser.add_argument(
        "--reference-dir",
        default="demo/check/computation",
        help="Reference CSV directory for CLI mode",
    )
    parser.add_argument(
        "--reference-dir-python",
        default="demo/check/computation",
        help=(
            "Reference CSV directory for python mode. "
            "Uses shared CLI references: V->cubicalripser_*.csv, T->tcubicalripser_*.csv"
        ),
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
    parser.add_argument(
        "--python-filtrations",
        nargs="+",
        choices=DEFAULT_FILTRATIONS,
        default=DEFAULT_FILTRATIONS,
        help="Filtrations for cripser.compute_ph in python mode (default: V T)",
    )
    parser.add_argument(
        "--python-maxdim",
        type=int,
        default=None,
        help="Optional maxdim for cripser.compute_ph in python mode",
    )
    parser.add_argument(
        "--python-location",
        default="yes",
        help="location argument for cripser.compute_ph in python mode (default: yes)",
    )
    parser.add_argument(
        "--float-atol",
        type=float,
        default=1e-6,
        help=(
            "Absolute tolerance for python-mode birth/death comparison "
            "(default: 1e-6)"
        ),
    )
    parser.add_argument(
        "--keep-generated",
        action="store_true",
        help="Keep generated CSV files under <reference_dir>/generated",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.float_atol < 0:
        raise ValueError("--float-atol must be >= 0")

    repo_root = Path(__file__).resolve().parent.parent.parent
    sample_dir = (repo_root / args.sample_dir).resolve()
    reference_dir_cli = (repo_root / args.reference_dir).resolve()
    reference_dir_python = (repo_root / args.reference_dir_python).resolve()

    needed_paths = [sample_dir]
    if args.mode in {"cli", "all"}:
        needed_paths.append(reference_dir_cli)
    if args.mode in {"python", "all"}:
        needed_paths.append(reference_dir_python)
    for needed in needed_paths:
        if not needed.exists():
            raise FileNotFoundError(f"Required path not found: {needed}")

    output_dir_cli: Path | None = None
    output_dir_python: Path | None = None
    if args.keep_generated:
        if args.mode in {"cli", "all"}:
            output_dir_cli = reference_dir_cli / "generated"
            output_dir_cli.mkdir(parents=True, exist_ok=True)
        if args.mode in {"python", "all"}:
            output_dir_python = reference_dir_python / "generated"
            output_dir_python.mkdir(parents=True, exist_ok=True)

    results: list[CheckResult] = []
    with tempfile.TemporaryDirectory(prefix="check_computation_") as temp_dir:
        temp_path = Path(temp_dir)

        if args.mode in {"cli", "all"}:
            for binary_name in args.binaries:
                binary_path = resolve_binary(repo_root, binary_name, args)
                if not binary_path.exists():
                    raise FileNotFoundError(f"Binary not found: {binary_path}")

                for dataset in args.datasets:
                    dataset_name, input_path = resolve_dataset(sample_dir, dataset)
                    if not input_path.exists():
                        raise FileNotFoundError(f"Input not found: {input_path}")

                    ref_csv = reference_dir_cli / f"{binary_name}_{dataset_name}.csv"
                    if not ref_csv.exists():
                        raise FileNotFoundError(f"Reference CSV not found: {ref_csv}")

                    new_csv = temp_path / f"{binary_name}_{dataset_name}.csv"
                    run_one(binary_path, input_path, new_csv)

                    if output_dir_cli is not None:
                        (output_dir_cli / new_csv.name).write_bytes(new_csv.read_bytes())

                    ref_sha = sha1sum(ref_csv)
                    new_sha = sha1sum(new_csv)
                    ref_lines = line_count(ref_csv)
                    new_lines = line_count(new_csv)
                    ok = ref_sha == new_sha

                    note = ""
                    if not ok:
                        note = "content mismatch"
                        if ref_lines != new_lines:
                            note = f"line mismatch ({ref_lines} vs {new_lines})"

                    results.append(
                        CheckResult(
                            binary=binary_name,
                            dataset=dataset_name,
                            ok=ok,
                            ref_lines=ref_lines,
                            new_lines=new_lines,
                            ref_sha1=ref_sha,
                            new_sha1=new_sha,
                            note=note,
                        )
                    )

        if args.mode in {"python", "all"}:
            import numpy as np
            import cripser

            for dataset in args.datasets:
                dataset_name, input_path = resolve_dataset(sample_dir, dataset)
                if not input_path.exists():
                    raise FileNotFoundError(f"Input not found: {input_path}")
                arr = np.load(input_path)

                for filtration in args.python_filtrations:
                    runner_name = f"py_compute_ph_{filtration}"
                    reference_binary = PYTHON_FILTRATION_TO_REFERENCE[filtration]
                    ref_csv = reference_dir_python / f"{reference_binary}_{dataset_name}.csv"
                    if not ref_csv.exists():
                        raise FileNotFoundError(f"Reference CSV not found: {ref_csv}")

                    kwargs = {
                        "filtration": filtration,
                        "location": args.python_location,
                    }
                    if args.python_maxdim is not None:
                        kwargs["maxdim"] = args.python_maxdim

                    ph = cripser.compute_ph(arr, **kwargs)

                    new_csv = temp_path / f"{runner_name}_{dataset_name}.csv"
                    np.savetxt(new_csv, ph, delimiter=",", fmt="%.17g")

                    if output_dir_python is not None:
                        (output_dir_python / new_csv.name).write_bytes(new_csv.read_bytes())

                    ref_sha = sha1sum(ref_csv)
                    new_sha = sha1sum(new_csv)
                    ref_lines = line_count(ref_csv)
                    new_lines = line_count(new_csv)
                    ok, note = compare_python_against_cli_reference(ref_csv, new_csv, args.float_atol)

                    if ok:
                        ref_sha = ""
                        new_sha = ""
                        note = (
                            f"full row match (float atol={args.float_atol:g}, "
                            "death inf normalized)"
                        )
                    elif note == "":
                        note = "full-row mismatch"

                    results.append(
                        CheckResult(
                            binary=runner_name,
                            dataset=dataset_name,
                            ok=ok,
                            ref_lines=ref_lines,
                            new_lines=new_lines,
                            ref_sha1=ref_sha,
                            new_sha1=new_sha,
                            note=note,
                        )
                    )

    print("binary,dataset,status,ref_lines,new_lines,ref_sha1,new_sha1,note")
    for r in results:
        print(
            f"{r.binary},{r.dataset},{'OK' if r.ok else 'NG'},"
            f"{r.ref_lines},{r.new_lines},{r.ref_sha1},{r.new_sha1},{r.note}"
        )

    failures = [r for r in results if not r.ok]
    if failures:
        print(f"\nFAILED: {len(failures)} mismatch(es).")
        return 1

    print(f"\nPASS: all {len(results)} comparisons matched ground truth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
