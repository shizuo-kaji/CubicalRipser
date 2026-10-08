# Development and validation

[Manual](README.md) · Previous: [PyTorch](torch.md)

## Repository map

| Path | Purpose |
| --- | --- |
| `src/` | Shared C++ persistence core, bindings, CLI, and legacy Makefile |
| `cripser/` | Python wrappers, I/O, plots, vectorization, and PyTorch integration |
| `tests/` | Correctness, memory, layout, threading, and helper tests |
| `demo/` | Tutorial notebook, conversion scripts, and local benchmark tools |
| `sample/` | Benchmark volumes such as `bonsai128.npy`, created on demand (not tracked) |
| `docs/` | This manual |

Read [AGENTS.md](../AGENTS.md) before modifying the core. In particular, preserve
persistence pairs, birth/death values, coordinate conventions, V/T duality,
C/Fortran layout handling, and the absence of representative-tracking overhead
when representatives are disabled.

## Build and test

Prepare the [development environment](installation.md#build-from-source), then:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
python -m pip install -e . --no-build-isolation
python -m pytest
```

Rebuild Python extensions after changes under `src/`. Run the full suite after
any change to `src/` or `cripser/`: the C++ core is shared by all entry points.
Tests of CLI-only options run the programs in `build/`, or in the directory
named by `CRIPSER_CLI_DIR`, and are skipped when they are absent.
Optional tests may require the packages in the
[dependency table](installation.md#optional-dependencies).

Coverage includes known 3D/4D topology, Alexander duality, C/Fortran array
layouts, creator/destroyer locations, input limits, memory behavior, threading,
representative-cycle boundaries, GUDHI comparisons, and Python helpers.

## Timing and memory regression checks

Measure performance-sensitive changes before and after on the same nontrivial
input and configuration. Use Release binaries, repeat measurements, and record
both runtime and peak memory. Preserve barcode correctness as well as speed.

The standard CLI comparison with `demo/compare_gudhi.py` is below. A missing
`sample/bonsai128.npy` is created from
[`cripser.datasets.fetch("bonsai")`](python-api.md#downloaded-volumes) at
stride 2 (`bonsai256` at full resolution; other volume names likewise):

```bash
python demo/compare_gudhi.py \
  --methods cli \
  --sample-datasets sample/bonsai128.npy \
  --cubicalripser-bin build/cubicalripser \
  --tcubicalripser-bin build/tcubicalripser \
  --runs 5 --warmup 1 \
  --csv-out demo/logs/timing_check.csv
```

When a reference CSV exists, add
`--reference-csv demo/logs/reference_timing.csv --max-slowdown 1.10 --fail-on-regression`.
Use `--methods cripser gudhi cli` to include the Python and GUDHI paths.
`--json-out PATH` records detailed results; `--output-mode tmpcsv` includes
temporary CSV output in CLI timings instead of suppressing file creation.

Run logs are written to `demo/logs/`, which is not tracked; they record the
host name and command line. Reference logs and local `demo/COMPARE.md` /
`improvements.md` may not be included in every checkout or source
distribution. Check their availability before running these
commands; use the script's `--help` for its options. Local improvement notes
include abandoned experiments, so verify claims against current code.

Avoid committing build artifacts, compiled extensions, generated benchmark
logs, or large locally generated sample arrays. Report actual before/after
numbers for code changes; for documentation-only changes, state that no
performance-sensitive path was touched.
