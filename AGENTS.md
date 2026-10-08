# AGENTS.md

Instructions for AI coding agents (Claude Code, Codex, etc.) working in this repository.
Read this before making any change, and re-check the **Non-negotiables** section before
touching anything under `src/`, `cripser/`, or `demo/compare_gudhi.py`.

## What this project is

CubicalRipser computes persistent homology of 1D/2D/3D/4D cubical complexes (time series,
images, volumes). The core is C++ (`src/`), exposed to Python via nanobind
(`src/cubicalripser_pybind.cpp` → `cripser/_cripser*.abi3.so`, `cripser/tcripser*.abi3.so`),
plus a pure-Python layer (`cripser/*.py`) for I/O, vectorization, plotting, and a
differentiable PyTorch wrapper. Two CLI binaries (`cubicalripser`, `tcubicalripser`) are
built from the same C++ core via CMake.

This is numerical/algorithmic infrastructure, not application code: it is used inside
research pipelines and downstream libraries (see [PyTorch integration](docs/torch.md)),
often on large volumes (hundreds of MB+ arrays). **Correctness of the persistence
computation and its performance characteristics are the product.**

## Non-negotiables when modifying code

These apply to any change touching the C++ core (`src/`), the pybind/nanobind glue
(`src/cubicalripser_pybind.*`), or hot paths in `cripser/*.py`:

1. **Do not silently change numerical/algorithmic behavior.** Persistence pairs, birth/death
   values, creator/destroyer coordinates, and cycle representatives have a precise, documented
   meaning (see [output semantics](docs/concepts.md)). If a change
   alters any output for existing tests or sample data, that is a regression unless it is the
   explicit goal of the task — flag it explicitly, don't let it pass quietly.
2. **Never regress speed or memory usage.** This library exists because it is faster and
   leaner than the general-purpose alternatives it's benchmarked against
   (see [related software](docs/related-software.md)). Before proposing or landing a change to hot code:
   - Understand the current complexity/allocation pattern (caching in `compute_pairs.cpp`,
     union-find in `union_find.h`, radix sort in `radix_sort.h`, the dense grid representation
     in `dense_cubical_grids*.cpp`) before changing it.
   - Prefer changes that avoid new heap allocations, extra passes over the grid, or extra
     copies of the (potentially huge) input array in hot loops.
   - When a change plausibly affects performance, run a timing comparison
     (`demo/compare_gudhi.py`, see below) before/after on at least one non-trivial sample
     (e.g. `sample/bonsai128.npy` or `sample/bonsai256.npy`) and report the numbers. Don't
     assert "this should be faster/should not matter" without measuring — surprising
     regressions in this codebase have historically come from innocuous-looking changes
     (allocation patterns, cache thresholds, sort algorithm choice).
   - If a genuine speed/memory tradeoff is unavoidable, surface it explicitly to the user
     with numbers; do not decide unilaterally that it's acceptable.
3. **Never assume array memory layout.** CubicalRipser supports both C- and Fortran-contiguous
   NumPy arrays; v0.0.15 fixed a real correctness bug from assuming `C_CONTIGUOUS`. Any code
   touching raw buffers must handle both layouts (or explicitly and visibly reject one).
4. **Preserve the V/T-construction and Alexander-duality relationships.** `cubicalripser` vs
   `tcubicalripser`, and `--embedded`, encode specific mathematical dualities
   ([V and T constructions](docs/concepts.md#v-and-t-constructions)). Do not "simplify" or
   unify code paths in ways that break this correspondence
   without discussing it first.
5. **Representative-cycle computation (`representatives.cpp`) is opt-in and must stay
   zero-cost when disabled.** It intentionally trades speed for extra bookkeeping only when
   `representatives=True` is requested, and is incompatible with `top_dim=True`. Don't let
   changes here leak cost into the default (no-representatives) path.

## Before you start

- Skim `README.md` first, then consult the [manual](docs/README.md) for behavior, CLI
  flags, output format, and terminology (creator/destroyer, V/T-construction,
  embedded/Alexander duality). Don't guess at semantics; if something is ambiguous, check
  the relevant manual chapter and test file before changing behavior.
- Check `improvements.md` for known planned work / open issues before assuming a gap is
  unnoticed.
- For anything beyond a trivial fix, state your plan (what you intend to change, and how you
  will verify correctness *and* performance) before editing C++ hot paths.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Produces `build/cubicalripser` (V-construction) and `build/tcubicalripser` (T-construction).
**Always build `Release`** (`-O3`/`-O2`) when measuring performance — a debug build's timings
are meaningless for regression checks.

Rebuild the Python extension after any change under `src/`:

```bash
pip install -e . --no-build-isolation
```

(Legacy path: `cd src && make all` — keep in sync with CMake if you touch build config, but
CMake is the canonical build.)

## Test

```bash
pytest
```

`tests/` covers correctness (dimension-by-dimension checks against known topology in
`test_3d_hole.py`, `test_4d_hole.py`, `test_alexander.py`), Fortran/C layout handling, memory
behavior (`test_cripser_memory.py`), representative cycles, vectorization, and comparison
against GUDHI (`test_compare_gudhi.py`). **Run the full suite after any change to `src/` or
`cripser/`, not just tests that look related** — the C++ core is shared across every binding
and CLI entry point, so a change can break something non-obviously.

For any change that touches performance-sensitive code, additionally run a timing comparison:

```bash
python demo/compare_gudhi.py \
  --methods cli \
  --sample-datasets sample/bonsai128.npy \
  --cubicalripser-bin build/cubicalripser \
  --tcubicalripser-bin build/tcubicalripser \
  --runs 5 --warmup 1 \
  --csv-out demo/logs/timing_check.csv
```

Compare against `demo/logs/reference_timing.csv` with `--reference-csv ... --max-slowdown 1.10
--fail-on-regression` when one exists. See `demo/COMPARE.md` for details. Report the actual
before/after numbers to the user rather than a qualitative impression.

## Code conventions

- C++17 (CMake enforces this for nanobind compatibility); keep new code compiling cleanly
  under both GCC/Clang and MSVC (see `if(MSVC)` branch in `CMakeLists.txt`) — no
  compiler-specific extensions without a guarded fallback.
- Match existing naming/style in the file you're editing rather than introducing a new
  convention; this codebase mixes an older C++ style (raw arrays, manual memory management
  in places) with newer nanobind/Python-facing code — don't "modernize" style incidentally
  while fixing something unrelated.
- Python: follow the conventions already in `cripser/*.py` (type hints where present, numpy
  docstring style). Run `pytest` as the correctness gate; there is no enforced linter/formatter
  configured in this repo — don't introduce one without asking.
- Don't add abstractions, config options, or generality the task doesn't need. Three similar
  lines of numerical code is better than a premature template/abstraction in a hot loop.

## Scope discipline

- Don't refactor unrelated code, rename things, or "clean up" style while doing a bug fix or
  feature addition — each of those is a separate reviewable change in a codebase where subtle
  correctness and performance properties are easy to break invisibly.
- Don't add error handling/validation for inputs that can't occur (e.g. internal buffers whose
  size invariants are established elsewhere) — validate only at real boundaries (Python-facing
  API surface, CLI argument parsing, file I/O).
- If you find a genuine bug or risk outside the current task's scope, report it instead of
  fixing it inline.

## Commit / PR practices

- Follow the existing log style (`git log`): short imperative prefix + summary, e.g.
  `fix: 1d tcripser`, `change: sorting`, `add: cycle representative`.
- In the PR/commit description, state explicitly: what correctness property was preserved
  (which tests confirm it), and what the performance impact is (numbers, or "no
  performance-sensitive path touched" if genuinely true).
- Never commit build artifacts (`build/`, `*.so`, `*.egg-info`, sample data you generated for
  local testing) — check `git status` before staging.

## Releasing a version

Pushing a tag publishes a release: `.github/workflows/build.yml` builds the sdist and the
wheels (cibuildwheel on Linux, macOS, Windows) and uploads them to PyPI (`cripser`) through
the `pypi` environment, which has no approval step. Pull requests and manual runs
(`workflow_dispatch`) build without uploading. PyPI never accepts a version number twice.

**Do not bump the version, tag, or push without the user's explicit go-ahead for that
release in the current conversation.** Prepare steps 1–4 locally, show the user the release
commit and the planned tag, and wait.

1. Start from `main`, up to date with `origin/main` (`git fetch`, `git status`), with
   nothing unintended staged.
2. Verify: run the full `pytest` with `CRIPSER_CLI_DIR` pointing at a fresh Release build so
   the CLI tests run, and the timing check above if `src/` or hot paths in `cripser/` changed
   since the last tag (`git diff --stat <last tag> -- src cripser`).
3. Bump `version` in `pyproject.toml` — the only place it is set (`setup.py` passes it to
   CMake, which compiles it into `cripser.__version__`). Use the next patch version
   (`0.0.N` → `0.0.N+1`) unless the user says otherwise; check the latest with
   `pip index versions cripser`. In `docs/release-notes.md`, rename the `**Unreleased**`
   entry to `**vX.Y.Z**`.
4. Commit as `release: vX.Y.Z - <summary>`, with the correctness and performance statement
   required above.
5. After the go-ahead, push `main` (`git push origin main`), then check the wheel build
   before tagging: `gh workflow run build.yml --ref main` and `gh run watch`. This catches
   platform-specific failures (v0.0.36 failed on MSVC after it was tagged) without publishing.
6. Tag the release commit and push the tag; tags use the `v` prefix:
   `git tag -a vX.Y.Z -m "Release vX.Y.Z"` and `git push origin vX.Y.Z`.
7. Watch the tag run (`gh run list --workflow build.yml --limit 1`, then `gh run watch <id>`)
   and confirm the upload: `pip index versions cripser` lists `X.Y.Z`, and in a fresh
   environment `pip install cripser==X.Y.Z` gives `cripser.__version__ == "X.Y.Z"`.
8. If the tag run fails before the upload job, fix the problem on `main`; then, with the
   user's go-ahead, move the tag (`git push origin :refs/tags/vX.Y.Z`, `git tag -d vX.Y.Z`,
   re-tag the fixed commit, push). If any file of `X.Y.Z` reached PyPI, do not reuse the
   number: release the fix as the next version.

## Data & platform notes

- `sample/` holds only the large benchmark volumes (`bonsai128.npy`, `bonsai256.npy`; not
  tracked). `demo/compare_gudhi.py` creates them when missing from
  `cripser.datasets.fetch("bonsai")`, which downloads and caches real 3D/4D volumes
  (SHA-512 checked). Don't add binary fixtures; generate test and example inputs on the
  fly with `cripser.datasets` (noise, Gaussian random fields, and sphere / torus /
  point-set distance fields with known topology — see `tests/test_datasets.py`). Tests
  must not need the network; the download test runs only with `CRIPSER_TEST_DOWNLOADS=1`.
- Wheel builds target multiple Python versions and OSes (`pyproject.toml`
  `[tool.cibuildwheel]`) with an `abi3` extension (nanobind, Python ≥ 3.9) — avoid
  version-specific or platform-specific C API usage in the pybind layer.
