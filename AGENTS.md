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

## Data & platform notes

- Sample arrays under `sample/` can be large (`bonsai256.npy`, etc.) — don't casually add new
  large binary fixtures; prefer small synthetic arrays with known ground-truth topology for
  new tests (see how `test_3d_hole.py` constructs its input).
- Wheel builds target multiple Python versions and OSes (`pyproject.toml`
  `[tool.cibuildwheel]`) with an `abi3` extension (nanobind, Python ≥ 3.9) — avoid
  version-specific or platform-specific C API usage in the pybind layer.
