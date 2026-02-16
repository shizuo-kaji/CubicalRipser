# Check Protocol

## 1) Computation correctness: CLI binaries (CSV exact match)

Ground-truth CSV files (shared by CLI + Python checks) are stored in:

- `demo/check/computation/`

Run the regression check:

```bash
python3 demo/check/check_computation.py
```

This runs:

- `src/cubicalripser` and `src/tcubicalripser`
- on `sample/bonsai128.npy`, `sample/bonsai256.npy`, `sample/rand4d.npy`

and compares generated CSVs against ground truth by SHA1/line counts.

Note:

- Python mode is checked against the same full-row reference output as CLI mode.

Useful options:

```bash
# only one dataset / one binary
python3 demo/check/check_computation.py --datasets rand4d --binaries cubicalripser

# custom binary paths
python3 demo/check/check_computation.py \
  --cubicalripser-bin /path/to/cubicalripser \
  --tcubicalripser-bin /path/to/tcubicalripser
```

## 2) Performance comparison: CLI binaries (timing baseline)

Reference timing CSV is:

- `demo/check/performance/reference_timing.csv`

Run timing and compare against baseline:

```bash
python3 demo/check/timing_cr.py \
  --runs 3 --warmup 1 --output-mode none \
  --reference-csv demo/check/performance/reference_timing.csv \
  --max-slowdown 1.10 \
  --fail-on-regression \
  -o timing_current_vs_reference.csv
```

Interpretation:

- `ratio = current_mean / reference_mean`
- `PASS` if `ratio <= --max-slowdown`, otherwise `FAIL`

Useful options:

```bash
# quick smoke test
python3 demo/check/timing_cr.py rand4d --runs 1 --warmup 0 \
  --reference-csv demo/check/performance/reference_timing.csv

# include temp CSV writing in timing run (instead of --output none)
python3 demo/check/timing_cr.py --output-mode tmpcsv
```

## 3) Computation correctness: Python module (`cripser.compute_ph`)

Python module uses the same reference CSVs as CLI mode:

- `demo/check/computation/cubicalripser_*.csv` for `filtration="V"`
- `demo/check/computation/tcubicalripser_*.csv` for `filtration="T"`

Run the module regression check:

```bash
python3 demo/check/check_computation.py --mode python
```

This runs:

- `cripser.compute_ph(..., filtration=\"V\")`
- `cripser.compute_ph(..., filtration=\"T\")`
- on `sample/bonsai128.npy`, `sample/bonsai256.npy`, `sample/rand4d.npy`

Comparison rule:

- all rows/columns must match
- `dimension` and coordinate columns are matched as integer values
- finite `birth/death` values are matched with absolute tolerance `1e-6` (default)
- "infinite" death values are normalized so `inf` and `1.79769e+308` are treated as equal

Useful options:

```bash
# only one dataset / one filtration
python3 demo/check/check_computation.py --mode python \
  --datasets rand4d --python-filtrations V

# tighten/loosen float tolerance used for birth/death comparison
python3 demo/check/check_computation.py --mode python --float-atol 1e-7

# check both CLI and Python in one run
python3 demo/check/check_computation.py --mode all
```

## 4) Performance comparison: Python module

Python timing baseline CSV is:

- `demo/check/performance/py_reference_timing.csv`

Run timing and compare against Python baseline:

```bash
python3 demo/check/timing_cr.py \
  --mode python \
  --runs 3 --warmup 1 --output-mode none \
  --reference-csv demo/check/performance/py_reference_timing.csv \
  --max-slowdown 1.10 \
  --fail-on-regression \
  -o timing_python_current_vs_reference.csv
```

Useful options:

```bash
# quick smoke test
python3 demo/check/timing_cr.py --mode python rand4d --runs 1 --warmup 0 \
  --reference-csv demo/check/performance/py_reference_timing.csv

# benchmark CLI + Python together
python3 demo/check/timing_cr.py --mode all --runs 1 --warmup 0
```
