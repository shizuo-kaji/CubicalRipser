# Release notes

[Manual](README.md) · [Installation](installation.md)

- **Unreleased**:
  - New `compute_zigzag` for [zigzag persistence](zigzag.md) of sequences of binary masks (V/T constructions, intersection or union connections).
  - New `cripser.datasets` [generators of synthetic arrays](python-api.md#synthetic-arrays): uniform noise, Gaussian random fields, and distances to a sphere, a torus, or a set of points. The small files under `sample/` were removed; the examples generate their inputs.
  - New `cripser.datasets.fetch` [downloads real volumes](python-api.md#downloaded-volumes) on first use: nine 3D CT, MRI, and simulation volumes (including the bonsai used in the benchmarks) and a 4D fMRI time series. `demo/compare_gudhi.py` creates a missing `sample/bonsai128.npy` or `sample/bonsai256.npy` from it.
  - Faster 4D computations (H₁ about 15–25%) and about 5–10% faster 3D computations.
  - Fix: with tied values, creator/destroyer coordinates could point to a voxel outside the creating or destroying cell. They now always lie on that cell; persistence pairs and values are unchanged.
  - Locations outside the input array are now `-1` in every coordinate column, in Python and in CLI CSV/NPY output. This changes the destroyer of an essential class (previously `0`, or `-1` / `4294967295` with `--embedded`) and creators in the embedded boundary padding (previously `-1`, `4294967295` or one past the last index, depending on the output path). Persistence pairs and values are unchanged.
  - `top_dim` now supports the T-construction and 4D input, and its pairs equal those of the ordinary computation in the top dimension. Previously the V-construction shortcut returned different pairs (for a random 30×31 image, 53 H₁ pairs instead of 17), and in Python the T-construction crashed the process and 4D input returned an empty table. In 3D it takes about a quarter to a third of the time and about a quarter of the memory of the previous shortcut.
  - Fix: CLI `--threshold t` now truncates the filtration in every dimension: the output equals that of the computation without a threshold, restricted to classes born below `t`, with deaths at or above `t` reported as `t` and no destroyer. Previously input values above `t` could still destroy classes (for example 2D H₁ deaths above `t`), only one essential H₀ class was reported when the threshold left several components, and with `--embedded` the boundary padding entered at `-t` instead of first. The creator of an essential class of dimension 1 or more is now a voxel with the birth value; previously it was the cell's anchor.
  - Fix: when values at `DBL_MAX` or `inf` separate the input into several components, each is reported as an essential H₀ class; previously only one was.
  - CLI: malformed input files (a dimension outside 1–4, fewer values than the header states, ragged CSV rows, a non-float64 `.npy`) are reported as errors; previously some gave undefined results or crashed. All errors exit with status 1.
  - Removed CLI options: `--algorithm compute_pairs` (it gave wrong T-construction H₀ results) and `--coface-table` / `--no-coface-table`.
- **v0.0.36**:
  - The Python binding now releases the GIL during the computation, so `ThreadPoolExecutor` over many images scales across cores.
  - New `n_threads` argument (`--threads` on the CLI) for intra-computation threading.
  - Fix: 3D/4D inputs with an axis longer than 32767 segfaulted. Such shapes now raise `ValueError`.
  - Fix: a single-voxel 3D/4D input reported `birth = DBL_MAX` instead of the voxel value.
- **v0.0.35**: Added support for computation of cycle representatives (homology cycles) for each persistence interval.
- **v0.0.34**: Switched the Python binding layer from pybind11 to [nanobind](https://github.com/wjakob/nanobind) so a single `cp312-abi3` wheel covers Python 3.12+. **Python 3.8 is dropped** (nanobind requires ≥ 3.9).
- **v0.0.31**: Changed module structure (hopefully, backward compatible)
- **v0.0.30**: Improved cache resulting in large speedup
- **v0.0.24**: Repository renamed from `CubicalRipser_3dim` to `CubicalRipser`.
  - update old remote if needed:
    ```bash
    git remote set-url origin https://github.com/shizuo-kaji/CubicalRipser.git
    ```
- **v0.0.23**: Added torch integration
- **v0.0.22**: Changed birth coordinates for T-construction to better match GUDHI for permanent cycles
- **v0.0.19**: Added support for 4D cubical complexes
- **v0.0.15**: Added support for Fortran-indexed numpy arrays (F_CONTIGUOUS arrays): Up to v0.0.14, C_CONTIGUOUS was assumed, which caused incorrect results for Fortran-indexed arrays.
- **v0.0.8**: Fixed memory leak in Python bindings (pointed out by Nicholas Byrne)
- **v0.0.7**: Speed improvements
- **v0.0.6**: Changed [birth/death location definition](concepts.md#creator-and-destroyer-cells)
- **up to v0.0.5**, differences from the [original version](https://github.com/CubicalRipser/CubicalRipser_3dim):
  - optimized implementation (lower memory footprint and faster on some data)
  - improved Python usability
  - much larger practical input sizes
  - cache control
  - Alexander duality option for highest-degree PH
  - both V and T constructions
  - birth/death location output
