# Python API

[Manual](README.md) · Previous: [Installation](installation.md) · Next: [CLI](cli.md)

## First computation

This example needs no sample files. The zero-valued ring creates a loop that
is filled when the central values enter the filtration at 1.

```python
import numpy as np
import cripser

image = np.ones((7, 7), dtype=np.float64)
image[1:6, 1:6] = 0.0
image[2:5, 2:5] = 1.0

ph = cripser.compute_ph(image, filtration="V", maxdim=1)
print(ph[:, :3])  # dimension, birth, death
h1 = ph[ph[:, 0] == 1]
finite = ph[np.isfinite(ph[:, 2])]
lifetimes = finite[:, 2] - finite[:, 1]
```

`maxdim` is a **homology dimension**, not the number of array axes: use 0 for
components, 1 to include loops, and 2 to include voids in 3D.
For a 4D array, use `maxdim=3` to include H₃.

## Main function

```text
cripser.compute_ph(
    arr, *, filtration="V", maxdim=3, top_dim=False, embedded=False,
    location="yes", representatives=False, n_threads=1, inf_cutoff=True,
)
```

| Parameter | Meaning |
| --- | --- |
| `arr` | Numeric NumPy array with 1–4 axes; converted to `float64` if needed. |
| `filtration` | `"V"` (default) or `"T"`; see [constructions](concepts.md#v-and-t-constructions). |
| `maxdim` | Highest homology dimension to compute, from 0 to 3; bounded by input dimension minus one. |
| `top_dim` | Compute only the top dimension, by Alexander duality; see the [top-dimensional shortcut](concepts.md#top-dimensional-shortcut). |
| `embedded` | Compute with the embedded Alexander-dual convention; changes signs and coordinate interpretation. |
| `location` | Compatibility argument. The current Python binding always returns coordinate columns, even with `"none"`. |
| `representatives` | If true, return `(pairs, cycles)`; incompatible with `top_dim=True`. |
| `n_threads` | 1 for sequential execution, 0 for automatic selection, or a positive worker count. |
| `inf_cutoff` | Boolean switch: convert deaths ≥ `np.finfo(np.float64).max / 2` to `np.inf`. Set false to keep raw values. |

### Input shape and memory layout

- Each axis must be nonempty, and arrays must have 1–4 axes.
- Both C- and Fortran-contiguous arrays are supported. Conversion of dtype or
  noncontiguous views may allocate a copy; account for this with large volumes.
- For 3D/4D input, and 2D input with `top_dim=True` or `representatives=True`,
  each axis must be at most **32760**. These paths use packed 15-bit coordinates
  and reject larger axes with `ValueError`.
- Plain 1D/2D computations use wider coordinate encoding and do not have that
  axis limit.
- A scalar time series is a 1D array. RGB channels, batches, and time axes are
  not inferred: every array axis is treated as a cubical-complex direction.
  Convert color images to scalar values and process independent samples separately.

## Output table

The default return value is a `float64` NumPy array, one row per persistence
interval. Coordinate columns are also stored as floating-point numbers.

| Input | Shape | Columns |
| --- | --- | --- |
| 1D, 2D, or 3D | `(n_pairs, 9)` | `dim, birth, death, x1, y1, z1, x2, y2, z2` |
| 4D | `(n_pairs, 11)` | `dim, birth, death, x1, y1, z1, w1, x2, y2, z2, w2` |

For fewer than three axes, use only the coordinate columns corresponding to
actual input axes. Coordinates follow NumPy indexing order. A location
outside the input array, such as the destroyer of an essential class, is `-1`
in every coordinate column. Essential classes have infinite death in
`compute_ph`; filter them before calculating finite lifetimes. See
[output semantics](concepts.md) for details, including the embedded convention.

```python
np.save("ph.npy", ph)
np.savetxt("ph.csv", ph, delimiter=",")
```

With `representatives=True`, the result is `(pairs, cycles)` and `cycles[i]`
corresponds to `pairs[i]`. See [representative cycles](cycles.md).

## Low-level bindings

The low-level functions accept the same computation arguments except
`filtration` and `inf_cutoff`. Select the construction by choosing the function:

```python
raw_v = cripser.computePH(image, maxdim=1)
raw_t = cripser.computePH_T(image, maxdim=1)
```

`computePH_T` is also available as `cripser.tcripser.computePH`.
Raw essential deaths use `DBL_MAX` (the largest finite `float64`) rather than
`np.inf`. Prefer `compute_ph` for downstream analysis.

## Parallelism

The native computation releases the Python GIL. Process independent images
with threads, keeping each computation sequential to avoid nested worker pools:

```python
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cripser

rng = np.random.default_rng(0)
images = [rng.random((32, 32)) for _ in range(8)]
with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(lambda a: cripser.compute_ph(a, maxdim=1), images))
```

Within one computation, `n_threads` parallelizes grid scans and sorts:

- `n_threads=1` (default): sequential.
- `n_threads=0`: hardware concurrency, overridden by `CRIPSER_NUM_THREADS`.
- `n_threads=k`: at most `k` workers.

Results are independent of the worker count. Not all phases run in parallel;
measure your own data before choosing a worker count. Concurrent computations
also multiply the memory needed for grids and reduction state.

## Related helpers

[Zigzag persistence](zigzag.md) covers `compute_zigzag` for sequences of
binary masks. [Plotting and vectorization](analysis.md) covers `plot_diagrams`,
`to_gudhi_diagrams`, `to_gudhi_persistence`, `group_by_dim`, `persistence_image`,
and `create_PH_histogram_volume`. [I/O](io.md) covers loaders and transforms;
[PyTorch](torch.md) covers differentiable computation.
