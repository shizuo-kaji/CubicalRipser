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

## Synthetic arrays

`cripser.datasets` generates arrays for examples, tests, and benchmarks. Each
function returns a C-contiguous `float64` array. Random generators take
`seed`, an integer or a `numpy.random.Generator`.

| Function | Array |
| --- | --- |
| `uniform_noise(shape, levels=None, seed=None)` | Independent uniform values in [0, 1). With `levels=k`, integers 0, …, k−1 with many ties. |
| `gaussian_random_field(shape, sigma=2.0, seed=None)` | White noise convolved with a Gaussian kernel (periodic), standardized; features are about `sigma` grid units in size. |
| `sphere(shape, radius=None, center=None)` | Distance to a circle (2D), 2-sphere (3D), or 3-sphere (4D). The diagram has one long bar in dimension `ndim - 1`. |
| `torus(shape, major_radius=None, minor_radius=None, center=None)` | Distance to a torus surface in 3D, with the axis of revolution along the last axis. |
| `distance_to_points(shape, points=10, seed=None)` | Distance to the nearest of a set of points, random or given. The sublevel sets are unions of balls. |

The shape generators return distances, so `arr <= t` is the shape thickened
by `t`. Thresholding at about one grid unit gives a binary shape with known
Betti numbers:

```python
import numpy as np
import cripser
from cripser import datasets

shell = np.where(datasets.torus((32, 32, 16)) <= 0.85, 0.0, 1.0)
ph = cripser.compute_ph(shell, maxdim=2)
print([int(np.sum((ph[:, 0] == k) & (ph[:, 1] == 0))) for k in range(3)])  # [1, 2, 1]
```

## Downloaded volumes

`cripser.datasets.fetch(name)` loads a real 3D or 4D volume as a C-contiguous
`float64` array, downloading it on first use:

| Name | Shape | Content | Source |
| --- | --- | --- | --- |
| `bonsai` | 256×256×256 | CT of a bonsai tree | volvis.org, S. Roettger (University of Stuttgart) |
| `foot` | 256×256×256 | Rotational X-ray of a human foot | volvis.org, Philips Research |
| `skull` | 256×256×256 | Rotational X-ray of a skull phantom | volvis.org, Siemens Medical Solutions |
| `aneurism` | 256×256×256 | Rotational angiography of head arteries | volvis.org, Philips Research |
| `engine` | 128×256×256 | CT of an engine block | volvis.org, General Electric |
| `mri_ventricles` | 124×256×256 | MRI of a head, cerebrospinal fluid cavities | volvis.org, D. Bartz (University of Tübingen) |
| `fuel` | 64×64×64 | Simulated fuel injection | volvis.org, SFB 382 (DFG) |
| `hydrogen_atom` | 128×128×128 | Simulated electron probability density | volvis.org, SFB 382 (DFG) |
| `tacc_turbulence` | 256×256×256 | Enstrophy of isotropic turbulence (continuous values) | G. D. Abram, G. P. Johnson (TACC), D. A. Donzis |
| `fmri_haxby` | 40×64×64×121 | BOLD fMRI time series | Haxby et al. 2001, OpenNeuro ds000105 |

The 3D volumes come from the
[Open SciVis Datasets](https://klacansky.com/open-scivis-datasets/) collection
and are indexed `[z, y, x]`. `fmri_haxby` is indexed `[x, y, z, t]`, as in
nibabel. `datasets.VOLUMES[name]` holds the shape, description, and full
credit; acknowledge the source when publishing results.

```python
import cripser
from cripser import datasets

bonsai = datasets.fetch("bonsai")
ph = cripser.compute_ph(bonsai[::2, ::2, ::2], maxdim=2)  # 128³, as in the benchmarks
bold = datasets.fetch("fmri_haxby")
ph4 = cripser.compute_ph(bold[..., :10], maxdim=3)        # first 10 time points
```

Files are saved in `~/.cache/cripser` (`$XDG_CACHE_HOME/cripser` when set);
set `CRIPSER_DATA_DIR` or pass `data_home=` to use another directory. Each
download is checked against a SHA-512 digest stored in the package.

## Related helpers

[Zigzag persistence](zigzag.md) covers `compute_zigzag` for sequences of
binary masks. [Plotting and vectorization](analysis.md) covers `plot_diagrams`,
`to_gudhi_diagrams`, `to_gudhi_persistence`, `group_by_dim`, `persistence_image`,
and `create_PH_histogram_volume`. [I/O](io.md) covers loaders and transforms;
[PyTorch](torch.md) covers differentiable computation.
