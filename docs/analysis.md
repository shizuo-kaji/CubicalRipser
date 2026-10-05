# Plotting and vectorization

[Manual](README.md) · [Representative cycles](cycles.md) · Next: [PyTorch](torch.md)

## Prepare a persistence table

The examples below share this setup:

```python
import numpy as np
import cripser

image = np.ones((7, 7), dtype=np.float64)
image[1:6, 1:6] = 0.0
image[2:5, 2:5] = 1.0
ph = cripser.compute_ph(image, maxdim=1)
```

Use `compute_ph` so essential deaths are already `np.inf`. Raw `DBL_MAX`
values are finite floating-point numbers and should be normalized before
finite-pair filtering or feature extraction.

## Persistence diagrams

```python
import matplotlib.pyplot as plt

ax = cripser.plot_diagrams(ph, title="Ring filtration")
plt.show()
```

`plot_diagrams` accepts a PH table, a single `(n, 2)` birth/death array, or a
list of such arrays. It returns a Matplotlib axis; pass `ax=` to reuse one.
Use `labels=` to name each diagram or `show=True` to display directly.

## GUDHI conversion and grouping

```python
diagrams = cripser.to_gudhi_diagrams(ph, maxdim=1)
persistence = cripser.to_gudhi_persistence(ph)
```

- `diagrams[k]` is an `(n_k, 2)` array of birth/death values for Hₖ. Empty
  dimensions are retained by default (`include_empty=True`).
- `persistence` is a list of `(dimension, (birth, death))` tuples.
- Both converters normalize raw essential deaths to `np.inf`.

For GUDHI plotting, pass the tuple representation:

```python
import gudhi as gd

gd.plot_persistence_diagram(persistence=persistence)
```

`cripser.group_by_dim(ph)` groups full rows but currently accepts only a
9-column table (1D–3D input). For 4D tables, select directly with
`ph[ph[:, 0] == k]`.

## Persistence images

A persistence image is a smooth summary in birth/lifetime space:

```python
pi, metadata = cripser.persistence_image(
    ph,
    homology_dims=(0, 1),
    n_birth_bins=32,
    n_life_bins=32,
    birth_range=(0.0, 1.0),
    life_range=(0.0, 1.0),
    return_metadata=True,
)
print(pi.shape)  # (2, 32, 32): homology channel, lifetime, birth
```

| Parameter | Meaning |
| --- | --- |
| `homology_dims` | Channel order; otherwise inferred from finite dimensions present. |
| `n_birth_bins`, `n_life_bins` | Resolution; both default to 16. |
| `birth_range`, `life_range` | Coordinate ranges; otherwise inferred using quantiles `(0.1, 0.9)`. |
| `sigma` | Gaussian width, default 1.0. |
| `weight_power` | Weight by `max(lifetime, 0) ** weight_power`; default 1.0. |
| `drop_nonfinite` | Drop nonfinite rows by default. |
| `normalize` | If true, L1-normalize each channel. |
| `return_metadata` | Also return ranges, centers, and counting statistics. |

For a dataset, use the same dimensions, ranges, and bin counts for every
sample so feature coordinates remain comparable. The helper also accepts
PyTorch tensors and supports gradients through its smooth kernel calculation;
see [PyTorch integration](torch.md).

## Spatial PH histogram volumes

A histogram volume counts intervals at their creator or destroyer locations,
preserving the original image grid:

```python
hist = cripser.create_PH_histogram_volume(
    ph,
    image_shape=image.shape,
    homology_dims=(0, 1),
    n_birth_bins=4,
    n_life_bins=4,
    birth_range=(0.0, 1.0),
    life_range=(0.0, 1.0),
    location="birth",
)
print(hist.shape)  # (32, 7, 7)
```

The shape is `(channels, *image_shape)`, with
`channels = len(homology_dims) * n_life_bins * n_birth_bins`.
Choose `location="death"` for destroyer locations. Supply `image_shape` or
`reference_volume` to define the grid. Nonfinite intervals are dropped by
default; `return_metadata=True` adds bin edges, ranges, and counting statistics.

This helper requires coordinate columns and returns a NumPy array. Its dense
output can be large: estimate `channels * product(image_shape) * dtype.itemsize`
before applying it to a volume (default dtype is `float32`).
