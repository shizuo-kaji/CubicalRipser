# Representative homology cycles

[Manual](README.md) · [Output semantics](concepts.md) · [Plotting](analysis.md)

## Compute and select cycles

Pass `representatives=True` to either Python computation API. The return value
is `(pairs, cycles)`, where `cycles[i]` represents the interval in `pairs[i]`.
Keep the same indices when filtering or sorting these two outputs.

```python
import numpy as np
import cripser

image = np.ones((7, 7), dtype=np.float64)
image[1:6, 1:6] = 0.0
image[2:5, 2:5] = 1.0
pairs, cycles = cripser.compute_ph(image, maxdim=1, representatives=True)
selected = np.flatnonzero((pairs[:, 0] == 1) & np.isfinite(pairs[:, 2]))
h1_cycles = [cycles[i] for i in selected]
```

## Cell encoding

A cycle is a list of cells, each encoded as:

- 1D–3D: `[x, y, z, cell_type]`.
- 4D: `[x, y, z, w, cell_type]`.

All listed cells have coefficient one in F₂. `cell_type` identifies the
orientation within the cell's dimension. In planar H₁ cycles, type 0 means an
x-edge from `(x, y)` to `(x + 1, y)`, and type 1 means a y-edge from `(x, y)` to
`(x, y + 1)`. It is not a homology-dimension field.

Convert a chain to an array with `np.asarray(cycles[i], dtype=np.uint32)` if
convenient. A representative is a homology cycle, not a unique or necessarily
shortest geometric outline.

## Plot planar loops

With the variables from the first example:

```python
import matplotlib.pyplot as plt

ax = cripser.plot_cycles(
    h1_cycles,
    image=image,
    labels=[f"H₁ #{i}" for i in selected],
)
plt.show()
```

`plot_cycle` plots one cycle, while `plot_cycles` handles several. Both return
a Matplotlib axis. The supplied 2D image is drawn in the same `(x, y)` indexing
convention as the cycles; `overlay=False` omits the background. These helpers
are for planar H₁ representatives, not arbitrary higher-dimensional chains.

The [main notebook](../demo/cubicalripser.ipynb#representative-cycles) includes
an example with two figure-eight peaks.

## Cost and restrictions

Representative computation uses direct boundary reduction over F₂, adding
bookkeeping and memory for chains. It is intended for inspection and
visualization; start with small arrays before using it on a large volume.

The option is disabled by default. With it disabled, the optimized
cohomology/coboundary path allocates and tracks no representative chains.
`representatives=True` cannot be combined with `top_dim=True`, which uses a
separate Alexander-duality shortcut. The C++ CLI does not expose this option.
