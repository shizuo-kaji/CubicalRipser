# Constructions and output semantics

[Manual](README.md) · [Python API](python-api.md) · [Representative cycles](cycles.md)

## Filtration direction

CubicalRipser computes sublevel-set persistent homology: cells enter as the
filtration value increases. H₀ tracks connected components, H₁ tracks loops,
and H₂ tracks enclosed voids. Coefficients are in F₂.

For an interval `[birth, death)`, the class exists from `birth` until `death`.
Its lifetime is `death - birth`. An essential class never dies and has death
`np.inf` in the convenience Python API, or `DBL_MAX` in raw/CLI output.

To study bright structures first, compute on the negated scalar array:

```python
import numpy as np
import cripser

image = np.array([[0., 1., 0.], [1., 3., 1.], [0., 1., 0.]])
ph_bright = cripser.compute_ph(-image, maxdim=1)
```

The reported birth/death values are then in the negated filtration.
Negating the input and enabling `embedded` are different operations: embedding
also changes boundary treatment for Alexander duality.

## V and T constructions

| | V-construction | T-construction |
| --- | --- | --- |
| Input values belong to | Vertices (0-cells) | Top-dimensional cells |
| Other cells receive | Maximum of incident vertex values | Minimum of incident top-cell values |
| Pixel connectivity in 2D | 4-neighborhood | 8-neighborhood |
| Python | `filtration="V"` | `filtration="T"` |
| CLI | `cubicalripser` | `tcubicalripser` |

Diagonal pixels can therefore connect differently. Choose the construction
according to the meaning of your data, and use the same construction when
comparing with another library.

## Alexander duality and embedding

The following computations are related by Alexander duality:

```bash
./build/cubicalripser --output v.csv input.npy
./build/tcubicalripser --embedded --output t_embedded.csv input.npy
```

They are not identical tables. `--embedded` converts the input `I` to
`-I^infty` in the paper's notation; it changes filtration signs and the treatment
of permanent cycles. For corresponding **finite** intervals in an ambient
dimension `d`, the duality reverses dimensions `k ↔ d - 1 - k` and endpoints
`(birth, death) ↔ (-death, -birth)`. Essential classes need separate treatment.
The reverse V/T correspondence is tested in
[`test_alexander.py`](../tests/test_alexander.py).

For the mathematical construction, see
[Duality in Persistent Homology of Images](https://arxiv.org/abs/2005.04597)
by Adelie Garin et al.

`cripser.dual_embedding(arr)` is a separate array helper: it returns shape
`(n1 + 1, ..., nd + 1)` by taking minima of incident input cells. It does not
itself compute PH or perform the sign-changing `embedded=True` operation.

## Top-dimensional shortcut

`top_dim=True` / `--top_dim` computes only the top dimension: H₁ for 2D input,
H₂ for 3D, H₃ for 4D, and H₀ for 1D, where it is the ordinary computation. The
pairs are those of the ordinary computation in that dimension, for both
constructions and with or without `embedded`; `maxdim` is ignored.

By Alexander duality, the top-dimensional classes of one construction
correspond to the H₀ classes of the other construction with the embedding
flipped, with `(birth, death)` mapped to `(-death, -birth)`. These are computed
by union-find, which takes less time and memory than the ordinary computation.
An input with an axis of length 1 is computed in full and the top dimension
kept.

Creator and destroyer locations follow the conventions below, but with tied
values they can differ from those of the ordinary computation. `top_dim`
cannot be combined with representative cycles.

## Creator and destroyer cells

The creator cell gives birth to a class; the destroyer cell kills it. For H₀,
a component is born at its lowest filtration value, and a connecting cell can
kill the component with the higher birth value.

The output records input-grid locations associated with these cells: a voxel
whose value is the cell's filtration value, chosen among the cell's vertices
(V-construction) or the top-dimensional cells containing it (T-construction).
Such locations help localize a class, but do not describe an entire cycle; use
[representative cycles](cycles.md) for that. Creators and destroyers need not
be unique, particularly when input values tie.

In 3D, the default convention for a finite interval satisfies:

```text
arr[x2, y2, z2] - arr[x1, y1, z1] = death - birth
```

Here `(x1, y1, z1)` is the creator location and `(x2, y2, z2)` the destroyer.
With `embedded=True` / `--embedded`, the roles are swapped:

```text
arr[x1, y1, z1] - arr[x2, y2, z2] = death - birth
```

Use the analogous number of indices for other input dimensions. The `x`, `y`,
`z`, `w` names refer to NumPy axes 0, 1, 2, 3, not physical coordinates or a
plotting library's horizontal/vertical convention.

A location outside the input array is `-1` in every coordinate column. This
is the destroyer of an essential class, and, with `embedded=True`, the creator
of a class born in the boundary padding (the essential H₀ class and, for 2D–4D
input, the top-dimensional class born at `-DBL_MAX`). Every other location
indexes the input array.

Thanks to Nicholas Byrne for suggesting the location convention and providing
test code. See also [the output column reference](python-api.md#output-table).
