# Zigzag persistence

[Manual](README.md) · [Python API](python-api.md) · [Output semantics](concepts.md)

Zigzag persistence tracks topological features through a sequence of binary
images or volumes, such as segmentation masks of a video or a time-lapse
volume. Features may appear, disappear, and reappear; unlike ordinary
persistence, the frames need not grow monotonically.

## What is computed

Frame `t` of the input defines a cubical complex `K_t`, built with the
V-construction (mask entries are vertices) or the T-construction (mask entries
are top-dimensional cells), as in [V and T constructions](concepts.md#v-and-t-constructions).
Consecutive frames are joined through their intersection or their union:

```text
connect="intersection":  K_0 ← K_0 ∩ K_1 → K_1 ← K_1 ∩ K_2 → … → K_{T-1}
connect="union":         K_0 → K_0 ∪ K_1 ← K_1 → K_1 ∪ K_2 ← … ← K_{T-1}
```

The result is the barcode of this zigzag in each homology dimension, with
coefficients in F₂.

## Computation

```python
import numpy as np
import cripser

frames = np.zeros((4, 7, 7), dtype=bool)
frames[0:3, 1:6, 1:6] = True   # a square in frames 0-2
frames[1:3, 3, 3] = False      # with a hole in frames 1-2
frames[3, 6, 6] = True         # a dot in frame 3

zz = cripser.compute_zigzag(frames)
print(zz[:, :3])
# [[0.  0.  2. ]
#  [0.  3.  3. ]
#  [1.  0.5 2. ]]
```

```text
cripser.compute_zigzag(masks, *, filtration="V", connect="intersection", maxdim=3,
                       exhaustive=True)
```

| Parameter | Meaning |
| --- | --- |
| `masks` | Array of shape `(T, n1)`, `(T, n1, n2)`, or `(T, n1, n2, n3)`. The first axis is time. Boolean or integer; nonzero entries are present. |
| `filtration` | `"V"` (default) or `"T"`. |
| `connect` | `"intersection"` (default) or `"union"`. |
| `maxdim` | Highest homology dimension to compute; bounded by the number of spatial axes minus one. |
| `exhaustive` | Exhaustive reduction for two spatial axes (see [Method and cost](#method-and-cost)). The result does not depend on it. |

Real-valued data must be thresholded first, for example `masks = video <= level`
for sublevel sets. `cripser.load_series` stacks image files along axis 0, which
matches the expected layout. C- and Fortran-contiguous arrays are used without
copying; other views are copied to a contiguous array.

## Output table

The result is a `float64` array of shape `(n, 9)` with the same columns as
[`compute_ph`](python-api.md#output-table):

```text
dim, birth, death, x1, y1, z1, x2, y2, z2
```

`birth` and `death` are the first and last positions at which the class is
alive, in frame units. Position `t` is frame `t`; position `t + 0.5` is the
complex connecting frames `t` and `t + 1`. The interval includes both ends, so
a class present only in frame `t` is reported as `(t, t)`.

An integer end is a closed end at that frame. A half-integer end is an open
end: the class exists in the connecting complex but not in the neighboring
frame. In the example above, the hole is reported from `0.5`: the
intersection of frames 0 and 1 already lacks the center pixel, whereas frame 0
still contains it. The four combinations correspond to the closed-closed, closed-open,
open-closed, and open-open intervals of zigzag persistence.

Every class dies by the last frame: a class alive in frame `T - 1` has
`death = T - 1`, and no death is infinite.

`(x1, y1, z1)` and `(x2, y2, z2)` are voxel locations of the cells whose
addition or deletion creates and destroys the class: the lowest corner of the
cell, clamped to the image for the T-construction. Unused axes are 0. A class
alive in the last frame is not destroyed within the sequence; its destroyer is
`-1` in every coordinate column, as for an essential class of ordinary
persistence. As in ordinary persistence, these locations localize a class but
do not describe it completely.

Rows are sorted by dimension, birth, and death. The existing helpers accept the
table, for example `cripser.to_gudhi_diagrams(zz)` and `cripser.plot_diagrams(zz)`.

## Relation to ordinary persistence

- With a single frame, every interval is `(0, 0)` and the counts per dimension
  are the Betti numbers of `K_0`.
- When each frame contains the previous one and `connect="intersection"`, the
  intervals are those of the sublevel filtration of the first frame at which
  each voxel is present. An ordinary interval `[b, d)` appears as
  `(b, d - 0.5)`; intervals still alive in the last frame end at `T - 1`.
- Reversing the frames reverses every interval: `(b, d)` becomes
  `(T - 1 - d, T - 1 - b)`.

## Method and cost

The computation follows [FastZigzag](https://arxiv.org/abs/2204.11080)
(T. K. Dey and T. Hou, ESA 2022). Each maximal run of frames during which a
cell stays present becomes one cell of an up-down filtration, whose barcode is
read from the ordinary persistence of its coned filtration.

- Besides the input, memory holds about 6 bytes per cell of one frame's
  cubical grid (2^d cells per voxel for d spatial axes), about 60 bytes per
  run, and the columns kept during the reduction.
- Time depends mainly on the number of runs and on the structure of the
  sequence. Masks that change little between frames are fast to process;
  frame-to-frame noise multiplies the runs and the work.
- For two spatial axes, `exhaustive=True` (default) keeps the reduced columns
  of the dimension-1 computation exhaustively reduced. This shortens columns
  that are added many times, which is faster on noisy sequences; on smooth
  sequences it costs some time, so `exhaustive=False` can be faster there. The
  option has no effect for one or three spatial axes.
- The GIL is released during the computation, so independent sequences can
  be processed with threads as described in [Parallelism](python-api.md#parallelism).
- Inputs have one to three spatial axes. Representative cycles and the
  command-line programs do not support zigzag persistence.
