"""Slow reference computations of zigzag barcodes for tiny inputs.

Both functions return a Counter of ``(dim, birth, death)`` in coarse positions
``0..2T-2`` (twice the frame units used by ``cripser.compute_zigzag``).

- ``spacetime_cone_zigzag`` stacks the frames into a spacetime complex, adds a
  cone axis, and reads the ordinary persistence computed by
  ``cripser.compute_ph`` as extended persistence of the time function
  (Carlsson, de Silva, Morozov 2009).  V-construction with intersections only;
  1-2 spatial axes (the cone array has two more axes).
- ``explicit_fzz_zigzag`` runs FastZigzag (Dey, Hou 2022) with an explicit
  boundary matrix and plain column reduction.  Both constructions and both
  connection types.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations, product

import numpy as np

import cripser


def spacetime_cone_zigzag(masks: np.ndarray) -> Counter:
    masks = np.asarray(masks, dtype=bool)
    T = masks.shape[0]
    spatial = masks.shape[1:]
    # Axis -1 = 0: the spacetime complex X filtered by time (absent = inf).
    # Axis -1 = 1: a box filtered by reversed time, entered through the
    # cylinder X x [0, 1]; together they realise the cone over the
    # superlevel sets.  The extra time layer T seeds the box before X so that
    # it acts as the cone vertex in H_0.
    A = np.empty((T + 1,) + spatial + (2,), dtype=np.float64)
    t = np.arange(T, dtype=np.float64).reshape((T,) + (1,) * len(spatial))
    A[:T, ..., 0] = np.where(masks, t, np.inf)
    A[T, ..., 0] = np.inf
    A[:T, ..., 1] = 2 * T - 1 - t
    A[T, ..., 1] = -1.0
    ph = cripser.compute_ph(A, maxdim=len(spatial))
    out: Counter = Counter()
    for row in ph:
        k, b, d = int(row[0]), row[1], row[2]
        if b == -1.0:
            assert k == 0 and np.isinf(d)
            continue
        b, d = int(b), int(d)
        if d < T:  # ordinary: [b, d) closed-open
            key = (k, 2 * b, 2 * d - 1)
        elif b < T:  # extended
            s = 2 * T - 1 - d
            key = (k, 2 * b, 2 * s) if s >= b else (k - 1, 2 * s + 1, 2 * b - 1)
        else:  # relative: (s_d, s_b] open-closed, one dimension lower
            sb, sd = 2 * T - 1 - b, 2 * T - 1 - d
            key = (k - 1, 2 * sd + 1, 2 * sb)
        out[key] += 1
    return out


def _cells(extent, d):
    for k in range(d + 1):
        for axes in combinations(range(d), k):
            ranges = [range(n - 1) if a in axes else range(n) for a, n in enumerate(extent)]
            for anchor in product(*ranges):
                yield (anchor, axes)


def _faces(cell):
    anchor, axes = cell
    for a in axes:
        rest = tuple(x for x in axes if x != a)
        yield (anchor, rest)
        shifted = list(anchor)
        shifted[a] += 1
        yield (tuple(shifted), rest)


def _present(mask, cell, tconstruction):
    anchor, axes = cell
    shape = mask.shape
    if not tconstruction:  # all vertices
        for bits in product((0, 1), repeat=len(axes)):
            v = list(anchor)
            for a, bit in zip(axes, bits):
                v[a] += bit
            if not mask[tuple(v)]:
                return False
        return True
    free = [b for b in range(len(shape)) if b not in axes]
    for offsets in product((-1, 0), repeat=len(free)):  # any incident voxel
        v = list(anchor)
        for b, o in zip(free, offsets):
            v[b] += o
        if all(0 <= v[i] < shape[i] for i in range(len(shape))) and mask[tuple(v)]:
            return True
    return False


def _reduce(boundaries):
    pivot_of, columns, pairs = {}, [], []
    for j, boundary in enumerate(boundaries):
        column = set(boundary)
        while column and max(column) in pivot_of:
            column ^= columns[pivot_of[max(column)]]
        columns.append(column)
        if column:
            pivot_of[max(column)] = j
            pairs.append((max(column), j))
    return pairs


def explicit_fzz_zigzag(masks, filtration="V", connect="intersection") -> Counter:
    masks = np.asarray(masks, dtype=bool)
    T, d = masks.shape[0], masks.ndim - 1
    tcon = filtration == "T"
    extent = [n + 1 if tcon else n for n in masks.shape[1:]]
    cells = list(_cells(extent, d))
    frames = [{c for c in cells if _present(masks[t], c, tcon)} for t in range(T)]
    complexes = []
    for t in range(T):
        complexes.append(frames[t])
        if t + 1 < T:
            complexes.append(frames[t] & frames[t + 1] if connect == "intersection"
                             else frames[t] | frames[t + 1])
    P = len(complexes)

    # Cell-wise zigzag: faces are added before cofaces and deleted after them.
    ops, current = [], set()
    for p in range(P + 1):
        target = complexes[p] if p < P else set()
        for c in sorted(current - target, key=lambda c: (-len(c[1]), c)):
            ops.append(("-", c, p))
        for c in sorted(target - current, key=lambda c: (len(c[1]), c)):
            ops.append(("+", c, p))
        current = target

    # One cell per run; the boundary uses the face copies alive at addition.
    copy_of, runs, add_pos, del_pos, del_order = {}, [], [], {}, []
    for op, c, p in ops:
        if op == "+":
            copy_of[c] = len(runs)
            runs.append((len(c[1]), [copy_of[f] for f in _faces(c)]))
            add_pos.append(p)
        else:
            del_pos[copy_of[c]] = p
            del_order.append(copy_of[c])
    n = len(runs)

    # Coned filtration: w, the runs in addition order, the cones in reverse
    # deletion order.
    boundaries, dims = [set()], [0]
    for dim, faces in runs:
        boundaries.append({f + 1 for f in faces})
        dims.append(dim)
    cone_rank = {r: 2 * n - k for k, r in enumerate(del_order)}
    for r in reversed(del_order):
        dim, faces = runs[r]
        boundaries.append({r + 1} | ({0} if dim == 0 else {cone_rank[f] for f in faces}))
        dims.append(dim + 1)
    run_of = {**{q: q - 1 for q in range(1, n + 1)}, **{q: r for r, q in cone_rank.items()}}

    out: Counter = Counter()
    for birth, death in _reduce(boundaries):
        dim = dims[birth]
        b, k = run_of[birth], run_of[death]
        if death <= n:  # ordinary
            lo, hi = add_pos[b], add_pos[k] - 1
        elif birth <= n:  # extended
            if add_pos[b] < del_pos[k]:
                lo, hi = add_pos[b], del_pos[k] - 1
            else:
                lo, hi, dim = del_pos[k], add_pos[b] - 1, dim - 1
        else:  # relative
            lo, hi, dim = del_pos[k], del_pos[b] - 1, dim - 1
        if lo <= hi:
            out[(dim, lo, hi)] += 1
    return out
