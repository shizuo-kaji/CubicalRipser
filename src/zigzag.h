/* zigzag.h

Zigzag persistence of a sequence of binary cubical complexes

    K_0 <-> K_0 . K_1 <-> K_1 <-> K_1 . K_2 <-> ... <-> K_{T-1}

where "." is the intersection (K_t <- K_t ∩ K_{t+1} -> K_{t+1}) or the union
(K_t -> K_t ∪ K_{t+1} <- K_{t+1}).  Each K_t is built from frame t of a mask
stack by the V-construction (mask entries are vertices) or the T-construction
(mask entries are top-dimensional cells).

The computation follows FastZigzag (T. K. Dey and T. Hou, "Fast Computation of
Zigzag Persistence", ESA 2022): every maximal run of presence of a cell becomes
one cell of an up-down filtration, whose barcode is read off the ordinary
persistence of its coned filtration.  Coefficients are in F_2.

Intervals are reported on the coarse positions 0..2T-2 of the zigzag above:
even p is frame p/2 and odd p is the connecting complex between frames
(p-1)/2 and (p+1)/2.  [birth, death] are the first and last positions at
which the class is alive.
*/

#pragma once

#include <cstdint>
#include <vector>

// Read-only view of a (T, n_0[, n_1[, n_2]]) mask stack.  Strides are in
// elements, so both C- and Fortran-ordered buffers are addressed directly.
struct ZigzagMasks {
  const bool *data{nullptr};
  uint8_t spatial_dim{0};   // d = 1, 2 or 3
  uint32_t frames{0};       // T
  uint32_t shape[3]{1, 1, 1};
  int64_t frame_stride{0};
  int64_t stride[3]{0, 0, 0};
};

struct ZigzagInterval {
  uint8_t dim;
  uint32_t birth; // first coarse position at which the class is alive
  uint32_t death; // last coarse position at which the class is alive
  // Anchor voxel (lowest corner, clamped to the image for T) of the cell whose
  // addition or deletion creates / destroys the class; -1 on every axis for
  // the destroyer of a class alive in the last frame.
  int64_t birth_voxel[3];
  int64_t death_voxel[3];
};

// Intervals of dimension 0..min(maxdim, d-1), sorted by (dim, birth, death).
// `exhaustive` selects exhaustive reduction of cached columns for 2D frames
// (see zigzag_reduction.h); the intervals do not depend on it.
std::vector<ZigzagInterval> compute_zigzag(const ZigzagMasks &masks,
                                           bool tconstruction,
                                           bool union_connect, int maxdim,
                                           bool exhaustive = true);
