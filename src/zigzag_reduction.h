/* zigzag_reduction.h

Ordinary persistence of the coned filtration of FastZigzag

    {w} ⊂ L_1 ∪ {w} ⊂ ... ⊂ L_n ∪ {w} = K ⊂ K ∪ w·L_{2n-1} ⊂ ... ⊂ K ∪ w·K,

built over the run table: rank 0 is the cone vertex w, rank 1 + r is run r in
addition order, and rank 2n - k is the cone w·(deletion_order[k]).  A cell of
rank q enters at index q, so ranks double as filtration values.

H_0 is computed by union-find.  Higher dimensions use boundary-matrix
reduction with clearing (twist), from the top dimension down.  Cohomology, as
in the main CubicalRipser pipeline, is not used here: the cocycles of classes
killed by coning span large parts of the complex, whereas the corresponding
cycles stay small.  In dimension 1 the pairs among cone cells come from
union-find on the coned subcomplex, and its spanning forest turns every other
cone triangle into a cycle of base edges before the reduction.  For 2D
frames, cached columns of that dimension are by default reduced exhaustively
(every entry that is the pivot of another column is eliminated), which keeps
them short when they are added again and again.
*/

#pragma once

#include <cstdint>
#include <vector>

#include "zigzag_runs.h"

struct ConedPair {
  uint8_t dim;    // homological dimension in the coned complex
  uint32_t birth; // rank of the creating cell
  uint32_t death; // rank of the destroying cell
};

// All finite pairs of dimension 0..max_dim.  The only essential class, born by
// the cone vertex, is omitted.  The pairs do not depend on `exhaustive`.
std::vector<ConedPair> reduce_coned_filtration(const ZigzagRunTable &table,
                                               int max_dim, bool exhaustive);
