/* ph_2d.h
 *
 * Specialized fast path for effectively 1-D / 2-D persistent homology.
 *
 *  In 2-D, H_1 is computed via the dual union-find using Alexander duality on
 *  S^2. In 1-D, the same code provides a dedicated H_0-only fast path.
 */
#pragma once

#include <vector>

#include "config.h"
#include "dense_cubical_grids.h"
#include "write_pairs.h"

// Compute H_0 (and optionally H_1 in 2-D) directly from a loaded
// DenseCubicalGrids. Returns true on success.
//
// Preconditions:
//   - dcg->dim <= 2 and the image is effectively planar (az == 1, aw == 1)
//   - config.method == LINKFIND
//   - config.threshold == DBL_MAX (no early cutoff is currently honored
//     by the existing generic path either, beyond filtering the
//     "boundary = threshold" cells)
bool compute_PH_2d(DenseCubicalGrids* dcg,
                   std::vector<WritePairs>& writepairs,
                   const Config& config);
