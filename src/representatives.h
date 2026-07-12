/*
 * Optional direct-homology reduction used to recover representative cycles.
 *
 * The normal CubicalRipser path reduces implicit coboundaries.  It is much
 * faster, but produces cohomology information only.  This module is kept
 * entirely off that path: it explicitly reduces boundaries over F_2 and
 * tracks the column operations needed to obtain homology cycles.
 */

#pragma once

#include <cstdint>
#include <vector>

#include "config.h"
#include "cube.h"

class DenseCubicalGrids;

struct CycleRepresentative {
  uint8_t dim;
  double birth;
  double death;
  std::vector<Cube> cells;
};

// Compute all finite and essential representative homology cycles through
// config.maxdim.  This intentionally uses a separate, explicit reduction so
// callers that do not request representatives retain the optimized PH path.
std::vector<CycleRepresentative>
compute_homology_representatives(DenseCubicalGrids *dcg,
                                 const Config &config);
