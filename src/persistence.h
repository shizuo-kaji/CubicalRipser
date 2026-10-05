/* persistence.h

The persistent homology computation shared by the command-line programs and
the Python bindings.
*/

#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "config.h"
#include "write_pairs.h"

class DenseCubicalGrids;

// Converts grid voxels of a computed grid to input-array coordinates.
class InputCoordinates {
public:
  explicit InputCoordinates(const DenseCubicalGrids &dcg);

  // The input voxel at grid voxel (x, y, z, w), or -1 on every axis when it
  // lies outside the input: in the boundary padding, or NO_VOXEL.
  std::array<int64_t, 4> operator()(uint32_t x, uint32_t y, uint32_t z,
                                    uint32_t w) const {
    const uint32_t v[4] = {x, y, z, w};
    std::array<int64_t, 4> c{};
    for (int a = 0; a < 4; ++a) {
      const uint32_t u = v[a] - pad_[a]; // wraps below the input
      if (u >= size_[a])
        return {-1, -1, -1, -1};
      c[a] = u;
    }
    return c;
  }

private:
  uint32_t pad_[4];
  uint32_t size_[4];
};

// Persistence pairs, grouped by dimension, and the conversion of their voxels
// to input-array coordinates.
struct Persistence {
  std::vector<WritePairs> pairs;
  InputCoordinates input_voxel;
};

// Persistence of an input array of the shape dcg was constructed with;
// `values` has axis 0 varying fastest if fortran_order and the last axis
// otherwise.  config.method selects the ordinary computation (LINKFIND),
// which builds its grid in dcg, or the top dimension alone (ALEXANDER), which
// builds the grid of the dual construction instead.  config.maxdim must
// already be at most dim - 1.
Persistence compute_persistence(DenseCubicalGrids &dcg, const double *values,
                                bool fortran_order, Config &config);
