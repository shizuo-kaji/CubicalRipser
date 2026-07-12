/* Optional direct homology representative computation. */

#include "representatives.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dense_cubical_grids.h"

namespace {

constexpr uint32_t kNoColumn = std::numeric_limits<uint32_t>::max();

struct Cell {
  Cube cube;
  uint8_t dim;
};

uint8_t mask_count_for_cell_dim(const DenseCubicalGrids *dcg,
                                uint8_t cell_dim) {
  if (dcg->dim == 4) {
    switch (cell_dim) {
    case 0:
    case 4:
      return 1;
    case 1:
    case 3:
      return 4;
    case 2:
      return 6;
    default:
      return 0;
    }
  }

  // T-construction of a planar image is represented internally with az == 1.
  // There are no out-of-plane edges or squares in that complex.
  if (dcg->config->tconstruction && dcg->az == 1) {
    switch (cell_dim) {
    case 0:
    case 2:
      return 1;
    case 1:
      return 2;
    default:
      return 0;
    }
  }

  switch (cell_dim) {
  case 0:
  case 3:
    return 1;
  case 1:
  case 2:
    return 3;
  default:
    return 0;
  }
}

uint8_t axes_for(uint8_t ambient_dim, uint8_t cell_dim, uint8_t mask) {
  if (ambient_dim < 4) {
    switch (cell_dim) {
    case 0:
      return 0;
    case 1:
      return static_cast<uint8_t>(1U << mask); // x, y, z
    case 2:
      // xy, zx, yz
      return mask == 0 ? 0x3U : (mask == 1 ? 0x5U : 0x6U);
    case 3:
      return 0x7U;
    default:
      return 0;
    }
  }

  switch (cell_dim) {
  case 0:
    return 0;
  case 1:
    return static_cast<uint8_t>(1U << mask); // x, y, z, w
  case 2:
    // xy, zx, yz, wx, wy, wz
    switch (mask) {
    case 0:
      return 0x3U;
    case 1:
      return 0x5U;
    case 2:
      return 0x6U;
    case 3:
      return 0x9U;
    case 4:
      return 0xaU;
    default:
      return 0xcU;
    }
  case 3:
    // xyz, xyw, xzw, yzw
    switch (mask) {
    case 0:
      return 0x7U;
    case 1:
      return 0xbU;
    case 2:
      return 0xdU;
    default:
      return 0xeU;
    }
  case 4:
    return 0xfU;
  default:
    return 0;
  }
}

uint8_t mask_for_axes(uint8_t ambient_dim, uint8_t cell_dim, uint8_t axes) {
  if (ambient_dim < 4) {
    if (cell_dim == 0 || cell_dim == 3)
      return 0;
    if (cell_dim == 1) {
      if (axes == 0x1U)
        return 0;
      if (axes == 0x2U)
        return 1;
      return 2;
    }
    if (axes == 0x3U)
      return 0;
    if (axes == 0x5U)
      return 1;
    return 2;
  }

  if (cell_dim == 0 || cell_dim == 4)
    return 0;
  if (cell_dim == 1) {
    switch (axes) {
    case 0x1U:
      return 0;
    case 0x2U:
      return 1;
    case 0x4U:
      return 2;
    default:
      return 3;
    }
  }
  if (cell_dim == 2) {
    switch (axes) {
    case 0x3U:
      return 0;
    case 0x5U:
      return 1;
    case 0x6U:
      return 2;
    case 0x9U:
      return 3;
    case 0xaU:
      return 4;
    default:
      return 5;
    }
  }
  switch (axes) {
  case 0x7U:
    return 0;
  case 0xbU:
    return 1;
  case 0xdU:
    return 2;
  default:
    return 3;
  }
}

uint64_t packed_index(uint32_t x, uint32_t y, uint32_t z, uint32_t w,
                      uint8_t mask) {
  return static_cast<uint64_t>(x) | (static_cast<uint64_t>(y) << 15) |
         (static_cast<uint64_t>(z) << 30) | (static_cast<uint64_t>(w) << 45) |
         (static_cast<uint64_t>(mask) << 60);
}

// Symmetric difference of two ordered F_2 chains.
void xor_chains(std::vector<uint32_t> &chain,
                const std::vector<uint32_t> &addon,
                std::vector<uint32_t> &scratch) {
  scratch.clear();
  scratch.reserve(chain.size() + addon.size());
  size_t i = 0;
  size_t j = 0;
  while (i < chain.size() && j < addon.size()) {
    if (chain[i] == addon[j]) {
      ++i;
      ++j;
    } else if (chain[i] < addon[j]) {
      scratch.push_back(chain[i++]);
    } else {
      scratch.push_back(addon[j++]);
    }
  }
  while (i < chain.size())
    scratch.push_back(chain[i++]);
  while (j < addon.size())
    scratch.push_back(addon[j++]);
  chain.swap(scratch);
}

} // namespace

std::vector<CycleRepresentative>
compute_homology_representatives(DenseCubicalGrids *dcg,
                                 const Config &config) {
  if (dcg == nullptr) {
    throw std::invalid_argument("representatives: missing cubical grid");
  }

  const int requested_maxdim = std::max(0, config.maxdim);
  const uint8_t max_homology_dim = std::min<uint8_t>(
      static_cast<uint8_t>(requested_maxdim), dcg->dim - 1U);
  const uint8_t max_cell_dim = static_cast<uint8_t>(max_homology_dim + 1U);
  const double threshold = dcg->threshold;

  std::vector<Cell> cells;
  for (uint8_t cell_dim = 0; cell_dim <= max_cell_dim; ++cell_dim) {
    const uint8_t mask_count = mask_count_for_cell_dim(dcg, cell_dim);
    for (uint8_t mask = 0; mask < mask_count; ++mask) {
      for (uint32_t w = 0; w < dcg->aw; ++w) {
        for (uint32_t z = 0; z < dcg->az; ++z) {
          for (uint32_t y = 0; y < dcg->ay; ++y) {
            for (uint32_t x = 0; x < dcg->ax; ++x) {
              const double birth = dcg->getBirth(x, y, z, w, mask, cell_dim);
              if (birth < threshold) {
                cells.push_back(
                    {Cube(birth, x, y, z, w, mask), cell_dim});
              }
            }
          }
        }
      }
    }
  }

  // A face must precede a coface at a tied filtration value.  This ordering is
  // deliberately independent from the optimized coboundary reducer's order.
  std::sort(cells.begin(), cells.end(), [](const Cell &a, const Cell &b) {
    if (a.cube.birth != b.cube.birth)
      return a.cube.birth < b.cube.birth;
    if (a.dim != b.dim)
      return a.dim < b.dim;
    return a.cube.index < b.cube.index;
  });

  const size_t cell_count = cells.size();
  std::vector<std::unordered_map<uint64_t, uint32_t>> cell_lookup(
      static_cast<size_t>(max_cell_dim) + 1U);
  for (uint32_t i = 0; i < cell_count; ++i) {
    cell_lookup[cells[i].dim].emplace(cells[i].cube.index, i);
  }

  std::vector<std::vector<uint32_t>> reduced_boundaries(cell_count);
  std::vector<std::vector<uint32_t>> column_chains(cell_count);
  std::vector<uint32_t> pivot_column(cell_count, kNoColumn);
  std::vector<bool> is_positive_column(cell_count, false);
  std::vector<CycleRepresentative> representatives;
  std::vector<uint32_t> boundary;
  std::vector<uint32_t> chain;
  std::vector<uint32_t> boundary_scratch;
  std::vector<uint32_t> chain_scratch;

  for (uint32_t column = 0; column < cell_count; ++column) {
    const Cell &cell = cells[column];
    boundary.clear();
    chain.assign(1, column);

    if (cell.dim > 0) {
      const uint8_t axes = axes_for(dcg->dim, cell.dim, cell.cube.m());
      for (uint8_t axis = 0; axis < dcg->dim; ++axis) {
        if ((axes & (1U << axis)) == 0U)
          continue;

        const uint8_t face_axes = static_cast<uint8_t>(axes & ~(1U << axis));
        const uint8_t face_mask =
            mask_for_axes(dcg->dim, static_cast<uint8_t>(cell.dim - 1U),
                          face_axes);
        std::array<uint32_t, 4> coords = {
            cell.cube.x(), cell.cube.y(), cell.cube.z(), cell.cube.w()};
        for (uint8_t side = 0; side < 2; ++side) {
          if (side != 0)
            ++coords[axis];
          const uint64_t face_index = packed_index(
              coords[0], coords[1], coords[2], coords[3], face_mask);
          const auto found = cell_lookup[cell.dim - 1U].find(face_index);
          if (found == cell_lookup[cell.dim - 1U].end()) {
            throw std::runtime_error(
                "representatives: cubical boundary face missing from filtration");
          }
          boundary.push_back(found->second);
        }
      }
      std::sort(boundary.begin(), boundary.end());
    }

    bool reduced_to_zero = true;
    while (!boundary.empty()) {
      const uint32_t pivot = boundary.back();
      const uint32_t previous_column = pivot_column[pivot];
      if (previous_column == kNoColumn) {
        pivot_column[pivot] = column;
        reduced_boundaries[column] = std::move(boundary);
        column_chains[column] = std::move(chain);
        reduced_to_zero = false;

        const Cell &birth_cell = cells[pivot];
        if (birth_cell.dim <= max_homology_dim &&
            birth_cell.cube.birth != cell.cube.birth) {
          CycleRepresentative representative;
          representative.dim = birth_cell.dim;
          representative.birth = birth_cell.cube.birth;
          representative.death = cell.cube.birth;
          representative.cells.reserve(column_chains[pivot].size());
          for (uint32_t chain_cell : column_chains[pivot]) {
            representative.cells.push_back(cells[chain_cell].cube);
          }
          representatives.push_back(std::move(representative));
        }
        break;
      }

      xor_chains(boundary, reduced_boundaries[previous_column], boundary_scratch);
      xor_chains(chain, column_chains[previous_column], chain_scratch);
    }

    if (reduced_to_zero) {
      is_positive_column[column] = true;
      column_chains[column] = std::move(chain);
    }
  }

  // Zero columns not used as a pivot represent essential classes.
  for (uint32_t column = 0; column < cell_count; ++column) {
    if (!is_positive_column[column] || pivot_column[column] != kNoColumn ||
        cells[column].dim > max_homology_dim ||
        cells[column].cube.birth == threshold) {
      continue;
    }
    CycleRepresentative representative;
    representative.dim = cells[column].dim;
    representative.birth = cells[column].cube.birth;
    representative.death = threshold;
    representative.cells.reserve(column_chains[column].size());
    for (uint32_t chain_cell : column_chains[column]) {
      representative.cells.push_back(cells[chain_cell].cube);
    }
    representatives.push_back(std::move(representative));
  }

  return representatives;
}
