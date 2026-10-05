/* The persistent homology computation shared by the CLI and the bindings. */

#include "persistence.h"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#include "compute_pairs.h"
#include "cube.h"
#include "dense_cubical_grids.h"
#include "joint_pairs.h"
#include "ph_2d.h"

namespace {

class StageTimer {
public:
  explicit StageTimer(const Config &config) : verbose_(config.verbose) {}
  void report(const char *stage) {
    if (!verbose_)
      return;
    const auto now = std::chrono::steady_clock::now();
    std::cout << stage << " took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(now - start_).count()
              << " [msec]" << std::endl;
    start_ = now;
  }

private:
  bool verbose_;
  std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
};

// H_0 by union-find over the axis-aligned edges of a built grid.
void union_find_h0(DenseCubicalGrids &dcg, std::vector<WritePairs> &pairs,
                   std::vector<Cube> &ctr, Config &config) {
  JointPairs jp(&dcg, pairs, config);
  std::vector<uint8_t> edge_types(dcg.dim);
  std::iota(edge_types.begin(), edge_types.end(), uint8_t{0});
  jp.enum_edges(edge_types, ctr);
  jp.joint_pairs_main(ctr);
}

// The top dimension d - 1 by Alexander duality: its classes for one
// construction of the input, embedded or not, are the H_0 classes of the other
// construction with the embedding flipped, with (birth, death) mapped to
// (-death, -birth) and creator and destroyer exchanged.  Without the
// embedding, the H_0 class born in the boundary padding has no counterpart.
// The dual is computed without the threshold; truncating at it keeps the
// classes born below it and ends the others' lives there.
Persistence top_dimension(const DenseCubicalGrids &dcg, const double *values,
                          bool fortran_order, const Config &config) {
  Config dual = config;
  dual.embedded = !config.embedded;
  dual.maxdim = 0;
  dual.print = false;
  dual.threshold = DBL_MAX;
  DenseCubicalGrids grid(dual, dcg.dim, dcg.ax, dcg.ay, dcg.az, dcg.aw);
  dual.tconstruction = !config.tconstruction; // after the constructor sets it
  grid.gridFromArray(values, dual.embedded, fortran_order);
  grid.finalisePadding();

  Persistence result{{}, InputCoordinates(grid)};
  std::vector<WritePairs> &pairs = result.pairs; // H_0, then mapped in place
  {
    std::vector<Cube> ctr;
    union_find_h0(grid, pairs, ctr, dual);
  }
  const uint8_t top = static_cast<uint8_t>(grid.dim - 1);
  size_t n = 0;
  for (size_t i = 0; i < pairs.size(); ++i) {
    const WritePairs &p = pairs[i];
    const double birth = -p.death;
    const double death = -p.birth;
    if ((!config.embedded && p.death_x == NO_VOXEL) || !(birth < config.threshold))
      continue;
    if (death < config.threshold) {
      pairs[n++] = WritePairs(top, birth, death, p.death_x, p.death_y,
                              p.death_z, p.death_w, p.birth_x, p.birth_y,
                              p.birth_z, p.birth_w, config.print);
    } else {
      pairs[n++] = WritePairs(top, birth, config.threshold, p.death_x,
                              p.death_y, p.death_z, p.death_w, NO_VOXEL,
                              NO_VOXEL, NO_VOXEL, NO_VOXEL, config.print);
    }
  }
  pairs.erase(pairs.begin() + static_cast<std::ptrdiff_t>(n), pairs.end());
  return result;
}

} // namespace

Persistence compute_persistence(DenseCubicalGrids &dcg, const double *values,
                                bool fortran_order, Config &config) {
  StageTimer timer(config);
  bool top_only = false;
  if (config.method == ALEXANDER) {
    const uint32_t shape[4] = {dcg.ax, dcg.ay, dcg.az, dcg.aw};
    if (dcg.dim > 1 && std::all_of(shape, shape + dcg.dim,
                                   [](uint32_t n) { return n > 1; })) {
      Persistence result = top_dimension(dcg, values, fortran_order, config);
      timer.report("Top dimension");
      return result;
    }
    // The duality needs the input to span all d axes.  For 1D input and with
    // an axis of length 1, compute every dimension and keep the top one.
    config.method = LINKFIND;
    config.maxdim = static_cast<uint8_t>(dcg.dim - 1);
    top_only = true;
  }

  dcg.gridFromArray(values, config.embedded, fortran_order);
  dcg.finalisePadding();
  Persistence result{{}, InputCoordinates(dcg)};
  std::vector<WritePairs> &pairs = result.pairs;

  if (dcg.dim <= 2 && dcg.az == 1 && dcg.aw == 1 &&
      compute_PH_2d(&dcg, pairs, config)) {
    // 1D and 2D: dedicated union-find path for H_0 and H_1.
    timer.report("Computation");
  } else {
    // H_0 by union-find over the axis-aligned edges, then one cohomology
    // reduction per dimension.
    std::vector<Cube> ctr;
    union_find_h0(dcg, pairs, ctr, config);
    timer.report("Dimension 0");
    if (config.maxdim > 0) {
      ComputePairs cp(&dcg, pairs, config);
      cp.compute_pairs_main(ctr); // the edges left by union-find
      timer.report("Dimension 1");
      for (uint8_t d = 2; d <= config.maxdim; ++d) {
        cp.assemble_columns_to_reduce(ctr, d);
        cp.compute_pairs_main(ctr);
        timer.report(d == 2 ? "Dimension 2" : "Dimension 3");
      }
    }
  }
  if (top_only) {
    const uint8_t top = static_cast<uint8_t>(dcg.dim - 1);
    pairs.erase(std::remove_if(pairs.begin(), pairs.end(),
                               [top](const WritePairs &p) { return p.dim != top; }),
                pairs.end());
  }
  return result;
}

InputCoordinates::InputCoordinates(const DenseCubicalGrids &dcg)
    : pad_{(dcg.ax - dcg.img_x) / 2, (dcg.ay - dcg.img_y) / 2,
           (dcg.az - dcg.img_z) / 2, (dcg.aw - dcg.img_w) / 2},
      size_{dcg.img_x, dcg.img_y, dcg.img_z, dcg.img_w} {}
