/* Zigzag persistence of mask sequences (FastZigzag over cell runs). */

#include "zigzag.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "zigzag_reduction.h"
#include "zigzag_runs.h"

namespace {

// An addition or deletion of a run in the input zigzag.
struct Event {
  uint32_t run;
  bool addition;
};

void anchor_voxel(const ZigzagRunTable &table, const ZigzagMasks &masks,
                  bool tconstruction, uint32_t run, int64_t voxel[3]) {
  const ZigzagCellGrid &grid = table.grid;
  const uint64_t anchor = grid.anchor(table.runs[run].cell);
  for (int a = 0; a < 3; ++a) {
    uint32_t x = a < grid.dim ? grid.coord(anchor, a) : 0;
    // T-construction anchors live on the vertex grid, one larger per axis.
    if (tconstruction && x >= masks.shape[a])
      x = masks.shape[a] - 1;
    voxel[a] = x;
  }
}

} // namespace

std::vector<ZigzagInterval> compute_zigzag(const ZigzagMasks &masks,
                                           bool tconstruction,
                                           bool union_connect, int maxdim,
                                           bool exhaustive) {
  const int top_dim = std::min<int>(maxdim, masks.spatial_dim - 1);
  if (top_dim < 0)
    return {};

  const ZigzagRunTable table =
      build_zigzag_runs(masks, tconstruction, union_connect);
  // A zigzag interval of dimension p comes from a coned pair of dimension p or
  // p + 1.
  const std::vector<ConedPair> pairs =
      reduce_coned_filtration(table, top_dim + 1, exhaustive);

  const uint32_t n = static_cast<uint32_t>(table.runs.size());
  auto run_of = [&](uint32_t rank) {
    return rank > n ? table.deletion_order[2 * n - rank] : rank - 1;
  };
  // Coarse position at which an event takes effect: an addition is visible
  // from the start of its run, a deletion from just after its end.
  auto position = [&](const Event &e) -> int64_t {
    const ZigzagRun &run = table.runs[e.run];
    return e.addition ? int64_t{run.start} : int64_t{run.end} + 1;
  };

  const int64_t last_position = 2 * int64_t{masks.frames} - 2;
  std::vector<ZigzagInterval> intervals;
  intervals.reserve(pairs.size());
  for (const ConedPair &pair : pairs) {
    if (pair.birth == 0)
      throw std::logic_error("zigzag: the cone vertex has no finite pair");
    // Map the coned pair to the up-down filtration and then to the input
    // zigzag (Dey-Hou, Propositions 19 and 15).  Deletions in the up-down
    // filtration correspond to cone cells.
    const bool birth_cone = pair.birth > n;
    const bool death_cone = pair.death > n;
    int dim = pair.dim;
    Event creator{run_of(pair.birth), true};
    Event destroyer{run_of(pair.death), true};
    if (!birth_cone && !death_cone) {
      // Ordinary: closed-open.
    } else if (!birth_cone) {
      // Extended: closed-closed, or open-open one dimension lower when the
      // deletion precedes the addition in the input zigzag.
      destroyer.addition = false;
      if (position(creator) > position(destroyer)) {
        std::swap(creator, destroyer);
        --dim;
      }
    } else {
      // Relative: open-closed one dimension lower, roles swapped.
      creator = {run_of(pair.death), false};
      destroyer = {run_of(pair.birth), false};
      --dim;
    }
    if (dim < 0)
      throw std::logic_error("zigzag: negative interval dimension");
    if (dim > top_dim)
      continue;
    const int64_t birth = position(creator);
    const int64_t death = position(destroyer) - 1;
    if (birth > death)
      continue; // lives only between two coarse positions

    ZigzagInterval interval;
    interval.dim = static_cast<uint8_t>(dim);
    interval.birth = static_cast<uint32_t>(birth);
    interval.death = static_cast<uint32_t>(death);
    anchor_voxel(table, masks, tconstruction, creator.run,
                 interval.birth_voxel);
    if (death == last_position) {
      // Destroyed by the deletions after the last frame, outside the input.
      std::fill(interval.death_voxel, interval.death_voxel + 3, int64_t{-1});
    } else {
      anchor_voxel(table, masks, tconstruction, destroyer.run,
                   interval.death_voxel);
    }
    intervals.push_back(interval);
  }

  auto key = [](const ZigzagInterval &i) {
    return std::make_tuple(i.dim, i.birth, i.death, i.birth_voxel[0],
                           i.birth_voxel[1], i.birth_voxel[2], i.death_voxel[0],
                           i.death_voxel[1], i.death_voxel[2]);
  };
  std::sort(intervals.begin(), intervals.end(),
            [&](const ZigzagInterval &a, const ZigzagInterval &b) {
              return key(a) < key(b);
            });
  return intervals;
}
