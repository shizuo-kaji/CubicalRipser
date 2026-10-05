/* zigzag_runs.h

Presence runs of the cells of a cubical grid along a zigzag of mask frames.

A spatial cell is an anchor point x of the grid together with a set S of axes
(a bitmask); it spans x + [0,1]^S.  Its id is S * anchors + linear(x).  A run
is a maximal interval of coarse positions (see zigzag.h) on which the cell is
present.  FastZigzag treats every run as a distinct cell, so the runs are the
cells of the up-down filtration: additions in the order runs start, deletions
in the order they end.
*/

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "zigzag.h"

constexpr uint32_t ZIGZAG_NO_RUN = std::numeric_limits<uint32_t>::max();

struct ZigzagCellGrid {
  uint8_t dim{0};                // spatial dimension d
  uint32_t extent[3]{1, 1, 1};   // anchor-grid size per axis (1 if unused)
  uint64_t stride[3]{1, 1, 1};   // linear-anchor stride per axis
  uint64_t anchors{1};           // number of anchor points

  uint8_t axes(uint64_t cell) const {
    return static_cast<uint8_t>(cell / anchors);
  }
  uint64_t anchor(uint64_t cell) const { return cell % anchors; }
  uint32_t coord(uint64_t anchor, int axis) const {
    return static_cast<uint32_t>((anchor / stride[axis]) % extent[axis]);
  }
  uint8_t cell_dim(uint64_t cell) const {
    static const uint8_t popcount[8] = {0, 1, 1, 2, 1, 2, 2, 3};
    return popcount[axes(cell)];
  }
};

struct ZigzagRun {
  uint64_t cell;
  uint32_t start;    // first coarse position at which the cell is present
  uint32_t end;      // last coarse position at which the cell is present
  uint32_t deletion; // index of this run in deletion_order
};

struct ZigzagRunTable {
  ZigzagCellGrid grid;
  // Addition order: by start, then cell dimension, then cell id.
  std::vector<ZigzagRun> runs;
  // Run ids by end, then decreasing cell dimension, then cell id.
  std::vector<uint32_t> deletion_order;
  // Runs of cell c, ordered by start, are
  // cell_runs[cell_offset[c] .. cell_offset[c + 1]).
  std::vector<uint32_t> cell_offset;
  std::vector<uint32_t> cell_runs;

  // Calls f(q) for the run q of each facet of run r's cell that is alive
  // when r starts, i.e. the boundary of r in the up-down filtration.  Such a
  // run exists and contains all of r, since a face is present wherever its
  // coface is.
  template <class F> void for_each_face_run(uint32_t r, F &&f) const {
    const ZigzagRun &run = runs[r];
    const uint8_t axes = grid.axes(run.cell);
    const uint64_t anchor = grid.anchor(run.cell);
    for (int a = 0; a < grid.dim; ++a) {
      if ((axes & (1U << a)) == 0U)
        continue;
      const uint64_t face_base =
          static_cast<uint64_t>(axes & ~(1U << a)) * grid.anchors;
      f(run_at(face_base + anchor, run.start));
      f(run_at(face_base + anchor + grid.stride[a], run.start));
    }
  }

  // The run of `cell` containing `position`, or ZIGZAG_NO_RUN.
  uint32_t run_at(uint64_t cell, uint32_t position) const {
    const auto first = cell_runs.begin() + cell_offset[cell];
    const auto last = cell_runs.begin() + cell_offset[cell + 1];
    auto it = std::upper_bound(
        first, last, position,
        [this](uint32_t p, uint32_t q) { return p < runs[q].start; });
    if (it == first)
      return ZIGZAG_NO_RUN;
    --it;
    return runs[*it].end >= position ? *it : ZIGZAG_NO_RUN;
  }
};

ZigzagRunTable build_zigzag_runs(const ZigzagMasks &masks, bool tconstruction,
                                 bool union_connect);
