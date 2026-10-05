/* Presence runs of cubical cells along a zigzag of mask frames. */

#include "zigzag_runs.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

// Ranks of the up-down filtration (2 * runs + 1 cells) must fit in uint32_t.
constexpr size_t kMaxRuns = 0x7fffffffU;

uint8_t popcount3(uint8_t axes) {
  static const uint8_t popcount[8] = {0, 1, 1, 2, 1, 2, 2, 3};
  return popcount[axes];
}

// Visits every anchor point in linear order with its coordinates.
template <class F> void for_each_anchor(const ZigzagCellGrid &grid, F &&f) {
  uint64_t lin = 0;
  for (uint32_t x2 = 0; x2 < grid.extent[2]; ++x2)
    for (uint32_t x1 = 0; x1 < grid.extent[1]; ++x1)
      for (uint32_t x0 = 0; x0 < grid.extent[0]; ++x0)
        f(lin++, x0, x1, x2);
}

// Presence of every cell (one byte per cell id) in frame t.
//
// V-construction: a cell is present iff all its vertices are, computed by
// removing the highest axis: (x, S) = (x, S-a) & (x + e_a, S-a).
// T-construction: a cell is present iff some top cell containing it is,
// computed by adding the lowest missing axis: (x, S) = (x, S+b) | (x - e_b, S+b).
// Cells sticking out of the grid are never present.
void load_frame(const ZigzagMasks &masks, const ZigzagCellGrid &grid,
                bool tconstruction, uint32_t t,
                const std::vector<uint8_t> &up_types,
                std::vector<uint8_t> &present) {
  const uint64_t n = grid.anchors;
  const bool *frame = masks.data + static_cast<int64_t>(t) * masks.frame_stride;
  auto voxel = [&](uint32_t x0, uint32_t x1, uint32_t x2) -> uint8_t {
    return frame[static_cast<int64_t>(x0) * masks.stride[0] +
                 static_cast<int64_t>(x1) * masks.stride[1] +
                 static_cast<int64_t>(x2) * masks.stride[2]]
               ? 1
               : 0;
  };

  if (!tconstruction) {
    uint8_t *vertices = present.data();
    for_each_anchor(grid, [&](uint64_t lin, uint32_t x0, uint32_t x1,
                              uint32_t x2) { vertices[lin] = voxel(x0, x1, x2); });
    for (uint8_t axes : up_types) {
      if (axes == 0)
        continue;
      int a = 2;
      while ((axes & (1U << a)) == 0U)
        --a;
      const uint8_t *face = present.data() + (axes ^ (1U << a)) * n;
      uint8_t *cell = present.data() + axes * n;
      const uint64_t step = grid.stride[a];
      const uint32_t limit = grid.extent[a] - 1;
      for_each_anchor(grid, [&](uint64_t lin, uint32_t x0, uint32_t x1,
                                uint32_t x2) {
        const uint32_t x[3] = {x0, x1, x2};
        cell[lin] = x[a] < limit ? (face[lin] & face[lin + step]) : 0;
      });
    }
    return;
  }

  const uint8_t full = static_cast<uint8_t>((1U << grid.dim) - 1U);
  uint8_t *top = present.data() + full * n;
  for_each_anchor(grid, [&](uint64_t lin, uint32_t x0, uint32_t x1,
                            uint32_t x2) {
    const bool inside = x0 < masks.shape[0] && x1 < masks.shape[1] &&
                        x2 < masks.shape[2];
    top[lin] = inside ? voxel(x0, x1, x2) : 0;
  });
  for (auto it = up_types.rbegin(); it != up_types.rend(); ++it) {
    const uint8_t axes = *it;
    if (axes == full)
      continue;
    int b = 0;
    while ((axes & (1U << b)) != 0U)
      ++b;
    const uint8_t *coface = present.data() + (axes | (1U << b)) * n;
    uint8_t *cell = present.data() + axes * n;
    const uint64_t step = grid.stride[b];
    for_each_anchor(grid, [&](uint64_t lin, uint32_t x0, uint32_t x1,
                              uint32_t x2) {
      const uint32_t x[3] = {x0, x1, x2};
      cell[lin] = coface[lin] | (x[b] > 0 ? coface[lin - step] : 0);
    });
  }
}

} // namespace

ZigzagRunTable build_zigzag_runs(const ZigzagMasks &masks, bool tconstruction,
                                 bool union_connect) {
  ZigzagRunTable table;
  ZigzagCellGrid &grid = table.grid;
  grid.dim = masks.spatial_dim;
  uint64_t stride = 1;
  for (int a = 0; a < 3; ++a) {
    if (a < grid.dim)
      grid.extent[a] = masks.shape[a] + (tconstruction ? 1U : 0U);
    grid.stride[a] = stride;
    stride *= grid.extent[a];
  }
  grid.anchors = stride;

  const uint8_t type_count = static_cast<uint8_t>(1U << grid.dim);
  const size_t cell_count = static_cast<size_t>(type_count) * grid.anchors;

  // Additions go up in dimension (faces first), deletions go down.
  std::vector<uint8_t> up_types(type_count);
  for (uint8_t axes = 0; axes < type_count; ++axes)
    up_types[axes] = axes;
  std::stable_sort(up_types.begin(), up_types.end(), [](uint8_t a, uint8_t b) {
    return popcount3(a) < popcount3(b);
  });
  std::vector<uint8_t> down_types(up_types);
  std::stable_sort(down_types.begin(), down_types.end(),
                   [](uint8_t a, uint8_t b) {
                     return popcount3(a) > popcount3(b);
                   });

  std::vector<uint8_t> before(cell_count, 0);
  std::vector<uint8_t> now(cell_count, 0);
  // The run currently open for each cell; reused below as the CSR offsets.
  std::vector<uint32_t> open_run;
  open_run.reserve(cell_count + 1);
  open_run.resize(cell_count);

  // Opens a run for every cell present in `now` but not in `was`.
  auto open_runs = [&](const std::vector<uint8_t> *was, uint32_t position) {
    for (uint8_t axes : up_types) {
      const uint64_t base = static_cast<uint64_t>(axes) * grid.anchors;
      for (uint64_t cell = base; cell < base + grid.anchors; ++cell) {
        if (now[cell] == 0 || (was != nullptr && (*was)[cell] != 0))
          continue;
        if (table.runs.size() >= kMaxRuns)
          throw std::length_error("zigzag: too many cell runs (limit 2^31-1)");
        open_run[cell] = static_cast<uint32_t>(table.runs.size());
        table.runs.push_back({cell, position, 0, 0});
      }
    }
  };
  // Closes the run of every cell present in `was` but not in `is`.
  auto close_runs = [&](const std::vector<uint8_t> &was,
                        const std::vector<uint8_t> *is, uint32_t position) {
    for (uint8_t axes : down_types) {
      const uint64_t base = static_cast<uint64_t>(axes) * grid.anchors;
      for (uint64_t cell = base; cell < base + grid.anchors; ++cell) {
        if (was[cell] == 0 || (is != nullptr && (*is)[cell] != 0))
          continue;
        ZigzagRun &run = table.runs[open_run[cell]];
        run.end = position;
        run.deletion = static_cast<uint32_t>(table.deletion_order.size());
        table.deletion_order.push_back(open_run[cell]);
      }
    }
  };

  // Frame t sits at position 2t.  With intersections the connecting complex
  // at 2t+1 loses the cells that leave, and the cells that enter appear at
  // 2t+2; with unions they appear at 2t+1 and leave after it.
  load_frame(masks, grid, tconstruction, 0, up_types, now);
  open_runs(nullptr, 0);
  for (uint32_t t = 1; t < masks.frames; ++t) {
    before.swap(now);
    load_frame(masks, grid, tconstruction, t, up_types, now);
    if (union_connect) {
      open_runs(&before, 2 * t - 1);
      close_runs(before, &now, 2 * t - 1);
    } else {
      close_runs(before, &now, 2 * t - 2);
      open_runs(&before, 2 * t);
    }
  }
  close_runs(now, nullptr, 2 * masks.frames - 2);
  std::vector<uint8_t>().swap(before);
  std::vector<uint8_t>().swap(now);

  // Group run ids by cell (counting sort; ids increase with start).
  std::vector<uint32_t> &offset = open_run;
  offset.assign(cell_count + 1, 0);
  for (const ZigzagRun &run : table.runs)
    ++offset[run.cell + 1];
  for (size_t c = 0; c < cell_count; ++c)
    offset[c + 1] += offset[c];
  table.cell_runs.resize(table.runs.size());
  for (uint32_t r = 0; r < table.runs.size(); ++r)
    table.cell_runs[offset[table.runs[r].cell]++] = r;
  for (size_t c = cell_count; c > 0; --c)
    offset[c] = offset[c - 1];
  offset[0] = 0;
  table.cell_offset = std::move(open_run);
  return table;
}
