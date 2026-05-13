/* compute_pairs.cpp

This file is part of CubicalRipser_3dim.
Copyright 2017-2018 Takeki Sudo and Kazushi Ahara.
Modified by Shizuo Kaji

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License along
with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <time.h>
#include <unordered_map>
#include <utility>
#include <vector>
// #ifdef _OPENMP
// #include <omp.h>
// #endif

using namespace std;

#include "coboundary_enumerator.h"
#include "compute_pairs.h"
#include "cube.h"
#include "dense_cubical_grids.h"
#include "radix_sort.h"
#include "write_pairs.h"

namespace {

uint8_t mask_count_for_cell_dim(const DenseCubicalGrids *dcg, uint8_t cell_dim) {
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

  if (dcg->config->tconstruction && dcg->az == 1 && dcg->dim < 4) {
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

void sort_working_column(CachedColumn &column) {
  if (column.size() > 1) {
    std::sort(column.begin(), column.end(), CubeComparator());
  }
}

void normalize_column(CachedColumn &column) {
  sort_working_column(column);
  if (column.size() < 2) {
    return;
  }

  size_t write = 0;
  for (size_t read = 0; read < column.size();) {
    size_t next = read + 1;
    while (next < column.size() && column[next].index == column[read].index) {
      ++next;
    }
    if (((next - read) & 1U) != 0U) {
      column[write++] = column[read];
    }
    read = next;
  }
  column.resize(write);
}

void xor_sorted_columns(CachedColumn &column, const CachedColumn &addon,
                        CachedColumn &scratch) {
  scratch.clear();
  scratch.reserve(column.size() + addon.size());

  CubeComparator less;
  size_t i = 0;
  size_t j = 0;
  while (i < column.size() && j < addon.size()) {
    if (column[i].index == addon[j].index) {
      ++i;
      ++j;
    } else if (less(column[i], addon[j])) {
      scratch.push_back(column[i++]);
    } else {
      scratch.push_back(addon[j++]);
    }
  }

  while (i < column.size()) {
    scratch.push_back(column[i++]);
  }
  while (j < addon.size()) {
    scratch.push_back(addon[j++]);
  }

  column.swap(scratch);
}

size_t suggested_cache_reserve(size_t column_count, uint32_t cache_size) {
  const size_t bounded_limit = std::min<size_t>(column_count, cache_size);
  const size_t expected_reduced = column_count / 8U + 1024U;
  return std::min(bounded_limit, expected_reduced) + 10U;
}

} // namespace

void DensePivotTable::reset(DenseCubicalGrids *dcg, uint8_t target_dim) {
  clearing_bits.clear();
  ax = dcg->ax;
  ay = dcg->ay;
  az = dcg->az;
  aw = dcg->aw;
  cell_dim = target_dim;
  mask_count = mask_count_for_cell_dim(dcg, target_dim);
  if (mask_count == 0) {
    deactivate();
    return;
  }
  const size_t total_slots =
      static_cast<size_t>(ax) * static_cast<size_t>(ay) *
      static_cast<size_t>(az) * static_cast<size_t>(aw) *
      static_cast<size_t>(mask_count);
  slots.assign(total_slots, empty_value());
}

void DensePivotTable::compress_to_clearing_bits() {
  if (slots.empty()) {
    return;
  }
  const size_t total_slots = slots.size();
  clearing_bits.assign((total_slots + 63U) >> 6, 0ULL);
  for (size_t i = 0; i < total_slots; ++i) {
    if (slots[i] != empty_value()) {
      clearing_bits[i >> 6] |= 1ULL << (i & 63U);
    }
  }
  std::vector<uint32_t>().swap(slots);
}

ComputePairs::ComputePairs(DenseCubicalGrids *_dcg,
                           std::vector<WritePairs> &_wp, Config &_config)
    : dcg(_dcg), pivot_column_index(nullptr), dim(1), wp(&_wp),
      config(&_config) { // Initialize dim to 1 (default method is LINK_FIND,
                         // where we skip dim=0)
}

void ComputePairs::compute_pairs_main(vector<Cube> &ctr) {
  auto ctl_size = ctr.size();
  if (config->verbose) {
    cout << "# columns to reduce: " << ctl_size << endl;
  }

  if (!pivot_column_index) {
    pivot_column_index = std::make_unique<DensePivotTable>();
  }
  pivot_column_index->reset(dcg, dim + 1);

  const bool use_heap_working_column =
      (!config->vector_working_column);
  if (use_heap_working_column) {
    std::vector<WritePairs> local_wp;
    int num_apparent_pairs = 0;

    CachedColumn batch_column;
    batch_column.reserve((dcg->dim == 4) ? 8u : 6u);
    CoboundaryEnumerator cofaces(dcg, dim);
    unordered_map<uint32_t, CachedColumn> recorded_wc;
    queue<uint32_t> cached_column_idx;
    recorded_wc.max_load_factor(0.7f);
    recorded_wc.reserve(suggested_cache_reserve(ctl_size, config->cache_size));
    const bool bounded_cache = static_cast<size_t>(config->cache_size) < ctl_size;
    CubeQue working_coboundary;
    working_coboundary.reserve(64);

    for (uint32_t i = 0; i < ctl_size; ++i) {
      working_coboundary.clear();
      double birth = ctr[i].birth;
      auto j = i;
      Cube pivot;
      bool might_be_apparent_pair = true;
      bool found_apparent_pair = false;
      int num_recurse = 0;

      for (int k = 0; k < config->maxiter; ++k) {
        bool cache_hit = false;
        if (i != j) {
          auto findWc = recorded_wc.find(j);
          if (findWc != recorded_wc.end()) {
            cache_hit = true;
            for (const auto &c : findWc->second) {
              working_coboundary.push(c);
            }
          }
        }
        if (!cache_hit) {
          batch_column.clear();
          cofaces.setCoboundaryEnumerator(ctr[j]);
          const double column_birth = ctr[j].birth;
          while (cofaces.hasNextCoface()) {
            batch_column.push_back(cofaces.nextCoface);
            if (might_be_apparent_pair &&
                (column_birth == cofaces.nextCoface.birth)) {
              auto apparent =
                  pivot_column_index->insert(cofaces.nextCoface.index, i);
              if (apparent.second) {
                found_apparent_pair = true;
                ++num_apparent_pairs;
                break;
              }
              might_be_apparent_pair = false;
            }
          }
          if (found_apparent_pair) {
            break;
          }
          for (const auto &e : batch_column) {
            working_coboundary.push(e);
          }
        }
        pivot = get_pivot(working_coboundary);
        if (pivot.index != NONE) {
          auto insert_result = pivot_column_index->insert(pivot.index, i);
          if (!insert_result.second) {
            j = insert_result.first;
            num_recurse++;
            continue;
          }

          if (num_recurse >= config->min_recursion_to_cache) {
            add_cache(i, working_coboundary, recorded_wc);
            if (bounded_cache) {
              cached_column_idx.push(i);
              if (cached_column_idx.size() > config->cache_size) {
                recorded_wc.erase(cached_column_idx.front());
                cached_column_idx.pop();
              }
            }
          }
          double death = pivot.birth;
          if (birth != death) {
            local_wp.emplace_back(
                WritePairs(dim, ctr[i], pivot, dcg, config->print));
          }
          break;
        }

        if (birth != dcg->threshold) {
          local_wp.emplace_back(
              WritePairs(dim, birth, dcg->threshold, ctr[i].x(), ctr[i].y(),
                         ctr[i].z(), ctr[i].w(), 0, 0, 0, 0, config->print));
        }
        break;
      }
    }

    wp->insert(wp->end(), local_wp.begin(), local_wp.end());
    if (config->verbose) {
      cout << "# apparent pairs: " << num_apparent_pairs << endl;
    }
    if (config->explicit_clearing && pivot_column_index) {
      pivot_column_index->compress_to_clearing_bits();
    }
    return;
  }

  std::vector<WritePairs> local_wp;
  int num_apparent_pairs = 0;

  CachedColumn batch_column;
  batch_column.reserve((dcg->dim == 4) ? 8u : 6u);
  CoboundaryEnumerator cofaces(dcg, dim);
  unordered_map<uint32_t, CachedColumn> recorded_wc;
  queue<uint32_t> cached_column_idx;
  recorded_wc.max_load_factor(0.7f);
  recorded_wc.reserve(suggested_cache_reserve(ctl_size, config->cache_size));
  const bool bounded_cache = static_cast<size_t>(config->cache_size) < ctl_size;
  CachedColumn working_coboundary;
  working_coboundary.reserve(64);
  CachedColumn merge_scratch;
  merge_scratch.reserve(64);
  int local_apparent_pairs = 0;

  for (uint32_t i = 0; i < ctl_size; ++i) {
    working_coboundary.clear();
    double birth = ctr[i].birth;
    auto j = i;
    Cube pivot;
    bool might_be_apparent_pair = true;
    bool found_apparent_pair = false;
    int num_recurse = 0;

    for (int k = 0; k < config->maxiter; ++k) {
      bool cache_hit = false;
      if (i != j) {
        auto findWc = recorded_wc.find(j);
        if (findWc != recorded_wc.end()) {
          cache_hit = true;
          xor_sorted_columns(working_coboundary, findWc->second,
                             merge_scratch);
        }
      }
      if (!cache_hit) {
        batch_column.clear();
        cofaces.setCoboundaryEnumerator(ctr[j]);
        const double column_birth = ctr[j].birth;
        while (cofaces.hasNextCoface()) {
          batch_column.push_back(cofaces.nextCoface);
          if (might_be_apparent_pair &&
              (column_birth == cofaces.nextCoface.birth)) {
            auto apparent =
                pivot_column_index->insert(cofaces.nextCoface.index, i);
            if (apparent.second) { // inserted
              found_apparent_pair = true;
              ++local_apparent_pairs;
              break;
            }
            might_be_apparent_pair = false;
          }
        }
        if (found_apparent_pair)
          break;
        normalize_column(batch_column);
        xor_sorted_columns(working_coboundary, batch_column, merge_scratch);
      }
      pivot = get_pivot(working_coboundary);
      if (pivot.index != NONE) {
        auto insert_result = pivot_column_index->insert(pivot.index, i);
        if (!insert_result.second) { // found existing entry
          j = insert_result.first;
          num_recurse++;
          continue;
        } else { // new pivot inserted
          if (num_recurse >= config->min_recursion_to_cache) {
            add_cache(i, working_coboundary, recorded_wc);
            if (bounded_cache) {
              cached_column_idx.push(i);
              if (cached_column_idx.size() > config->cache_size) {
                recorded_wc.erase(cached_column_idx.front());
                cached_column_idx.pop();
              }
            }
          }
          double death = pivot.birth;
          if (birth != death) {
            local_wp.emplace_back(
                WritePairs(dim, ctr[i], pivot, dcg, config->print));
          }
          break;
        }
      } else {
        if (birth != dcg->threshold) {
          local_wp.emplace_back(
              WritePairs(dim, birth, dcg->threshold, ctr[i].x(), ctr[i].y(),
                         ctr[i].z(), ctr[i].w(), 0, 0, 0, 0, config->print));
        }
        break;
      }
    }
  }
  num_apparent_pairs += local_apparent_pairs;

  wp->insert(wp->end(), local_wp.begin(), local_wp.end());

  if (config->verbose) {
    cout << "# apparent pairs: " << num_apparent_pairs << endl;
  }
  if (config->explicit_clearing && pivot_column_index) {
    pivot_column_index->compress_to_clearing_bits();
  }
}

// cache a new reduced column after mod 2
void ComputePairs::add_cache(
    uint32_t i, CachedColumn &wc,
    unordered_map<uint32_t, CachedColumn> &recorded_wc) {
  recorded_wc.emplace(i, std::move(wc));
}

void ComputePairs::add_cache(
    uint32_t i, CubeQue &wc,
    unordered_map<uint32_t, CachedColumn> &recorded_wc) {
  CachedColumn clean_wc;
  clean_wc.reserve(wc.size());
  while (!wc.empty()) {
    auto c = wc.top();
    wc.pop();
    if (!wc.empty() && c.index == wc.top().index) {
      wc.pop();
    } else {
      clean_wc.push_back(c);
    }
  }
  recorded_wc.emplace(i, std::move(clean_wc));
}

// get the pivot from a column after mod 2
Cube ComputePairs::pop_pivot(CachedColumn &column) {
  if (column.empty()) {
    return Cube();
  }
  const Cube pivot = column.back();
  column.pop_back();
  return pivot;
}

Cube ComputePairs::get_pivot(CachedColumn &column) {
  if (column.empty()) {
    return Cube();
  }
  return column.back();
}

Cube ComputePairs::pop_pivot(CubeQue &column) {
  if (column.empty()) {
    return Cube();
  } else {
    auto pivot = column.top();
    column.pop();

    while (!column.empty() && column.top().index == pivot.index) {
      column.pop();
      if (column.empty()) {
        return Cube();
      } else {
        pivot = column.top();
        column.pop();
      }
    }
    return pivot;
  }
}

Cube ComputePairs::get_pivot(CubeQue &column) {
  Cube result = pop_pivot(column);
  if (result.index != NONE) {
    column.push(result);
  }
  return result;
}

// enumerate and sort columns for a new dimension
void ComputePairs::assemble_columns_to_reduce(vector<Cube> &ctr, uint8_t _dim) {
  dim = _dim;
  ctr.clear();
  double birth;
  uint8_t max_m = 0;
  // Determine number of mask types per target dimension based on ambient
  // dimension 3D: dim 0/1/2/3 => 1/3/3/1 4D: dim 0/1/2/3/4 => 1/4/6/4/1
  if (dcg->dim == 4) {
    switch (dim) {
    case 0:
      max_m = 1;
      break;
    case 1:
      max_m = 4;
      break;
    case 2:
      max_m = 6;
      break;
    case 3:
      max_m = 4;
      break;
    default:
      max_m = 1;
      break; // dim == 4
    }
  } else {
    switch (dim) {
    case 0:
      max_m = 1;
      break;
    case 1:
      max_m = 3;
      break;
    case 2:
      max_m = 3;
      break;
    default:
      max_m = 1;
      break; // dim == 3 (or lower)
    }
  }
  // Special-case: 2D image under T-construction (embedded in 3D with az==1)
  // Restrict mask variants to in-plane components
  if (dcg->config->tconstruction && dcg->az == 1 && dcg->dim < 4) {
    switch (dim) {
    case 0:
      max_m = 1;
      break; // 0-cells: single variant
    case 1:
      max_m = 2;
      break; // 1-cells: only x- and y-edges (no z)
    default:
      max_m = 1;
      break; // 2-cells: single square variant (xy)
    }
  }
  if (dim == 0) {
    if (pivot_column_index) {
      pivot_column_index->deactivate();
    }
  }
  const size_t max_ctr_size =
      static_cast<size_t>(max_m) * static_cast<size_t>(dcg->ax) *
      static_cast<size_t>(dcg->ay) * static_cast<size_t>(dcg->az) *
      static_cast<size_t>(dcg->aw);
  // Cap reserve to avoid over-allocating when many cells are filtered by
  // threshold.
  const size_t reserve_target =
      std::min(max_ctr_size, static_cast<size_t>(8000000));
  ctr.reserve(reserve_target);
  const double threshold = dcg->threshold;
  const bool skip_paired_columns =
      pivot_column_index && pivot_column_index->active_for(dim);
  // Note: Stage 2 Morse PH filtering was explored and found unsound -- dropping
  // V-paired cells from ctr or from the working coboundary is NOT equivalent
  // to xor-ing with a trivial column {tau} when a lower-birth column naturally
  // picks tau up as its pivot.  A correct integration needs a rebuilt Morse
  // boundary operator over V-paths.  See agents.md section 10.
  for (uint8_t m = 0; m < max_m; ++m) {
    for (uint32_t w = 0; w < dcg->aw; ++w) {
      for (uint32_t z = 0; z < dcg->az; ++z) {
        for (uint32_t y = 0; y < dcg->ay; ++y) {
          for (uint32_t x = 0; x < dcg->ax; ++x) {
            birth = dcg->getBirth(x, y, z, w, m, dim);
            //                        cout << x << "," << y << "," << z << ", "
            //                        << m << "," << birth << endl;
            if (birth < threshold) {
              const uint64_t index = static_cast<uint64_t>(x) |
                                     (static_cast<uint64_t>(y) << 15) |
                                     (static_cast<uint64_t>(z) << 30) |
                                     (static_cast<uint64_t>(w) << 45) |
                                     (static_cast<uint64_t>(m) << 60);
              if (!skip_paired_columns ||
                  !pivot_column_index->contains(index)) {
                ctr.emplace_back(birth, index);
              }
            }
          }
        }
      }
    }
  }
  clock_t start = clock();
  cubicalripser::radix_sort_cubes(ctr);
  if (config->verbose) {
    clock_t end = clock();
    const double time =
        static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;
    cout << "Sorting took: " << time << endl;
  }
}
