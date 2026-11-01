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
#include <cfloat>
#include <cstdint>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <time.h>
#include <type_traits>
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

// Two columns sorted in CubeComparator-descending order, without repeats,
// summed into out.
void xor_merge_descending(const Cube *a, const Cube *a_end, const Cube *b,
                          const Cube *b_end, CachedColumn &out) {
  const CubeComparator less;
  while (a != a_end && b != b_end) {
    if (a->index == b->index) {
      ++a;
      ++b;
    } else if (less(*b, *a)) {
      out.push_back(*a++);
    } else {
      out.push_back(*b++);
    }
  }
  out.insert(out.end(), a, a_end);
  out.insert(out.end(), b, b_end);
}

// The heap working column.  A cached column added to it is not copied into
// the heap: it stays in the cache, sorted pivot first, and is read through a
// cursor, with a second heap ordering the cursors by the cube they point at.
// Only freshly enumerated coboundaries go into the cube heap.  The pivot is
// held outside both heaps; the column added next has the same pivot, so the
// two copies cancel without a heap operation.
class LazyColumn {
  struct Cursor {
    Cube head;
    uint32_t span;
  };
  struct CursorLess {
    bool operator()(const Cursor &a, const Cursor &b) const {
      return CubeComparator()(a.head, b.head);
    }
  };

public:
  void reserve(size_t n) { fresh.reserve(n); }

  void clear() {
    fresh.clear();
    spans.clear();
    heads.clear();
    held = Cube();
  }

  // Add a cached column whose pivot is the held one.  Its storage must stay
  // in place until the column is cleared.
  void add_cached(const CachedColumn &column) {
    held = Cube(); // column.front() cancels it
    if (column.size() > 1) {
      spans.emplace_back(column.data() + 1, column.data() + column.size());
      heads.push_back({column[1], static_cast<uint32_t>(spans.size() - 1)});
      std::push_heap(heads.begin(), heads.end(), CursorLess());
    }
  }

  // Add a coboundary, which may repeat cubes.
  void add_fresh(const CachedColumn &cubes) {
    bool holding = held.index != NONE;
    for (const auto &cube : cubes) {
      if (holding && cube.index == held.index) {
        holding = false;
      } else {
        fresh.push(cube);
      }
    }
    if (holding) {
      fresh.push(held);
    }
    held = Cube();
  }

  // The largest cube present an odd number of times, which is then held;
  // Cube() for the zero column.
  Cube pop_pivot() {
    const CubeComparator less;
    const CursorLess cursor_less;
    while (!fresh.empty() || !heads.empty()) {
      const Cube top = (heads.empty() || (!fresh.empty() &&
                                          !less(fresh.top(), heads.front().head)))
                           ? fresh.top()
                           : heads.front().head;
      unsigned count = 0;
      while (!fresh.empty() && fresh.top().index == top.index) {
        fresh.pop();
        ++count;
      }
      while (!heads.empty() && heads.front().head.index == top.index) {
        const uint32_t s = heads.front().span;
        std::pop_heap(heads.begin(), heads.end(), cursor_less);
        heads.pop_back();
        ++count;
        auto &span = spans[s];
        if (++span.first != span.second) {
          heads.push_back({*span.first, s});
          std::push_heap(heads.begin(), heads.end(), cursor_less);
        }
      }
      if ((count & 1U) != 0U) {
        held = top;
        return held;
      }
    }
    held = Cube();
    return held;
  }

  // The column, sorted pivot first without repeats, as an exact-size vector.
  // The four arguments are scratch.
  CachedColumn reduced(CachedColumn &runs, CachedColumn &runs_tmp,
                       vector<size_t> &ends, vector<size_t> &ends_tmp);

private:
  CubeQue fresh;
  // [cursor, end) of each added cached column.
  vector<std::pair<const Cube *, const Cube *>> spans;
  vector<Cursor> heads; // a heap under CursorLess
  Cube held;
};

CachedColumn LazyColumn::reduced(CachedColumn &runs, CachedColumn &runs_tmp,
                                 vector<size_t> &ends,
                                 vector<size_t> &ends_tmp) {
  // The held pivot and the fresh cubes left, sorted and with pairs cancelled,
  // make the first run; what is left of each cached column is another.  The
  // runs are summed pairwise until one is left.
  vector<Cube> &items = fresh.container();
  const CubeComparator less;
  std::sort(items.begin(), items.end(),
            [&less](const Cube &a, const Cube &b) { return less(b, a); });
  runs.clear();
  ends.clear();
  if (held.index != NONE) {
    runs.push_back(held);
  }
  for (size_t k = 0; k < items.size();) {
    if (k + 1 < items.size() && items[k].index == items[k + 1].index) {
      k += 2;
    } else {
      runs.push_back(items[k++]);
    }
  }
  items.clear();
  ends.push_back(runs.size());
  for (const auto &span : spans) {
    if (span.first != span.second) {
      runs.insert(runs.end(), span.first, span.second);
      ends.push_back(runs.size());
    }
  }
  while (ends.size() > 1) {
    runs_tmp.clear();
    ends_tmp.clear();
    size_t start = 0;
    size_t r = 0;
    for (; r + 1 < ends.size(); r += 2) {
      xor_merge_descending(runs.data() + start, runs.data() + ends[r],
                           runs.data() + ends[r], runs.data() + ends[r + 1],
                           runs_tmp);
      ends_tmp.push_back(runs_tmp.size());
      start = ends[r + 1];
    }
    if (r < ends.size()) {
      runs_tmp.insert(runs_tmp.end(), runs.begin() + start,
                      runs.begin() + ends[r]);
      ends_tmp.push_back(runs_tmp.size());
    }
    runs.swap(runs_tmp);
    ends.swap(ends_tmp);
  }
  CachedColumn column(runs.begin(), runs.end());
  // Do not keep the scratch of an exceptionally long column.
  if (runs.capacity() > (1U << 16) || runs_tmp.capacity() > (1U << 16)) {
    CachedColumn().swap(runs);
    CachedColumn().swap(runs_tmp);
  }
  return column;
}

// The two working-column representations differ only in how a column is
// accumulated.  Overloads keep the reduction loop itself single-sourced while
// the template still emits one specialised copy per representation, exactly as
// the two hand-written loops used to.
void column_add(LazyColumn &column, const CachedColumn &addon, CachedColumn &) {
  column.add_cached(addon);
}

void column_add(CachedColumn &column, const CachedColumn &addon,
                CachedColumn &scratch) {
  xor_sorted_columns(column, addon, scratch);
}

// A freshly enumerated coboundary still has duplicates; the sorted-vector
// representation must normalise it first, the heap absorbs them on pop.
void column_add_fresh(LazyColumn &column, CachedColumn &batch, CachedColumn &) {
  column.add_fresh(batch);
}

void column_add_fresh(CachedColumn &column, CachedColumn &batch,
                      CachedColumn &scratch) {
  normalize_column(batch);
  xor_sorted_columns(column, batch, scratch);
}

// The pivot stays in the column: the sorted vector's last cube, or held.
Cube get_pivot(LazyColumn &column) { return column.pop_pivot(); }

Cube get_pivot(CachedColumn &column) {
  if (column.empty()) {
    return Cube();
  }
  return column.back();
}

// Cache a column whose pivot is new.
void add_cache(uint32_t i, LazyColumn &wc,
               unordered_map<uint32_t, CachedColumn> &recorded_wc,
               CachedColumn &runs, CachedColumn &runs_tmp,
               vector<size_t> &ends, vector<size_t> &ends_tmp) {
  // The cached columns wc reads stay in place: unordered_map does not move
  // its elements on rehash, and the bounded cache erases one only after this.
  recorded_wc.emplace(i, wc.reduced(runs, runs_tmp, ends, ends_tmp));
}

void add_cache(uint32_t i, CachedColumn &wc,
               unordered_map<uint32_t, CachedColumn> &recorded_wc,
               CachedColumn &, CachedColumn &, vector<size_t> &,
               vector<size_t> &) {
  recorded_wc.emplace(i, std::move(wc));
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
  mask_count = dcg->cell_type_count(target_dim);
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

// Standard left-to-right cohomology column reduction.  `Column` selects the
// working-column representation: heaps reading cached columns in place
// (LazyColumn), or a sorted vector XOR-merged in place.  Both instantiations
// are emitted separately, so neither pays for the other.
template <typename Column>
void ComputePairs::reduce_columns(vector<Cube> &ctr, size_t ctl_size,
                                  int &num_apparent_pairs) {
  std::vector<WritePairs> local_wp;

  CachedColumn batch_column;
  batch_column.reserve((dcg->dim == 4) ? 8u : 6u);
  CoboundaryEnumerator cofaces(dcg, dim);
  unordered_map<uint32_t, CachedColumn> recorded_wc;
  queue<uint32_t> cached_column_idx;
  recorded_wc.max_load_factor(0.7f);
  recorded_wc.reserve(suggested_cache_reserve(ctl_size, config->cache_size));
  const bool bounded_cache = static_cast<size_t>(config->cache_size) < ctl_size;
  Column working_coboundary;
  working_coboundary.reserve(64);
  CachedColumn merge_scratch;
  merge_scratch.reserve(64);
  // Scratch for LazyColumn::reduced().  As members of LazyColumn, or of one
  // struct, these made 4D H1 4-8% slower than as separate variables (Apple
  // clang, arm64).
  CachedColumn runs, runs_tmp;
  vector<size_t> run_ends, run_ends_tmp;
  // Emergent columns are never cached.  Their pivot-table entries carry this
  // tag in the top bit, so adding one skips the cache lookup.  Column indices
  // need that bit from 2^31 - 1 columns on, and then there is no tag.
  const uint32_t emergent_tag = (ctl_size < 0x7fffffffU) ? 0x80000000U : 0U;

  for (uint32_t i = 0; i < ctl_size; ++i) {
    working_coboundary.clear();
    double birth = ctr[i].birth;
    auto j = i;
    bool j_emergent = false;
    Cube pivot;
    bool might_be_apparent_pair = true;
    bool found_apparent_pair = false;
    int num_recurse = 0;

    for (int k = 0; k < config->maxiter; ++k) {
      bool cache_hit = false;
      if (i != j && !j_emergent) {
        auto findWc = recorded_wc.find(j);
        if (findWc != recorded_wc.end()) {
          cache_hit = true;
          column_add(working_coboundary, findWc->second, merge_scratch);
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
            auto apparent = pivot_column_index->insert(
                cofaces.nextCoface.index, i | emergent_tag);
            if (apparent.second) { // inserted
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
        column_add_fresh(working_coboundary, batch_column, merge_scratch);
      }
      pivot = get_pivot(working_coboundary);
      if (pivot.index != NONE) {
        auto insert_result = pivot_column_index->insert(pivot.index, i);
        if (!insert_result.second) { // found existing entry
          j = insert_result.first & ~emergent_tag;
          j_emergent = (insert_result.first & emergent_tag) != 0U;
          num_recurse++;
          continue;
        }
        // new pivot inserted
        if (num_recurse >= config->min_recursion_to_cache) {
          add_cache(i, working_coboundary, recorded_wc, runs, runs_tmp,
                    run_ends, run_ends_tmp);
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
        const auto b = dcg->ParentVoxel(dim, ctr[i]);
        local_wp.emplace_back(
            WritePairs(dim, birth, dcg->threshold, b[0], b[1], b[2], b[3],
                       NO_VOXEL, NO_VOXEL, NO_VOXEL, NO_VOXEL, config->print));
      }
      break;
    }
  }

  wp->insert(wp->end(), local_wp.begin(), local_wp.end());
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

  int num_apparent_pairs = 0;
  if (config->vector_working_column) {
    reduce_columns<CachedColumn>(ctr, ctl_size, num_apparent_pairs);
  } else {
    reduce_columns<LazyColumn>(ctr, ctl_size, num_apparent_pairs);
  }

  if (config->verbose) {
    cout << "# apparent pairs: " << num_apparent_pairs << endl;
  }
  if (config->explicit_clearing && pivot_column_index) {
    pivot_column_index->compress_to_clearing_bits();
  }
}

// enumerate and sort columns for a new dimension
void ComputePairs::assemble_columns_to_reduce(vector<Cube> &ctr, uint8_t _dim) {
  dim = _dim;
  ctr.clear();
  double birth;
  const uint8_t max_m = dcg->cell_type_count(dim);
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
  // boundary operator over V-paths.  See improvements.md section 10.
  //
  // Under the V-construction the anchor is one of a cell's vertices, so a
  // cell anchored at the threshold never enters: one value decides it.  The
  // scan tests it only with a threshold under the V-construction; otherwise
  // the loop is the plain one.
  auto scan = [&](auto skip_by_anchor) {
    for (uint8_t m = 0; m < max_m; ++m) {
      for (uint32_t w = 0; w < dcg->aw; ++w) {
        for (uint32_t z = 0; z < dcg->az; ++z) {
          for (uint32_t y = 0; y < dcg->ay; ++y) {
            for (uint32_t x = 0; x < dcg->ax; ++x) {
              if (decltype(skip_by_anchor)::value &&
                  *dcg->cell_ptr(x, y, z, w) >= threshold) {
                continue;
              }
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
  };
  if (!dcg->tconstruction() && threshold < DBL_MAX) {
    scan(std::true_type{});
  } else {
    scan(std::false_type{});
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
