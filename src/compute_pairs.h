/* compute_pairs.h

This file is part of CubicalRipser
Copyright 2017-2018 Takeki Sudo and Kazushi Ahara.
Modified by Shizuo Kaji

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License along
with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#include "config.h"
#include "cube.h"
#include <cstddef>
#include <limits>
#include <unordered_map>
#include <vector>

// #define GOOGLE_HASH

#ifdef GOOGLE_HASH
#include "sparsehash/dense_hash_map"
#endif

#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

typedef vector<Cube> CachedColumn;
class CoboundaryEnumerator;
class DenseCubicalGrids;
class WritePairs;

class CubeQue : public priority_queue<Cube, vector<Cube>, CubeComparator> {
public:
  using priority_queue<Cube, vector<Cube>, CubeComparator>::priority_queue;

  void reserve(size_t n) { this->c.reserve(n); }
  void clear() { this->c.clear(); }
};

class DensePivotTable {
private:
  static constexpr uint32_t empty_value() {
    return std::numeric_limits<uint32_t>::max();
  }

  std::vector<uint32_t> slots;
  uint32_t ax{0};
  uint32_t ay{0};
  uint32_t az{0};
  uint32_t aw{0};
  uint8_t mask_count{0};
  uint8_t cell_dim{std::numeric_limits<uint8_t>::max()};

  size_t linearize(uint64_t packed_index) const {
    const size_t x = static_cast<size_t>(packed_index & 0x7fffULL);
    const size_t y = static_cast<size_t>((packed_index >> 15) & 0x7fffULL);
    const size_t z = static_cast<size_t>((packed_index >> 30) & 0x7fffULL);
    const size_t w = static_cast<size_t>((packed_index >> 45) & 0x7fffULL);
    const size_t m = static_cast<size_t>((packed_index >> 60) & 0xfULL);
    return (((w * static_cast<size_t>(az) + z) * static_cast<size_t>(ay) + y) *
                static_cast<size_t>(ax) +
            x) *
               static_cast<size_t>(mask_count) +
           m;
  }

public:
  struct InsertResult {
    uint32_t first;
    bool second;
  };

  void reset(DenseCubicalGrids *dcg, uint8_t target_dim);

  void deactivate() {
    slots.clear();
    ax = ay = az = aw = 0;
    mask_count = 0;
    cell_dim = std::numeric_limits<uint8_t>::max();
  }

  bool active_for(uint8_t target_dim) const {
    return cell_dim == target_dim && !slots.empty();
  }

  InsertResult insert(uint64_t k, uint32_t v) {
    const size_t idx = linearize(k);
    if (slots[idx] == empty_value()) {
      slots[idx] = v;
      return {v, true};
    }
    return {slots[idx], false};
  }

  bool contains(uint64_t k) const {
    return !slots.empty() && slots[linearize(k)] != empty_value();
  }
};

class ComputePairs {
private:
  DenseCubicalGrids *dcg;
  std::unique_ptr<DensePivotTable> pivot_column_index;
  uint8_t dim;
  vector<WritePairs> *wp;
  Config *config;

public:
  ComputePairs(DenseCubicalGrids *_dcg, vector<WritePairs> &_wp, Config &);
  void compute_pairs_main(vector<Cube> &ctr);
  void assemble_columns_to_reduce(vector<Cube> &ctr, uint8_t _dim);
  void add_cache(uint32_t i, CachedColumn &wc,
                 unordered_map<uint32_t, CachedColumn> &recorded_wc);
  void add_cache(uint32_t i, CubeQue &wc,
                 unordered_map<uint32_t, CachedColumn> &recorded_wc);
  Cube pop_pivot(vector<Cube> &column);
  Cube get_pivot(vector<Cube> &column);
  Cube pop_pivot(CubeQue &column);
  Cube get_pivot(CubeQue &column);
};
