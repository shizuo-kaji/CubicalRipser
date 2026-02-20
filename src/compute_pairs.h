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
#include <queue>
#include <unordered_map>
#include <vector>

// #define GOOGLE_HASH

#ifdef GOOGLE_HASH
#include "sparsehash/dense_hash_map"
#endif

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>

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

#include <array>
#include <mutex>
class ConcurrentHashMap {
private:
  std::vector<std::atomic<uint64_t>> keys;
  std::vector<std::atomic<uint32_t>> values;
  size_t capacity;
  size_t capacity_mask;

public:
  ConcurrentHashMap(size_t cap) {
    size_t power = 1;
    while (power < cap && power > 0) {
      power *= 2;
    }
    if (power == 0 || power > 1000000000)
      power = 1ULL << 30;
    capacity = power;
    capacity_mask = capacity - 1;

    keys = std::vector<std::atomic<uint64_t>>(capacity);
    values = std::vector<std::atomic<uint32_t>>(capacity);

#pragma omp parallel for schedule(static)
    for (long long i = 0; i < capacity; ++i) {
      keys[i].store(uint64_t(-1), std::memory_order_relaxed);
      values[i].store(uint32_t(-1), std::memory_order_relaxed);
    }
  }

  struct InsertResult {
    uint32_t first;
    bool second;
  };

  InsertResult insert(uint64_t k, uint32_t v) {
    uint64_t h = k;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;

    size_t idx = h & capacity_mask;
    while (true) {
      uint64_t empty_key = uint64_t(-1);
      uint64_t current_key = keys[idx].load(std::memory_order_acquire);
      if (current_key == empty_key) {
        if (keys[idx].compare_exchange_strong(empty_key, k,
                                              std::memory_order_relaxed,
                                              std::memory_order_relaxed)) {
          values[idx].store(v, std::memory_order_release);
          return {v, true};
        }
        current_key = keys[idx].load(std::memory_order_acquire);
      }
      if (current_key == k) {
        uint32_t existing_v;
        int spin = 0;
        while ((existing_v = values[idx].load(std::memory_order_acquire)) ==
               uint32_t(-1)) {
          if (spin++ > 32) {
            std::this_thread::yield();
          }
        }
        return {existing_v, false};
      }
      idx = (idx + 1) & capacity_mask;
    }
  }

  bool contains(uint64_t k) {
    uint64_t h = k;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;

    size_t idx = h & capacity_mask;
    while (true) {
      uint64_t current_key = keys[idx].load(std::memory_order_acquire);
      if (current_key == uint64_t(-1))
        return false;
      if (current_key == k)
        return true;
      idx = (idx + 1) & capacity_mask;
    }
  }

  void clear() {
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < capacity; ++i) {
      keys[i].store(uint64_t(-1), std::memory_order_relaxed);
      values[i].store(uint32_t(-1), std::memory_order_relaxed);
    }
  }
};

class ComputePairs {
private:
  DenseCubicalGrids *dcg;
  std::unique_ptr<ConcurrentHashMap> pivot_column_index;
  uint8_t dim;
  vector<WritePairs> *wp;
  Config *config;

public:
  ComputePairs(DenseCubicalGrids *_dcg, vector<WritePairs> &_wp, Config &);
  void compute_pairs_main(vector<Cube> &ctr);
  void assemble_columns_to_reduce(vector<Cube> &ctr, uint8_t _dim);
  void add_cache(uint32_t i, CubeQue &wc,
                 unordered_map<uint32_t, CachedColumn> &recorded_wc);
  Cube pop_pivot(vector<Cube> &column);
  Cube get_pivot(vector<Cube> &column);
  Cube pop_pivot(CubeQue &column);
  Cube get_pivot(CubeQue &column);
};
