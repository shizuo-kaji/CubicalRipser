/* parallel.h

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

// std::thread rather than OpenMP: CMakeLists.txt disables OpenMP on Apple
// platforms outright, and mixing another libomp into a Python process that has
// already loaded one (via numpy/torch) is a known source of crashes.  ph_2d.cpp
// already uses std::thread for the H0/H1 split.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <string>
#include <thread>
#include <vector>

namespace cubicalripser {

// Number of workers for a job of `n` items.
//   requested  1 -> always sequential (byte-for-byte the old code path)
//              0 -> auto: hardware concurrency, overridable by CRIPSER_NUM_THREADS
//             >1 -> that many, capped by hardware concurrency
// Returns 1 when the job is too small to pay for thread startup, so small
// inputs keep their current timing exactly.
inline unsigned worker_count(int requested, size_t n, size_t min_items_per_worker) {
  if (requested == 1 || n < 2 * min_items_per_worker) {
    return 1;
  }
  if (requested <= 0) {
    static const int env_threads = [] {
      const char *raw = std::getenv("CRIPSER_NUM_THREADS");
      if (raw == nullptr) return 0;
      const long parsed = std::strtol(raw, nullptr, 10);
      return (parsed > 0 && parsed < 4096) ? static_cast<int>(parsed) : 0;
    }();
    requested = env_threads;
  }
  unsigned hw = std::thread::hardware_concurrency();
  if (hw == 0) {
    hw = 1;
  }
  const unsigned want =
      (requested <= 0) ? hw
                       : std::min(static_cast<unsigned>(requested), hw);
  const unsigned by_work = static_cast<unsigned>(n / min_items_per_worker);
  return std::max(1u, std::min(want, by_work));
}

// Split [0, n) into `workers` contiguous blocks and run fn(worker, begin, end).
// With one worker fn is invoked inline and no thread is created.
template <typename Fn>
void parallel_blocks(size_t n, unsigned workers, Fn &&fn) {
  if (n == 0) {
    return;
  }
  if (workers <= 1) {
    fn(0u, size_t{0}, n);
    return;
  }

  const size_t chunk = (n + workers - 1) / workers;
  std::vector<std::exception_ptr> errors(workers);
  std::vector<std::thread> pool;
  pool.reserve(workers - 1);

  for (unsigned t = 1; t < workers; ++t) {
    const size_t begin = std::min(n, chunk * t);
    const size_t end = std::min(n, begin + chunk);
    if (begin >= end) {
      break;
    }
    // An exception escaping a std::thread calls std::terminate, which would
    // take the host Python process down; capture and rethrow after joining.
    pool.emplace_back([&fn, &errors, t, begin, end] {
      try {
        fn(t, begin, end);
      } catch (...) {
        errors[t] = std::current_exception();
      }
    });
  }
  try {
    fn(0u, size_t{0}, std::min(n, chunk));
  } catch (...) {
    errors[0] = std::current_exception();
  }
  for (auto &thread : pool) {
    thread.join();
  }
  for (auto &error : errors) {
    if (error) {
      std::rethrow_exception(error);
    }
  }
}

// One stable LSD radix pass over [0, n).
//
// `digit_of(i)` yields the bucket of element i; `place(i, pos)` moves element i
// to destination slot pos.  Elements are visited in increasing i within each
// block, and block t's slice of a bucket is placed after every earlier block's,
// so the resulting permutation is exactly the one a single-threaded stable LSD
// pass produces -- for any worker count.
//
// `histogram` is caller-owned scratch of size workers * buckets; reusing it
// across passes keeps this allocation-free in the inner loop.
template <typename DigitOf, typename Place>
void radix_pass(size_t n, unsigned buckets, unsigned workers,
                std::vector<uint32_t> &histogram, DigitOf digit_of,
                Place place) {
  histogram.assign(static_cast<size_t>(workers) * buckets, 0u);

  // __restrict matters here: without it the compiler must assume the histogram
  // may alias the arrays that `place` writes to, and reloads the bucket cursor
  // on every element.  That alone costs ~10% on the single-threaded path.
  parallel_blocks(n, workers, [&](unsigned worker, size_t begin, size_t end) {
    uint32_t *__restrict counts =
        histogram.data() + static_cast<size_t>(worker) * buckets;
    for (size_t i = begin; i < end; ++i) {
      ++counts[digit_of(i)];
    }
  });

  // Exclusive prefix sum in (bucket, worker) order.
  size_t offset = 0;
  for (unsigned bucket = 0; bucket < buckets; ++bucket) {
    for (unsigned worker = 0; worker < workers; ++worker) {
      uint32_t &slot = histogram[static_cast<size_t>(worker) * buckets + bucket];
      const uint32_t count = slot;
      slot = static_cast<uint32_t>(offset);
      offset += count;
    }
  }

  parallel_blocks(n, workers, [&](unsigned worker, size_t begin, size_t end) {
    uint32_t *__restrict cursor =
        histogram.data() + static_cast<size_t>(worker) * buckets;
    for (size_t i = begin; i < end; ++i) {
      place(i, cursor[digit_of(i)]++);
    }
  });
}

} // namespace cubicalripser
