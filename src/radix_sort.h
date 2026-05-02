#pragma once

#include "cube.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace cubicalripser {

inline uint64_t double_to_monotone_uint(double value) {
  uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits >> 63) != 0 ? ~bits : (bits ^ (1ULL << 63));
}

template <typename KeyFn>
inline void radix_sort_cubes_by_u64(std::vector<Cube> &cubes, KeyFn key_fn) {
  constexpr unsigned kBitsPerPass = 11;
  constexpr size_t kBucketCount = size_t{1} << kBitsPerPass;
  constexpr uint64_t kBucketMask = kBucketCount - 1;
  constexpr unsigned kPassCount = (64 + kBitsPerPass - 1) / kBitsPerPass;

  std::vector<Cube> buffer(cubes.size());
  std::array<size_t, kBucketCount> histogram{};
  auto *src = &cubes;
  auto *dst = &buffer;

  for (unsigned pass = 0; pass < kPassCount; ++pass) {
    histogram.fill(0);
    const unsigned shift = pass * kBitsPerPass;

    for (const auto &cube : *src) {
      const size_t bucket =
          static_cast<size_t>((key_fn(cube) >> shift) & kBucketMask);
      ++histogram[bucket];
    }

    size_t offset = 0;
    for (size_t bucket = 0; bucket < kBucketCount; ++bucket) {
      const size_t count = histogram[bucket];
      histogram[bucket] = offset;
      offset += count;
    }

    for (const auto &cube : *src) {
      const size_t bucket =
          static_cast<size_t>((key_fn(cube) >> shift) & kBucketMask);
      (*dst)[histogram[bucket]++] = cube;
    }

    std::swap(src, dst);
  }

  if (src != &cubes) {
    cubes.swap(*src);
  }
}

inline void radix_sort_cubes(std::vector<Cube> &cubes) {
  if (cubes.size() < 4096) {
    std::sort(cubes.begin(), cubes.end(), CubeComparator());
    return;
  }

  // Sort the actual Cube array, not an indirection table, so later sweeps stay
  // fully sequential in memory.
  radix_sort_cubes_by_u64(cubes, [](const Cube &cube) {
    return ~double_to_monotone_uint(cube.birth);
  });

  for (size_t begin = 0; begin < cubes.size();) {
    size_t end = begin + 1;
    while (end < cubes.size() && cubes[end].birth == cubes[begin].birth) {
      ++end;
    }
    if (end - begin > 1) {
      std::sort(cubes.begin() + static_cast<ptrdiff_t>(begin),
                cubes.begin() + static_cast<ptrdiff_t>(end),
                [](const Cube &lhs, const Cube &rhs) {
                  return lhs.index < rhs.index;
                });
    }
    begin = end;
  }
}

} // namespace cubicalripser
