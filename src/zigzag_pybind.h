/* Python binding for zigzag persistence of mask sequences. */

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "zigzag.h"

namespace nb = nanobind;

// masks: (T, n_0[, n_1[, n_2]]) boolean array, C- or Fortran-contiguous.
// Returns an (n, 9) float64 array
//   [dim, birth, death, birth_x, birth_y, birth_z, death_x, death_y, death_z]
// with birth/death in frame units (coarse position / 2).
inline nb::object computeZigzag(nb::ndarray<const bool, nb::any_contig> masks,
                                bool tconstruction, bool union_connect,
                                int maxdim, bool exhaustive) {
  const size_t nd = masks.ndim();
  if (nd < 2 || nd > 4) {
    throw std::invalid_argument(
        "computeZigzag: masks must have shape (T, n1[, n2[, n3]])");
  }
  ZigzagMasks view;
  view.data = masks.data();
  view.spatial_dim = static_cast<uint8_t>(nd - 1);
  for (size_t i = 0; i < nd; ++i) {
    if (masks.shape(i) == 0) {
      throw std::invalid_argument("computeZigzag: axis " + std::to_string(i) +
                                  " has length 0; every axis must be non-empty");
    }
    if (masks.shape(i) > (size_t{1} << 31)) {
      throw std::invalid_argument("computeZigzag: axis " + std::to_string(i) +
                                  " is longer than 2^31");
    }
  }
  view.frames = static_cast<uint32_t>(masks.shape(0));
  view.frame_stride = masks.stride(0);
  for (size_t a = 0; a + 1 < nd; ++a) {
    view.shape[a] = static_cast<uint32_t>(masks.shape(a + 1));
    view.stride[a] = masks.stride(a + 1);
  }

  std::vector<ZigzagInterval> intervals;
  {
    nb::gil_scoped_release no_gil;
    intervals = compute_zigzag(view, tconstruction, union_connect, maxdim,
                               exhaustive);
  }

  const size_t rows = intervals.size();
  constexpr size_t cols = 9;
  double *data = new double[rows * cols];
  for (size_t i = 0; i < rows; ++i) {
    const ZigzagInterval &z = intervals[i];
    double *row = data + i * cols;
    row[0] = z.dim;
    row[1] = 0.5 * z.birth;
    row[2] = 0.5 * z.death;
    for (int a = 0; a < 3; ++a) {
      row[3 + a] = static_cast<double>(z.birth_voxel[a]);
      row[6 + a] = static_cast<double>(z.death_voxel[a]);
    }
  }
  nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<double *>(p); });
  const size_t shape[2] = {rows, cols};
  return nb::cast(nb::ndarray<nb::numpy, double, nb::ndim<2>>(data, 2, shape, owner));
}
