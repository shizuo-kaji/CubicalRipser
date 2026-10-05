/* image_io.h

Input files of the command-line programs: NumPy (.npy), Perseus (.txt),
CSV (.csv) and DIPHA (.complex).
*/

#pragma once

#include <cstdint>
#include <vector>

#include "config.h"

// An input array: shape[a] for the dim axes (1 for the others), with axis 0
// varying fastest in `values` if fortran_order and the last axis otherwise.
struct InputImage {
  uint8_t dim{0};
  uint32_t shape[4]{1, 1, 1, 1};
  bool fortran_order{true};
  std::vector<double> values;
};

// Reads config.filename in config.format.  Throws std::runtime_error for an
// unreadable or malformed file.
InputImage read_input_image(const Config &config);
