/* Input files of the command-line programs. */

#include "image_io.h"

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "npy.hpp"

namespace {

[[noreturn]] void fail(const Config &config, const std::string &what) {
  throw std::runtime_error(config.filename + ": " + what);
}

// Checks the header of a format that states its dimension and axis lengths.
void check_shape(const Config &config, int64_t dim, const int64_t *lengths) {
  if (dim < 1 || dim > 4) {
    fail(config, "the dimension must be 1 to 4, not " + std::to_string(dim));
  }
  for (int64_t a = 0; a < dim; ++a) {
    if (lengths[a] < 1 || lengths[a] > INT32_MAX) {
      fail(config, "invalid length " + std::to_string(lengths[a]) +
                       " of axis " + std::to_string(a));
    }
  }
}

size_t element_count(const InputImage &image) {
  return static_cast<size_t>(image.shape[0]) * image.shape[1] *
         image.shape[2] * image.shape[3];
}

// DIPHA: magic, type, count, dim, lengths (int64), then doubles with axis 0
// fastest.
InputImage read_dipha(const Config &config) {
  std::ifstream in(config.filename, std::ios::binary);
  int64_t header[4] = {0, 0, 0, 0};
  in.read(reinterpret_cast<char *>(header), sizeof(header));
  if (!in || header[0] != 8067171840 || header[1] != 1) {
    fail(config, "not a DIPHA image file");
  }
  int64_t lengths[4] = {1, 1, 1, 1};
  if (header[3] >= 1 && header[3] <= 4) {
    in.read(reinterpret_cast<char *>(lengths),
            static_cast<std::streamsize>(header[3]) *
                static_cast<std::streamsize>(sizeof(int64_t)));
  }
  check_shape(config, header[3], lengths);
  InputImage image;
  image.dim = static_cast<uint8_t>(header[3]);
  for (int a = 0; a < image.dim; ++a)
    image.shape[a] = static_cast<uint32_t>(lengths[a]);
  image.values.resize(element_count(image));
  in.read(reinterpret_cast<char *>(image.values.data()),
          static_cast<std::streamsize>(image.values.size() * sizeof(double)));
  if (!in) {
    fail(config, "fewer values than the header states");
  }
  return image;
}

// Perseus: dim, the lengths, then the values with axis 0 fastest; -1 marks a
// cell that never enters (the threshold).
InputImage read_perseus(const Config &config) {
  std::ifstream in(config.filename);
  int64_t dim = 0;
  int64_t lengths[4] = {1, 1, 1, 1};
  in >> dim;
  for (int64_t a = 0; a < dim && a < 4; ++a)
    in >> lengths[a];
  if (!in) {
    fail(config, "unreadable Perseus header");
  }
  check_shape(config, dim, lengths);
  InputImage image;
  image.dim = static_cast<uint8_t>(dim);
  for (int a = 0; a < image.dim; ++a)
    image.shape[a] = static_cast<uint32_t>(lengths[a]);
  image.values.resize(element_count(image));
  for (double &v : image.values) {
    if (!(in >> v)) {
      fail(config, "fewer values than the header states");
    }
    if (v == -1) {
      v = config.threshold;
    }
  }
  return image;
}

// CSV: a 2D array, one row per line; a row is axis 1, a column axis 0.
InputImage read_csv(const Config &config) {
  std::ifstream in(config.filename);
  if (!in) {
    fail(config, "cannot open the file");
  }
  InputImage image;
  image.dim = 2;
  size_t columns = 0;
  size_t rows = 0;
  std::string line;
  while (std::getline(in, line)) {
    if (line.find_first_not_of(" \t\r") == std::string::npos) {
      continue;
    }
    std::istringstream stream(line);
    std::string field;
    size_t count = 0;
    while (std::getline(stream, field, ',')) {
      image.values.push_back(std::stod(field));
      ++count;
    }
    if (rows > 0 && count != columns) {
      fail(config, "row " + std::to_string(rows + 1) + " has " +
                       std::to_string(count) + " values, not " +
                       std::to_string(columns));
    }
    columns = count;
    ++rows;
  }
  if (rows == 0 || columns == 0) {
    fail(config, "no values");
  }
  image.shape[0] = static_cast<uint32_t>(columns);
  image.shape[1] = static_cast<uint32_t>(rows);
  return image;
}

InputImage read_numpy(const Config &config) {
  std::vector<unsigned long> shape;
  InputImage image;
  try {
    npy::LoadArrayFromNumpy(config.filename.c_str(), shape,
                            image.fortran_order, image.values);
  } catch (const std::exception &e) {
    fail(config, std::string("cannot read a float64 NumPy array (") +
                     e.what() + ")");
  }
  int64_t lengths[4] = {1, 1, 1, 1};
  for (size_t a = 0; a < shape.size() && a < 4; ++a)
    lengths[a] = static_cast<int64_t>(shape[a]);
  check_shape(config, static_cast<int64_t>(shape.size()), lengths);
  image.dim = static_cast<uint8_t>(shape.size());
  for (int a = 0; a < image.dim; ++a)
    image.shape[a] = static_cast<uint32_t>(lengths[a]);
  return image;
}

} // namespace

InputImage read_input_image(const Config &config) {
  switch (config.format) {
  case DIPHA:
    return read_dipha(config);
  case PERSEUS:
    return read_perseus(config);
  case CSV:
    return read_csv(config);
  case NUMPY:
    return read_numpy(config);
  }
  fail(config, "unknown input format");
}
