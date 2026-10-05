/* cubical_cells.h

Cell types of the cubical grid, shared by every part of the computation.

A cell is anchored at a grid point and has a type m.  For each cell
dimension the axis-aligned types are numbered by the axes they span:

  3D (inputs with 1-3 axes)  dim 1: x y z    dim 2: xy zx yz    dim 3: xyz
  4D                         dim 1: x y z w  dim 2: xy zx yz wx wy wz
                             dim 3: xyz xyw xzw yzw             dim 4: xyzw

The birth of a cell combines values of the padded input array: the maximum
over its vertices (V-construction) or the minimum over the top-dimensional
cells containing it (T-construction).  build_stencils lists those values as
offsets from the cell's anchor.
*/

#pragma once

#include <cstdint>

namespace cubical_cells {

constexpr int kMaxTypes = 6; // the most types of any dimension (2-cells, 4D)

// Number of axis-aligned types of cell_dim-cells.
inline uint8_t type_count(bool four_d, uint8_t cell_dim) {
  static const uint8_t count3[4] = {1, 3, 3, 1};
  static const uint8_t count4[5] = {1, 4, 6, 4, 1};
  if (four_d)
    return cell_dim <= 4 ? count4[cell_dim] : 0;
  return cell_dim <= 3 ? count3[cell_dim] : 0;
}

// Axes (bit a for axis a) spanned by axis-aligned type m.
inline uint8_t axes_of(bool four_d, uint8_t cell_dim, uint8_t m) {
  static const uint8_t axes3[4][3] = {{0}, {1, 2, 4}, {3, 5, 6}, {7}};
  static const uint8_t axes4[5][6] = {
      {0}, {1, 2, 4, 8}, {3, 5, 6, 9, 10, 12}, {7, 11, 13, 14}, {15}};
  return four_d ? axes4[cell_dim][m] : axes3[cell_dim][m];
}

// Type of the cell_dim-cell spanning `axes`.
inline uint8_t type_of(bool four_d, uint8_t cell_dim, uint8_t axes) {
  for (uint8_t m = 0; m < type_count(four_d, cell_dim); ++m)
    if (axes_of(four_d, cell_dim, m) == axes)
      return m;
  return 0;
}

// Offset (dx, dy, dz, dw) from the anchor to the other end of an edge of
// type m.
inline const int8_t *edge_offset(uint8_t m) {
  static const int8_t edge[4][4] = {
      {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
  return edge[m];
}

// Cofaces of a cell_dim-cell of type m: rows (dx, dy, dz, dw, m') giving the
// anchor offset and type of each coface, in the order the reduction relies on.
// Returns nullptr when there are none.
inline const int8_t (*coface_table(bool four_d, uint8_t cell_dim, uint8_t m,
                                   uint8_t &count))[5] {
  static const int8_t cofaces3[3][3][6][5] = {
      {{{0, 0, 0, 0, 2}, {0, 0, -1, 0, 2}, {0, 0, 0, 0, 1}, {0, -1, 0, 0, 1}, {0, 0, 0, 0, 0}, {-1, 0, 0, 0, 0}}},
      {{{0, 0, 0, 0, 1}, {0, 0, -1, 0, 1}, {0, 0, 0, 0, 0}, {0, -1, 0, 0, 0}},
       {{0, 0, 0, 0, 2}, {0, 0, -1, 0, 2}, {0, 0, 0, 0, 0}, {-1, 0, 0, 0, 0}},
       {{0, 0, 0, 0, 2}, {0, -1, 0, 0, 2}, {0, 0, 0, 0, 1}, {-1, 0, 0, 0, 1}}},
      {{{0, 0, 0, 0, 0}, {0, 0, -1, 0, 0}},
       {{0, 0, 0, 0, 0}, {0, -1, 0, 0, 0}},
       {{0, 0, 0, 0, 0}, {-1, 0, 0, 0, 0}}}};
  static const uint8_t count3[3] = {6, 4, 2};
  static const int8_t cofaces4_dim0[8][5] = {
      {0, 0, 0, 0, 3}, {0, 0, 0, -1, 3}, {0, 0, 0, 0, 2}, {0, 0, -1, 0, 2},
      {0, 0, 0, 0, 1}, {0, -1, 0, 0, 1}, {0, 0, 0, 0, 0}, {-1, 0, 0, 0, 0}};
  static const int8_t cofaces4_dim1[4][6][5] = {
      {{0, 0, 0, 0, 3}, {0, 0, 0, -1, 3}, {0, 0, 0, 0, 1}, {0, 0, -1, 0, 1}, {0, 0, 0, 0, 0}, {0, -1, 0, 0, 0}},
      {{0, 0, 0, 0, 4}, {0, 0, 0, -1, 4}, {0, 0, 0, 0, 2}, {0, 0, -1, 0, 2}, {0, 0, 0, 0, 0}, {-1, 0, 0, 0, 0}},
      {{0, 0, 0, 0, 5}, {0, 0, 0, -1, 5}, {0, 0, 0, 0, 2}, {0, -1, 0, 0, 2}, {0, 0, 0, 0, 1}, {-1, 0, 0, 0, 1}},
      {{0, 0, 0, 0, 5}, {0, 0, -1, 0, 5}, {0, 0, 0, 0, 4}, {0, -1, 0, 0, 4}, {0, 0, 0, 0, 3}, {-1, 0, 0, 0, 3}}};
  static const int8_t cofaces4_dim2[6][4][5] = {
      {{0, 0, 0, 0, 1}, {0, 0, 0, -1, 1}, {0, 0, 0, 0, 0}, {0, 0, -1, 0, 0}},
      {{0, 0, 0, 0, 2}, {0, 0, 0, -1, 2}, {0, 0, 0, 0, 0}, {0, -1, 0, 0, 0}},
      {{0, 0, 0, 0, 3}, {0, 0, 0, -1, 3}, {0, 0, 0, 0, 0}, {-1, 0, 0, 0, 0}},
      {{0, 0, 0, 0, 2}, {0, 0, -1, 0, 2}, {0, 0, 0, 0, 1}, {0, -1, 0, 0, 1}},
      {{0, 0, 0, 0, 3}, {0, 0, -1, 0, 3}, {0, 0, 0, 0, 1}, {-1, 0, 0, 0, 1}},
      {{0, 0, 0, 0, 3}, {0, -1, 0, 0, 3}, {0, 0, 0, 0, 2}, {-1, 0, 0, 0, 2}}};
  static const int8_t cofaces4_dim3[4][2][5] = {
      {{0, 0, 0, 0, 0}, {0, 0, 0, -1, 0}},
      {{0, 0, 0, 0, 0}, {0, 0, -1, 0, 0}},
      {{0, 0, 0, 0, 0}, {0, -1, 0, 0, 0}},
      {{0, 0, 0, 0, 0}, {-1, 0, 0, 0, 0}}};

  count = 0;
  if (m >= type_count(four_d, cell_dim))
    return nullptr;
  if (!four_d) {
    if (cell_dim > 2)
      return nullptr;
    count = count3[cell_dim];
    return cofaces3[cell_dim][m];
  }
  switch (cell_dim) {
  case 0:
    count = 8;
    return cofaces4_dim0;
  case 1:
    count = 6;
    return cofaces4_dim1[m];
  case 2:
    count = 4;
    return cofaces4_dim2[m];
  case 3:
    count = 2;
    return cofaces4_dim3[m];
  default:
    return nullptr;
  }
}

// Offsets, relative to a cell's anchor value in the padded array, of the
// values whose maximum (V) or minimum (T) is the cell's birth.
struct CellStencil {
  uint8_t count{0};
  int32_t offset[16];
};

inline void build_stencils(bool tconstruction, bool four_d,
                           const int64_t stride[4],
                           CellStencil stencil[5][kMaxTypes]) {
  const int n_axes = four_d ? 4 : 3;
  auto offset_of = [&](const int delta[4]) {
    return static_cast<int32_t>(delta[0] * stride[0] + delta[1] * stride[1] +
                                delta[2] * stride[2] + delta[3] * stride[3]);
  };
  for (int d = 0; d < 5; ++d)
    for (int m = 0; m < kMaxTypes; ++m)
      stencil[d][m].count = 0;

  for (uint8_t d = 0; d <= n_axes; ++d) {
    for (uint8_t m = 0; m < type_count(four_d, d); ++m) {
      const uint8_t axes = axes_of(four_d, d, m);
      CellStencil &st = stencil[d][m];
      // V: the vertices anchor + {0,1}^axes.  T: the top cells
      // anchor + {-1,0}^(other axes).
      for (int bits = 0; bits < (1 << n_axes); ++bits) {
        int delta[4] = {0, 0, 0, 0};
        bool member = true;
        for (int a = 0; a < n_axes && member; ++a) {
          const bool spans = ((axes >> a) & 1) != 0;
          const bool bit = ((bits >> a) & 1) != 0;
          member = !bit || (tconstruction ? !spans : spans);
          delta[a] = bit ? (tconstruction ? -1 : 1) : 0;
        }
        if (member)
          st.offset[st.count++] = offset_of(delta);
      }
    }
  }
}

// Voxels inspected, in order, when locating the voxel that defines a cell's
// birth: rows (dx, dy, dz, dw) relative to the cell's anchor.
inline const int8_t (*parent_voxel_order(bool tconstruction, bool four_d,
                                         uint8_t &count))[4] {
  static const int8_t v3[8][4] = {
      {0, 0, 0, 0}, {1, 0, 0, 0}, {1, 1, 0, 0}, {0, 1, 0, 0},
      {0, 0, 1, 0}, {1, 0, 1, 0}, {0, 1, 1, 0}, {1, 1, 1, 0}};
  static const int8_t v4[16][4] = {
      {0, 0, 0, 0}, {1, 0, 0, 0}, {1, 1, 0, 0}, {0, 1, 0, 0},
      {0, 0, 1, 0}, {1, 0, 1, 0}, {0, 1, 1, 0}, {1, 1, 1, 0},
      {0, 0, 0, 1}, {1, 0, 0, 1}, {1, 1, 0, 1}, {0, 1, 0, 1},
      {0, 0, 1, 1}, {1, 0, 1, 1}, {0, 1, 1, 1}, {1, 1, 1, 1}};
  static const int8_t t3[8][4] = {
      {0, 0, 0, 0},   {-1, 0, 0, 0},  {-1, -1, 0, 0}, {-1, -1, -1, 0},
      {-1, 0, -1, 0}, {0, -1, 0, 0},  {0, -1, -1, 0}, {0, 0, -1, 0}};
  static const int8_t t4[16][4] = {
      {0, 0, 0, 0},    {-1, 0, 0, 0},    {-1, -1, 0, 0},  {-1, -1, -1, 0},
      {-1, 0, -1, 0},  {0, -1, 0, 0},    {0, -1, -1, 0},  {0, 0, -1, 0},
      {0, 0, 0, -1},   {-1, 0, 0, -1},   {-1, -1, 0, -1}, {-1, -1, -1, -1},
      {-1, 0, -1, -1}, {0, -1, 0, -1},   {0, -1, -1, -1}, {0, 0, -1, -1}};
  if (tconstruction) {
    count = four_d ? 16 : 8;
    return four_d ? t4 : t3;
  }
  count = four_d ? 16 : 8;
  return four_d ? v4 : v3;
}

} // namespace cubical_cells
