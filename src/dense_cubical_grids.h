/* dense_cubical_grids.h

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

#include <string>
#include <vector>
#include <iostream>
#include <memory>
#include <array>
#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include "config.h"
#include "cube.h"
#include "cubical_cells.h"

using namespace std;

template<typename T>
class NDArray {
private:
    std::vector<T> data_;
    std::vector<size_t> dimensions_;
    std::vector<size_t> strides_;

public:
    const T *data() const { return data_.data(); }
    T *data() { return data_.data(); }
    size_t stride(size_t i) const { return strides_[i]; }

    NDArray(std::initializer_list<size_t> dims) : dimensions_(dims) {
        size_t total_size = 1;
        strides_.resize(dims.size());

        // Calculate strides (row-major order)
        for (int i = dims.size() - 1; i >= 0; --i) {
            strides_[i] = total_size;
            total_size *= dimensions_[i];
        }
        data_.resize(total_size);
    }

    // Fixed-arity indexing used by union-find and the planar path.
    T& operator()(size_t i, size_t j, size_t k) {
        return data_[i * strides_[0] + j * strides_[1] + k * strides_[2]];
    }

    const T& operator()(size_t i, size_t j, size_t k) const {
        return data_[i * strides_[0] + j * strides_[1] + k * strides_[2]];
    }

    T& operator()(size_t i, size_t j, size_t k, size_t l) {
        return data_[i * strides_[0] + j * strides_[1] + k * strides_[2] + l * strides_[3]];
    }

    const T& operator()(size_t i, size_t j, size_t k, size_t l) const {
        return data_[i * strides_[0] + j * strides_[1] + k * strides_[2] + l * strides_[3]];
    }

    template<typename... Indices>
    T& operator()(Indices... indices) {
        std::array<size_t, sizeof...(indices)> idx_array = {static_cast<size_t>(indices)...};
        size_t flat_index = 0;
        for (size_t i = 0; i < idx_array.size(); ++i) {
            flat_index += idx_array[i] * strides_[i];
        }
        return data_[flat_index];
    }

    template<typename... Indices>
    const T& operator()(Indices... indices) const {
        std::array<size_t, sizeof...(indices)> idx_array = {static_cast<size_t>(indices)...};
        size_t flat_index = 0;
        for (size_t i = 0; i < idx_array.size(); ++i) {
            flat_index += idx_array[i] * strides_[i];
        }
        return data_[flat_index];
    }
};

class DenseCubicalGrids{
public:
	Config *config;
	double threshold;
	uint8_t dim;
	uint32_t img_x, img_y, img_z, img_w;
	uint32_t ax, ay, az, aw;
    uint32_t axy, axyz, ayz, azw, axw, ayw;
	std::unique_ptr<NDArray<double>> dense;
    std::vector<double> planar_fastpath_dense;

	// Birth evaluation over `dense` (see cubical_cells.h), set up by
	// build_birth_stencils() once the padded array exists.
	bool tconstruction_{false};
	const double *birth_base_{nullptr};
	int64_t birth_stride_[4]{0, 0, 0, 0};
	cubical_cells::CellStencil stencil_[5][cubical_cells::kMaxTypes];

	template <bool T>
	static double fold_stencil(const double *p, const cubical_cells::CellStencil &st) {
		switch (st.count) {
		case 1: return fold<1, T>(p, st.offset);
		case 2: return fold<2, T>(p, st.offset);
		case 4: return fold<4, T>(p, st.offset);
		case 8: return fold<8, T>(p, st.offset);
		default: return fold<16, T>(p, st.offset);
		}
	}

	void build_birth_stencils() {
		const bool four_d = dim == 4;
		tconstruction_ = config->tconstruction;
		birth_base_ = dense->data();
		for (int a = 0; a < 4; ++a)
			birth_stride_[a] = a < (four_d ? 4 : 3) ? static_cast<int64_t>(dense->stride(static_cast<size_t>(a))) : 0;
		cubical_cells::build_stencils(tconstruction_, four_d, birth_stride_, stencil_);
	}

    // Overloaded constructor allowing explicit shape initialization
    DenseCubicalGrids(Config&, uint8_t dim, uint32_t ax, uint32_t ay = 1, uint32_t az = 1, uint32_t aw = 1);
	~DenseCubicalGrids() = default; // NDArray uses RAII, no manual cleanup needed

	// Pointer to the padded value at a cell's anchor.  The +1 padding shift is
	// done in uint32_t, so a coface at -1 (UINT32_MAX) lands on padding 0.
	const double *cell_ptr(uint32_t x, uint32_t y, uint32_t z, uint32_t w) const {
		return birth_base_ +
		    static_cast<int64_t>(static_cast<uint32_t>(x + 1)) * birth_stride_[0] +
		    static_cast<int64_t>(static_cast<uint32_t>(y + 1)) * birth_stride_[1] +
		    static_cast<int64_t>(static_cast<uint32_t>(z + 1)) * birth_stride_[2] +
		    static_cast<int64_t>(static_cast<uint32_t>(w + 1)) * birth_stride_[3];
	}
	int64_t stride(int axis) const { return birth_stride_[axis]; }
	bool tconstruction() const { return tconstruction_; }
	const cubical_cells::CellStencil &stencil(uint8_t cell_dim, uint8_t cm) const {
		return stencil_[cell_dim][cm];
	}

	// Fold of p[o[0]], ..., p[o[N-1]] by max (V) or min (T), as a balanced
	// tree.
	template <int N, bool T>
	static double fold(const double *p, const int32_t *o) {
		if constexpr (N == 1) {
			return p[o[0]];
		} else {
			const double a = fold<N / 2, T>(p, o);
			const double b = fold<N / 2, T>(p, o + N / 2);
			if constexpr (T) return (a < b) ? a : b;
			else return (a < b) ? b : a;
		}
	}

	// Birth of the cell of dimension cell_dim and type cm whose anchor value
	// is at p (see cubical_cells.h).
	double birth_at(const double *p, uint8_t cell_dim, uint8_t cm) const {
		const cubical_cells::CellStencil &st = stencil_[cell_dim][cm];
		return tconstruction_ ? fold_stencil<true>(p, st) : fold_stencil<false>(p, st);
	}
	double getBirth(uint32_t x, uint32_t y, uint32_t z, uint32_t w, uint8_t cm, uint8_t cell_dim) const {
		return birth_at(cell_ptr(x, y, z, w), cell_dim, cm);
	}

	// Number of types of cell_dim-cells in this grid.  A 2D image under the
	// T-construction is stored with one layer of voxels, which has no
	// out-of-plane cells.
	uint8_t cell_type_count(uint8_t cell_dim) const {
		if (config->tconstruction && az == 1 && dim < 4) {
			static const uint8_t planar[5] = {1, 2, 1, 0, 0};
			return cell_dim < 5 ? planar[cell_dim] : 0;
		}
		return cubical_cells::type_count(dim == 4, cell_dim);
	}

	// Voxel whose value is the birth of the cell_dim-cell c, as (x, y, z, w):
	// the first, in a fixed order, among the cell's vertices (V) or the top
	// cells containing it (T) with that value.
	std::array<uint32_t, 4> ParentVoxel(uint8_t cell_dim, const Cube &c) const {
		const bool four_d = dim == 4;
		const uint8_t axes = cubical_cells::axes_of(four_d, cell_dim, c.m());
		auto belongs = [&](const int8_t *r) {
			for (int a = 0; a < 4; ++a) {
				const bool spans = ((axes >> a) & 1) != 0;
				if (r[a] != 0 && r[a] != (tconstruction_ ? -1 : 1)) return false;
				if (r[a] != 0 && spans == tconstruction_) return false;
			}
			return true;
		};
		uint8_t count = 0;
		const int8_t (*order)[4] =
		    cubical_cells::parent_voxel_order(tconstruction_, four_d, count);
		const double *anchor = cell_ptr(c.x(), c.y(), c.z(), c.w());
		for (uint8_t i = 0; i < count; ++i) {
			const int8_t *r = order[i];
			if (!belongs(r)) continue;
			if (c.birth == anchor[r[0] * birth_stride_[0] + r[1] * birth_stride_[1] +
			                      r[2] * birth_stride_[2] + r[3] * birth_stride_[3]]) {
				return {c.x() + static_cast<uint32_t>(r[0]), c.y() + static_cast<uint32_t>(r[1]),
				        c.z() + static_cast<uint32_t>(r[2]),
				        dim == 4 ? c.w() + static_cast<uint32_t>(r[3]) : 0u};
			}
		}
		cerr << "parent voxel not found!" << endl;
		return {0, 0, 0, 0};
	}

	void finalisePadding(){
		// T-construction (the number of vertices = that of the top cells plus one, in each dimension)
		if(config->tconstruction){
			if(dim>3) aw++;
			if(dim>2) az++;
			ax++;
			ay++;
		}
		axy = ax * ay;
		ayz = ay * az;
		azw = az * aw;
		axw = ax * aw;
		ayw = ay * aw;
		axyz = ax * ay * az;
	}

	// True when the computation will go through the planar fast path in
	// ph_2d.cpp, which uses its own 31-bit packing and so is not bound by the
	// 15-bit Cube encoding.  Mirrors the guards in compute_PH_2d() and its
	// callers.  `representatives` forces the generic path as well, since
	// compute_homology_representatives() builds Cubes for every cell.
	bool usesPlanarFastPath() const {
		return dim <= 2 && az == 1 && aw == 1 &&
		       config->method == LINKFIND && !config->representatives;
	}

	// Reject degenerate empty axes, and shapes the 15-bit cell encoding in Cube
	// cannot represent.  Every entry point (the CLI and the Python binding
	// alike) funnels through gridFromArray, so this single check covers them all
	// and runs once per computation.
	void validateShape() const {
		const uint32_t axes[4] = {ax, ay, az, aw};
		const bool cube_encoded = !usesPlanarFastPath();
		for (int i = 0; i < 4; ++i) {
			if (axes[i] == 0) {
				throw std::invalid_argument(
					"input axis " + std::to_string(i) +
					" has length 0; every axis must be non-empty");
			}
			if (cube_encoded && axes[i] > CUBE_MAX_AXIS) {
				throw std::invalid_argument(
					"input axis " + std::to_string(i) + " has length " +
					std::to_string(axes[i]) + ", which exceeds the maximum " +
					std::to_string(CUBE_MAX_AXIS) +
					" supported for this computation (the cell coordinate "
					"encoding uses 15 bits per axis)");
			}
		}
	}

	// construct volume with boundary
	void gridFromArray(const double *arr, bool embedded, bool fortran_order){
		validateShape();
		img_x = ax;
		img_y = ay;
		img_z = az;
		img_w = aw;
        planar_fastpath_dense.clear();
		const double sgn = embedded ? -1 : 1;
        const bool use_planar_fastpath_storage =
            !embedded &&
            config->method == LINKFIND &&
            !config->representatives &&
            dim <= 2 &&
            az == 1 &&
            aw == 1;
        if (use_planar_fastpath_storage) {
            const size_t planar_size =
                static_cast<size_t>(ax) * static_cast<size_t>(ay);
            planar_fastpath_dense.resize(planar_size);

            auto arrIndexFortran2D = [&](uint32_t ox, uint32_t oy) -> size_t {
                return static_cast<size_t>(ox) +
                       static_cast<size_t>(ax) * static_cast<size_t>(oy);
            };
            auto arrIndexC2D = [&](uint32_t ox, uint32_t oy) -> size_t {
                return static_cast<size_t>(oy) +
                       static_cast<size_t>(ay) * static_cast<size_t>(ox);
            };

            for (uint32_t y = 0; y < ay; ++y) {
                for (uint32_t x = 0; x < ax; ++x) {
                    const size_t src_idx = fortran_order
                        ? arrIndexFortran2D(x, y)
                        : arrIndexC2D(x, y);
                    planar_fastpath_dense[
                        static_cast<size_t>(x) + static_cast<size_t>(ax) * y
                    ] = std::min(sgn * arr[src_idx], threshold);
                }
            }
            return;
        }
		// Padded copy: an outer layer at the threshold on every axis and, when
		// embedded, an inner layer at -DBL_MAX on x, y and the other axes of
		// length > 1.  Inputs with fewer than 4 axes have no w padding.  Values
		// at or above the threshold become the threshold: cells never entering.
		const bool four_d = dim == 4;
		const uint32_t n[4] = {ax, ay, az, aw};
		uint32_t lo[4];   // index of the first input value on each axis
		uint32_t size[4]; // padded length
		for (int a = 0; a < 4; ++a) {
			if (a == 3 && !four_d) {
				lo[a] = 0;
				size[a] = 1;
			} else {
				lo[a] = (embedded && (a < 2 || n[a] > 1)) ? 2 : 1;
				size[a] = n[a] + 2 * lo[a];
			}
		}
		dense = std::make_unique<NDArray<double>>(
		    std::initializer_list<size_t>{size[0], size[1], size[2], size[3]});
		auto source_index = [&](uint32_t ox, uint32_t oy, uint32_t oz, uint32_t ow) -> size_t {
			return fortran_order
			    ? ox + static_cast<size_t>(n[0]) * (oy + static_cast<size_t>(n[1]) * (oz + static_cast<size_t>(n[2]) * ow))
			    : ow + static_cast<size_t>(n[3]) * (oz + static_cast<size_t>(n[2]) * (oy + static_cast<size_t>(n[1]) * ox));
		};
		auto inside = [&](int a, uint32_t c) { return c >= lo[a] && c < lo[a] + n[a]; };
		auto on_outer = [&](int a, uint32_t c) {
			return (a < 3 || four_d) && (c == 0 || c == size[a] - 1);
		};
		// The loops visit `dense` in storage order (w fastest).
		double *out = dense->data();
		for (uint32_t x = 0; x < size[0]; ++x) {
			const bool in_x = inside(0, x), out_x = on_outer(0, x);
			for (uint32_t y = 0; y < size[1]; ++y) {
				const bool in_y = in_x && inside(1, y), out_y = out_x || on_outer(1, y);
				for (uint32_t z = 0; z < size[2]; ++z) {
					const bool in_z = in_y && inside(2, z), out_z = out_y || on_outer(2, z);
					for (uint32_t w = 0; w < size[3]; ++w) {
						if (in_z && inside(3, w)) {
							*out++ = std::min(sgn * arr[source_index(x - lo[0], y - lo[1], z - lo[2], w - lo[3])], threshold);
						} else if (out_z || on_outer(3, w)) {
							*out++ = config->threshold;
						} else {
							*out++ = -DBL_MAX; // inner boundary (only when embedded)
						}
					}
				}
			}
		}
		ax = n[0] + 2 * (lo[0] - 1);
		ay = n[1] + 2 * (lo[1] - 1);
		az = n[2] + 2 * (lo[2] - 1);
		aw = four_d ? n[3] + 2 * (lo[3] - 1) : n[3];
		build_birth_stencils();
	}
};
