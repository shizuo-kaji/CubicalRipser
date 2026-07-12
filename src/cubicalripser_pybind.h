/*
This file is part of CubicalRipser
Copyright 2017-2018 Takeki Sudo and Kazushi Ahara.
Modified by Shizuo Kaji

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License along
with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <queue>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <memory>

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#include "cube.h"
#include "write_pairs.h"
#include "joint_pairs.h"
#include "compute_pairs.h"
#include "config.h"
#include "dense_cubical_grids.h"
#include "ph_2d.h"
#include "representatives.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
using namespace std;

// Returned ndarray exposes the stable ABI-friendly numpy framework.
using ResultArray = nb::ndarray<nb::numpy, double, nb::ndim<2>>;

/////////////////////////////////////////////
inline nb::object computePH(
    nb::ndarray<const double, nb::any_contig> img,
    int maxdim = 3,
    bool top_dim = false,
    bool embedded = false,
    const std::string &location = "yes",
    bool representatives = false)
{
    // we ignore "location" argument
    if (representatives && top_dim) {
        throw std::invalid_argument(
            "representatives=True is not supported with top_dim=True; "
            "top_dim uses an Alexander-duality shortcut rather than direct homology");
    }
    Config config;
    config.format = NUMPY;
    config.representatives = representatives;

    vector<WritePairs> writepairs; // (dim birth death x y z)
    writepairs.reserve(1000);

    std::unique_ptr<DenseCubicalGrids> dcg;
    vector<Cube> ctr;

    const size_t nd = img.ndim();
    if (nd < 1 || nd > 4) {
        throw std::invalid_argument("computePH: input array must have 1 to 4 dimensions");
    }

    // any_contig accepts either C- or F-contiguous arrays. Determine which.
    bool fortran_order = false;
    if (nd > 1) {
        // strides are reported in element units; trailing stride 1 ⇒ C-contig,
        // leading stride 1 ⇒ F-contig.
        fortran_order = (img.stride(0) == 1);
    }

    const uint8_t ndim = static_cast<uint8_t>(nd);
    config.maxdim = maxdim;
    const uint32_t sx = static_cast<uint32_t>(img.shape(0));
    const uint32_t sy = (nd > 1) ? static_cast<uint32_t>(img.shape(1)) : 1u;
    const uint32_t sz = (nd > 2) ? static_cast<uint32_t>(img.shape(2)) : 1u;
    const uint32_t sw = (nd > 3) ? static_cast<uint32_t>(img.shape(3)) : 1u;
    dcg = std::make_unique<DenseCubicalGrids>(config, ndim, sx, sy, sz, sw);
    config.maxdim = std::min<uint8_t>(config.maxdim, dcg->dim - 1);
    if (top_dim && dcg->dim > 1) {
        config.method = ALEXANDER;
        config.embedded = !embedded;
    } else {
        config.embedded = embedded;
    }

    dcg->gridFromArray(img.data(), embedded, fortran_order);
    dcg->finalisePadding();

    // compute PH
    if (config.method == ALEXANDER) {
        auto jp = std::make_unique<JointPairs>(dcg.get(), writepairs, config);
        if (dcg->dim == 1) {
            jp->enum_edges({0}, ctr);
            jp->joint_pairs_main(ctr, 0); // dim0
        } else if (dcg->dim == 2) {
            jp->enum_edges({0, 1, 3, 4}, ctr);
            jp->joint_pairs_main(ctr, 1); // dim1
        } else if (dcg->dim == 3) {
            jp->enum_edges({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, ctr);
            jp->joint_pairs_main(ctr, 2); // dim2
        }
    } else {
        // 2-D fast path: avoids the generic LINKFIND + ComputePairs machinery
        // entirely (dual union-find Alexander-duality algorithm).
        bool fastpath_handled = false;
        if (dcg->dim <= 2 && dcg->az == 1 && dcg->aw == 1) {
            fastpath_handled = compute_PH_2d(dcg.get(), writepairs, config);
        }
        if (!fastpath_handled) {
            auto jp = std::make_unique<JointPairs>(dcg.get(), writepairs, config);
            std::vector<uint32_t> betti;
            if (dcg->dim == 1) {
                jp->enum_edges({0}, ctr);
            } else if (dcg->dim == 2) {
                jp->enum_edges({0, 1}, ctr);
            } else if (dcg->dim == 3) {
                jp->enum_edges({0, 1, 2}, ctr);
            } else if (dcg->dim == 4) {
                jp->enum_edges({0, 1, 2, 3}, ctr);
            }
            jp->joint_pairs_main(ctr, 0); // dim0
            betti.push_back(writepairs.size());
            if (config.maxdim > 0) {
                ComputePairs cp(dcg.get(), writepairs, config);
                cp.compute_pairs_main(ctr); // dim1
                betti.push_back(writepairs.size() - betti[0]);
                if (config.maxdim > 1) {
                    cp.assemble_columns_to_reduce(ctr, 2);
                    cp.compute_pairs_main(ctr); // dim2
                    betti.push_back(writepairs.size() - betti[0] - betti[1]);
                    if (config.maxdim > 2) {
                        cp.assemble_columns_to_reduce(ctr, 3);
                        cp.compute_pairs_main(ctr); // dim3
                        betti.push_back(writepairs.size() - betti[0] - betti[1] - betti[2]);
                    }
                }
            }
        }
    }

    // result
    // determine shift between dcg and the voxel coordinates
    auto pad_x = (dcg->ax - dcg->img_x) / 2;
    auto pad_y = (dcg->ay - dcg->img_y) / 2;
    auto pad_z = (dcg->az - dcg->img_z) / 2;
    auto pad_w = (dcg->aw - dcg->img_w) / 2;
    const int64_t p = static_cast<int64_t>(writepairs.size());
    const int num_column = (dcg->dim > 3) ? 11 : 9;

    // Allocate an owned buffer; nanobind takes ownership via the capsule and
    // frees it once the Python array has no remaining references.
    double *data_ptr = new double[static_cast<size_t>(p) * static_cast<size_t>(num_column)];
    for (int64_t i = 0; i < p; ++i) {
        const int offset = static_cast<int>(i * num_column);
        data_ptr[offset + 0] = writepairs[i].dim;
        data_ptr[offset + 1] = writepairs[i].birth;
        data_ptr[offset + 2] = writepairs[i].death;
        data_ptr[offset + 3] = writepairs[i].birth_x - pad_x;
        data_ptr[offset + 4] = writepairs[i].birth_y - pad_y;
        data_ptr[offset + 5] = writepairs[i].birth_z - pad_z;

        if (dcg->dim > 3) {
            data_ptr[offset + 6] = writepairs[i].birth_w - pad_w;
            data_ptr[offset + 7] = writepairs[i].death_x - pad_x;
            data_ptr[offset + 8] = writepairs[i].death_y - pad_y;
            data_ptr[offset + 9] = writepairs[i].death_z - pad_z;
            data_ptr[offset + 10] = writepairs[i].death_w - pad_w;
        } else {
            data_ptr[offset + 6] = writepairs[i].death_x - pad_x;
            data_ptr[offset + 7] = writepairs[i].death_y - pad_y;
            data_ptr[offset + 8] = writepairs[i].death_z - pad_z;
        }
    }

    nb::capsule owner(data_ptr, [](void *p) noexcept { delete[] static_cast<double *>(p); });
    const size_t shape[2] = { static_cast<size_t>(p), static_cast<size_t>(num_column) };
    ResultArray result(data_ptr, 2, shape, owner);
    if (!representatives) {
        return nb::cast(result);
    }

    // The ordinary PH calculation above deliberately remains unchanged.  Only
    // this opt-in branch performs direct boundary reduction with column
    // tracking, which is what supplies homology (rather than cohomology)
    // cycles.
    const auto representative_cycles =
        compute_homology_representatives(dcg.get(), config);

    struct RepresentativeKey {
        uint8_t dim;
        uint64_t birth;
        uint64_t death;

        bool operator==(const RepresentativeKey &other) const {
            return dim == other.dim && birth == other.birth && death == other.death;
        }
    };
    struct RepresentativeKeyHash {
        size_t operator()(const RepresentativeKey &key) const {
            const uint64_t mixed = key.birth ^ (key.death + 0x9e3779b97f4a7c15ULL +
                                                 (key.birth << 6U) + (key.birth >> 2U));
            return std::hash<uint64_t>{}(mixed ^ (static_cast<uint64_t>(key.dim) << 56U));
        }
    };
    const auto double_bits = [](double value) {
        uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    };

    // Multiple intervals can have equal endpoints.  Keep each group in its
    // reduction order and consume one cycle per ordinary output row, thereby
    // preserving the established persistence-table order for the public API.
    std::unordered_map<RepresentativeKey, std::vector<size_t>, RepresentativeKeyHash>
        cycles_by_interval;
    cycles_by_interval.reserve(representative_cycles.size());
    for (size_t i = 0; i < representative_cycles.size(); ++i) {
        const auto &cycle = representative_cycles[i];
        cycles_by_interval[{cycle.dim, double_bits(cycle.birth), double_bits(cycle.death)}]
            .push_back(i);
    }
    std::unordered_map<RepresentativeKey, size_t, RepresentativeKeyHash> next_cycle;

    nb::list python_cycles;
    for (const auto &pair : writepairs) {
        const RepresentativeKey key = {
            pair.dim, double_bits(pair.birth), double_bits(pair.death)};
        const auto found = cycles_by_interval.find(key);
        const size_t used = next_cycle[key]++;
        if (found == cycles_by_interval.end() || used >= found->second.size()) {
            throw std::runtime_error(
                "representatives: direct homology reduction did not reproduce a persistence interval");
        }

        const auto &cycle = representative_cycles[found->second[used]];
        nb::list python_chain;
        for (const Cube &cell : cycle.cells) {
            // Lists keep the variable ambient dimension simple while remaining
            // directly usable as ``np.asarray(cycle, dtype=np.uint32)``.
            nb::list encoded_cell;
            encoded_cell.append(cell.x());
            encoded_cell.append(cell.y());
            encoded_cell.append(cell.z());
            if (dcg->dim == 4) {
                encoded_cell.append(cell.w());
                encoded_cell.append(cell.m());
            } else {
                encoded_cell.append(cell.m());
            }
            python_chain.append(std::move(encoded_cell));
        }
        python_cycles.append(std::move(python_chain));
    }

    return nb::make_tuple(nb::cast(result), std::move(python_cycles));
}
