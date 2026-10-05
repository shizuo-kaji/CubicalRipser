/* coboundary_enumerator.cpp
This file is part of CubicalRipser
Copyright 2017-2018 Takeki Sudo and Kazushi Ahara.
Modified by Shizuo Kaji
This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License along
with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


#include <cstdint>

#include "cube.h"
#include "dense_cubical_grids.h"
#include "coboundary_enumerator.h"
#include "cubical_cells.h"

using namespace std;

CoboundaryEnumerator::CoboundaryEnumerator(DenseCubicalGrids* _dcg, uint8_t _dim)
    : position(0), dim(_dim), table_count(0), dcg(_dcg),
      table_offsets(nullptr), anchor_ptr(nullptr), delta(deltas[0]),
      nextCoface(Cube()) {
    // Coface anchors differ from the cube's by fixed offsets, so the step in
    // the padded array is computed once per cube type.
    const bool four_d = dcg->dim >= 4;
    for (uint8_t m = 0; m < cubical_cells::type_count(four_d, dim); ++m) {
        uint8_t count = 0;
        const int8_t (*cofaces)[5] = cubical_cells::coface_table(four_d, dim, m, count);
        for (uint8_t i = 0; i < count; ++i) {
            const int8_t *off = cofaces[i];
            deltas[m][i] = off[0] * dcg->stride(0) + off[1] * dcg->stride(1) +
                           off[2] * dcg->stride(2) + off[3] * dcg->stride(3);
        }
    }
}

void CoboundaryEnumerator::setCoboundaryEnumerator(Cube& _s) {
    cube = _s;
    // current position of coface search
    if (dcg->az == 1 && dcg->config->tconstruction && dim < 2) {
        // For 2D images under T-construction, skip out-of-plane (z) cofaces
        position = 2;
    } else {
        position = 0;
    }
    table_offsets = cubical_cells::coface_table(dcg->dim >= 4, dim, cube.m(), table_count);
    if (table_offsets != nullptr) {
        anchor_ptr = dcg->cell_ptr(cube.x(), cube.y(), cube.z(), cube.w());
        delta = deltas[cube.m()];
    }
}

// All cofaces have dimension dim + 1, so their births fold the same number N
// of values; scan<N, T> keeps that fold inlined.
template <int N, bool T>
bool CoboundaryEnumerator::scan() {
    const double threshold = dcg->threshold;
    const uint8_t coface_dim = static_cast<uint8_t>(dim + 1);
    for (uint8_t i = position; i < table_count; ++i) {
        const int8_t *off = table_offsets[i];
        const uint8_t nm = static_cast<uint8_t>(off[4]);
        const double birth = DenseCubicalGrids::fold<N, T>(
            anchor_ptr + delta[i], dcg->stencil(coface_dim, nm).offset);
        if (birth != threshold) {
            // An offset of -1 wraps to UINT32_MAX, as the padding expects.
            nextCoface = Cube(birth, cube.x() + static_cast<uint32_t>(off[0]),
                              cube.y() + static_cast<uint32_t>(off[1]),
                              cube.z() + static_cast<uint32_t>(off[2]),
                              cube.w() + static_cast<uint32_t>(off[3]), nm);
            position = i + 1;
            return true;
        }
    }
    return false;
}

bool CoboundaryEnumerator::hasNextCoface() {
    if (table_offsets == nullptr || position >= table_count) return false;
    const uint8_t n = dcg->stencil(static_cast<uint8_t>(dim + 1),
                                   static_cast<uint8_t>(table_offsets[0][4])).count;
    const bool t = dcg->tconstruction();
    switch (n) {
    case 1: return t ? scan<1, true>() : scan<1, false>();
    case 2: return t ? scan<2, true>() : scan<2, false>();
    case 4: return t ? scan<4, true>() : scan<4, false>();
    case 8: return t ? scan<8, true>() : scan<8, false>();
    default: return t ? scan<16, true>() : scan<16, false>();
    }
}
