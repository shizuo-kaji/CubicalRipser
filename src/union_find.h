/* union_find.h

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
#include <limits>
#include <stdexcept>
#include <vector>
#include "dense_cubical_grids.h"

using namespace std;

class UnionFind{
private:
	vector<uint32_t> parent;
public:
	vector<double> birthtime;
	UnionFind(DenseCubicalGrids* _dcg);
	uint64_t find(uint64_t x);
	void link(uint64_t x, uint64_t y);
};

inline UnionFind::UnionFind(DenseCubicalGrids* _dcg) {
	uint64_t n = _dcg->ax * _dcg->ay * _dcg->az * _dcg->aw;
	if (n > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
		throw std::length_error("UnionFind supports at most 2^32-1 vertices");
	}
	parent.resize(static_cast<size_t>(n));
	birthtime.resize(static_cast<size_t>(n));
	//cout << n << " vertices" << endl;

	uint32_t i=0;
	auto append_vertex = [&](double birth) {
		parent[i] = i;
		birthtime[i] = birth;
		++i;
	};
    const bool tconstruction = _dcg->config->tconstruction;
    if (!tconstruction) {
        if (!_dcg->planar_fastpath_dense.empty()) {
            for (uint32_t y = 0; y < _dcg->ay; ++y) {
                const size_t row = static_cast<size_t>(_dcg->img_x) * y;
                for (uint32_t x = 0; x < _dcg->ax; ++x) {
                    append_vertex(_dcg->planar_fastpath_dense[row + x]);
                }
            }
            return;
        }
        if (_dcg->dim < 4) {
            for (uint32_t w = 0; w < _dcg->aw; ++w) {
                (void)w;
                for (uint32_t z = 0; z < _dcg->az; ++z) {
                    for (uint32_t y = 0; y < _dcg->ay; ++y) {
                        for (uint32_t x = 0; x < _dcg->ax; ++x) {
                            append_vertex((*_dcg->dense)(x + 1, y + 1, z + 1));
                        }
                    }
                }
            }
        } else {
            for (uint32_t w = 0; w < _dcg->aw; ++w) {
                for (uint32_t z = 0; z < _dcg->az; ++z) {
                    for (uint32_t y = 0; y < _dcg->ay; ++y) {
                        for (uint32_t x = 0; x < _dcg->ax; ++x) {
                            append_vertex((*_dcg->dense)(x + 1, y + 1, z + 1, w + 1));
                        }
                    }
                }
            }
        }
    } else {
        for (uint32_t w = 0; w < _dcg->aw; ++w) {
            for (uint32_t z = 0; z < _dcg->az; ++z) {
                for (uint32_t y = 0; y < _dcg->ay; ++y) {
                    for (uint32_t x = 0; x < _dcg->ax; ++x) {
                        append_vertex(_dcg->getBirth(x, y, z, w, 0, 0));
                    }
                }
            }
        }
    }
}

// find the root of a node x (specified by the index)
inline uint64_t UnionFind::find(uint64_t x){
	uint32_t root = static_cast<uint32_t>(x);
	uint32_t next = parent[root];
	while (next != root) {
		root = next;
		next = parent[root];
	}
	uint32_t y = parent[static_cast<uint32_t>(x)];
	while (root != y) {
		parent[static_cast<uint32_t>(x)] = root;
		x = y;
		y = parent[static_cast<uint32_t>(x)];
	}
	return root;
}

// merge nodes x and y (they should be root nodes); older will be the new parent
inline void UnionFind::link(uint64_t x, uint64_t y){
	if (x == y) return;
	const double bx = birthtime[x];
	const double by = birthtime[y];
	if (bx >= by){
		parent[static_cast<uint32_t>(x)] = static_cast<uint32_t>(y);
	} else {
		parent[static_cast<uint32_t>(y)] = static_cast<uint32_t>(x);
	}
}
