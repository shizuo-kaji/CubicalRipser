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


#include <vector>
#include "dense_cubical_grids.h"

using namespace std;

class UnionFind{
private:
	vector<uint64_t> parent;
	vector<double> time_max;
public:
	vector<double> birthtime;
	UnionFind(DenseCubicalGrids* _dcg);
	uint64_t find(uint64_t x);
	void link(uint64_t x, uint64_t y);
};

UnionFind::UnionFind(DenseCubicalGrids* _dcg) {
	uint64_t n = _dcg->ax * _dcg->ay * _dcg->az * _dcg->aw;
	parent.resize(n);
	birthtime.resize(n);
	time_max.resize(n);
	//cout << n << " vertices" << endl;

	uint64_t i=0;
    const bool tconstruction = _dcg->config->tconstruction;
    if (!tconstruction) {
        if (_dcg->dim < 4) {
            for (uint32_t w = 0; w < _dcg->aw; ++w) {
                (void)w;
                for (uint32_t z = 0; z < _dcg->az; ++z) {
                    for (uint32_t y = 0; y < _dcg->ay; ++y) {
                        for (uint32_t x = 0; x < _dcg->ax; ++x) {
                            parent[i] = i;
                            birthtime[i] = (*_dcg->dense)(x + 1, y + 1, z + 1);
                            time_max[i] = birthtime[i]; // maximum filtration value for the group
                            i++;
                        }
                    }
                }
            }
        } else {
            for (uint32_t w = 0; w < _dcg->aw; ++w) {
                for (uint32_t z = 0; z < _dcg->az; ++z) {
                    for (uint32_t y = 0; y < _dcg->ay; ++y) {
                        for (uint32_t x = 0; x < _dcg->ax; ++x) {
                            parent[i] = i;
                            birthtime[i] = (*_dcg->dense)(x + 1, y + 1, z + 1, w + 1);
                            time_max[i] = birthtime[i]; // maximum filtration value for the group
                            i++;
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
                        parent[i] = i;
                        birthtime[i] = _dcg->getBirth(x, y, z, w, 0, 0);
                        time_max[i] = birthtime[i]; // maximum filtration value for the group
                        i++;
                    }
                }
            }
        }
    }
}

// find the root of a node x (specified by the index)
uint64_t UnionFind::find(uint64_t x){
	uint64_t y = x, z = parent[y];
	while (z != y) {
		y = z;
		z = parent[y];
	}
	// reassign parents to the found root z
	y = parent[x];
	while (z != y) {
		parent[x] = z;
		x = y;
		y = parent[x];
	}
	return z;
}

// merge nodes x and y (they should be root nodes); older will be the new parent
void UnionFind::link(uint64_t x, uint64_t y){
	if (x == y) return;
	if (birthtime[x] >= birthtime[y]){
		parent[x] = y;
		time_max[y] = std::max(time_max[x], time_max[y]);
	} else if(birthtime[x] < birthtime[y]) {
		parent[y] = x;
		time_max[x] = std::max(time_max[x], time_max[y]);
	}
}
