/* dense_cubical_grids_T.cpp (for T-construction)

This file is part of CubicalRipser
Copyright 2017-2018 Takeki Sudo and Kazushi Ahara.
Modified by Shizuo Kaji

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License along
with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <algorithm>
#include <string>
#include <initializer_list>

// switch V and T constructions
#include "dense_cubical_grids.h"

using namespace std;

namespace {
inline double min2(const double a, const double b) {
    return (a < b) ? a : b;
}

inline double min4(
    const double a, const double b, const double c, const double d) {
    return min2(min2(a, b), min2(c, d));
}

inline double min8(
    const double a, const double b, const double c, const double d,
    const double e, const double f, const double g, const double h) {
    return min2(min4(a, b, c, d), min4(e, f, g, h));
}

inline double min16(
    const double a, const double b, const double c, const double d,
    const double e, const double f, const double g, const double h,
    const double i, const double j, const double k, const double l,
    const double m, const double n, const double o, const double p) {
    return min2(min8(a, b, c, d, e, f, g, h), min8(i, j, k, l, m, n, o, p));
}
} // namespace


DenseCubicalGrids::DenseCubicalGrids(Config& _config)  {
    config = &_config;
    threshold = config->threshold;
    config->tconstruction = true;
}

// Explicit-shape constructor (T-construction)
DenseCubicalGrids::DenseCubicalGrids(Config& _config, uint8_t d, uint32_t x, uint32_t y, uint32_t z, uint32_t w)  {
    config = &_config;
    threshold = config->threshold;
    config->tconstruction = true;
    dim = d;
    ax = x; ay = y; az = z; aw = w;
    img_x = ax; img_y = ay; img_z = az; img_w = aw;
}

// return filtlation value for a cube
double DenseCubicalGrids::getBirth(uint32_t cx, uint32_t cy, uint32_t cz){
	return min8(
        (*dense)(cx, cy, cz), (*dense)(cx+1, cy, cz),
        (*dense)(cx+1, cy+1, cz), (*dense)(cx, cy+1, cz),
        (*dense)(cx, cy, cz+1), (*dense)(cx+1, cy, cz+1),
        (*dense)(cx+1, cy+1, cz+1), (*dense)(cx, cy+1, cz+1));
}

double DenseCubicalGrids::getBirth(uint32_t cx, uint32_t cy, uint32_t cz, uint32_t cw, uint8_t cm, uint8_t dim) {
	// beware of the shift due to the boundary
    if (this->dim < 4) {
        switch (dim) {
            case 3:
                return (*dense)(cx+1, cy+1, cz+1);
            case 2:
				switch (cm) {
					case 0: // fix z
						return min((*dense)(cx+1, cy+1, cz), (*dense)(cx + 1, cy + 1, cz + 1));
					case 1: // fix y
						return min((*dense)(cx+1, cy, cz+1), (*dense)(cx + 1, cy + 1, cz + 1));
					case 2: // fix x
						return min((*dense)(cx, cy+1, cz+1), (*dense)(cx + 1, cy + 1, cz + 1));
					break;
				}
            case 1:
                switch (cm) {
					case 0: // x,x+1
                        return min4(
                            (*dense)(cx+1, cy+1, cz+1), (*dense)(cx+1, cy+1, cz),
                            (*dense)(cx+1, cy, cz+1), (*dense)(cx+1, cy, cz));
					case 1: // y,y+1
                        return min4(
                            (*dense)(cx+1, cy+1, cz+1), (*dense)(cx, cy+1, cz+1),
                            (*dense)(cx+1, cy+1, cz), (*dense)(cx, cy+1, cz));
					case 2: // z,z+1
                        return min4(
                            (*dense)(cx+1, cy+1, cz+1), (*dense)(cx, cy+1, cz+1),
                            (*dense)(cx+1, cy, cz+1), (*dense)(cx, cy, cz+1));
					break;
                }
            case 0:
                // 0-cells in 3D T-construction: min over 8 adjacent voxels
                return getBirth(cx, cy, cz);
        }
    } else {
		// 4D case - T-construction
		switch (dim) {
			case 4:
				return (*dense)(cx+1, cy+1, cz+1, cw+1);
			case 3:
				switch (cm) {
					case 0: // fix w
						return min((*dense)(cx+1, cy+1, cz+1, cw), (*dense)(cx + 1, cy + 1, cz + 1, cw + 1));
					case 1: // fix z
						return min((*dense)(cx+1, cy+1, cz, cw+1), (*dense)(cx + 1, cy + 1, cz + 1, cw + 1));
					case 2: // fix y
						return min((*dense)(cx+1, cy, cz+1, cw+1), (*dense)(cx + 1, cy + 1, cz + 1, cw + 1));
					case 3: // fix x
						return min((*dense)(cx, cy+1, cz+1, cw+1), (*dense)(cx + 1, cy + 1, cz + 1, cw + 1));
					break;
				}
			case 2:
				switch (cm) {
					// 2D faces in 4D: min over 4 adjacent 4D voxels
					case 0: // m=0: x-y plane (normals: z,w)
                        return min4(
                            (*dense)(cx+1, cy+1, cz+1, cw+1), (*dense)(cx+1, cy+1, cz,   cw+1),
                            (*dense)(cx+1, cy+1, cz+1, cw  ), (*dense)(cx+1, cy+1, cz,   cw  ));
					case 1: // m=1: z-x plane (normals: y,w)
                        return min4(
                            (*dense)(cx+1, cy+1, cz+1, cw+1), (*dense)(cx+1, cy,   cz+1, cw+1),
                            (*dense)(cx+1, cy+1, cz+1, cw  ), (*dense)(cx+1, cy,   cz+1, cw  ));
					case 2: // m=2: y-z plane (normals: x,w)
                        return min4(
                            (*dense)(cx+1, cy+1, cz+1, cw+1), (*dense)(cx,   cy+1, cz+1, cw+1),
                            (*dense)(cx+1, cy+1, cz+1, cw  ), (*dense)(cx,   cy+1, cz+1, cw  ));
					case 3: // m=3: w-x plane (normals: y,z)
                        return min4(
                            (*dense)(cx+1, cy+1, cz+1, cw+1), (*dense)(cx+1, cy,   cz+1, cw+1),
                            (*dense)(cx+1, cy+1, cz,   cw+1), (*dense)(cx+1, cy,   cz,   cw+1));
					case 4: // m=4: w-y plane (normals: x,z)
                        return min4(
                            (*dense)(cx+1, cy+1, cz+1, cw+1), (*dense)(cx,   cy+1, cz+1, cw+1),
                            (*dense)(cx+1, cy+1, cz,   cw+1), (*dense)(cx,   cy+1, cz,   cw+1));
					case 5: // m=5: w-z plane (normals: x,y)
                        return min4(
                            (*dense)(cx+1, cy+1, cz+1, cw+1), (*dense)(cx,   cy+1, cz+1, cw+1),
                            (*dense)(cx+1, cy,   cz+1, cw+1), (*dense)(cx,   cy,   cz+1, cw+1));
				}
			case 1:
				switch (cm) {
					// 1D edges in 4D: min over 8 adjacent 4D voxels (toggle other 3 axes)
					case 0: // edge along x (toggle y,z,w)
                        return min8(
                            (*dense)(cx+1, cy+1, cz+1, cw+1), (*dense)(cx+1, cy,   cz+1, cw+1),
                            (*dense)(cx+1, cy+1, cz,   cw+1), (*dense)(cx+1, cy,   cz,   cw+1),
                            (*dense)(cx+1, cy+1, cz+1, cw  ), (*dense)(cx+1, cy,   cz+1, cw  ),
                            (*dense)(cx+1, cy+1, cz,   cw  ), (*dense)(cx+1, cy,   cz,   cw  ));
					case 1: // edge along y (toggle x,z,w)
                        return min8(
                            (*dense)(cx+1, cy+1, cz+1, cw+1), (*dense)(cx,   cy+1, cz+1, cw+1),
                            (*dense)(cx+1, cy+1, cz,   cw+1), (*dense)(cx,   cy+1, cz,   cw+1),
                            (*dense)(cx+1, cy+1, cz+1, cw  ), (*dense)(cx,   cy+1, cz+1, cw  ),
                            (*dense)(cx+1, cy+1, cz,   cw  ), (*dense)(cx,   cy+1, cz,   cw  ));
					case 2: // edge along z (toggle x,y,w)
                        return min8(
                            (*dense)(cx+1, cy+1, cz+1, cw+1), (*dense)(cx,   cy+1, cz+1, cw+1),
                            (*dense)(cx+1, cy,   cz+1, cw+1), (*dense)(cx,   cy,   cz+1, cw+1),
                            (*dense)(cx+1, cy+1, cz+1, cw  ), (*dense)(cx,   cy+1, cz+1, cw  ),
                            (*dense)(cx+1, cy,   cz+1, cw  ), (*dense)(cx,   cy,   cz+1, cw  ));
					case 3: // edge along w (toggle x,y,z)
                        return min8(
                            (*dense)(cx+1, cy+1, cz+1, cw+1), (*dense)(cx,   cy+1, cz+1, cw+1),
                            (*dense)(cx+1, cy,   cz+1, cw+1), (*dense)(cx,   cy,   cz+1, cw+1),
                            (*dense)(cx+1, cy+1, cz,   cw+1), (*dense)(cx,   cy+1, cz,   cw+1),
                            (*dense)(cx+1, cy,   cz,   cw+1), (*dense)(cx,   cy,   cz,   cw+1));
				}
			case 0:
				// All 16 vertices of the 4D hypercube for 0-cells
                return min16(
                    (*dense)(cx, cy, cz, cw), (*dense)(cx+1, cy, cz, cw),
                    (*dense)(cx+1, cy+1, cz, cw), (*dense)(cx, cy+1, cz, cw),
                    (*dense)(cx, cy, cz+1, cw), (*dense)(cx+1, cy, cz+1, cw),
                    (*dense)(cx+1, cy+1, cz+1, cw), (*dense)(cx, cy+1, cz+1, cw),
                    (*dense)(cx, cy, cz, cw+1), (*dense)(cx+1, cy, cz, cw+1),
                    (*dense)(cx+1, cy+1, cz, cw+1), (*dense)(cx, cy+1, cz, cw+1),
                    (*dense)(cx, cy, cz+1, cw+1), (*dense)(cx+1, cy, cz+1, cw+1),
                    (*dense)(cx+1, cy+1, cz+1, cw+1), (*dense)(cx, cy+1, cz+1, cw+1));
		}
	}
	return threshold; // fallback
}

// (x,y,z) of the voxel which defines the birthtime of the cube
vector<uint32_t> DenseCubicalGrids::ParentVoxel(uint8_t _dim, Cube &c){
	uint32_t cx = c.x();
	uint32_t cy = c.y();
	uint32_t cz = c.z();
	uint32_t cw = c.w();

	if (dim < 4) {
		static const int rel[][3] = {
			{0,0,0},{-1,0,0},{-1,-1,0},{-1,-1,-1},{-1,0,-1},{0,-1,0},{0,-1,-1},{0,0,-1}
		};
		for (auto &r : rel) {
			int dx=r[0], dy=r[1], dz=r[2];
			if (c.birth == (*dense)(cx+1+dx, cy+1+dy, cz+1+dz))
				return {cx+dx, cy+dy, cz+dz};
		}
	} else {
		// 4D case
		static const int rel4d[][4] = {
			{0,0,0,0},{-1,0,0,0},{-1,-1,0,0},{-1,-1,-1,0},
			{-1,0,-1,0},{0,-1,0,0},{0,-1,-1,0},{0,0,-1,0},
			{0,0,0,-1},{-1,0,0,-1},{-1,-1,0,-1},{-1,-1,-1,-1},
			{-1,0,-1,-1},{0,-1,0,-1},{0,-1,-1,-1},{0,0,-1,-1}
		};
		for (const auto &r : rel4d) {
			if (c.birth == (*dense)(cx+1+r[0], cy+1+r[1], cz+1+r[2], cw+1+r[3]))
				return { uint32_t(cx+r[0]), uint32_t(cy+r[1]), uint32_t(cz+r[2]), uint32_t(cw+r[3]) };
		}
	}
	cerr << "parent voxel not found!" << endl;
	return dim < 4 ? vector<uint32_t>{0,0,0} : vector<uint32_t>{0,0,0,0};
}
