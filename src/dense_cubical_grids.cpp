/* dense_cubical_grids.cpp

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
inline double max2(const double a, const double b) {
    return (a < b) ? b : a;
}

inline double max4(
    const double a, const double b, const double c, const double d) {
    return max2(max2(a, b), max2(c, d));
}

inline double max8(
    const double a, const double b, const double c, const double d,
    const double e, const double f, const double g, const double h) {
    return max2(max4(a, b, c, d), max4(e, f, g, h));
}

inline double max16(
    const double a, const double b, const double c, const double d,
    const double e, const double f, const double g, const double h,
    const double i, const double j, const double k, const double l,
    const double m, const double n, const double o, const double p) {
    return max2(max8(a, b, c, d, e, f, g, h), max8(i, j, k, l, m, n, o, p));
}
} // namespace


DenseCubicalGrids::DenseCubicalGrids(Config& _config)  {
    config = &_config;
    threshold = config->threshold;
    config->tconstruction = false;
}

// Explicit-shape constructor (V-construction)
DenseCubicalGrids::DenseCubicalGrids(Config& _config, uint8_t d, uint32_t x, uint32_t y, uint32_t z, uint32_t w)  {
    config = &_config;
    threshold = config->threshold;
    config->tconstruction = false;
    dim = d;
    ax = x; ay = y; az = z; aw = w;
    img_x = ax; img_y = ay; img_z = az; img_w = aw;
}

// return filtlation value for a cube
// (cx,cy,cz) is the voxel coordinates in the original image
double DenseCubicalGrids::getBirth(uint32_t cx, uint32_t cy, uint32_t cz){
	return (*dense)(cx+1, cy+1, cz+1);
}

double DenseCubicalGrids::getBirth(uint32_t cx, uint32_t cy, uint32_t cz, uint32_t cw, uint8_t cm, uint8_t dim) {
	// beware of the shift due to the boundary
	if (this->dim < 4) {
		switch (dim) {
			case 0:
				return (*dense)(cx+1, cy+1, cz+1);
			case 1: {
				static const int off[13][3] = {
					{1,0,0},{0,1,0},{0,0,1},{1,1,0},{1,-1,0},
					{0,-1,1},{0,1,1},{1,-1,1},{1,0,1},{1,1,1},
					{1,-1,-1},{1,0,-1},{1,1,-1}
				};
				if (cm < 13) {
					const double b = (*dense)(cx+1, cy+1, cz+1);
					const int *o = off[cm];
					return max(b, (*dense)(cx+1+o[0], cy+1+o[1], cz+1+o[2]));
				}
				// fallthrough on invalid cm
			}
			case 2:
				switch (cm) {
				case 0: // x - y (fix z)
                    return max4(
                        (*dense)(cx+1, cy+1, cz+1), (*dense)(cx+2, cy+1, cz+1),
                        (*dense)(cx+2, cy+2, cz+1), (*dense)(cx+1, cy+2, cz+1));
				case 1: // z - x (fix y)
                    return max4(
                        (*dense)(cx+1, cy+1, cz+1), (*dense)(cx+1, cy+1, cz+2),
                        (*dense)(cx+2, cy+1, cz+2), (*dense)(cx+2, cy+1, cz+1));
				case 2: // y - z (fix x)
                    return max4(
                        (*dense)(cx+1, cy+1, cz+1), (*dense)(cx+1, cy+2, cz+1),
                        (*dense)(cx+1, cy+2, cz+2), (*dense)(cx+1, cy+1, cz+2));
				}
			case 3:
                return max8(
                    (*dense)(cx+1, cy+1, cz+1), (*dense)(cx+2, cy+1, cz+1),
                    (*dense)(cx+2, cy+2, cz+1), (*dense)(cx+1, cy+2, cz+1),
                    (*dense)(cx+1, cy+1, cz+2), (*dense)(cx+2, cy+1, cz+2),
                    (*dense)(cx+2, cy+2, cz+2), (*dense)(cx+1, cy+2, cz+2));
			}
	} else {
		// 4D case
		switch (dim) {
			case 0:
				return (*dense)(cx+1, cy+1, cz+1, cw+1);
			case 1: {
				static const int off4d[13][4] = {
					{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1},{1,1,0,0},{1,-1,0,0},
					{0,-1,1,0},{0,1,1,0},{1,-1,1,0},{1,0,1,0},{1,1,1,0},
					{1,-1,-1,0},{1,0,-1,0}
				};
				if (cm < 13) {
					const double b = (*dense)(cx+1, cy+1, cz+1, cw+1);
					const int *o = off4d[cm];
					return max(b, (*dense)(cx+1+o[0], cy+1+o[1], cz+1+o[2], cw+1+o[3]));
				}
				// fallthrough on invalid cm
			}
			case 2:
				switch (cm) {
				case 0: // x - y (fix z,w)
                    return max4(
                        (*dense)(cx+1, cy+1, cz+1, cw+1), (*dense)(cx+2, cy+1, cz+1, cw+1),
                        (*dense)(cx+2, cy+2, cz+1, cw+1), (*dense)(cx+1, cy+2, cz+1, cw+1));
				case 1: // z - x (fix y,w)
                    return max4(
                        (*dense)(cx+1, cy+1, cz+1, cw+1), (*dense)(cx+1, cy+1, cz+2, cw+1),
                        (*dense)(cx+2, cy+1, cz+2, cw+1), (*dense)(cx+2, cy+1, cz+1, cw+1));
				case 2: // y - z (fix x,w)
                    return max4(
                        (*dense)(cx+1, cy+1, cz+1, cw+1), (*dense)(cx+1, cy+2, cz+1, cw+1),
                        (*dense)(cx+1, cy+2, cz+2, cw+1), (*dense)(cx+1, cy+1, cz+2, cw+1));
				case 3: // w - x (fix y,z)
                    return max4(
                        (*dense)(cx+1, cy+1, cz+1, cw+1), (*dense)(cx+1, cy+1, cz+1, cw+2),
                        (*dense)(cx+2, cy+1, cz+1, cw+2), (*dense)(cx+2, cy+1, cz+1, cw+1));
				case 4: // w - y (fix x,z)
                    return max4(
                        (*dense)(cx+1, cy+1, cz+1, cw+1), (*dense)(cx+1, cy+1, cz+1, cw+2),
                        (*dense)(cx+1, cy+2, cz+1, cw+2), (*dense)(cx+1, cy+2, cz+1, cw+1));
				case 5: // w - z (fix x,y)
                    return max4(
                        (*dense)(cx+1, cy+1, cz+1, cw+1), (*dense)(cx+1, cy+1, cz+1, cw+2),
                        (*dense)(cx+1, cy+1, cz+2, cw+2), (*dense)(cx+1, cy+1, cz+2, cw+1));
				}
			case 3:
				// 3D faces in 4D space - there are 8 such faces
				switch (cm) {
				case 0: // x-y-z (fix w)
                    return max8(
                        (*dense)(cx+1, cy+1, cz+1, cw+1), (*dense)(cx+2, cy+1, cz+1, cw+1),
                        (*dense)(cx+2, cy+2, cz+1, cw+1), (*dense)(cx+1, cy+2, cz+1, cw+1),
                        (*dense)(cx+1, cy+1, cz+2, cw+1), (*dense)(cx+2, cy+1, cz+2, cw+1),
                        (*dense)(cx+2, cy+2, cz+2, cw+1), (*dense)(cx+1, cy+2, cz+2, cw+1));
				case 1: // x-y-w (fix z)
                    return max8(
                        (*dense)(cx+1, cy+1, cz+1, cw+1), (*dense)(cx+2, cy+1, cz+1, cw+1),
                        (*dense)(cx+2, cy+2, cz+1, cw+1), (*dense)(cx+1, cy+2, cz+1, cw+1),
                        (*dense)(cx+1, cy+1, cz+1, cw+2), (*dense)(cx+2, cy+1, cz+1, cw+2),
                        (*dense)(cx+2, cy+2, cz+1, cw+2), (*dense)(cx+1, cy+2, cz+1, cw+2));
				case 2: // x-z-w (fix y)
                    return max8(
                        (*dense)(cx+1, cy+1, cz+1, cw+1), (*dense)(cx+2, cy+1, cz+1, cw+1),
                        (*dense)(cx+1, cy+1, cz+2, cw+1), (*dense)(cx+2, cy+1, cz+2, cw+1),
                        (*dense)(cx+1, cy+1, cz+1, cw+2), (*dense)(cx+2, cy+1, cz+1, cw+2),
                        (*dense)(cx+1, cy+1, cz+2, cw+2), (*dense)(cx+2, cy+1, cz+2, cw+2));
				case 3: // y-z-w (fix x)
                    return max8(
                        (*dense)(cx+1, cy+1, cz+1, cw+1), (*dense)(cx+1, cy+2, cz+1, cw+1),
                        (*dense)(cx+1, cy+1, cz+2, cw+1), (*dense)(cx+1, cy+2, cz+2, cw+1),
                        (*dense)(cx+1, cy+1, cz+1, cw+2), (*dense)(cx+1, cy+2, cz+1, cw+2),
                        (*dense)(cx+1, cy+1, cz+2, cw+2), (*dense)(cx+1, cy+2, cz+2, cw+2));
				}
			case 4:
				// 4D hypercube
                return max16(
                    (*dense)(cx+1, cy+1, cz+1, cw+1), (*dense)(cx+2, cy+1, cz+1, cw+1),
                    (*dense)(cx+2, cy+2, cz+1, cw+1), (*dense)(cx+1, cy+2, cz+1, cw+1),
                    (*dense)(cx+1, cy+1, cz+2, cw+1), (*dense)(cx+2, cy+1, cz+2, cw+1),
                    (*dense)(cx+2, cy+2, cz+2, cw+1), (*dense)(cx+1, cy+2, cz+2, cw+1),
                    (*dense)(cx+1, cy+1, cz+1, cw+2), (*dense)(cx+2, cy+1, cz+1, cw+2),
                    (*dense)(cx+2, cy+2, cz+1, cw+2), (*dense)(cx+1, cy+2, cz+1, cw+2),
                    (*dense)(cx+1, cy+1, cz+2, cw+2), (*dense)(cx+2, cy+1, cz+2, cw+2),
                    (*dense)(cx+2, cy+2, cz+2, cw+2), (*dense)(cx+1, cy+2, cz+2, cw+2));
			}
	}
	return threshold;
}


// (x,y,z) or (x,y,z,w) of the voxel which defines the birthtime of the cube
vector<uint32_t> DenseCubicalGrids::ParentVoxel(uint8_t _dim, Cube &c){
	(void)_dim;
	uint32_t cx = c.x(), cy = c.y(), cz = c.z(), cw = c.w();

	if (dim < 4) {
		static const int rel[][3] = {
			{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{0,1,1},{1,1,1},
			{1,-1,0},{0,-1,1},{1,-1,1},{1,-1,-1},{1,0,-1},{1,1,-1}
		};
		for (auto &r : rel) {
			int dx=r[0], dy=r[1], dz=r[2];
			if (c.birth == (*dense)(cx+1+dx, cy+1+dy, cz+1+dz))
				return {cx+dx, cy+dy, cz+dz};
		}
	} else {
		// 4D case
		static const int rel4d[][4] = {
			{0,0,0,0},{1,0,0,0},{1,1,0,0},{0,1,0,0},{0,0,1,0},{1,0,1,0},{0,1,1,0},{1,1,1,0},
			{0,0,0,1},{1,0,0,1},{1,1,0,1},{0,1,0,1},{0,0,1,1},{1,0,1,1},{0,1,1,1},{1,1,1,1},
			{1,-1,0,0},{0,-1,1,0},{1,-1,1,0},{1,-1,-1,0},{1,0,-1,0},{1,1,-1,0}
		};
		for (auto &r : rel4d) {
			int dx=r[0], dy=r[1], dz=r[2], dw=r[3];
			if (c.birth == (*dense)(cx+1+dx, cy+1+dy, cz+1+dz, cw+1+dw))
				return {uint32_t(cx+dx), uint32_t(cy+dy), uint32_t(cz+dz), uint32_t(cw+dw)};
		}
	}
	cerr << "parent voxel not found!" << endl;
	return dim < 4 ? vector<uint32_t>{0,0,0} : vector<uint32_t>{0,0,0,0};
}
