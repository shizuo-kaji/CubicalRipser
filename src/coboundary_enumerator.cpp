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


#include <algorithm>
#include <initializer_list>
#include <vector>
#include <cstdlib>
#include <iostream>

#include "cube.h"
#include "dense_cubical_grids.h"
#include "coboundary_enumerator.h"

using namespace std;

CoboundaryEnumerator::CoboundaryEnumerator(DenseCubicalGrids* _dcg, uint8_t _dim)
    : dcg(_dcg), dim(_dim), nextCoface(Cube()) {}

void CoboundaryEnumerator::setCoboundaryEnumerator(Cube& _s) {
	cube = _s;
	// current position of coface search
    if (dcg->az == 1 && dcg->config->tconstruction && dim < 2) {
        // For 2D images under T-construction, skip out-of-plane (z) cofaces
        position = 2;
    } else {
        position = 0;
    }
}

bool CoboundaryEnumerator::hasNextCoface() {
	double birth=0;
    const double threshold = dcg->threshold;
	auto cx = cube.x();
	auto cy = cube.y();
	auto cz = cube.z();
	auto cw = cube.w();

	if (dcg->dim < 4) {
		// offsets[dim][variant(m)][index][(dx,dy,dz,m')] location relative to (cx,cy,cz) which has the smallest index
		static const int8_t offsets[3][3][6][4] = {
			// the descending order is important!!
			// dim 0 (point) : 6 cofaces (duplicate rows for m-variants)
			{
				{{0,0,0,2},{0,0,-1,2},{0,0,0,1},{0,-1,0,1},{0,0,0,0},{-1,0,0,0}},
				{{0,0,0,0},{0,0, 0,0},{0,0,0,0},{0, 0,0,0},{0,0,0,0},{ 0,0,0,0}}, //pad
				{{0,0,0,0},{0,0, 0,0},{0,0,0,0},{0, 0,0,0},{0,0,0,0},{ 0,0,0,0}}, //pad
			},
			// dim 1 (edge) : 4 cofaces (pad remaining with zeros)
			{
				{{0,0,0,1},{0,0,-1,1},{0,0,0,0},{0,-1,0,0},{0,0,0,0},{0,0,0,0}},
				{{0,0,0,2},{0,0,-1,2},{0,0,0,0},{-1,0,0,0},{0,0,0,0},{0,0,0,0}},
				{{0,0,0,2},{0,-1,0,2},{0,0,0,1},{-1,0,0,1},{0,0,0,0},{0,0,0,0}},
			},
			// dim 2 (square) : 2 cofaces (pad)
			{
				{{0,0,0,0},{0,0,-1,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}},
				{{0,0,0,0},{0,-1,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}},
				{{0,0,0,0},{-1,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}},
			}
		};
		static const uint8_t counts[3] = {6,4,2};

        if (dim > 2) return false;
        uint8_t variant = cube.m();
        uint8_t cnt = counts[dim];

        for (uint8_t i = position; i < cnt; ++i) {
            int x = cx + offsets[dim][variant][i][0];
            int y = cy + offsets[dim][variant][i][1];
            int z = cz + offsets[dim][variant][i][2];
            int m = offsets[dim][variant][i][3];
            birth = dcg->getBirth(x,y,z,cw,m, dim+1);
            if (birth != threshold) {
                nextCoface = Cube(birth, x,y,z,cw, m);
                //cube.print();nextCoface.print();
                position = i + 1; return true; }
        }
        return false;
	} else { // dcg->dim == 4
		// Explicit 4D coface enumeration matching getBirth() m-encodings
		if (dim > 3) return false;
		static bool debug = (std::getenv("CR_DEBUG") != nullptr);
		static size_t debug_emitted = 0;

		// Helper lambdas to emit candidate and advance position
		auto emit = [&](int nx, int ny, int nz, int nw, uint8_t nm) -> bool {
			birth = dcg->getBirth(nx, ny, nz, nw, nm, dim + 1);
			if (birth != threshold) {
				nextCoface = Cube(birth, nx, ny, nz, nw, nm);
				//cube.print();
				//nextCoface.print();
				return true;
			}
			return false;
		};

		// Enumerate by dim/variant with fixed tables
		uint8_t m = cube.m();
		if (dim == 0) {
			static const int off[8][5] = {
				{ 0, 0, 0, 0, 3}, { 0, 0, 0, -1, 3}, { 0, 0, 0, 0, 2}, { 0, 0, -1, 0, 2}, { 0, 0, 0, 0, 1}, { 0, -1, 0, 0, 1},{ 0, 0, 0, 0, 0}, { -1, 0, 0, 0, 0}};
			for (uint8_t i = position; i < 8; ++i) {
				int nx = (int)cx + off[i][0];
				int ny = (int)cy + off[i][1];
				int nz = (int)cz + off[i][2];
				int nw = (int)cw + off[i][3];
				uint8_t nm = off[i][4];
				if (emit(nx, ny, nz, nw, nm)) { position = i + 1; return true; }
			}
			return false;
		}
		if (dim == 1) {
			// Edge along m in {0:x,1:y,2:z,3:w}
			// 6 squares: for each other axis, ± side
			// Table per m: 6 entries of (dx,dy,dz,dw,m2)
			static const int off1[4][6][5] = {
				// m=0 (x): pair with y->m2=0 (xy), z->m2=1 (zx), w->m2=3 (wx)
				{{0, 0, 0, 0, 3},{0, 0, 0,-1, 3},{0, 0, 0, 0, 1},{0, 0,-1, 0, 1},{0, 0, 0, 0, 0},{0,-1, 0, 0, 0}},
				// m=1 (y): pair with x->m2=0 (xy), z->m2=2 (yz), w->m2=4 (wy)
				{{0, 0, 0, 0, 4},{0, 0, 0,-1, 4},{0, 0, 0, 0, 2},{0, 0,-1, 0, 2},{0, 0, 0, 0, 0},{-1,0, 0, 0, 0}},
				// m=2 (z): pair with x->m2=1 (zx), y->m2=2 (yz), w->m2=5 (wz)
				{{0, 0, 0, 0, 5},{0, 0, 0,-1, 5},{0, 0, 0, 0, 2},{0,-1, 0, 0, 2},{0, 0, 0, 0, 1},{-1,0, 0, 0, 1}},
				// m=3 (w): pair with x->m2=3 (wx), y->m2=4 (wy), z->m2=5 (wz)
				{{0, 0, 0, 0, 5},{0, 0,-1, 0, 5},{0, 0, 0, 0, 4},{0,-1, 0, 0, 4},{0, 0, 0, 0, 3},{-1,0, 0, 0, 3}},
			};
			for (uint8_t i = position; i < 6; ++i) {
				int nx = (int)cx + off1[m][i][0];
				int ny = (int)cy + off1[m][i][1];
				int nz = (int)cz + off1[m][i][2];
				int nw = (int)cw + off1[m][i][3];
				uint8_t nm = off1[m][i][4];
				if (emit(nx, ny, nz, nw, nm)) { position = i + 1; return true; }
			}
			return false;
		}
		if (dim == 2) {
			// Square variant m in {0:xy,1:zx,2:yz,3:wx,4:wy,5:wz}
			// 4 cubes: for each missing axis, ± side
			static const int off2[6][4][5] = {
				// m=0 xy -> add z (cube xyz:1?) index 0 and w (xyw:1)
				{{0,0,0,0,1},{0,0,0,-1,1},{0,0,0,0,0},{0,0,-1,0,0}},
				// m=1 zx -> add y (xyz:0) and w (xzw:2)
				{{0,0,0,0,2},{0,0,0,-1,2},{0,0,0,0,0},{0,-1,0,0,0}},
				// m=2 yz -> add x (xyz:0) and w (yzw:3)
				{{0,0,0,0,3},{0,0,0,-1,3},{0,0,0,0,0},{-1,0,0,0,0}},
				// m=3 wx -> add y (xyw:1) and z (xzw:2)
				{{0,0,0,0,2},{0,0,-1,0,2},{0,0,0,0,1},{0,-1,0,0,1}},
				// m=4 wy -> add x (xyw:1) and z (yzw:3)
				{{0,0,0,0,3},{0,0,-1,0,3},{0,0,0,0,1},{-1,0,0,0,1}},
				// m=5 wz -> add x (xzw:2) and y (yzw:3)
				{{0,0,0,0,3},{0,-1,0,0,3},{0,0,0,0,2},{-1,0,0,0,2}},
			};
			for (uint8_t i = position; i < 4; ++i) {
				int nx = (int)cx + off2[m][i][0];
				int ny = (int)cy + off2[m][i][1];
				int nz = (int)cz + off2[m][i][2];
				int nw = (int)cw + off2[m][i][3];
				uint8_t nm = off2[m][i][4];
				if (emit(nx, ny, nz, nw, nm)) { position = i + 1; return true; }
			}
			return false;
		}
		// dim == 3
		static const int off3[4][2][5] = {
			// m=0 xyz -> add w -> 4D (m=0)
			{{0,0,0,0,0},{0,0,0,-1,0}},
			// m=1 xyw -> add z -> 4D (m=0)
			{{0,0,0,0,0},{0,0,-1,0,0}},
			// m=2 xzw -> add y -> 4D (m=0)
			{{0,0,0,0,0},{0,-1,0,0,0}},
			// m=3 yzw -> add x -> 4D (m=0)
			{{0,0,0,0,0},{-1,0,0,0,0}},
		};
		for (uint8_t i = position; i < 2; ++i) {
			int nx = (int)cx + off3[m][i][0];
			int ny = (int)cy + off3[m][i][1];
			int nz = (int)cz + off3[m][i][2];
			int nw = (int)cw + off3[m][i][3];
			uint8_t nm = off3[m][i][4];
			if (emit(nx, ny, nz, nw, nm)) { position = i + 1; return true; }
		}
		return false;
	}
}
