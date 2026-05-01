/* write_pairs.h

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
#include <cstdint>
#include "dense_cubical_grids.h"

class WritePairs
{
public:
	uint8_t dim;
	double birth;
	double death;
	uint32_t birth_x,birth_y,birth_z,birth_w;
	uint32_t death_x,death_y,death_z,death_w;

	WritePairs(uint8_t _dim, double _birth, double _death, uint32_t _birth_x,uint32_t _birth_y,uint32_t _birth_z, uint32_t _birth_w, uint32_t _death_x,uint32_t _death_y,uint32_t _death_z, uint32_t _death_w, bool print = false){
        dim = _dim;
        birth = _birth;
        death = _death;
        birth_x = _birth_x;
        birth_y = _birth_y;
        birth_z = _birth_z;
        birth_w = _birth_w;
        death_x = _death_x;
        death_y = _death_y;
        death_z = _death_z;
        death_w = _death_w;
        if (print == true) {
            std::cout << "[" << birth << "," << death << ")" << " birth loc. (" << birth_x << "," << birth_y << "," << birth_z << "," << birth_w << "), " << " death loc. (" << death_x << "," << death_y << "," << death_z << "," << death_w << ")" << std::endl;
        }
    }
    WritePairs(uint8_t _dim, Cube _birthC, Cube _deathC, DenseCubicalGrids* _dcg, bool print = false){
        dim = _dim;
        birth = _birthC.birth;
        death = _deathC.birth;
        auto b =  _dcg->ParentVoxel(dim, _birthC);
        auto d =  _dcg->ParentVoxel(dim, _deathC);
        birth_x=b[0];
        birth_y=b[1];
        birth_z=b[2];
        birth_w=b[3];
        death_x=d[0];
        death_y=d[1];
        death_z=d[2];
        death_w=d[3];
        if (print == true) {
            //_birthC.print();
            //_deathC.print();
            if (_dcg->dim < 4)
                std::cout << "[" << birth << "," << death << ")" << " birth loc. (" << birth_x << "," << birth_y << "," << birth_z << "), " << " death loc. (" << death_x << "," << death_y << "," << death_z << ")" << std::endl;
            else
                std::cout << "[" << birth << "," << death << ")" << " birth loc. (" << birth_x << "," << birth_y << "," << birth_z << "," << birth_w << "), " << " death loc. (" << death_x << "," << death_y << "," << death_z << "," << death_w << ")" << std::endl;
        }
    }

};
