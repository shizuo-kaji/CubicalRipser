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

// The V- and T-constructions share all code; the library linked into a
// program decides which one DenseCubicalGrids builds.  This one is the
// V-construction: input values sit on vertices.

#include "dense_cubical_grids.h"

DenseCubicalGrids::DenseCubicalGrids(Config& _config, uint8_t d, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    config = &_config;
    threshold = config->threshold;
    config->tconstruction = false;
    dim = d;
    ax = x; ay = y; az = z; aw = w;
    img_x = ax; img_y = ay; img_z = az; img_w = aw;
}
