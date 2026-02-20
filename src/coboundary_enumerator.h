/* coboundary_enumerator.h

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
#include "cube.h"
#include <cstdint>

class DenseCubicalGrids;

class CoboundaryEnumerator {
private:
  uint8_t position;
  uint8_t dim;
  DenseCubicalGrids *dcg;

public:
  Cube cube;
  Cube nextCoface;

  CoboundaryEnumerator(DenseCubicalGrids *_dcg, uint8_t dim);
  void setCoboundaryEnumerator(Cube &_s);

  bool hasNextCoface();
};
