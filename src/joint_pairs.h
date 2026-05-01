/* joint_pairs.h

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

#include <vector>
#include <cstdint>
#include "config.h"
#include "cube.h"          // Needed for std::vector<Cube>
#include "write_pairs.h"   // Needed for std::vector<WritePairs>

class DenseCubicalGrids;

class JointPairs {
private:
    std::vector<WritePairs>* wp;  // Pointer to vector of WritePairs for storing results
    Config* config;               // Pointer to configuration settings
    DenseCubicalGrids* dcg;        // Pointer to the dense cubical grids object

public:
    // Constructor for initializing JointPairs
    JointPairs(DenseCubicalGrids* _dcg, std::vector<WritePairs>& _wp, Config& _config);

    // Method to enumerate all edges based on provided types
    void enum_edges(const std::vector<uint8_t>& types, std::vector<Cube>& ctr);

    // Main method for computing PH0
    void joint_pairs_main(std::vector<Cube>& ctr, int current_dim);
};
