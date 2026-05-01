/* cube.h

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
#include <iostream>

constexpr uint64_t NONE = 0xffffffffffffffff;

class Cube {
public:
    double birth{0};
    uint64_t index{NONE};

    // Default constructor
    Cube() = default;

    // Copy constructor
    Cube(const Cube& other) = default;

    // Constructor with detailed initialization
    Cube(double _birth, uint32_t _x, uint32_t _y, uint32_t _z, uint32_t _w, uint8_t _m)
        : birth(_birth),
          index(static_cast<uint64_t>(_x)
                | (static_cast<uint64_t>(_y) << 15)
                | (static_cast<uint64_t>(_z) << 30)
                | (static_cast<uint64_t>(_w) << 45)
                | (static_cast<uint64_t>(_m) << 60)) {}

    // Constructor with index
    Cube(double _birth, uint64_t _index)
        : birth(_birth), index(_index) {}

    // Accessor methods
    uint32_t x() const { return index & 0x7fff; }
    uint32_t y() const { return (index >> 15) & 0x7fff; }
    uint32_t z() const { return (index >> 30) & 0x7fff; }
    uint32_t w() const { return (index >> 45) & 0x7fff; }
    uint8_t m() const { return (index >> 60) & 0xf; }

    // Copy method
    void copyCube(const Cube& other) {
        birth = other.birth;
        index = other.index;
    }

    // Print method
    void print() const {
        std::cout << "Cube(birth: " << birth << ", x: " << x() << ", y: " << y()
                  << ", z: " << z() << ", w: " << w() << ", m: " << static_cast<int>(m()) << ")\n";
    }

    // Equality operator
    bool operator==(const Cube& rhs) const {
        return index == rhs.index;
    }
};

// Comparator for sorting cubes
// true when b1>b2 (tie break i1<i2)
struct CubeComparator {
    bool operator()(const Cube& o1, const Cube& o2) const {
        if(o1.birth == o2.birth){
            return(o1.index < o2.index);
        } else {
            return(o1.birth > o2.birth);
        }
    }
};
