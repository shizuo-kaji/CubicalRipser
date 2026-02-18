/* compute_pairs.h

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
#include <unordered_map>
#include <queue>
#include <cstddef>
#include "config.h"
#include "cube.h"

// #define GOOGLE_HASH

#ifdef GOOGLE_HASH
#include "sparsehash/dense_hash_map"
#endif

using namespace std;

typedef vector<Cube> CachedColumn;
class CoboundaryEnumerator;

class CubeQue : public priority_queue<Cube, vector<Cube>, CubeComparator> {
public:
    using priority_queue<Cube, vector<Cube>, CubeComparator>::priority_queue;

    void reserve(size_t n) { this->c.reserve(n); }
    void clear() { this->c.clear(); }
};

class ComputePairs{
private:
	struct SparseColumn {
		std::vector<Cube> entries;

		void reserve(std::size_t n) { entries.reserve(n); }
		void normalize();
		Cube pivot() const { return entries.empty() ? Cube() : entries.front(); }
		void xor_with(const SparseColumn& rhs);
	};

	DenseCubicalGrids* dcg;
#ifdef GOOGLE_HASH
    google::dense_hash_map<uint64_t, uint32_t> pivot_column_index;
#else
    unordered_map<uint64_t, uint32_t> pivot_column_index;
#endif
	uint8_t dim;
	vector<WritePairs> *wp;
	Config* config;

	void compute_pairs_main_chunked(std::vector<Cube>& ctr);
	void assemble_initial_columns_chunk(
		std::vector<Cube>& ctr,
		uint32_t begin,
		uint32_t end,
		std::vector<SparseColumn>& out_columns);
	void speculative_pre_reduce_window(
		std::vector<SparseColumn>& chunk_columns,
		uint32_t chunk_begin,
		uint32_t window_begin,
		uint32_t window_end,
		const std::vector<SparseColumn>& reduced_columns);
	void make_initial_column(const Cube& cube, SparseColumn& out_column, CoboundaryEnumerator& cofaces);

public:
	ComputePairs(DenseCubicalGrids* _dcg, vector<WritePairs> &_wp, Config&);
	void compute_pairs_main(vector<Cube>& ctr);
	void assemble_columns_to_reduce(vector<Cube>& ctr, uint8_t _dim);
	void add_cache(uint32_t i, CubeQue &wc, unordered_map<uint32_t, CachedColumn>& recorded_wc);
	Cube pop_pivot(vector<Cube>& column);
	Cube get_pivot(vector<Cube>& column);
	Cube pop_pivot(CubeQue& column);
	Cube get_pivot(CubeQue& column);
};
