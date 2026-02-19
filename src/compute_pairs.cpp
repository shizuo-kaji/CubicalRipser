/* compute_pairs.cpp

This file is part of CubicalRipser_3dim.
Copyright 2017-2018 Takeki Sudo and Kazushi Ahara.
Modified by Shizuo Kaji

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License along
with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


#include <iostream>
#include <algorithm>
#include <queue>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdint>
#include <utility>
#include <stdexcept>
#include <time.h>

using namespace std;

#include "cube.h"
#include "dense_cubical_grids.h"
#include "coboundary_enumerator.h"
#include "write_pairs.h"
#include "compute_pairs.h"

ComputePairs::ComputePairs(DenseCubicalGrids* _dcg, std::vector<WritePairs> &_wp, Config& _config)
    : dcg(_dcg), dim(1), wp(&_wp), config(&_config) { // Initialize dim to 1 (default method is LINK_FIND, where we skip dim=0)

#ifdef GOOGLE_HASH
    pivot_column_index.set_empty_key(0xffffffff); // for Google hash
#else
    pivot_column_index.max_load_factor(0.7f);
#endif
}


void ComputePairs::compute_pairs_main(vector<Cube>& ctr){
	vector<Cube> coface_entries; // pivotIDs of cofaces
    coface_entries.reserve((dcg->dim == 4) ? 8u : 6u);
	auto ctl_size = ctr.size();
	if(config->verbose){
	    cout << "# columns to reduce: " << ctl_size << endl;
	}
	pivot_column_index.clear();
#ifdef GOOGLE_HASH
	pivot_column_index.resize(ctl_size); // googlehash
#else
    pivot_column_index.reserve(ctl_size);
#endif
	CoboundaryEnumerator cofaces(dcg,dim);
	unordered_map<uint32_t, CachedColumn> recorded_wc;
	queue<uint32_t> cached_column_idx;
	recorded_wc.max_load_factor(0.7f);
	recorded_wc.reserve(ctl_size);
    int num_apparent_pairs = 0;
    CubeQue working_coboundary;
    working_coboundary.reserve(64);

	for(uint32_t i = 0; i < ctl_size; ++i){  // descending order of birth
        working_coboundary.clear();   // non-zero entries of the column
		double birth = ctr[i].birth;
//        cout << i << endl;  ctr[i].print();   // debug

		auto j = i;
		Cube pivot;
		bool might_be_apparent_pair = true;
		bool found_apparent_pair = false;
		int num_recurse = 0;

		for(int k = 0; k < config->maxiter; ++k) { // for each column{}
            bool cache_hit = false;
            if(i!=j){
                auto findWc = recorded_wc.find(j);
                if(findWc != recorded_wc.end()){ // If the reduced form of the pivot column is cached
                    cache_hit = true;
                    const auto& wc = findWc->second;
                    for (const auto& c : wc) { // add the cached pivot column
                        working_coboundary.push(c);
                    }
                }
//				assert(might_be_apparent_pair == false); // As there is always cell-coface pair with the same birthtime, the flag should be set by the next block.
			}
            if(!cache_hit){
                // make the column by enumerating cofaces
                coface_entries.clear();
                cofaces.setCoboundaryEnumerator(ctr[j]);
                const double column_birth = ctr[j].birth;
                while (cofaces.hasNextCoface()) {
                    coface_entries.push_back(cofaces.nextCoface);
                    // cout << "coface: " << j << endl;
                    // ctr[j].print();
                    // cofaces.nextCoface.print();
                    if (might_be_apparent_pair && (column_birth == cofaces.nextCoface.birth)) { // we cannot find this coface on the left (Short-Circuit Evaluation)
                        const auto apparent =
                            pivot_column_index.insert(std::make_pair(cofaces.nextCoface.index, i));
                        if (apparent.second) { // If coface is not in pivot list
                            found_apparent_pair = true;
                            ++num_apparent_pairs;
                            break;
                        }
                        might_be_apparent_pair = false;
                    }
                }
                if (found_apparent_pair) {
                    break;
                }
                for(const auto& e : coface_entries){
                    working_coboundary.push(e);
                }
            }
            pivot = get_pivot(working_coboundary);
            if (pivot.index != NONE){ // if the column is not reduced to zero
                const auto insert_result =
                    pivot_column_index.insert(std::make_pair(pivot.index, i));
                if (!insert_result.second) {	// found entry to reduce
                    j = insert_result.first->second;
					num_recurse++;
//                        cout << i << " to " << j << " " << pivot.index << endl;
                    continue;
                } else { // If the pivot is new
                    if(num_recurse >= config->min_recursion_to_cache){
                        add_cache(i, working_coboundary, recorded_wc);
						cached_column_idx.push(i);
						if(cached_column_idx.size()>config->cache_size){
							recorded_wc.erase(cached_column_idx.front());
							cached_column_idx.pop();
						}
                    }
                    double death = pivot.birth;
                    if (birth != death) {
						wp->emplace_back(WritePairs(dim, ctr[i], pivot, dcg, config->print));
                    }
//                        cout << pivot.index << ",f," << i << endl;
                    break;
                }
            } else { // the column is reduced to zero, which means it corresponds to a permanent cycle
                if (birth != dcg->threshold) {
					wp->emplace_back(WritePairs(dim, birth, dcg->threshold, ctr[i].x(), ctr[i].y(), ctr[i].z(), ctr[i].w(), 0, 0, 0, 0, config->print));
                }
                break;
            }
		}
	}
    if(config->verbose){
        cout << "# apparent pairs: " << num_apparent_pairs << endl;
    }
}

// cache a new reduced column after mod 2
void ComputePairs::add_cache(uint32_t i, CubeQue &wc, unordered_map<uint32_t, CachedColumn>& recorded_wc){
	CachedColumn clean_wc;
    clean_wc.reserve(wc.size());
	while(!wc.empty()){
		auto c = wc.top();
		wc.pop();
		if(!wc.empty() && c.index==wc.top().index){
			wc.pop();
		}else{
			clean_wc.push_back(c);
		}
	}
	recorded_wc.emplace(i, std::move(clean_wc));
}

// get the pivot from a column after mod 2
Cube ComputePairs::pop_pivot(CubeQue& column){
    if (column.empty()) {
        return Cube();
    } else {
        auto pivot = column.top();
        column.pop();

        while (!column.empty() && column.top().index == pivot.index) {
            column.pop();
            if (column.empty())
                return Cube();
            else {
                pivot = column.top();
                column.pop();
            }
        }
        return pivot;
    }
}

Cube ComputePairs::get_pivot(CubeQue& column) {
	Cube result = pop_pivot(column);
	if (result.index != NONE) {
		column.push(result);
	}
	return result;
}

// enumerate and sort columns for a new dimension
void ComputePairs::assemble_columns_to_reduce(vector<Cube>& ctr, uint8_t _dim) {
	dim = _dim;
	ctr.clear();
	double birth;
    uint8_t max_m = 0;
    // Determine number of mask types per target dimension based on ambient dimension
    // 3D: dim 0/1/2/3 => 1/3/3/1
    // 4D: dim 0/1/2/3/4 => 1/4/6/4/1
    if (dcg->dim == 4) {
        switch (dim) {
            case 0: max_m = 1; break;
            case 1: max_m = 4; break;
            case 2: max_m = 6; break;
            case 3: max_m = 4; break;
            default: max_m = 1; break; // dim == 4
        }
    } else {
        switch (dim) {
            case 0: max_m = 1; break;
            case 1: max_m = 3; break;
            case 2: max_m = 3; break;
            default: max_m = 1; break; // dim == 3 (or lower)
        }
    }
    // Special-case: 2D image under T-construction (embedded in 3D with az==1)
    // Restrict mask variants to in-plane components
    if (dcg->config->tconstruction && dcg->az == 1 && dcg->dim < 4) {
        switch (dim) {
            case 0: max_m = 1; break;      // 0-cells: single variant
            case 1: max_m = 2; break;      // 1-cells: only x- and y-edges (no z)
            default: max_m = 1; break;     // 2-cells: single square variant (xy)
        }
    }
    if (dim == 0) {
        pivot_column_index.clear();
    }
    const size_t max_ctr_size =
        static_cast<size_t>(max_m) *
        static_cast<size_t>(dcg->ax) *
        static_cast<size_t>(dcg->ay) *
        static_cast<size_t>(dcg->az) *
        static_cast<size_t>(dcg->aw);
    // Cap reserve to avoid over-allocating when many cells are filtered by threshold.
    const size_t reserve_target = std::min(max_ctr_size, static_cast<size_t>(8000000));
    ctr.reserve(reserve_target);
    const double threshold = dcg->threshold;
    for (uint8_t m = 0; m < max_m; ++m) {
        for(uint32_t w = 0; w < dcg->aw; ++w){
            for(uint32_t z = 0; z < dcg->az; ++z){
                for (uint32_t y = 0; y < dcg->ay; ++y) {
                    for (uint32_t x = 0; x < dcg->ax; ++x) {
                        birth = dcg -> getBirth(x,y,z,w,m, dim);
//                        cout << x << "," << y << "," << z << ", " << m << "," << birth << endl;
                        if (birth < threshold) {
                            const uint64_t index =
                                static_cast<uint64_t>(x)
                                | (static_cast<uint64_t>(y) << 15)
                                | (static_cast<uint64_t>(z) << 30)
                                | (static_cast<uint64_t>(w) << 45)
                                | (static_cast<uint64_t>(m) << 60);
                            if (pivot_column_index.find(index) == pivot_column_index.end()) {
                                ctr.emplace_back(birth, index);
                            }
                        }
                    }
                }
            }
        }
    }
    clock_t start = clock();
    sort(ctr.begin(), ctr.end(), CubeComparator());
	if(config->verbose){
		clock_t end = clock();
		const double time = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;
		cout << "Sorting took: " <<  time << endl;
	}
}
