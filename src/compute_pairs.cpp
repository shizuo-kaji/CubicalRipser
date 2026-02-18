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
#include <thread>
#include <mutex>
#include <exception>
#include <stdexcept>
#include <future>
#include <time.h>

using namespace std;

#include "cube.h"
#include "dense_cubical_grids.h"
#include "coboundary_enumerator.h"
#include "write_pairs.h"
#include "compute_pairs.h"

namespace {
inline bool pivot_order_less(const Cube& lhs, const Cube& rhs) {
    if (lhs.birth == rhs.birth) {
        return lhs.index > rhs.index;
    }
    return lhs.birth < rhs.birth;
}

// Reuse merge scratch per thread to avoid repeated heap allocations in xor_with.
thread_local std::vector<Cube> g_xor_merge_scratch;
} // namespace


void ComputePairs::SparseColumn::normalize() {
    if (entries.empty()) {
        return;
    }
    std::sort(entries.begin(), entries.end(), pivot_order_less);
    std::size_t write = 0;
    std::size_t read = 0;
    while (read < entries.size()) {
        std::size_t next = read + 1;
        while (next < entries.size() && entries[next].index == entries[read].index) {
            ++next;
        }
        if (((next - read) & 1u) == 1u) {
            entries[write++] = entries[read];
        }
        read = next;
    }
    entries.resize(write);
}

void ComputePairs::SparseColumn::xor_with(const SparseColumn& rhs) {
    if (rhs.entries.empty()) {
        return;
    }
    if (entries.empty()) {
        entries = rhs.entries;
        return;
    }
    auto& merged = g_xor_merge_scratch;
    merged.clear();
    merged.reserve(entries.size() + rhs.entries.size());
    std::size_t i = 0;
    std::size_t j = 0;
    while (i < entries.size() && j < rhs.entries.size()) {
        const Cube& left = entries[i];
        const Cube& right = rhs.entries[j];
        if (left.index == right.index) {
            ++i;
            ++j;
            continue;
        }
        if (pivot_order_less(left, right)) {
            merged.push_back(left);
            ++i;
        } else {
            merged.push_back(right);
            ++j;
        }
    }
    while (i < entries.size()) {
        merged.push_back(entries[i++]);
    }
    while (j < rhs.entries.size()) {
        merged.push_back(rhs.entries[j++]);
    }
    entries.swap(merged);
}

ComputePairs::ComputePairs(DenseCubicalGrids* _dcg, std::vector<WritePairs> &_wp, Config& _config)
    : dcg(_dcg), dim(1), wp(&_wp), config(&_config) { // Initialize dim to 1 (default method is LINK_FIND, where we skip dim=0)

#ifdef GOOGLE_HASH
    pivot_column_index.set_empty_key(0xffffffff); // for Google hash
#else
    pivot_column_index.max_load_factor(0.7f);
#endif
}

void ComputePairs::make_initial_column(
    const Cube& cube,
    SparseColumn& out_column,
    CoboundaryEnumerator& cofaces) {
    out_column.entries.clear();
    out_column.reserve((dcg->dim == 4) ? 8u : 6u);
    Cube cube_copy = cube;
    cofaces.setCoboundaryEnumerator(cube_copy);
    while (cofaces.hasNextCoface()) {
        out_column.entries.push_back(cofaces.nextCoface);
    }
    out_column.normalize();
}

void ComputePairs::assemble_initial_columns_chunk(
    std::vector<Cube>& ctr,
    uint32_t begin,
    uint32_t end,
    std::vector<SparseColumn>& out_columns) {
    out_columns.clear();
    if (begin >= end) {
        return;
    }
    const uint32_t count = end - begin;
    out_columns.resize(count);

    uint32_t worker_count = config->experimental_chunk_workers;
    if (worker_count == 0) {
        worker_count = std::thread::hardware_concurrency();
    }
    if (worker_count == 0) {
        worker_count = 1;
    }
    worker_count = std::min(worker_count, count);

    if (worker_count == 1) {
        CoboundaryEnumerator cofaces(dcg, dim);
        for (uint32_t idx = begin; idx < end; ++idx) {
            make_initial_column(ctr[idx], out_columns[idx - begin], cofaces);
        }
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    std::exception_ptr worker_error = nullptr;
    std::mutex error_mutex;

    for (uint32_t worker_id = 0; worker_id < worker_count; ++worker_id) {
        const uint32_t local_begin = begin + (count * worker_id) / worker_count;
        const uint32_t local_end = begin + (count * (worker_id + 1)) / worker_count;
        workers.emplace_back([&, local_begin, local_end]() {
            try {
                CoboundaryEnumerator cofaces(dcg, dim);
                for (uint32_t idx = local_begin; idx < local_end; ++idx) {
                    make_initial_column(ctr[idx], out_columns[idx - begin], cofaces);
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!worker_error) {
                    worker_error = std::current_exception();
                }
            }
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }
    if (worker_error) {
        std::rethrow_exception(worker_error);
    }
}

void ComputePairs::speculative_pre_reduce_window(
    std::vector<SparseColumn>& chunk_columns,
    uint32_t chunk_begin,
    uint32_t window_begin,
    uint32_t window_end,
    const std::vector<SparseColumn>& reduced_columns) {
    if (window_begin >= window_end || window_end > chunk_columns.size()) {
        return;
    }

    uint32_t worker_count = config->experimental_chunk_workers;
    if (worker_count == 0) {
        worker_count = std::thread::hardware_concurrency();
    }
    if (worker_count == 0) {
        worker_count = 1;
    }
    worker_count = std::min<uint32_t>(worker_count, window_end - window_begin);

    auto reduce_range = [&](uint32_t begin, uint32_t end) {
        for (uint32_t idx = begin; idx < end; ++idx) {
            auto& column = chunk_columns[idx];
            const uint32_t global_i = chunk_begin + idx;
            for (int iter = 0; iter < config->maxiter; ++iter) {
                const Cube pivot = column.pivot();
                if (pivot.index == NONE) {
                    break;
                }
                const auto found = pivot_column_index.find(pivot.index);
                if (found == pivot_column_index.end()) {
                    break;
                }
                const auto owner = found->second;
                if (owner >= global_i) {
                    break;
                }
                const auto& owner_column = reduced_columns[owner];
                if (owner_column.entries.empty()) {
                    break;
                }
                column.xor_with(owner_column);
            }
        }
    };

    if (worker_count == 1) {
        reduce_range(window_begin, window_end);
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    std::exception_ptr worker_error = nullptr;
    std::mutex error_mutex;
    const uint32_t total = window_end - window_begin;

    for (uint32_t worker_id = 0; worker_id < worker_count; ++worker_id) {
        const uint32_t local_begin = window_begin + (total * worker_id) / worker_count;
        const uint32_t local_end = window_begin + (total * (worker_id + 1)) / worker_count;
        workers.emplace_back([&, local_begin, local_end]() {
            try {
                reduce_range(local_begin, local_end);
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!worker_error) {
                    worker_error = std::current_exception();
                }
            }
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }
    if (worker_error) {
        std::rethrow_exception(worker_error);
    }

    // Jacobi-style local relaxation inside the current window:
    // build a local pivot-owner snapshot and reduce columns in parallel against
    // that snapshot. Final commitment remains serial and authoritative.
    const uint32_t kLocalRelaxRounds = config->experimental_relax_rounds;
    const uint32_t window_size = window_end - window_begin;
    if (window_size <= 1 || kLocalRelaxRounds == 0) {
        return;
    }

    std::unordered_map<uint64_t, uint32_t> local_owner_by_pivot;
    local_owner_by_pivot.max_load_factor(0.7f);
    std::vector<SparseColumn> owner_snapshot(window_size);
    std::vector<uint8_t> owner_present(window_size, 0);

    for (uint32_t round = 0; round < kLocalRelaxRounds; ++round) {
        local_owner_by_pivot.clear();
        local_owner_by_pivot.reserve(window_size * 2u);
        std::fill(owner_present.begin(), owner_present.end(), 0);

        for (uint32_t idx = window_begin; idx < window_end; ++idx) {
            const Cube pivot = chunk_columns[idx].pivot();
            if (pivot.index == NONE) {
                continue;
            }
            const uint32_t global_i = chunk_begin + idx;
            const auto inserted = local_owner_by_pivot.emplace(pivot.index, global_i);
            if (!inserted.second && global_i < inserted.first->second) {
                inserted.first->second = global_i;
            }
        }

        if (local_owner_by_pivot.empty()) {
            break;
        }

        for (const auto& entry : local_owner_by_pivot) {
            const uint32_t owner_global_i = entry.second;
            if (owner_global_i < (chunk_begin + window_begin) || owner_global_i >= (chunk_begin + window_end)) {
                continue;
            }
            const uint32_t local_owner_offset = owner_global_i - (chunk_begin + window_begin);
            if (!owner_present[local_owner_offset]) {
                owner_snapshot[local_owner_offset] = chunk_columns[window_begin + local_owner_offset];
                owner_present[local_owner_offset] = 1;
            }
        }

        std::atomic<bool> any_changed(false);
        auto relax_range = [&](uint32_t begin_idx, uint32_t end_idx) {
            for (uint32_t idx = begin_idx; idx < end_idx; ++idx) {
                auto& column = chunk_columns[idx];
                const uint32_t global_i = chunk_begin + idx;
                bool changed_this_column = false;
                for (int iter = 0; iter < config->maxiter; ++iter) {
                    const Cube pivot = column.pivot();
                    if (pivot.index == NONE) {
                        break;
                    }

                    const auto global_found = pivot_column_index.find(pivot.index);
                    if (global_found != pivot_column_index.end()) {
                        const uint32_t owner = global_found->second;
                        if (owner < global_i) {
                            const auto& owner_column = reduced_columns[owner];
                            if (!owner_column.entries.empty()) {
                                column.xor_with(owner_column);
                                changed_this_column = true;
                                continue;
                            }
                        }
                    }

                    const auto local_found = local_owner_by_pivot.find(pivot.index);
                    if (local_found == local_owner_by_pivot.end()) {
                        break;
                    }

                    const uint32_t local_owner_global_i = local_found->second;
                    if (local_owner_global_i >= global_i) {
                        break;
                    }
                    if (local_owner_global_i < (chunk_begin + window_begin) ||
                        local_owner_global_i >= (chunk_begin + window_end)) {
                        break;
                    }

                    const uint32_t local_owner_offset = local_owner_global_i - (chunk_begin + window_begin);
                    if (!owner_present[local_owner_offset]) {
                        break;
                    }
                    const auto& owner_column = owner_snapshot[local_owner_offset];
                    if (owner_column.entries.empty()) {
                        break;
                    }
                    column.xor_with(owner_column);
                    changed_this_column = true;
                }
                if (changed_this_column) {
                    any_changed.store(true, std::memory_order_relaxed);
                }
            }
        };

        if (worker_count == 1) {
            relax_range(window_begin, window_end);
        } else {
            std::vector<std::thread> relax_workers;
            relax_workers.reserve(worker_count);
            std::exception_ptr relax_error = nullptr;
            std::mutex relax_error_mutex;
            const uint32_t relax_total = window_end - window_begin;
            for (uint32_t worker_id = 0; worker_id < worker_count; ++worker_id) {
                const uint32_t local_begin = window_begin + (relax_total * worker_id) / worker_count;
                const uint32_t local_end = window_begin + (relax_total * (worker_id + 1)) / worker_count;
                relax_workers.emplace_back([&, local_begin, local_end]() {
                    try {
                        relax_range(local_begin, local_end);
                    } catch (...) {
                        std::lock_guard<std::mutex> lock(relax_error_mutex);
                        if (!relax_error) {
                            relax_error = std::current_exception();
                        }
                    }
                });
            }
            for (auto& worker : relax_workers) {
                worker.join();
            }
            if (relax_error) {
                std::rethrow_exception(relax_error);
            }
        }

        if (!any_changed.load(std::memory_order_relaxed)) {
            break;
        }
    }
}

void ComputePairs::compute_pairs_main_chunked(std::vector<Cube>& ctr) {
    const auto ctl_size = ctr.size();
    if (config->verbose) {
        cout << "# columns to reduce: " << ctl_size << endl;
        cout << "# experimental chunked reduction enabled"
             << " (chunk_size=" << config->experimental_chunk_size
             << ", window_size=" << config->experimental_window_size
             << ", relax_rounds=" << config->experimental_relax_rounds
             << ", workers="
             << (config->experimental_chunk_workers == 0
                     ? static_cast<uint32_t>(std::thread::hardware_concurrency())
                     : config->experimental_chunk_workers)
             << ")" << endl;
    }

    pivot_column_index.clear();
#ifdef GOOGLE_HASH
    pivot_column_index.resize(ctl_size); // googlehash
#else
    pivot_column_index.reserve(ctl_size);
#endif

    std::vector<SparseColumn> reduced_columns(ctl_size);
    const uint32_t chunk_size = std::max<uint32_t>(1u, config->experimental_chunk_size);
    int num_apparent_pairs = 0;

    if (ctl_size == 0) {
        if (config->verbose) {
            cout << "# apparent pairs: " << num_apparent_pairs << " (disabled in experimental chunked path)" << endl;
        }
        return;
    }

    auto launch_chunk_builder = [&](uint32_t begin, uint32_t end) {
        return std::async(std::launch::async, [&, begin, end]() {
            std::vector<SparseColumn> cols;
            assemble_initial_columns_chunk(ctr, begin, end, cols);
            return cols;
        });
    };

    uint32_t begin = 0;
    uint32_t end = std::min<uint32_t>(ctl_size, begin + chunk_size);
    auto inflight = launch_chunk_builder(begin, end);

    while (begin < ctl_size) {
        std::vector<SparseColumn> chunk_columns = inflight.get();
        const uint32_t next_begin = end;
        const uint32_t next_end = std::min<uint32_t>(ctl_size, next_begin + chunk_size);
        if (next_begin < ctl_size) {
            inflight = launch_chunk_builder(next_begin, next_end);
        }

        const uint32_t chunk_count = static_cast<uint32_t>(chunk_columns.size());
        const uint32_t configured_window_size = std::max<uint32_t>(1u, config->experimental_window_size);
        const uint32_t window_size =
            std::max<uint32_t>(1u, std::min<uint32_t>(configured_window_size, chunk_count));
        for (uint32_t window_begin = 0; window_begin < chunk_count; window_begin += window_size) {
            const uint32_t window_end = std::min<uint32_t>(chunk_count, window_begin + window_size);
            speculative_pre_reduce_window(
                chunk_columns, begin, window_begin, window_end, reduced_columns);

            for (uint32_t offset = window_begin; offset < window_end; ++offset) {
                const uint32_t i = begin + offset;
                const double birth = ctr[i].birth;
                SparseColumn working = std::move(chunk_columns[offset]);
                bool resolved = false;

                for (int iter = 0; iter < config->maxiter; ++iter) {
                    const Cube pivot = working.pivot();
                    if (pivot.index == NONE) {
                        if (birth != dcg->threshold) {
                            wp->emplace_back(WritePairs(
                                dim,
                                birth,
                                dcg->threshold,
                                ctr[i].x(),
                                ctr[i].y(),
                                ctr[i].z(),
                                ctr[i].w(),
                                0,
                                0,
                                0,
                                0,
                                config->print));
                        }
                        resolved = true;
                        break;
                    }

                    const auto insert_result = pivot_column_index.insert(std::make_pair(pivot.index, i));
                    if (insert_result.second) {
                        reduced_columns[i] = std::move(working);
                        const double death = pivot.birth;
                        if (birth != death) {
                            wp->emplace_back(WritePairs(dim, ctr[i], pivot, dcg, config->print));
                        }
                        resolved = true;
                        break;
                    }

                    const auto owner = insert_result.first->second;
                    if (owner == i) {
                        resolved = true;
                        break;
                    }
                    working.xor_with(reduced_columns[owner]);
                }

                if (!resolved) {
                    throw std::runtime_error("Maximum iterations reached in experimental chunked reduction");
                }
            }
        }

        begin = next_begin;
        end = next_end;
    }

    if (config->verbose) {
        cout << "# apparent pairs: " << num_apparent_pairs << " (disabled in experimental chunked path)" << endl;
    }
}


void ComputePairs::compute_pairs_main(vector<Cube>& ctr){
    // Current experimental path is tuned for dim=1 where reduction dominates.
    // For higher dimensions, keep the legacy path to avoid regressions.
    if (config->experimental_chunked_reduction && dim == 1) {
        compute_pairs_main_chunked(ctr);
        return;
    }

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
