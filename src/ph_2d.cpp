/* ph_2d.cpp -- specialized 2-D persistent homology computation
 *   H_1 is computed by Alexander duality on S^2 -- union-find on the 2-cells plus
 *            one virtual "outside" component, sweeping edges by descending
 *            birth. This avoids the priority-queue-based column reduction
 *            entirely.
 *
 * Implementation notes:
 *   - We sort a compact 16-byte (t, edge_index) key array once and iterate
 *     forward for H_0 and backward for H_1, so the sort cost is paid only
 *     once.
 *   - Edge records do not store the "creator pixel" coordinates; they are
 *     recomputed at emit time (the emit path is short relative to the
 *     reduction).
 */

#include "ph_2d.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "config.h"
#include "cube.h"
#include "dense_cubical_grids.h"
#include "write_pairs.h"

namespace {

constexpr uint32_t INVALID_ID = std::numeric_limits<uint32_t>::max();

struct PlanarDenseView {
    const double* data{nullptr};
    uint32_t width{0};
    uint32_t height{0};

    double operator()(uint32_t x, uint32_t y) const {
        return data[static_cast<size_t>(x) + static_cast<size_t>(width) * y];
    }
};

// Path-compression find for a flat parent array.
inline uint32_t uf_find(std::vector<uint32_t>& parent, uint32_t x) {
    uint32_t r = x;
    while (parent[r] != r) r = parent[r];
    while (parent[x] != r) {
        uint32_t nxt = parent[x];
        parent[x] = r;
        x = nxt;
    }
    return r;
}

// Compact 24-byte edge record. v1/v2 are vertex endpoints (for H_0);
// s1/s2 are adjacent square ids (or INVALID_ID for outside; for H_1).
struct EdgeRec {
    double t;
    uint32_t v1;
    uint32_t v2;
    uint32_t s1;
    uint32_t s2;
};

// 16-byte (t, idx) sort key. We sort this array instead of the edge
// records to keep the sort working set small (~32 MB for 2 M edges
// rather than ~64 MB).
struct SortKey {
    double t;
    uint32_t idx;
    uint32_t pad;  // alignment / cache-line friendly
};

// Convert a double to a uint64_t whose unsigned ordering matches the
// double's natural ordering (handles negative numbers and -0.0 correctly).
inline uint64_t double_to_radix(double d) {
    uint64_t u;
    std::memcpy(&u, &d, sizeof(u));
    return (u & (1ULL << 63)) ? ~u : (u | (1ULL << 63));
}

inline bool try_uint_bucket(double d, uint32_t max_value, uint32_t& out) {
    if (!(d >= 0.0) || d > static_cast<double>(max_value)) return false;
    const uint32_t v = static_cast<uint32_t>(d);
    if (static_cast<double>(v) != d) return false;
    out = v;
    return true;
}

bool detect_counting_sort_limit(const std::vector<SortKey>& keys, uint32_t& max_value) {
    uint32_t observed_max = 0u;
    for (const auto& key : keys) {
        uint32_t bucket = 0;
        if (!try_uint_bucket(key.t, 0xffffu, bucket)) return false;
        observed_max = std::max(observed_max, bucket);
    }
    max_value = (observed_max <= 0xffu) ? 0xffu : 0xffffu;
    return true;
}

void counting_sort_by_t(std::vector<SortKey>& keys, uint32_t max_value) {
    const size_t N = keys.size();
    if (N <= 1) return;

    const size_t bucket_count = static_cast<size_t>(max_value) + 1;
    std::vector<uint32_t> counts(bucket_count, 0u);
    for (const auto& key : keys) {
        uint32_t bucket = 0;
        const bool ok = try_uint_bucket(key.t, max_value, bucket);
        (void)ok;
        counts[bucket]++;
    }

    uint32_t sum = 0;
    for (size_t i = 0; i < bucket_count; ++i) {
        const uint32_t c = counts[i];
        counts[i] = sum;
        sum += c;
    }

    std::vector<SortKey> tmp(N);
    for (const auto& key : keys) {
        uint32_t bucket = 0;
        const bool ok = try_uint_bucket(key.t, max_value, bucket);
        (void)ok;
        tmp[counts[bucket]++] = key;
    }
    keys.swap(tmp);
}

// LSD radix sort of `keys` ascending by `.t`. Stable for ties (preserves
// input order).
//
// Six passes of 11-bit digits cover 66 bits (more than the 64 needed) and
// keep each per-pass histogram small enough (8 KiB) to live in L1.
void radix_sort_by_t(std::vector<SortKey>& keys) {
    const size_t N = keys.size();
    if (N <= 1) return;

    constexpr int RADIX_BITS = 11;
    constexpr int N_PASSES = 6;  // even, so output ends back in `keys`
    constexpr uint64_t BUCKETS = 1ULL << RADIX_BITS;
    constexpr uint64_t MASK = BUCKETS - 1;

    std::vector<SortKey> tmp(N);
    SortKey* a = keys.data();
    SortKey* b = tmp.data();

    // Precompute radix keys; carry them along during the passes so we don't
    // recompute the bit-twiddle on every pass.
    std::vector<uint64_t> rk_a(N), rk_b(N);
    uint64_t* ka = rk_a.data();
    uint64_t* kb = rk_b.data();
    for (size_t i = 0; i < N; ++i) ka[i] = double_to_radix(a[i].t);

    uint32_t hist[BUCKETS];
    for (int pass = 0; pass < N_PASSES; ++pass) {
        const int shift = pass * RADIX_BITS;
        std::memset(hist, 0, sizeof(hist));
        for (size_t i = 0; i < N; ++i) ++hist[(ka[i] >> shift) & MASK];
        uint32_t sum = 0;
        for (uint64_t i = 0; i < BUCKETS; ++i) {
            uint32_t c = hist[i];
            hist[i] = sum;
            sum += c;
        }
        for (size_t i = 0; i < N; ++i) {
            const uint64_t bk = (ka[i] >> shift) & MASK;
            const uint32_t pos = hist[bk]++;
            b[pos] = a[i];
            kb[pos] = ka[i];
        }
        std::swap(a, b);
        std::swap(ka, kb);
    }
    // Even number of passes => sorted output is in `keys.data()` already.
    if (a != keys.data()) {
        std::memcpy(keys.data(), a, N * sizeof(SortKey));
    }
}

void sort_keys_by_t(std::vector<SortKey>& keys) {
    if (keys.size() <= 1) return;

    uint32_t counting_limit = 0u;
    if (detect_counting_sort_limit(keys, counting_limit)) {
        counting_sort_by_t(keys, counting_limit);
        return;
    }
    radix_sort_by_t(keys);
}

} // namespace

bool compute_PH_2d(DenseCubicalGrids* dcg,
                   std::vector<WritePairs>& writepairs,
                   const Config& config) {
    if (dcg->dim == 0 || dcg->dim >= 3) return false;
    if (dcg->az != 1 || dcg->aw != 1) return false;
    if (config.method != LINKFIND) return false;

    const bool one_dim = (dcg->dim == 1);
    const bool tcon = config.tconstruction;
    const double threshold = dcg->threshold;
    const bool print = config.print;

    const uint32_t IH = tcon ? (dcg->ax - 1u) : dcg->ax;
    const uint32_t IW = one_dim ? 1u : (tcon ? (dcg->ay - 1u) : dcg->ay);
    if (IH == 0 || IW == 0) return false;

    const uint32_t VH = tcon ? (IH + 1u) : IH;
    const uint32_t VW = one_dim ? 1u : (tcon ? (IW + 1u) : IW);
    const uint32_t SH = one_dim ? 0u : (tcon ? IH : (IH > 0 ? IH - 1u : 0u));
    const uint32_t SW = one_dim ? 0u : (tcon ? IW : (IW > 0 ? IW - 1u : 0u));

    std::vector<double> pix_storage;
    PlanarDenseView pix;
    if (!dcg->planar_fastpath_dense.empty()) {
        pix = {dcg->planar_fastpath_dense.data(), IH, IW};
    } else {
        pix_storage.resize(static_cast<size_t>(IH) * IW);
        const auto& dense = *dcg->dense;
        for (uint32_t y = 0; y < IW; ++y) {
            for (uint32_t x = 0; x < IH; ++x) {
                pix_storage[static_cast<size_t>(x) + static_cast<size_t>(IH) * y] =
                    dense(x + 1, y + 1, 1);
            }
        }
        pix = {pix_storage.data(), IH, IW};
    }
    auto pix_at = [&](uint32_t x, uint32_t y) -> double {
        return pix(x, y);
    };
    auto sq_lin = [&](uint32_t sx, uint32_t sy) -> uint32_t {
        return static_cast<uint32_t>(static_cast<size_t>(sx) +
                                     static_cast<size_t>(SH) * sy);
    };
    auto vx_lin = [&](uint32_t x, uint32_t y) -> uint32_t {
        return static_cast<uint32_t>(static_cast<size_t>(x) +
                                     static_cast<size_t>(VH) * y);
    };

    const size_t nvert = static_cast<size_t>(VH) * VW;
    std::vector<double> v_birth(nvert);
    if (one_dim) {
        for (uint32_t vx = 0; vx < VH; ++vx) {
            double m = threshold;
            if (tcon) {
                if (vx > 0) m = std::min(m, pix_at(vx - 1, 0));
                if (vx < IH) m = std::min(m, pix_at(vx, 0));
            } else {
                m = pix_at(vx, 0);
            }
            v_birth[vx_lin(vx, 0)] = m;
        }
    } else if (tcon) {
        for (uint32_t vy = 0; vy < VW; ++vy) {
            for (uint32_t vx = 0; vx < VH; ++vx) {
                double m = threshold;
                if (vx > 0 && vy > 0) m = std::min(m, pix_at(vx - 1, vy - 1));
                if (vx < IH && vy > 0) m = std::min(m, pix_at(vx, vy - 1));
                if (vx > 0 && vy < IW) m = std::min(m, pix_at(vx - 1, vy));
                if (vx < IH && vy < IW) m = std::min(m, pix_at(vx, vy));
                v_birth[vx_lin(vx, vy)] = m;
            }
        }
    } else {
        for (uint32_t vy = 0; vy < VW; ++vy) {
            for (uint32_t vx = 0; vx < VH; ++vx) {
                v_birth[vx_lin(vx, vy)] = pix_at(vx, vy);
            }
        }
    }

    const size_t nsq = static_cast<size_t>(SH) * SW;
    std::vector<double> sq_birth(nsq);
    std::vector<uint32_t> sq_maxx;
    std::vector<uint32_t> sq_maxy;
    if (!one_dim && tcon) {
        for (uint32_t sy = 0; sy < SW; ++sy) {
            for (uint32_t sx = 0; sx < SH; ++sx) {
                sq_birth[sq_lin(sx, sy)] = pix_at(sx, sy);
            }
        }
    } else if (!one_dim) {
        sq_maxx.assign(nsq, 0u);
        sq_maxy.assign(nsq, 0u);
        for (uint32_t sy = 0; sy < SW; ++sy) {
            for (uint32_t sx = 0; sx < SH; ++sx) {
                const double p00 = pix_at(sx, sy);
                const double p10 = pix_at(sx + 1, sy);
                const double p11 = pix_at(sx + 1, sy + 1);
                const double p01 = pix_at(sx, sy + 1);
                double mv = p00;
                uint32_t mx = sx;
                uint32_t my = sy;
                if (p10 > mv) { mv = p10; mx = sx + 1; my = sy; }
                if (p11 > mv) { mv = p11; mx = sx + 1; my = sy + 1; }
                if (p01 > mv) { mv = p01; mx = sx; my = sy + 1; }
                const uint32_t lin = sq_lin(sx, sy);
                sq_birth[lin] = mv;
                sq_maxx[lin] = mx;
                sq_maxy[lin] = my;
            }
        }
    }

    const uint32_t EX_RANGE_X = one_dim ? (tcon ? IH : (IH > 0 ? IH - 1u : 0u)) : SH;
    const uint32_t EX_RANGE_Y = one_dim ? 1u : (tcon ? VW : IW);
    const uint32_t EY_RANGE_X = one_dim ? 0u : (tcon ? VH : IH);
    const uint32_t EY_RANGE_Y = one_dim ? 0u : SW;

    const size_t cap = one_dim
        ? static_cast<size_t>(EX_RANGE_X)
        : static_cast<size_t>(EX_RANGE_X) * EX_RANGE_Y +
              static_cast<size_t>(EY_RANGE_X) * EY_RANGE_Y;
    std::vector<EdgeRec> edges;
    edges.reserve(cap);
    std::vector<uint32_t> ecoord;
    ecoord.reserve(cap);

    auto add_xedge = [&](uint32_t ex, uint32_t ey) {
        if (one_dim) {
            const double t = tcon ? pix_at(ex, 0) : std::max(pix_at(ex, 0), pix_at(ex + 1, 0));
            if (t >= threshold) return;
            edges.push_back({t, vx_lin(ex, 0), vx_lin(ex + 1, 0), INVALID_ID, INVALID_ID});
            ecoord.push_back((0u << 30) | (ex << 15));
            return;
        }
        const bool has_above = (ey >= 1);
        const bool has_below = (ey < SW);
        if (!has_above && !has_below) return;
        uint32_t s_a = INVALID_ID, s_b = INVALID_ID;
        double pa = threshold, pb = threshold;
        if (has_above) {
            s_a = sq_lin(ex, ey - 1);
            pa = tcon ? pix_at(ex, ey - 1) : sq_birth[s_a];
        }
        if (has_below) {
            s_b = sq_lin(ex, ey);
            pb = tcon ? pix_at(ex, ey) : sq_birth[s_b];
        }
        double t;
        if (tcon) {
            t = std::min(pa, pb);
        } else {
            const double e0 = pix_at(ex, ey);
            const double e1 = pix_at(ex + 1, ey);
            t = std::max(e0, e1);
        }
        if (t >= threshold) return;
        edges.push_back({t, vx_lin(ex, ey), vx_lin(ex + 1, ey), s_a, s_b});
        ecoord.push_back((0u << 30) | (ex << 15) | ey);
    };

    auto add_yedge = [&](uint32_t ex, uint32_t ey) {
        const bool has_left = (ex >= 1);
        const bool has_right = (ex < SH);
        if (!has_left && !has_right) return;
        uint32_t s_l = INVALID_ID, s_r = INVALID_ID;
        double pl = threshold, pr = threshold;
        if (has_left) {
            s_l = sq_lin(ex - 1, ey);
            pl = tcon ? pix_at(ex - 1, ey) : sq_birth[s_l];
        }
        if (has_right) {
            s_r = sq_lin(ex, ey);
            pr = tcon ? pix_at(ex, ey) : sq_birth[s_r];
        }
        double t;
        if (tcon) {
            t = std::min(pl, pr);
        } else {
            const double e0 = pix_at(ex, ey);
            const double e1 = pix_at(ex, ey + 1);
            t = std::max(e0, e1);
        }
        if (t >= threshold) return;
        edges.push_back({t, vx_lin(ex, ey), vx_lin(ex, ey + 1), s_l, s_r});
        ecoord.push_back((1u << 30) | (ex << 15) | ey);
    };

    for (uint32_t ey = 0; ey < EX_RANGE_Y; ++ey) {
        for (uint32_t ex = 0; ex < EX_RANGE_X; ++ex) {
            add_xedge(ex, ey);
        }
    }
    for (uint32_t ey = 0; ey < EY_RANGE_Y; ++ey) {
        for (uint32_t ex = 0; ex < EY_RANGE_X; ++ex) {
            add_yedge(ex, ey);
        }
    }

    auto creator_pixel = [&](uint32_t i, uint32_t& cx, uint32_t& cy) {
        const uint32_t pk = ecoord[i];
        const uint32_t etype = pk >> 30;
        const uint32_t ex = (pk >> 15) & 0x7fffu;
        const uint32_t ey = pk & 0x7fffu;
        if (one_dim) {
            if (tcon) {
                cx = ex;
                cy = 0;
            } else {
                const double e0 = pix_at(ex, 0);
                const double e1 = pix_at(ex + 1, 0);
                if (e0 >= e1) { cx = ex; cy = 0; } else { cx = ex + 1; cy = 0; }
            }
            return;
        }
        if (etype == 0) {
            if (tcon) {
                const bool has_above = (ey >= 1);
                const bool has_below = (ey < SW);
                if (has_below && has_above) {
                    const double pb = pix_at(ex, ey);
                    const double pa = pix_at(ex, ey - 1);
                    if (pb <= pa) { cx = ex; cy = ey; } else { cx = ex; cy = ey - 1; }
                } else if (has_below) {
                    cx = ex; cy = ey;
                } else {
                    cx = ex; cy = ey - 1;
                }
            } else {
                const double e0 = pix_at(ex, ey);
                const double e1 = pix_at(ex + 1, ey);
                if (e0 >= e1) { cx = ex; cy = ey; } else { cx = ex + 1; cy = ey; }
            }
        } else {
            if (tcon) {
                const bool has_left = (ex >= 1);
                const bool has_right = (ex < SH);
                if (has_left && has_right) {
                    const double pr = pix_at(ex, ey);
                    const double pl = pix_at(ex - 1, ey);
                    if (pr <= pl) { cx = ex; cy = ey; } else { cx = ex - 1; cy = ey; }
                } else if (has_right) {
                    cx = ex; cy = ey;
                } else {
                    cx = ex - 1; cy = ey;
                }
            } else {
                const double e0 = pix_at(ex, ey);
                const double e1 = pix_at(ex, ey + 1);
                if (e0 >= e1) { cx = ex; cy = ey; } else { cx = ex; cy = ey + 1; }
            }
        }
    };

    std::vector<SortKey> keys(edges.size());
    for (size_t i = 0; i < edges.size(); ++i) {
        keys[i].t = edges[i].t;
        keys[i].idx = static_cast<uint32_t>(i);
    }
    sort_keys_by_t(keys);
    {
        std::vector<EdgeRec> sorted_edges(edges.size());
        std::vector<uint32_t> sorted_coords(edges.size());
        for (size_t i = 0; i < edges.size(); ++i) {
            sorted_edges[i] = edges[keys[i].idx];
            sorted_coords[i] = ecoord[keys[i].idx];
        }
        edges.swap(sorted_edges);
        ecoord.swap(sorted_coords);
    }

    auto vertex_parent_pixel = [&](uint32_t vx, uint32_t vy, double b,
                                   uint32_t& bx, uint32_t& by) {
        if (one_dim) {
            if (!tcon) {
                bx = vx;
                by = 0;
                return;
            }
            if (vx < IH && pix_at(vx, 0) == b) { bx = vx; by = 0; return; }
            if (vx > 0 && pix_at(vx - 1, 0) == b) { bx = vx - 1; by = 0; return; }
            bx = (vx == 0) ? 0u : vx - 1u;
            by = 0;
            return;
        }
        if (!tcon) {
            bx = vx;
            by = vy;
            return;
        }
        if (vx < IH && vy < IW && pix_at(vx, vy) == b) { bx = vx; by = vy; return; }
        if (vx > 0 && vy < IW && pix_at(vx - 1, vy) == b) { bx = vx - 1; by = vy; return; }
        if (vx > 0 && vy > 0 && pix_at(vx - 1, vy - 1) == b) { bx = vx - 1; by = vy - 1; return; }
        if (vx < IH && vy > 0 && pix_at(vx, vy - 1) == b) { bx = vx; by = vy - 1; return; }
        bx = (vx == 0) ? 0u : vx - 1u;
        by = (vy == 0) ? 0u : vy - 1u;
    };

    auto compute_h0_pairs = [&]() {
        std::vector<WritePairs> pairs;
        std::vector<uint32_t> v_parent(nvert);
        std::iota(v_parent.begin(), v_parent.end(), 0u);
        std::vector<double> v_root_birth(nvert);
        std::vector<uint32_t> v_root_v(nvert);
        for (size_t i = 0; i < nvert; ++i) {
            v_root_birth[i] = v_birth[i];
            v_root_v[i] = static_cast<uint32_t>(i);
        }

        for (size_t i = 0; i < edges.size(); ++i) {
            const EdgeRec& e = edges[i];
            uint32_t r1 = uf_find(v_parent, e.v1);
            uint32_t r2 = uf_find(v_parent, e.v2);
            if (r1 == r2) continue;
            uint32_t younger;
            uint32_t older;
            if (v_root_birth[r1] > v_root_birth[r2]) { younger = r1; older = r2; }
            else if (v_root_birth[r1] < v_root_birth[r2]) { younger = r2; older = r1; }
            else { younger = (r1 > r2) ? r1 : r2; older = (younger == r1) ? r2 : r1; }
            const double b = v_root_birth[younger];
            const double d = e.t;
            if (b != d) {
                const uint32_t v_id = v_root_v[younger];
                const uint32_t vx = v_id % VH;
                const uint32_t vy = v_id / VH;
                uint32_t bx, by, dx, dy;
                vertex_parent_pixel(vx, vy, b, bx, by);
                creator_pixel(static_cast<uint32_t>(i), dx, dy);
                pairs.emplace_back(0, b, d,
                                   bx, by, 0u, 0u,
                                   dx, dy, 0u, 0u, print);
            }
            v_parent[younger] = older;
        }

        double min_b = std::numeric_limits<double>::infinity();
        uint32_t min_v = 0;
        for (size_t i = 0; i < nvert; ++i) {
            if (v_parent[i] == static_cast<uint32_t>(i) && v_root_birth[i] < min_b) {
                min_b = v_root_birth[i];
                min_v = v_root_v[i];
            }
        }
        if (min_b < threshold) {
            const uint32_t vx = min_v % VH;
            const uint32_t vy = min_v / VH;
            uint32_t bx, by;
            vertex_parent_pixel(vx, vy, min_b, bx, by);
            pairs.emplace_back(0, min_b, threshold,
                               bx, by, 0u, 0u,
                               0u, 0u, 0u, 0u, print);
        }
        return pairs;
    };

    if (one_dim || config.maxdim < 1) {
        auto h0_pairs = compute_h0_pairs();
        writepairs.insert(writepairs.end(), h0_pairs.begin(), h0_pairs.end());
        return true;
    }

    auto compute_h1_pairs = [&]() {
        std::vector<WritePairs> pairs;
        const uint32_t outside_id = static_cast<uint32_t>(nsq);
        std::vector<uint32_t> s_parent(nsq + 1);
        std::vector<double> s_max_pix(nsq + 1);
        std::vector<uint32_t> s_max_x(nsq + 1, 0u);
        std::vector<uint32_t> s_max_y(nsq + 1, 0u);
        std::iota(s_parent.begin(), s_parent.end(), 0u);
        for (size_t i = 0; i < nsq; ++i) {
            s_max_pix[i] = sq_birth[i];
            if (tcon) {
                s_max_x[i] = static_cast<uint32_t>(i % SH);
                s_max_y[i] = static_cast<uint32_t>(i / SH);
            } else {
                s_max_x[i] = sq_maxx[i];
                s_max_y[i] = sq_maxy[i];
            }
        }
        s_max_pix[outside_id] = std::numeric_limits<double>::infinity();

        for (size_t i = edges.size(); i-- > 0;) {
            const EdgeRec& e = edges[i];
            const uint32_t a = (e.s1 == INVALID_ID) ? outside_id : e.s1;
            const uint32_t b = (e.s2 == INVALID_ID) ? outside_id : e.s2;
            uint32_t r1 = uf_find(s_parent, a);
            uint32_t r2 = uf_find(s_parent, b);
            if (r1 == r2) continue;
            uint32_t younger;
            uint32_t older;
            if (s_max_pix[r1] < s_max_pix[r2]) { younger = r1; older = r2; }
            else if (s_max_pix[r1] > s_max_pix[r2]) { younger = r2; older = r1; }
            else { younger = (r1 < r2) ? r1 : r2; older = (younger == r1) ? r2 : r1; }
            const double bp = e.t;
            const double dp = s_max_pix[younger];
            if (bp != dp) {
                uint32_t cx, cy;
                creator_pixel(static_cast<uint32_t>(i), cx, cy);
                pairs.emplace_back(1, bp, dp,
                                   cx, cy, 0u, 0u,
                                   s_max_x[younger], s_max_y[younger], 0u, 0u,
                                   print);
            }
            s_parent[younger] = older;
        }
        return pairs;
    };

    std::vector<WritePairs> h0_pairs;
    std::vector<WritePairs> h1_pairs;
    std::thread h0_thread([&]() { h0_pairs = compute_h0_pairs(); });
    std::thread h1_thread([&]() { h1_pairs = compute_h1_pairs(); });
    h0_thread.join();
    h1_thread.join();

    writepairs.insert(writepairs.end(), h0_pairs.begin(), h0_pairs.end());
    writepairs.insert(writepairs.end(), h1_pairs.begin(), h1_pairs.end());
    return true;
}
