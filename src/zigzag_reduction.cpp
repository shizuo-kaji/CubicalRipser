/* Persistence of the FastZigzag coned filtration over run cells. */

#include "zigzag_reduction.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "union_find.h"

namespace {

constexpr uint32_t kNone = std::numeric_limits<uint32_t>::max();

// Working column as a max-heap with lazy F_2 cancellation.  Adding a column
// only pushes its entries, so a long working column (a cycle being pushed
// down past many cells) is not copied on every addition.
class WorkingColumn {
public:
  void clear() { heap_.clear(); }

  void add(const std::vector<uint32_t> &column) {
    for (uint32_t cell : column) {
      heap_.push_back(cell);
      std::push_heap(heap_.begin(), heap_.end());
    }
  }

  // Largest entry with odd multiplicity (the pivot), or kNone if the column
  // is zero.  Cancelled pairs above it are discarded.
  uint32_t pivot() {
    while (!heap_.empty()) {
      const uint32_t top = pop();
      if (!heap_.empty() && heap_.front() == top) {
        pop();
        continue;
      }
      heap_.push_back(top);
      std::push_heap(heap_.begin(), heap_.end());
      return top;
    }
    return kNone;
  }

  // Removes the entry just returned by pivot().
  void remove_pivot() { pop(); }

  // The reduced column in increasing order; leaves the working column empty.
  void drain(std::vector<uint32_t> &out) {
    out.clear();
    for (uint32_t cell = pivot(); cell != kNone; cell = pivot()) {
      remove_pivot();
      out.push_back(cell);
    }
    std::reverse(out.begin(), out.end());
  }

private:
  std::vector<uint32_t> heap_;

  uint32_t pop() {
    std::pop_heap(heap_.begin(), heap_.end());
    const uint32_t top = heap_.back();
    heap_.pop_back();
    return top;
  }
};

// Spanning forest of the graph L formed by vertex and edge runs, grown in the
// order their cones enter the coned filtration.  Vertices are numbered as they
// are added, so a lower number means an older vertex.  Components are tracked
// by union-find whose roots are their oldest vertices; the trees are kept as
// parent pointers, so the path between two vertices of one tree is read off
// in time proportional to its length.
class ConeForest {
public:
  explicit ConeForest(size_t runs) : index_(runs, kNone) {}

  void add_vertex(uint32_t run) {
    const uint32_t v = static_cast<uint32_t>(vertex_run_.size());
    index_[run] = v;
    vertex_run_.push_back(run);
    component_.push_back(v);
    size_.push_back(1);
    tree_parent_.push_back(v);
    tree_edge_.push_back(kNone);
    seen_.push_back(0);
  }

  // Adds the edge run e between vertex runs a and b.  Returns the oldest
  // vertex run of the younger component if e joins two components, and kNone
  // otherwise.
  uint32_t join(uint32_t a, uint32_t b, uint32_t e) {
    const uint32_t u = index_[a];
    const uint32_t w = index_[b];
    const uint32_t ru = uf_find(component_, u);
    const uint32_t rw = uf_find(component_, w);
    if (ru == rw)
      return kNone;
    // Re-rooting the smaller tree keeps the total work O(n log n).
    if (size_[ru] < size_[rw])
      attach(u, w, e);
    else
      attach(w, u, e);
    const uint32_t older = std::min(ru, rw);
    const uint32_t younger = std::max(ru, rw);
    size_[older] = size_[ru] + size_[rw];
    component_[younger] = older;
    return vertex_run_[younger];
  }

  // The edge runs on the tree path between vertex runs a and b, which share a
  // tree.
  void path(uint32_t a, uint32_t b, std::vector<uint32_t> &edges) {
    const uint32_t u = index_[a];
    const uint32_t w = index_[b];
    // Climb from both ends in turn, marking visited vertices; the first
    // vertex reached by both is the lowest common ancestor.
    ++epoch_;
    const uint32_t mark_u = epoch_ << 1;
    const uint32_t mark_w = mark_u | 1U;
    uint32_t x = u;
    uint32_t y = w;
    seen_[x] = mark_u;
    seen_[y] = mark_w;
    uint32_t ancestor = kNone;
    while (ancestor == kNone) {
      if (tree_parent_[x] != x) {
        x = tree_parent_[x];
        if (seen_[x] == mark_w)
          ancestor = x;
        seen_[x] = mark_u;
      }
      if (ancestor == kNone && tree_parent_[y] != y) {
        y = tree_parent_[y];
        if (seen_[y] == mark_u)
          ancestor = y;
        seen_[y] = mark_w;
      }
    }
    edges.clear();
    for (uint32_t v = u; v != ancestor; v = tree_parent_[v])
      edges.push_back(tree_edge_[v]);
    for (uint32_t v = w; v != ancestor; v = tree_parent_[v])
      edges.push_back(tree_edge_[v]);
  }

private:
  std::vector<uint32_t> index_;      // run id -> vertex number
  std::vector<uint32_t> vertex_run_; // vertex number -> run id
  std::vector<uint32_t> component_;  // union-find parent
  std::vector<uint32_t> size_;       // component size, valid at roots
  std::vector<uint32_t> tree_parent_;
  std::vector<uint32_t> tree_edge_;  // edge run to tree_parent_
  std::vector<uint32_t> seen_;       // path search marks
  uint32_t epoch_{0};

  // Re-roots the tree of x at x and hangs it below y through edge e.
  void attach(uint32_t x, uint32_t y, uint32_t e) {
    uint32_t parent = y;
    uint32_t edge = e;
    while (true) {
      const uint32_t next = tree_parent_[x];
      const uint32_t next_edge = tree_edge_[x];
      tree_parent_[x] = parent;
      tree_edge_[x] = edge;
      if (next == x)
        break;
      parent = x;
      edge = next_edge;
      x = next;
    }
  }
};

class ConedReducer {
public:
  ConedReducer(const ZigzagRunTable &table, bool exhaustive)
      : table_(table), grid_(table.grid), exhaustive_(exhaustive),
        n_(static_cast<uint32_t>(table.runs.size())),
        birth_column_(2 * static_cast<size_t>(n_) + 1, kNone) {}

  std::vector<ConedPair> run(int max_dim) {
    // H_0 comes first: it identifies the negative (spanning-tree) edges, which
    // the dimension-2 reduction leaves out.
    pair_components();
    // Pairs of dimension p are found by reducing the (p + 1)-cells.  Going
    // down from the top lets every birth found in one dimension skip its own
    // column in the next (clearing).
    const int top = std::min(max_dim + 1, grid_.dim + 1);
    for (int dim = top; dim >= 2; --dim)
      pair_boundaries(static_cast<uint8_t>(dim), dim == top);
    return std::move(pairs_);
  }

private:
  const ZigzagRunTable &table_;
  const ZigzagCellGrid &grid_;
  const bool exhaustive_;
  const uint32_t n_;
  // birth_column_[q]: the column whose pivot is cell q, i.e. q is the birth of
  // a pair; kNone otherwise.
  std::vector<uint32_t> birth_column_;
  std::vector<ConedPair> pairs_;
  // Base edge runs that merge two components in H_0, by run id.
  std::vector<bool> negative_edge_;

  uint32_t base_rank(uint32_t run) const { return run + 1; }
  uint32_t cone_rank(uint32_t run) const {
    return 2 * n_ - table_.runs[run].deletion;
  }
  uint32_t cone_run(uint32_t rank) const {
    return table_.deletion_order[2 * n_ - rank];
  }
  uint8_t run_dim(uint32_t run) const {
    return grid_.cell_dim(table_.runs[run].cell);
  }

  // Boundary as increasing ranks: d(w·r) = r + w·(dr), with d(w·v) = v + w.
  void boundary(uint32_t rank, std::vector<uint32_t> &out) const {
    out.clear();
    if (rank > n_) {
      const uint32_t run = cone_run(rank);
      out.push_back(base_rank(run));
      if (run_dim(run) == 0)
        out.push_back(0);
      else
        table_.for_each_face_run(
            run, [&](uint32_t q) { out.push_back(cone_rank(q)); });
    } else {
      table_.for_each_face_run(
          rank - 1, [&](uint32_t q) { out.push_back(base_rank(q)); });
    }
    std::sort(out.begin(), out.end());
  }

  // Reduction state of one dimension.
  struct Reduction {
    uint8_t dim;
    bool top;
    // Columns hold only positive edges (dimension 2; see pair_boundaries).
    bool positive_edges_only;
    // Cached columns are reduced exhaustively (see pair_boundaries).
    bool exhaustive;
    // Reduced columns that differ from the boundary of their cell.
    std::unordered_map<uint32_t, std::vector<uint32_t>> reduced;
    std::vector<uint32_t> column;
    WorkingColumn working;
  };

  void drop_negative_edges(std::vector<uint32_t> &column) const {
    column.erase(std::remove_if(column.begin(), column.end(),
                                [this](uint32_t rank) {
                                  return rank <= n_ &&
                                         negative_edge_[rank - 1];
                                }),
                 column.end());
  }

  // Adds the reduced column of `owner` to the working column.
  void add_column_of(uint32_t owner, Reduction &red) {
    const auto cached = red.reduced.find(owner);
    if (cached != red.reduced.end()) {
      red.working.add(cached->second);
      return;
    }
    boundary(owner, red.column);
    if (red.positive_edges_only)
      drop_negative_edges(red.column);
    red.working.add(red.column);
  }

  // Reduces the column of `cell` already loaded into red.working and records
  // its pair.  `cache` keeps the reduced column even if no column was added.
  void reduce(uint32_t cell, bool cache, Reduction &red) {
    for (uint32_t pivot = red.working.pivot(); pivot != kNone;
         pivot = red.working.pivot()) {
      const uint32_t other = birth_column_[pivot];
      if (other == kNone) {
        birth_column_[pivot] = cell;
        pairs_.push_back({static_cast<uint8_t>(red.dim - 1), pivot, cell});
        if (cache)
          store_reduced(cell, red);
        return;
      }
      add_column_of(other, red);
      cache = true;
    }
    // A zero column creates a dim-cycle.  Below the top dimension such a cell
    // has already been cleared, because the coned complex is contractible.
    if (!red.top)
      throw std::logic_error("zigzag: unexpected essential class");
  }

  // Moves the working column, whose pivot has just been paired, into the
  // cache.  With exhaustive reduction, every entry below the pivot that is the
  // pivot of another column is eliminated first, so cached columns stay short.
  void store_reduced(uint32_t cell, Reduction &red) {
    std::vector<uint32_t> &stored = red.reduced[cell];
    if (!red.exhaustive) {
      red.working.drain(stored);
      return;
    }
    stored.assign(1, red.working.pivot());
    red.working.remove_pivot();
    for (uint32_t entry = red.working.pivot(); entry != kNone;
         entry = red.working.pivot()) {
      const uint32_t other = birth_column_[entry];
      if (other == kNone) {
        red.working.remove_pivot();
        stored.push_back(entry);
      } else {
        add_column_of(other, red); // cancels entry
      }
    }
    std::reverse(stored.begin(), stored.end());
  }

  void reduce_boundary(uint32_t cell, Reduction &red) {
    boundary(cell, red.column);
    if (red.positive_edges_only)
      drop_negative_edges(red.column);
    red.working.clear();
    red.working.add(red.column);
    reduce(cell, false, red);
  }

  // Pairs of dimension dim - 1 by reducing the boundaries of the dim-cells in
  // increasing rank.  Cells already found as births are positive and skipped.
  //
  // In dimension 2 every column is a 1-cycle of base edges (see
  // pair_cone_triangles), whose newest edge -- its pivot -- is positive.
  // Restricting the columns to positive edges is linear and keeps all
  // pivots, so the negative edges are dropped.  For two-dimensional frames,
  // where cached columns of noisy sequences are added again and again, they
  // are also reduced exhaustively unless disabled.  In measurements this cut
  // the time on noisy 2D sequences by up to a third and cost about 15% on
  // smooth ones; it slowed down three-dimensional frames, and enabling it only
  // once reuse becomes frequent was slower than either fixed choice.
  void pair_boundaries(uint8_t dim, bool top) {
    Reduction red;
    red.dim = dim;
    red.top = top;
    red.positive_edges_only = dim == 2;
    red.exhaustive = exhaustive_ && dim == 2 && grid_.dim == 2;
    for (uint32_t run = 0; run < n_; ++run) {
      const uint32_t cell = base_rank(run);
      if (run_dim(run) == dim && birth_column_[cell] == kNone)
        reduce_boundary(cell, red);
    }
    if (dim == 2) {
      pair_cone_triangles(red);
      return;
    }
    for (uint32_t k = n_; k-- > 0;) { // cones, rank 2n - k
      const uint32_t cell = 2 * n_ - k;
      if (run_dim(table_.deletion_order[k]) + 1 == dim &&
          birth_column_[cell] == kNone)
        reduce_boundary(cell, red);
    }
  }

  // The cone triangles w·e (dimension 2) without column reduction of their
  // cone parts.  Cone cells enter in reverse deletion order and form the
  // coned subcomplex w·L; its cone edges and triangles behave like the
  // vertices and edges of the graph L.  A triangle joining two components of
  // L therefore kills the cone edge of the younger component's oldest vertex
  // (elder rule).  For a triangle w·e inside one component, adding the
  // triangles along the path between the ends of e in a spanning forest of L
  // cancels the cone part, leaving the cycle e + path in K, which is reduced
  // against the base columns as usual.
  void pair_cone_triangles(Reduction &red) {
    ConeForest forest(n_);
    for (uint32_t k = n_; k-- > 0;) { // increasing rank 2n - k
      const uint32_t run = table_.deletion_order[k];
      const uint32_t cell = 2 * n_ - k;
      if (run_dim(run) == 0) {
        forest.add_vertex(run);
        continue;
      }
      if (run_dim(run) != 1)
        continue;
      uint32_t ends[2];
      int i = 0;
      table_.for_each_face_run(run, [&](uint32_t q) { ends[i++] = q; });
      const uint32_t younger = forest.join(ends[0], ends[1], run);
      const bool cleared = birth_column_[cell] != kNone;
      if (younger != kNone) {
        if (cleared) // a positive triangle lies inside one component
          throw std::logic_error("zigzag: positive cone triangle joins L");
        birth_column_[cone_rank(younger)] = cell;
        pairs_.push_back({1, cone_rank(younger), cell});
        continue;
      }
      if (cleared)
        continue;
      forest.path(ends[0], ends[1], red.column);
      red.column.push_back(run);
      for (uint32_t &edge : red.column)
        edge = base_rank(edge);
      drop_negative_edges(red.column);
      red.working.clear();
      red.working.add(red.column);
      // The column is not the boundary of its cell, so it is always kept.
      reduce(cell, true, red);
    }
  }

  // H_0 by union-find over the cone vertex (rank 0), the vertex runs, the edge
  // runs and the cone edges w·v, in increasing rank.
  void pair_components() {
    negative_edge_.assign(n_, false);
    std::vector<uint32_t> parent(static_cast<size_t>(n_) + 1);
    std::iota(parent.begin(), parent.end(), 0U);
    auto join = [&](uint32_t a, uint32_t b, uint32_t edge) {
      const uint32_t ra = uf_find(parent, a);
      const uint32_t rb = uf_find(parent, b);
      if (ra == rb)
        return false; // the edge creates a 1-cycle
      // Roots are the oldest (lowest-rank) vertices of their components, so
      // the younger root dies.
      const uint32_t younger = std::max(ra, rb);
      parent[younger] = std::min(ra, rb);
      pairs_.push_back({0, younger, edge});
      return true;
    };

    for (uint32_t r = 0; r < n_; ++r) {
      if (run_dim(r) != 1)
        continue;
      uint32_t ends[2];
      int i = 0;
      table_.for_each_face_run(r, [&](uint32_t q) { ends[i++] = q; });
      if (join(base_rank(ends[0]), base_rank(ends[1]), base_rank(r)))
        negative_edge_[r] = true;
    }
    // Cone edges enter in reverse deletion order.
    for (uint32_t k = n_; k-- > 0;) {
      const uint32_t r = table_.deletion_order[k];
      if (run_dim(r) == 0)
        join(0, base_rank(r), cone_rank(r));
    }
  }
};

} // namespace

std::vector<ConedPair> reduce_coned_filtration(const ZigzagRunTable &table,
                                               int max_dim, bool exhaustive) {
  return ConedReducer(table, exhaustive).run(max_dim);
}
