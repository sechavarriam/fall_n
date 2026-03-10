#ifndef FALL_N_SIMPLEX_CELL_HH
#define FALL_N_SIMPLEX_CELL_HH

// =============================================================================
//  SimplexCell.hh — Reference simplex with Lagrange basis
// =============================================================================
//
//  The reference simplex in D dimensions is defined on the standard domain:
//
//    S_D = { (ξ₁, …, ξ_D) : ξ_i ≥ 0, ξ₁ + … + ξ_D ≤ 1 }
//
//  with vertices at the origin and the D unit vectors:
//
//    v₀ = (0,0,…,0),  v₁ = (1,0,…,0),  v₂ = (0,1,…,0),  …, v_D = (0,…,0,1)
//
//  ─── Barycentric coordinates ───────────────────────────────────────────
//
//    λ₀ = 1 − ξ₁ − ξ₂ − … − ξ_D
//    λ_i = ξ_i   (i = 1, …, D)
//
//  ─── Lagrange shape functions ──────────────────────────────────────────
//
//  Order 1 (linear):  N_i = λ_i  (D+1 nodes = vertices)
//
//  Order 2 (quadratic):
//    Vertices  (D+1):  N_i = λ_i (2λ_i − 1)
//    Midpoints (D(D+1)/2):  N_{ij} = 4 λ_i λ_j  for i < j
//    Total nodes: (D+1)(D+2)/2
//
//  ─── Subentity topology ────────────────────────────────────────────────
//
//  A simplex of dimension D has:
//    C(D+1, k+1) sub-simplices of dimension k.
//  Faces (codim 1): D+1 sub-simplices of dimension D-1.
//
//  The f-th face (f = 0..D) is the sub-simplex opposite vertex f,
//  i.e. the face spanned by all vertices except v_f.
//
// =============================================================================

#include <array>
#include <cstddef>

#include "Point.hh"

namespace geometry::simplex {


// =============================================================================
//  SimplexNodeLayout — compile-time node enumeration
// =============================================================================

/// Number of nodes in a simplex of dimension D at Lagrange order p.
///
///   nNodes(D, p)  = C(D + p, D)
///
/// For p = 1:  D + 1           (vertices only)
/// For p = 2:  (D+1)(D+2)/2    (vertices + edge midpoints)
consteval std::size_t simplex_num_nodes(std::size_t D, std::size_t p) {
    // C(D + p, D) = C(D + p, p) = (D+p)! / (D! · p!)
    std::size_t result = 1;
    for (std::size_t i = 0; i < p; ++i) {
        result = result * (D + p - i) / (i + 1);
    }
    return result;
}

/// Number of faces (codim-1 sub-simplices) of a D-simplex = D + 1.
consteval std::size_t simplex_num_faces(std::size_t D) {
    return D + 1;
}


// =============================================================================
//  Reference node coordinates (compile-time)
// =============================================================================

/// Compute reference nodal coordinates for a D-simplex of order p.
///
///   Order 1:  D+1 vertices.
///   Order 2:  vertices + edge midpoints (total = (D+1)(D+2)/2).
///
/// Nodes are ordered: vertices first (same order as barycentric coords),
/// then edge midpoints in lexicographic order of the vertex pairs.
template <std::size_t D, std::size_t Order>
consteval auto simplex_reference_nodes() {
    constexpr std::size_t N = simplex_num_nodes(D, Order);
    std::array<Point<D>, N> nodes{};

    if constexpr (Order == 1) {
        // Vertex 0: origin (0,0,...,0)
        nodes[0] = Point<D>{};
        // Vertex i (i=1..D): unit vector e_i
        for (std::size_t i = 1; i <= D; ++i) {
            std::array<double, D> c{};
            c[i - 1] = 1.0;
            nodes[i] = Point<D>(c);
        }
    }
    else if constexpr (Order == 2) {
        // Vertices (same as order 1)
        nodes[0] = Point<D>{};
        for (std::size_t i = 1; i <= D; ++i) {
            std::array<double, D> c{};
            c[i - 1] = 1.0;
            nodes[i] = Point<D>(c);
        }
        // Edge midpoints: for each pair (i, j) with i < j, midpoint = (v_i + v_j) / 2
        std::size_t idx = D + 1;
        for (std::size_t i = 0; i <= D; ++i) {
            for (std::size_t j = i + 1; j <= D; ++j) {
                std::array<double, D> c{};
                for (std::size_t d = 0; d < D; ++d) {
                    c[d] = (nodes[i].coord()[d] + nodes[j].coord()[d]) * 0.5;
                }
                nodes[idx++] = Point<D>(c);
            }
        }
    }

    return nodes;
}


// =============================================================================
//  SimplexBasis — Lagrange shape functions on the reference simplex
// =============================================================================
//
//  Provides shape_function(i) and shape_function_derivative(i, j) as
//  callables taking an std::array<double, D> (natural coordinates ξ).
//

template <std::size_t D, std::size_t Order>
class SimplexBasis {
public:
    static constexpr std::size_t dim       = D;
    static constexpr std::size_t order     = Order;
    static constexpr std::size_t num_nodes = simplex_num_nodes(D, Order);

    // ── Barycentric coordinates from natural coordinates ────────────
    //   λ₀ = 1 − ξ₁ − … − ξ_D
    //   λ_i = ξ_i  for i = 1..D
    static constexpr double lambda(std::size_t i, const std::array<double, D>& xi) noexcept {
        if (i == 0) {
            double sum = 0.0;
            for (std::size_t d = 0; d < D; ++d) sum += xi[d];
            return 1.0 - sum;
        }
        return xi[i - 1];
    }

    // ── d(λ_i)/d(ξ_j) ──────────────────────────────────────────────
    static constexpr double dlambda_dxi(std::size_t i, std::size_t j) noexcept {
        if (i == 0) return -1.0;   // dλ₀/dξ_j = −1  for all j
        return (i == j + 1) ? 1.0 : 0.0;  // dλ_i/dξ_j = δ_{i,j+1}
    }

    // ── Shape function N_i(ξ) ───────────────────────────────────────
    constexpr auto shape_function(std::size_t i) const noexcept {
        if constexpr (Order == 1) {
            // Linear: N_i = λ_i
            return [i](const std::array<double, D>& xi) -> double {
                return lambda(i, xi);
            };
        }
        else if constexpr (Order == 2) {
            // Unified lambda: decode vertex vs midpoint at evaluation time
            // to ensure a single return type.
            constexpr std::size_t num_verts = D + 1;

            // Pre-decode edge midpoint (a, b) if applicable
            std::size_t a = 0, b = 0;
            if (i >= num_verts) {
                std::size_t mid_idx = i - num_verts;
                std::size_t cnt = 0;
                for (std::size_t ii = 0; ii <= D; ++ii) {
                    for (std::size_t jj = ii + 1; jj <= D; ++jj) {
                        if (cnt == mid_idx) { a = ii; b = jj; }
                        ++cnt;
                    }
                }
            }

            return [i, a, b](const std::array<double, D>& xi) -> double {
                constexpr std::size_t nv = D + 1;
                if (i < nv) {
                    double L = lambda(i, xi);
                    return L * (2.0 * L - 1.0);
                } else {
                    return 4.0 * lambda(a, xi) * lambda(b, xi);
                }
            };
        }
    }

    // ── Shape function derivative dN_i/dξ_j ────────────────────────
    constexpr auto shape_function_derivative(std::size_t i, std::size_t j) const noexcept {
        if constexpr (Order == 1) {
            // dN_i/dξ_j = dλ_i/dξ_j
            return [i, j](const std::array<double, D>& /*xi*/) -> double {
                return dlambda_dxi(i, j);
            };
        }
        else if constexpr (Order == 2) {
            // Unified lambda: decode vertex vs midpoint at evaluation time
            constexpr std::size_t num_verts = D + 1;

            std::size_t a = 0, b = 0;
            if (i >= num_verts) {
                std::size_t mid_idx = i - num_verts;
                std::size_t cnt = 0;
                for (std::size_t ii = 0; ii <= D; ++ii) {
                    for (std::size_t jj = ii + 1; jj <= D; ++jj) {
                        if (cnt == mid_idx) { a = ii; b = jj; }
                        ++cnt;
                    }
                }
            }

            return [i, j, a, b](const std::array<double, D>& xi) -> double {
                constexpr std::size_t nv = D + 1;
                if (i < nv) {
                    double L = lambda(i, xi);
                    return (4.0 * L - 1.0) * dlambda_dxi(i, j);
                } else {
                    return 4.0 * (dlambda_dxi(a, j) * lambda(b, xi) +
                                  lambda(a, xi) * dlambda_dxi(b, j));
                }
            };
        }
    }

    /// Interpolate a nodal field F at point xi.
    constexpr double interpolate(const auto& F, const std::array<double, D>& xi) const noexcept {
        double val = 0.0;
        for (std::size_t i = 0; i < num_nodes; ++i) {
            val += F[i] * shape_function(i)(xi);
        }
        return val;
    }

    constexpr SimplexBasis() = default;
};


// =============================================================================
//  SimplexCell — Reference simplex (analogous to LagrangianCell for hexes)
// =============================================================================

template <std::size_t D, std::size_t Order>
class SimplexCell {
public:
    static constexpr std::size_t dim       = D;
    static constexpr std::size_t order     = Order;
    static constexpr std::size_t num_nodes_value = simplex_num_nodes(D, Order);

    static constexpr auto reference_nodes = simplex_reference_nodes<D, Order>();
    static constexpr SimplexBasis<D, Order> basis{};

    // ── Subentity topology ──────────────────────────────────────────
    //
    //  Faces of a D-simplex: D+1 sub-simplices of dimension D-1.
    //  The f-th face is opposite vertex f (all vertices except f).
    //
    static constexpr std::size_t num_faces    = D + 1;
    static constexpr std::size_t num_vertices = D + 1;

    /// Number of nodes on the f-th face.
    ///
    ///   Order 1: D nodes per face   (just vertices)
    ///   Order 2: simplex_num_nodes(D-1, 2) nodes per face
    static consteval std::size_t face_num_nodes(std::size_t /*f*/) {
        return simplex_num_nodes(D - 1, Order);
    }

    /// Node indices of the f-th face (opposite vertex f).
    ///
    /// For order 1:
    ///   face f → all vertex indices except f.
    ///
    /// For order 2:
    ///   face f → all vertex indices except f, plus all
    ///   edge midpoints whose defining pair does NOT include f.
    ///
    /// Returns a fixed-size array padded with zeros beyond the actual count.
    struct FaceNodeIndices {
        std::array<std::size_t, num_nodes_value> indices{};
        std::size_t size = 0;
    };

    static consteval FaceNodeIndices face_node_indices(std::size_t f) {
        FaceNodeIndices result{};

        if constexpr (Order == 1) {
            // Vertices except f
            for (std::size_t v = 0; v <= D; ++v) {
                if (v != f) result.indices[result.size++] = v;
            }
        }
        else if constexpr (Order == 2) {
            // Vertices except f
            for (std::size_t v = 0; v <= D; ++v) {
                if (v != f) result.indices[result.size++] = v;
            }
            // Edge midpoints: include those whose pair (a,b) does NOT contain f
            std::size_t mid_start = D + 1;
            std::size_t mid_idx = 0;
            for (std::size_t a = 0; a <= D; ++a) {
                for (std::size_t b = a + 1; b <= D; ++b) {
                    if (a != f && b != f) {
                        result.indices[result.size++] = mid_start + mid_idx;
                    }
                    ++mid_idx;
                }
            }
        }

        return result;
    }

    // ── Faces subentity descriptor ──────────────────────────────────
    //
    //  Provides the same interface expected by ElementGeometry's
    //  OwningModel for face topology queries: subentity_num_nodes,
    //  NodeIndices, node_indices.
    //
    //  All faces of a D-simplex are (D-1)-simplices of the same Order.
    //
    struct Faces {
        static constexpr std::size_t count = num_faces;

        // Each face has the same number of nodes (a (D-1)-simplex of the same Order)
        static consteval std::size_t subentity_num_nodes(std::size_t /*f*/) {
            return simplex_num_nodes(D - 1, Order);
        }

        // Same NodeIndices structure used by Cell.hh SubentityDescriptor
        struct NodeIndices {
            std::array<std::size_t, num_nodes_value> indices{};
            std::size_t size = 0;
        };

        static consteval NodeIndices node_indices(std::size_t f) {
            auto fni = face_node_indices(f);
            NodeIndices result{};
            for (std::size_t i = 0; i < fni.size; ++i)
                result.indices[i] = fni.indices[i];
            result.size = fni.size;
            return result;
        }
    };

    // Also expose Vertices (codim = D sub-simplices = points)
    struct Vertices {
        static constexpr std::size_t count = D + 1;
    };

    consteval SimplexCell() = default;
};


} // namespace geometry::simplex

#endif // FALL_N_SIMPLEX_CELL_HH
