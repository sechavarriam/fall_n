#ifndef FALL_N_NODE_SELECTOR_HH
#define FALL_N_NODE_SELECTOR_HH

// ═══════════════════════════════════════════════════════════════════════════════
//  NodeSelector.hh — Compile-time polymorphic node selection predicates
// ═══════════════════════════════════════════════════════════════════════════════
//
//  A NodeSelector is any callable that accepts a const Node<dim>& and returns
//  bool.  Built-in selectors cover common geometric predicates.  Users can
//  pass lambdas, function objects, or any other callable for custom logic.
//
//  ─── Design Rationale ───────────────────────────────────────────────────
//
//  • No virtual dispatch — selectors are template parameters, resolved at
//    compile time through concepts.  After inlining, a PlaneSelector
//    collapses to a single floating-point comparison.
//
//  • Composable — selectors can be combined with select_and, select_or,
//    select_not.  Any lambda qualifies as a selector.
//
//  • Consistent with fall_n policy-based architecture — selectors are
//    lightweight value types, cheap to copy and pass by value.
//
//  ─── Usage ──────────────────────────────────────────────────────────────
//
//    // Fix u_x = 0 at x=0 plane (roller in x):
//    M.constrain_dof_where(fn::PlaneSelector<3>{0, 0.0}, 0);
//
//    // Fix all DOFs at x=0 (clamped):
//    M.constrain_where(fn::PlaneSelector<3>{0, 0.0});
//
//    // Impose displacement u_x = 0.01 at x=10:
//    M.constrain_dof_where(fn::PlaneSelector<3>{0, 10.0}, 0, 0.01);
//
//    // Custom selector via lambda:
//    M.constrain_where([](const auto& n){ return n.coord(2) > 5.0; });
//
//    // Composite: node at intersection of two planes:
//    auto corner = fn::select_and(fn::PlaneSelector<3>{0, 0.0},
//                                 fn::PlaneSelector<3>{1, 0.0});
//    M.constrain_where(corner);
//
// ═══════════════════════════════════════════════════════════════════════════════

#include <array>
#include <cmath>
#include <cstddef>
#include <concepts>
#include <vector>
#include <algorithm>
#include <initializer_list>
#include <utility>

namespace fn {

// ─── Concept ─────────────────────────────────────────────────────────────────
//
//  Any callable  f(const NodeT&) → bool  qualifies as a NodeSelector.
//  This includes PlaneSelector, BoxSelector, lambdas, std::function, etc.

template <typename S, typename NodeT>
concept NodeSelector = requires(const S& s, const NodeT& n) {
    { s(n) } -> std::convertible_to<bool>;
};


// ═════════════════════════════════════════════════════════════════════════════
//  Built-in Selectors
// ═════════════════════════════════════════════════════════════════════════════

// ─── PlaneSelector ───────────────────────────────────────────────────────────
//
//  Selects nodes lying on a coordinate hyperplane:
//    |node.coord(axis) - value| < tolerance
//
//  Equivalent to the geometric predicate in the old fix_orthogonal_plane().

template <std::size_t dim>
struct PlaneSelector {
    int    axis;
    double value;
    double tolerance;

    constexpr PlaneSelector(int ax, double val, double tol = 1.0e-6)
        : axis{ax}, value{val}, tolerance{tol} {}

    template <typename NodeT>
    constexpr bool operator()(const NodeT& node) const {
        return std::abs(node.coord(axis) - value) < tolerance;
    }
};


// ─── BoxSelector ─────────────────────────────────────────────────────────────
//
//  Selects nodes inside an axis-aligned bounding box [min, max].
//  Each coordinate is checked with tolerance.

template <std::size_t dim>
struct BoxSelector {
    std::array<double, dim> min_corner;
    std::array<double, dim> max_corner;
    double tolerance;

    constexpr BoxSelector(std::array<double, dim> lo, std::array<double, dim> hi,
                          double tol = 1.0e-6)
        : min_corner{lo}, max_corner{hi}, tolerance{tol} {}

    template <typename NodeT>
    constexpr bool operator()(const NodeT& node) const {
        for (std::size_t i = 0; i < dim; ++i) {
            if (node.coord(i) < min_corner[i] - tolerance ||
                node.coord(i) > max_corner[i] + tolerance)
                return false;
        }
        return true;
    }
};


// ─── SphereSelector ──────────────────────────────────────────────────────────
//
//  Selects nodes within a sphere of given center and radius.

template <std::size_t dim>
struct SphereSelector {
    std::array<double, dim> center;
    double radius;
    double tolerance;

    constexpr SphereSelector(std::array<double, dim> c, double r, double tol = 1.0e-6)
        : center{c}, radius{r}, tolerance{tol} {}

    template <typename NodeT>
    constexpr bool operator()(const NodeT& node) const {
        double r2 = 0.0;
        for (std::size_t i = 0; i < dim; ++i) {
            double d = node.coord(i) - center[i];
            r2 += d * d;
        }
        return std::sqrt(r2) <= radius + tolerance;
    }
};


// ─── NodeIdSelector ──────────────────────────────────────────────────────────
//
//  Selects nodes by their ID from an explicit set.

struct NodeIdSelector {
    std::vector<std::size_t> ids;

    NodeIdSelector(std::initializer_list<std::size_t> id_list)
        : ids(id_list) {}

    explicit NodeIdSelector(std::vector<std::size_t> id_vec)
        : ids(std::move(id_vec)) {}

    template <typename NodeT>
    bool operator()(const NodeT& node) const {
        return std::find(ids.begin(), ids.end(), node.id()) != ids.end();
    }
};


// ═════════════════════════════════════════════════════════════════════════════
//  Combinators
// ═════════════════════════════════════════════════════════════════════════════

template <typename A, typename B>
struct SelectorAnd {
    A a;
    B b;

    template <typename NodeT>
    constexpr bool operator()(const NodeT& node) const {
        return a(node) && b(node);
    }
};

template <typename A, typename B>
struct SelectorOr {
    A a;
    B b;

    template <typename NodeT>
    constexpr bool operator()(const NodeT& node) const {
        return a(node) || b(node);
    }
};

template <typename S>
struct SelectorNot {
    S s;

    template <typename NodeT>
    constexpr bool operator()(const NodeT& node) const {
        return !s(node);
    }
};

/// Combine two selectors with AND logic.
template <typename A, typename B>
constexpr auto select_and(A a, B b) {
    return SelectorAnd<A, B>{std::move(a), std::move(b)};
}

/// Combine two selectors with OR logic.
template <typename A, typename B>
constexpr auto select_or(A a, B b) {
    return SelectorOr<A, B>{std::move(a), std::move(b)};
}

/// Negate a selector.
template <typename S>
constexpr auto select_not(S s) {
    return SelectorNot<S>{std::move(s)};
}


// ═════════════════════════════════════════════════════════════════════════════
//  Convenience Factories
// ═════════════════════════════════════════════════════════════════════════════

/// Selector for nodes on a coordinate plane.
template <std::size_t dim>
constexpr auto on_plane(int axis, double value, double tol = 1.0e-6) {
    return PlaneSelector<dim>{axis, value, tol};
}

/// Selector for nodes inside an axis-aligned box.
template <std::size_t dim>
constexpr auto in_box(std::array<double, dim> lo, std::array<double, dim> hi,
                      double tol = 1.0e-6) {
    return BoxSelector<dim>{lo, hi, tol};
}

/// Selector for nodes inside a sphere.
template <std::size_t dim>
constexpr auto in_sphere(std::array<double, dim> center, double radius,
                         double tol = 1.0e-6) {
    return SphereSelector<dim>{center, radius, tol};
}

/// Selector for an explicit set of node IDs.
inline auto node_ids(std::initializer_list<std::size_t> ids) {
    return NodeIdSelector{ids};
}

} // namespace fn

#endif // FALL_N_NODE_SELECTOR_HH
