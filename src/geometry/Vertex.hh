#ifndef FALL_N_VERTEX_HH
#define FALL_N_VERTEX_HH

// ============================================================================
//  Vertex<Dim>  —  mesh vertex = geometry + topological identity
// ============================================================================
//
//  A Vertex is intentionally smaller and conceptually cleaner than Node:
//
//    Point   -> coordinates only
//    Vertex  -> Point + stable mesh/topology identifier
//    Node    -> analysis-layer decorator carrying DoFs and constraints
//
//  The purpose of introducing Vertex is architectural: Domain should be able
//  to own pure geometric/topological entities without importing analysis
//  concerns such as nodal DoF storage or boundary-condition state.
//
//  This header does not yet replace Node in Domain.  It defines the target
//  abstraction for the Domain refactor and lets the rest of the codebase refer
//  to a geometry-only mesh vertex type.
//
// ============================================================================

#include <concepts>
#include <cstddef>

#include "Point.hh"
#include "Topology.hh"

namespace geometry {

template <std::size_t Dim>
    requires topology::EmbeddableInSpace<Dim>
class Vertex : public Point<Dim> {
    std::size_t id_{};

public:
    static constexpr std::size_t dim = Dim;

    [[nodiscard]] constexpr std::size_t id() const noexcept { return id_; }
    constexpr void set_id(std::size_t id) noexcept { id_ = id; }

    constexpr Vertex() = default;

    constexpr Vertex(std::size_t tag, const std::array<double, Dim>& coords) noexcept
        : Point<Dim>(coords),
          id_{tag} {}

    template <std::floating_point... Args>
        requires (sizeof...(Args) == Dim)
    constexpr Vertex(std::integral auto tag, Args... args) noexcept
        : Point<Dim>(args...),
          id_{static_cast<std::size_t>(tag)} {}

    template <std::floating_point... Args>
        requires (sizeof...(Args) == Dim)
    constexpr Vertex(std::size_t tag, Args... args) noexcept
        : Point<Dim>(args...),
          id_{tag} {}
};

template <typename T>
concept VertexT = PointT<T> && requires(T v) {
    { v.id() } -> std::convertible_to<std::size_t>;
    { v.set_id(std::size_t{}) } -> std::same_as<void>;
};

static_assert(VertexT<Vertex<1>>);
static_assert(VertexT<Vertex<2>>);
static_assert(VertexT<Vertex<3>>);

} // namespace geometry

using geometry::VertexT;

#endif // FALL_N_VERTEX_HH
