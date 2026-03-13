#ifndef FALL_N_SIMPLEX_ELEMENT_HH
#define FALL_N_SIMPLEX_ELEMENT_HH

// =============================================================================
//  SimplexElement.hh — Simplex finite element + integration strategy
// =============================================================================
//
//  SimplexElement<Dim, TopDim, Order>
//    Dim    : embedding spatial dimension (1, 2, or 3)
//    TopDim : topological dimension of the simplex (1 = segment, 2 = tri, 3 = tet)
//    Order  : polynomial order (1 = linear, 2 = quadratic)
//
//  Example instantiations:
//    SimplexElement<3, 3, 1>  — linear tetrahedron in 3D           (4 nodes)
//    SimplexElement<3, 3, 2>  — quadratic tetrahedron in 3D       (10 nodes)
//    SimplexElement<3, 2, 1>  — linear triangle in 3D (face)       (3 nodes)
//    SimplexElement<3, 2, 2>  — quadratic triangle in 3D (face)    (6 nodes)
//    SimplexElement<2, 2, 1>  — linear triangle in 2D              (3 nodes)
//    SimplexElement<2, 1, 1>  — linear segment in 2D (edge)        (2 nodes)
//
//  SimplexIntegrator<TopDim, Order>
//    Wraps a SimplexQuadratureRule and exposes the interface expected
//    by ElementGeometry's OwningModel.
//
// =============================================================================

#include <array>
#include <cstddef>
#include <iostream>
#include <memory>
#include <optional>
#include <algorithm>
#include <span>
#include <ranges>
#include <type_traits>

#include "../Node.hh"

#include "../../geometry/Topology.hh"
#include "../../geometry/SimplexCell.hh"
#include "../../geometry/Point.hh"

#include "../../utils/small_math.hh"
#include "../../numerics/numerical_integration/SimplexIntegrator.hh"


// =============================================================================
//  Concept: is_SimplexElement
// =============================================================================
template <typename T>
concept private_Simplex_check_ = requires(T t) {
    requires std::same_as<decltype(t._is_SimplexElement()), bool>;
};

template <typename T>
struct SimplexConceptTester {
    static inline constexpr bool _is_in_Simplex_Family = private_Simplex_check_<T>;
};

template <typename T>
concept is_SimplexElement = SimplexConceptTester<T>::_is_in_Simplex_Family;


// =============================================================================
//  SimplexElement<Dim, TopDim, Order>
// =============================================================================

template <std::size_t Dim, std::size_t TopDim, std::size_t Order>
    requires (topology::EmbeddableInSpace<TopDim> && TopDim <= Dim)
class SimplexElement
{
    template <typename T> friend struct SimplexConceptTester;

    static inline constexpr bool _is_SimplexElement() { return true; }

    using SpatialArray = std::array<double, Dim>;
    using NaturalArray = std::array<double, TopDim>;

public:

    using ReferenceCell = geometry::simplex::SimplexCell<TopDim, Order>;

    static inline constexpr std::size_t dim             = Dim;
    static inline constexpr std::size_t topological_dim = TopDim;
    static inline constexpr std::size_t order           = Order;
    static inline constexpr std::size_t num_nodes       = geometry::simplex::simplex_num_nodes(TopDim, Order);

    static inline constexpr ReferenceCell reference_element_{};

    using pPointArray = std::optional<std::array<const geometry::Point<dim>*, num_nodes>>;
    using pNodeArray = std::optional<std::array<Node<dim>*, num_nodes>>;

    std::size_t tag_{0};
    // Geometry-only binding used by interpolation and Jacobian evaluation.
    pPointArray points_p{};
    // Analysis-layer binding used for DoFs, constraints, and PETSc layout.
    pNodeArray  nodes_p{};

    std::array<PetscInt, num_nodes> nodes_{}; // Global node numbers in Plex

private:

    std::array<PetscInt, num_nodes> local_index_{
        []<std::size_t... I>(std::index_sequence<I...>) {
            return std::array{static_cast<PetscInt>(I)...};
        }(std::make_index_sequence<num_nodes>{})
    };

    void set_local_index(const PetscInt idxs[]) noexcept {
        for (std::size_t i = 0; i < num_nodes; ++i) local_index_[i] = idxs[i];
    }

public:

    auto id()                      const noexcept { return tag_; }
    void set_id(std::size_t id)          noexcept { tag_ = id; }

    PetscInt                  node  (std::size_t i) const noexcept { return nodes_[i]; }
    std::span<const PetscInt> nodes ()              const noexcept { return std::span<const PetscInt>(nodes_); }
    const geometry::Point<dim>& point_p(std::size_t i) const noexcept { return *points_p.value()[i]; }
    Node<dim>&          node_p(std::size_t i) const noexcept { return *nodes_p.value()[i]; }

    void bind_point(std::size_t i, const geometry::Point<dim>* point) noexcept {
        if (points_p.has_value()) {
            points_p.value()[i] = point;
        } else {
            points_p = std::array<const geometry::Point<dim>*, num_nodes>{};
            points_p.value()[i] = point;
        }
    }

    void bind_node(std::size_t i, Node<dim>* node) noexcept {
        bind_point(i, node);
        if (nodes_p.has_value()) {
            nodes_p.value()[i] = node;
        } else {
            nodes_p = std::array<Node<dim>*, num_nodes>{};
            nodes_p.value()[i] = node;
        }
    }

    // ─── Shape functions ────────────────────────────────────────────────

    constexpr inline double H(std::size_t i, const NaturalArray& X) const noexcept {
        return reference_element_.basis.shape_function(local_index_[i])(X);
    }

    constexpr inline double dH_dx(std::size_t i, std::size_t j, const NaturalArray& X) const noexcept {
        return reference_element_.basis.shape_function_derivative(local_index_[i], j)(X);
    }

    // ─── Coordinate utilities ───────────────────────────────────────────

    constexpr inline auto coord_array() const noexcept {
        using CoordArray = std::array<std::array<double, num_nodes>, dim>;
        CoordArray coords{};
        for (std::size_t i = 0; i < dim; ++i) {
            for (std::size_t j = 0; j < num_nodes; ++j) {
                coords[i][j] = points_p.value()[j]->coord(i);
            }
        }
        return coords;
    }

    /// Map from reference simplex to physical space: x = ∑ N_i(ξ) · x_i
    constexpr inline SpatialArray map_local_point(const NaturalArray& xi) const noexcept {
        SpatialArray X{};
        auto F = coord_array();
        for (std::size_t i = 0; i < dim; ++i) {
            X[i] = reference_element_.basis.interpolate(F[i], xi);
        }
        return X;
    }

    /// Evaluate the Jacobian matrix J_{ij} = ∂x_i / ∂ξ_j  (dim × topological_dim)
    constexpr inline auto evaluate_jacobian(const NaturalArray& X) const noexcept {
        Eigen::Matrix<double, dim, topological_dim> J =
            Eigen::Matrix<double, dim, topological_dim>::Zero();

        SpatialArray x{};
        for (std::size_t k = 0; k < num_nodes; ++k) {
            x = points_p.value()[k]->coord();
            for (std::size_t i = 0; i < dim; ++i) {
                for (std::size_t j = 0; j < topological_dim; ++j) {
                    J(i, j) += x[i] * dH_dx(k, j, X);
                }
            }
        }
        return J;
    }

    constexpr inline auto detJ(const NaturalArray& X) const noexcept
        requires (topological_dim == dim)
    {
        return evaluate_jacobian(X).determinant();
    }

    // ─── Constructors ───────────────────────────────────────────────────

    constexpr SimplexElement() = default;

    constexpr SimplexElement(pNodeArray nodes)
        : nodes_p{std::forward<pNodeArray>(nodes)}
    {
        if (nodes_p.has_value()) {
            points_p = std::array<const geometry::Point<dim>*, num_nodes>{};
            for (std::size_t i = 0; i < num_nodes; ++i) {
                points_p.value()[i] = nodes_p.value()[i];
            }
        }
    }

    constexpr SimplexElement(std::size_t tag, std::ranges::input_range auto&& node_ids)
        requires (std::same_as<std::ranges::range_value_t<decltype(node_ids)>, PetscInt>)
        : tag_{tag}
    {
        std::ranges::copy(node_ids, nodes_.begin());
    }

    constexpr SimplexElement(std::size_t tag, std::ranges::input_range auto&& node_ids,
                             std::ranges::input_range auto&& local_ordering)
        requires (std::same_as<std::ranges::range_value_t<decltype(node_ids)>, PetscInt>)
        : tag_{tag}
    {
        std::ranges::copy(node_ids, nodes_.begin());
        std::ranges::copy(local_ordering, local_index_.begin());
    }

    // Copy / move
    constexpr SimplexElement(const SimplexElement&)            = default;
    constexpr SimplexElement(SimplexElement&&)                 = default;
    constexpr SimplexElement& operator=(const SimplexElement&) = default;
    constexpr SimplexElement& operator=(SimplexElement&&)      = default;
    constexpr ~SimplexElement()                                = default;


    // ─── Debug printing ─────────────────────────────────────────────────

    void print_info() const noexcept {
        std::cout << "SimplexElement Tag    : " << tag_ << std::endl;
        std::cout << "Dim/TopDim/Order      : " << dim << "/" << topological_dim << "/" << Order << std::endl;
        std::cout << "Number of Nodes       : " << num_nodes << std::endl;
        std::cout << "Nodes ID              : ";
        for (std::size_t i = 0; i < num_nodes; ++i)
            std::cout << nodes_[i] << " ";
        std::cout << std::endl;
    }
};


// Convenience aliases
template <std::size_t Order> using SimplexElement1D = SimplexElement<1, 1, Order>;
template <std::size_t Order> using SimplexElement2D = SimplexElement<2, 2, Order>;
template <std::size_t Order> using SimplexElement3D = SimplexElement<3, 3, Order>;


#endif // FALL_N_SIMPLEX_ELEMENT_HH
