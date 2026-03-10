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
#include <memory>
#include <optional>
#include <span>
#include <ranges>
#include <type_traits>

#include "../Node.hh"

#include "../../geometry/Topology.hh"
#include "../../geometry/SimplexCell.hh"
#include "../../geometry/Point.hh"

#include "../../utils/small_math.hh"
#include "../../numerics/numerical_integration/SimplexQuadrature.hh"

#include "../../numerics/linear_algebra/LinalgOperations.hh"


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

    using pNodeArray = std::optional<std::array<Node<dim>*, num_nodes>>;

    std::size_t tag_;
    pNodeArray  nodes_p;

    std::array<PetscInt, num_nodes> nodes_; // Global node numbers in Plex

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

    PetscInt            node  (std::size_t i) const noexcept { return nodes_[i]; }
    std::span<PetscInt> nodes ()              const noexcept { return std::span<PetscInt>(nodes_); }
    Node<dim>&          node_p(std::size_t i) const noexcept { return *nodes_p.value()[i]; }

    void bind_node(std::size_t i, Node<dim>* node) noexcept {
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
                coords[i][j] = nodes_p.value()[j]->coord(i);
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
            x = nodes_p.value()[k]->coord();
            for (std::size_t i = 0; i < dim; ++i) {
                for (std::size_t j = 0; j < topological_dim; ++j) {
                    J(i, j) += x[i] * dH_dx(k, j, X);
                }
            }
        }
        return J;
    }

    constexpr inline auto detJ(const NaturalArray& X) const noexcept {
        return evaluate_jacobian(X).determinant();
    }

    // ─── Constructors ───────────────────────────────────────────────────

    constexpr SimplexElement() = default;

    constexpr SimplexElement(pNodeArray nodes)
        : nodes_p{std::forward<pNodeArray>(nodes)} {}

    constexpr SimplexElement(std::size_t& tag, const std::ranges::range auto& node_ids)
        requires (std::same_as<std::ranges::range_value_t<decltype(node_ids)>, PetscInt>)
        : tag_{tag}
    {
        std::copy(node_ids.begin(), node_ids.end(), nodes_.begin());
    }

    constexpr SimplexElement(std::size_t&& tag, std::ranges::range auto&& node_ids)
        requires (std::same_as<std::ranges::range_value_t<decltype(node_ids)>, PetscInt>)
        : tag_{tag}
    {
        std::move(node_ids.begin(), node_ids.end(), nodes_.begin());
    }

    constexpr SimplexElement(std::size_t&& tag, std::ranges::range auto&& node_ids,
                             std::ranges::range auto&& local_ordering)
        requires (std::same_as<std::ranges::range_value_t<decltype(node_ids)>, PetscInt>)
        : tag_{tag}
    {
        std::move(node_ids.begin(), node_ids.end(), nodes_.begin());
        set_local_index(local_ordering.data());
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


// =============================================================================
//  SimplexIntegrator<TopDim, Order>
// =============================================================================
//
//  Wraps the appropriate simplex quadrature rule and exposes:
//    - num_integration_points
//    - reference_integration_point(i)  → span<const double>
//    - weight(i)                       → double
//    - operator()(f)                   → ∑ w_i · f(ξ_i)
//

template <std::size_t TopDim, std::size_t Order>
class SimplexIntegrator
{
    using Rule = decltype(simplex_quadrature::default_simplex_rule<TopDim, Order>());

    static constexpr Rule rule_ = simplex_quadrature::default_simplex_rule<TopDim, Order>();

    using NaturalArray   = std::array<double, TopDim>;
    using LocalCoordView = std::span<const double>;

public:

    static constexpr std::size_t num_integration_points = Rule::num_points;

    static constexpr auto reference_integration_point(std::size_t i) noexcept {
        const auto& p = rule_.get_point_coords(i);
        return LocalCoordView{p.data(), p.size()};
    }

    static constexpr double weight(std::size_t i) noexcept {
        return rule_.get_point_weight(i);
    }

    /// Pure quadrature: ∑ w_i · f(ξ_i)
    /// Does NOT multiply by |J| — the element applies its own differential measure.
    constexpr decltype(auto) operator()(std::invocable<LocalCoordView> auto&& f) const noexcept {
        using ReturnType = std::invoke_result_t<decltype(f), LocalCoordView>;

        if constexpr (std::is_arithmetic_v<std::decay_t<ReturnType>>) {
            double result = 0.0;
            for (std::size_t i = 0; i < num_integration_points; ++i) {
                const auto& pt = rule_.get_point_coords(i);
                const LocalCoordView xv{pt.data(), pt.size()};
                result += rule_.get_point_weight(i) * f(xv);
            }
            return result;
        }
        else if constexpr (std::is_base_of_v<
                               Eigen::MatrixBase<ReturnType>, ReturnType>) {
            // Eigen matrices/vectors — return an evaluated copy (no aliasing)
            const auto& pt0 = rule_.get_point_coords(0);
            const LocalCoordView xv0{pt0.data(), pt0.size()};
            auto result = (f(xv0) * rule_.get_point_weight(0)).eval();
            for (std::size_t i = 1; i < num_integration_points; ++i) {
                const auto& pt = rule_.get_point_coords(i);
                const LocalCoordView xv{pt.data(), pt.size()};
                result += f(xv) * rule_.get_point_weight(i);
            }
            return result;
        }
        else {
            const auto& pt0 = rule_.get_point_coords(0);
            const LocalCoordView xv0{pt0.data(), pt0.size()};
            auto result = f(xv0) * rule_.get_point_weight(0);
            for (std::size_t i = 1; i < num_integration_points; ++i) {
                const auto& pt = rule_.get_point_coords(i);
                const LocalCoordView xv{pt.data(), pt.size()};
                result += f(xv) * rule_.get_point_weight(i);
            }
            return result;
        }
    }

    constexpr SimplexIntegrator() noexcept = default;
    constexpr ~SimplexIntegrator() noexcept = default;
};


// Convenience aliases
template <std::size_t Order> using SimplexElement1D = SimplexElement<1, 1, Order>;
template <std::size_t Order> using SimplexElement2D = SimplexElement<2, 2, Order>;
template <std::size_t Order> using SimplexElement3D = SimplexElement<3, 3, Order>;


#endif // FALL_N_SIMPLEX_ELEMENT_HH
