#ifndef FALL_N_SERENDIPITY_ELEMENT_HH
#define FALL_N_SERENDIPITY_ELEMENT_HH

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <iostream>
#include <optional>
#include <ranges>
#include <span>
#include <type_traits>

#include "../Node.hh"

#include "../../geometry/SerendipityCell.hh"

template <typename T>
concept private_Serendipity_check_ = requires(T t) {
    requires std::same_as<decltype(t._is_SerendipityElement()), bool>;
};

template <typename T>
struct SerendipityConceptTester {
    static inline constexpr bool _is_in_Serendipity_Family = private_Serendipity_check_<T>;
};

template <typename T>
concept is_SerendipityElement = SerendipityConceptTester<T>::_is_in_Serendipity_Family;

template <std::size_t Dim, std::size_t TopDim, std::size_t Order>
    requires (topology::EmbeddableInSpace<TopDim> && TopDim <= Dim &&
              (Order == 1 || Order == 2))
class SerendipityElement {
    template <typename T> friend struct SerendipityConceptTester;

    static inline constexpr bool _is_SerendipityElement() { return true; }

    using SpatialArray = std::array<double, Dim>;
    using NaturalArray = std::array<double, TopDim>;

public:
    using ReferenceCell = geometry::cell::SerendipityCell<TopDim, Order>;

    static inline constexpr std::size_t dim             = Dim;
    static inline constexpr std::size_t topological_dim = TopDim;
    static inline constexpr std::size_t order           = Order;
    static inline constexpr std::size_t num_nodes       = ReferenceCell::num_nodes;

    static inline constexpr ReferenceCell reference_element_{};

    using PointArray    = std::array<const geometry::Point<dim>*, num_nodes>;
    using CoordPtrArray = std::array<const double*, num_nodes>;
    using NodeArray     = std::array<Node<dim>*, num_nodes>;
    using pPointArray   = std::optional<PointArray>;
    using pNodeArray    = std::optional<NodeArray>;

    std::size_t tag_{0};
    // Geometry-only binding used by interpolation and Jacobian evaluation.
    PointArray points_{};
    // Cache raw coordinate pointers to avoid repeated coord()/coord_ref() calls
    // inside map_local_point() and evaluate_jacobian().
    CoordPtrArray coord_data_{};
    // Analysis-layer binding used for DoFs, constraints, and PETSc layout.
    NodeArray nodes_p_{};
    bool has_nodes_{false};
    std::array<PetscInt, num_nodes> nodes_{};

private:
    std::array<PetscInt, num_nodes> local_index_{
        []<std::size_t... I>(std::index_sequence<I...>) {
            return std::array{static_cast<PetscInt>(I)...};
        }(std::make_index_sequence<num_nodes>{})
    };

public:
    auto id() const noexcept { return tag_; }
    void set_id(std::size_t id) noexcept { tag_ = id; }

    PetscInt node(std::size_t i) const noexcept { return nodes_[i]; }
    std::span<const PetscInt> nodes() const noexcept { return std::span<const PetscInt>(nodes_); }
    const geometry::Point<dim>& point_p(std::size_t i) const noexcept { return *points_[i]; }
    Node<dim>& node_p(std::size_t i) const noexcept { return *nodes_p_[i]; }

    void bind_point(std::size_t i, const geometry::Point<dim>* point) noexcept {
        points_[i] = point;
        coord_data_[i] = point->data();
    }

    void bind_node(std::size_t i, Node<dim>* node) noexcept {
        bind_point(i, node);
        nodes_p_[i] = node;
        has_nodes_ = true;
    }

    constexpr double H(std::size_t i, const NaturalArray& X) const noexcept {
        return reference_element_.basis.shape(local_index_[i], X);
    }

    constexpr double dH_dx(std::size_t i, std::size_t j,
                           const NaturalArray& X) const noexcept {
        return reference_element_.basis.shape_derivative(local_index_[i], j, X);
    }

    constexpr auto coord_array() const noexcept {
        using CoordArray = std::array<std::array<double, num_nodes>, dim>;
        CoordArray coords{};
        for (std::size_t i = 0; i < dim; ++i) {
            for (std::size_t j = 0; j < num_nodes; ++j) {
                coords[i][j] = coord_data_[j][i];
            }
        }
        return coords;
    }

    constexpr SpatialArray map_local_point(const NaturalArray& xi) const noexcept {
        SpatialArray X{};
        auto coords = coord_array();
        for (std::size_t i = 0; i < dim; ++i) {
            X[i] = reference_element_.basis.interpolate(coords[i], xi);
        }
        return X;
    }

    constexpr auto evaluate_jacobian(const NaturalArray& X) const noexcept {
        Eigen::Matrix<double, dim, topological_dim> J =
            Eigen::Matrix<double, dim, topological_dim>::Zero();

        for (std::size_t k = 0; k < num_nodes; ++k) {
            const auto* x = coord_data_[k];
            for (std::size_t i = 0; i < dim; ++i) {
                for (std::size_t j = 0; j < topological_dim; ++j) {
                    J(i, j) += x[i] * dH_dx(k, j, X);
                }
            }
        }
        return J;
    }

    constexpr auto detJ(const NaturalArray& X) const noexcept requires (topological_dim == dim)
    {
        return evaluate_jacobian(X).determinant();
    }

    constexpr SerendipityElement() = default;

    constexpr SerendipityElement(pPointArray points) {
        if (points.has_value()) {
            for (std::size_t i = 0; i < num_nodes; ++i) {
                bind_point(i, points.value()[i]);
            }
        }
    }

    constexpr SerendipityElement(pNodeArray nodes) {
        if (nodes.has_value()) {
            for (std::size_t i = 0; i < num_nodes; ++i) {
                bind_node(i, nodes.value()[i]);
            }
        }
    }

    constexpr SerendipityElement(std::size_t tag, std::ranges::input_range auto&& node_ids)
        requires (std::same_as<std::ranges::range_value_t<decltype(node_ids)>, PetscInt>)
        : tag_{tag}
    {
        std::ranges::copy(node_ids, nodes_.begin());
    }

    constexpr SerendipityElement(std::size_t tag, std::ranges::input_range auto&& node_ids,
                                 std::ranges::input_range auto&& local_ordering)
        requires (std::same_as<std::ranges::range_value_t<decltype(node_ids)>, PetscInt>)
        : tag_{tag}
    {
        std::ranges::copy(node_ids, nodes_.begin());
        std::ranges::copy(local_ordering, local_index_.begin());
    }

    constexpr SerendipityElement(const SerendipityElement&) = default;
    constexpr SerendipityElement(SerendipityElement&&) = default;
    constexpr SerendipityElement& operator=(const SerendipityElement&) = default;
    constexpr SerendipityElement& operator=(SerendipityElement&&) = default;
    constexpr ~SerendipityElement() = default;

    void print_info() const noexcept {
        std::cout << "SerendipityElement Tag : " << tag_ << '\n';
        std::cout << "Dim/TopDim/Order       : "
                  << dim << "/" << topological_dim << "/" << order << '\n';
        std::cout << "Number of Nodes        : " << num_nodes << '\n';
        std::cout << "Nodes ID               : ";
        for (std::size_t i = 0; i < num_nodes; ++i) {
            std::cout << nodes_[i] << ' ';
        }
        std::cout << std::endl;
    }
};

template <std::size_t Order>
using SerendipityElement1D = SerendipityElement<1, 1, Order>;

template <std::size_t Order>
using SerendipityElement2D = SerendipityElement<2, 2, Order>;

template <std::size_t Order>
using SerendipityElement3D = SerendipityElement<3, 3, Order>;

#endif // FALL_N_SERENDIPITY_ELEMENT_HH
