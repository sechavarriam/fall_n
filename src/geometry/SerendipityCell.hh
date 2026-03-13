#ifndef FALL_N_SERENDIPITY_CELL_HH
#define FALL_N_SERENDIPITY_CELL_HH

#include <array>
#include <cstddef>

#include "Point.hh"
#include "Topology.hh"

namespace geometry::cell {

consteval std::size_t serendipity_num_nodes(std::size_t top_dim,
                                            std::size_t order) {
    if (order == 1) {
        return std::size_t{1} << top_dim;
    }
    if (order == 2) {
        return (std::size_t{1} << top_dim) + top_dim * (std::size_t{1} << (top_dim - 1));
    }
    return 0;
}

template <std::size_t TopDim, std::size_t Order>
    requires (topology::EmbeddableInSpace<TopDim> && (Order == 1 || Order == 2))
consteval auto serendipity_reference_nodes() {
    constexpr std::size_t num_nodes = serendipity_num_nodes(TopDim, Order);
    std::array<Point<TopDim>, num_nodes> nodes{};

    if constexpr (TopDim == 1) {
        nodes[0] = Point<1>(std::array<double, 1>{-1.0});
        if constexpr (Order == 1) {
            nodes[1] = Point<1>(std::array<double, 1>{1.0});
        } else {
            nodes[1] = Point<1>(std::array<double, 1>{0.0});
            nodes[2] = Point<1>(std::array<double, 1>{1.0});
        }
    } else if constexpr (TopDim == 2) {
        nodes[0] = Point<2>(std::array<double, 2>{-1.0, -1.0});
        nodes[1] = Point<2>(std::array<double, 2>{ 1.0, -1.0});
        nodes[2] = Point<2>(std::array<double, 2>{ 1.0,  1.0});
        nodes[3] = Point<2>(std::array<double, 2>{-1.0,  1.0});
        if constexpr (Order == 2) {
            nodes[4] = Point<2>(std::array<double, 2>{ 0.0, -1.0});
            nodes[5] = Point<2>(std::array<double, 2>{ 1.0,  0.0});
            nodes[6] = Point<2>(std::array<double, 2>{ 0.0,  1.0});
            nodes[7] = Point<2>(std::array<double, 2>{-1.0,  0.0});
        }
    } else if constexpr (TopDim == 3) {
        nodes[0] = Point<3>(std::array<double, 3>{-1.0, -1.0, -1.0});
        nodes[1] = Point<3>(std::array<double, 3>{ 1.0, -1.0, -1.0});
        nodes[2] = Point<3>(std::array<double, 3>{ 1.0,  1.0, -1.0});
        nodes[3] = Point<3>(std::array<double, 3>{-1.0,  1.0, -1.0});
        nodes[4] = Point<3>(std::array<double, 3>{-1.0, -1.0,  1.0});
        nodes[5] = Point<3>(std::array<double, 3>{ 1.0, -1.0,  1.0});
        nodes[6] = Point<3>(std::array<double, 3>{ 1.0,  1.0,  1.0});
        nodes[7] = Point<3>(std::array<double, 3>{-1.0,  1.0,  1.0});
        if constexpr (Order == 2) {
            nodes[ 8] = Point<3>(std::array<double, 3>{ 0.0, -1.0, -1.0});
            nodes[ 9] = Point<3>(std::array<double, 3>{ 1.0,  0.0, -1.0});
            nodes[10] = Point<3>(std::array<double, 3>{ 0.0,  1.0, -1.0});
            nodes[11] = Point<3>(std::array<double, 3>{-1.0,  0.0, -1.0});
            nodes[12] = Point<3>(std::array<double, 3>{ 0.0, -1.0,  1.0});
            nodes[13] = Point<3>(std::array<double, 3>{ 1.0,  0.0,  1.0});
            nodes[14] = Point<3>(std::array<double, 3>{ 0.0,  1.0,  1.0});
            nodes[15] = Point<3>(std::array<double, 3>{-1.0,  0.0,  1.0});
            nodes[16] = Point<3>(std::array<double, 3>{-1.0, -1.0,  0.0});
            nodes[17] = Point<3>(std::array<double, 3>{ 1.0, -1.0,  0.0});
            nodes[18] = Point<3>(std::array<double, 3>{ 1.0,  1.0,  0.0});
            nodes[19] = Point<3>(std::array<double, 3>{-1.0,  1.0,  0.0});
        }
    }

    return nodes;
}

template <std::size_t TopDim, std::size_t Order>
    requires (topology::EmbeddableInSpace<TopDim> && (Order == 1 || Order == 2))
class SerendipityBasis {
public:
    static constexpr std::size_t dim       = TopDim;
    static constexpr std::size_t order     = Order;
    static constexpr std::size_t num_nodes = serendipity_num_nodes(TopDim, Order);

    using NaturalArray = std::array<double, TopDim>;

private:
    static constexpr auto reference_nodes_ = serendipity_reference_nodes<TopDim, Order>();

    static constexpr double inv_pow2(std::size_t exp) noexcept {
        double value = 1.0;
        for (std::size_t i = 0; i < exp; ++i) {
            value *= 0.5;
        }
        return value;
    }

    static constexpr std::size_t zero_axis(const NaturalArray& r) noexcept {
        for (std::size_t j = 0; j < TopDim; ++j) {
            if (r[j] == 0.0) {
                return j;
            }
        }
        return TopDim;
    }

    static constexpr double multilinear_shape(const NaturalArray& r,
                                              const NaturalArray& x) noexcept {
        double value = inv_pow2(TopDim);
        for (std::size_t j = 0; j < TopDim; ++j) {
            value *= (1.0 + r[j] * x[j]);
        }
        return value;
    }

    static constexpr double multilinear_derivative(const NaturalArray& r,
                                                   std::size_t deriv_dir,
                                                   const NaturalArray& x) noexcept {
        double value = inv_pow2(TopDim) * r[deriv_dir];
        for (std::size_t j = 0; j < TopDim; ++j) {
            if (j == deriv_dir) {
                continue;
            }
            value *= (1.0 + r[j] * x[j]);
        }
        return value;
    }

    static constexpr double corner_shape(const NaturalArray& r,
                                         const NaturalArray& x) noexcept {
        double prod = 1.0;
        double sum  = 0.0;
        for (std::size_t j = 0; j < TopDim; ++j) {
            prod *= (1.0 + r[j] * x[j]);
            sum  += r[j] * x[j];
        }
        return inv_pow2(TopDim) * prod * (1.0 + sum - static_cast<double>(TopDim));
    }

    static constexpr double corner_derivative(const NaturalArray& r,
                                              std::size_t deriv_dir,
                                              const NaturalArray& x) noexcept {
        double prod = 1.0;
        double prod_except = 1.0;
        double sum = 0.0;
        for (std::size_t j = 0; j < TopDim; ++j) {
            const double factor = 1.0 + r[j] * x[j];
            prod *= factor;
            if (j != deriv_dir) {
                prod_except *= factor;
            }
            sum += r[j] * x[j];
        }
        const double signed_dir = r[deriv_dir];
        const double tail = 1.0 + sum - static_cast<double>(TopDim);
        return inv_pow2(TopDim) * signed_dir * (prod_except * tail + prod);
    }

    static constexpr double edge_shape(const NaturalArray& r,
                                       std::size_t axis,
                                       const NaturalArray& x) noexcept {
        double value = inv_pow2(TopDim - 1) * (1.0 - x[axis] * x[axis]);
        for (std::size_t j = 0; j < TopDim; ++j) {
            if (j == axis) {
                continue;
            }
            value *= (1.0 + r[j] * x[j]);
        }
        return value;
    }

    static constexpr double edge_derivative(const NaturalArray& r,
                                            std::size_t axis,
                                            std::size_t deriv_dir,
                                            const NaturalArray& x) noexcept {
        const double factor = inv_pow2(TopDim - 1);
        double value = 1.0;

        if (deriv_dir == axis) {
            value = -2.0 * x[axis];
            for (std::size_t j = 0; j < TopDim; ++j) {
                if (j == axis) {
                    continue;
                }
                value *= (1.0 + r[j] * x[j]);
            }
            return factor * value;
        }

        value = (1.0 - x[axis] * x[axis]) * r[deriv_dir];
        for (std::size_t j = 0; j < TopDim; ++j) {
            if (j == axis || j == deriv_dir) {
                continue;
            }
            value *= (1.0 + r[j] * x[j]);
        }
        return factor * value;
    }

public:
    constexpr double shape(std::size_t i, const NaturalArray& x) const noexcept {
        const auto r = reference_nodes_[i].coord();

        if constexpr (Order == 1) {
            return multilinear_shape(r, x);
        } else {
            const auto axis = zero_axis(r);
            if (axis == TopDim) {
                return corner_shape(r, x);
            }
            return edge_shape(r, axis, x);
        }
    }

    constexpr double shape_derivative(std::size_t i, std::size_t deriv_dir,
                                      const NaturalArray& x) const noexcept {
        const auto r = reference_nodes_[i].coord();

        if constexpr (Order == 1) {
            return multilinear_derivative(r, deriv_dir, x);
        } else {
            const auto axis = zero_axis(r);
            if (axis == TopDim) {
                return corner_derivative(r, deriv_dir, x);
            }
            return edge_derivative(r, axis, deriv_dir, x);
        }
    }

    constexpr auto shape_function(std::size_t i) const noexcept {
        return [this, i](const NaturalArray& x) noexcept {
            return shape(i, x);
        };
    }

    constexpr auto shape_function_derivative(std::size_t i,
                                             std::size_t deriv_dir) const noexcept {
        return [this, i, deriv_dir](const NaturalArray& x) noexcept {
            return shape_derivative(i, deriv_dir, x);
        };
    }

    constexpr double interpolate(const auto& F,
                                 const NaturalArray& x) const noexcept {
        double value = 0.0;
        for (std::size_t i = 0; i < num_nodes; ++i) {
            value += F[i] * shape(i, x);
        }
        return value;
    }

    constexpr SerendipityBasis() = default;
};

template <std::size_t TopDim, std::size_t Order>
    requires (topology::EmbeddableInSpace<TopDim> && (Order == 1 || Order == 2))
class SerendipityCell {
public:
    static constexpr std::size_t dim       = TopDim;
    static constexpr std::size_t order     = Order;
    static constexpr std::size_t num_nodes = serendipity_num_nodes(TopDim, Order);
    static constexpr std::size_t num_faces = 2 * TopDim;

    static constexpr auto reference_nodes = serendipity_reference_nodes<TopDim, Order>();
    static constexpr SerendipityBasis<TopDim, Order> basis{};

    struct FaceNodeIndices {
        std::array<std::size_t, num_nodes> indices{};
        std::size_t size = 0;
    };

    static constexpr std::size_t face_num_nodes(std::size_t) {
        if constexpr (TopDim == 1) {
            return 1;
        } else if constexpr (TopDim == 2) {
            return (Order == 1) ? 2 : 3;
        } else {
            return (Order == 1) ? 4 : 8;
        }
    }

    static constexpr FaceNodeIndices face_node_indices(std::size_t f) {
        FaceNodeIndices result{};

        if constexpr (TopDim == 1) {
            result.size = 1;
            result.indices[0] = (f == 0) ? 0 : ((Order == 1) ? 1 : 2);
            return result;
        }

        if constexpr (TopDim == 2) {
            if constexpr (Order == 1) {
                constexpr std::array<std::array<std::size_t, 2>, 4> faces{{
                    {0, 1}, {1, 2}, {2, 3}, {3, 0}
                }};
                result.size = 2;
                for (std::size_t i = 0; i < 2; ++i) {
                    result.indices[i] = faces[f][i];
                }
            } else {
                constexpr std::array<std::array<std::size_t, 3>, 4> faces{{
                    {0, 4, 1},
                    {1, 5, 2},
                    {2, 6, 3},
                    {3, 7, 0}
                }};
                result.size = 3;
                for (std::size_t i = 0; i < 3; ++i) {
                    result.indices[i] = faces[f][i];
                }
            }
            return result;
        }

        if constexpr (Order == 1) {
            constexpr std::array<std::array<std::size_t, 4>, 6> faces{{
                {0, 1, 2, 3},
                {4, 5, 6, 7},
                {0, 1, 5, 4},
                {1, 2, 6, 5},
                {3, 2, 6, 7},
                {0, 3, 7, 4}
            }};
            result.size = 4;
            for (std::size_t i = 0; i < 4; ++i) {
                result.indices[i] = faces[f][i];
            }
            return result;
        }

        constexpr std::array<std::array<std::size_t, 8>, 6> faces{{
            {0, 1, 2, 3, 8,  9, 10, 11},
            {4, 5, 6, 7, 12, 13, 14, 15},
            {0, 1, 5, 4, 8, 17, 12, 16},
            {1, 2, 6, 5, 9, 18, 13, 17},
            {3, 2, 6, 7, 10, 18, 14, 19},
            {0, 3, 7, 4, 11, 19, 15, 16}
        }};

        result.size = 8;
        for (std::size_t i = 0; i < 8; ++i) {
            result.indices[i] = faces[f][i];
        }
        return result;
    }
};

} // namespace geometry::cell

#endif // FALL_N_SERENDIPITY_CELL_HH
