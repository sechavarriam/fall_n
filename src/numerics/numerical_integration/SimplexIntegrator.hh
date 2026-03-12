#ifndef FALL_N_SIMPLEX_INTEGRATOR_HH
#define FALL_N_SIMPLEX_INTEGRATOR_HH

#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>

#include <Eigen/Dense>

#include "SimplexQuadrature.hh"

template <std::size_t TopDim, std::size_t Order>
class SimplexIntegrator
{
    using Rule = decltype(simplex_quadrature::default_simplex_rule<TopDim, Order>());
    using LocalCoordView = std::span<const double>;

    static constexpr Rule rule_ = simplex_quadrature::default_simplex_rule<TopDim, Order>();

public:
    static constexpr std::size_t num_integration_points = Rule::num_points;

    static constexpr auto reference_integration_point(std::size_t i) noexcept {
        const auto& p = rule_.get_point_coords(i);
        return LocalCoordView{p.data(), p.size()};
    }

    static constexpr double weight(std::size_t i) noexcept {
        return rule_.get_point_weight(i);
    }

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
        } else if constexpr (std::is_base_of_v<Eigen::MatrixBase<ReturnType>, ReturnType>) {
            const auto& pt0 = rule_.get_point_coords(0);
            const LocalCoordView xv0{pt0.data(), pt0.size()};
            auto result = (f(xv0) * rule_.get_point_weight(0)).eval();
            for (std::size_t i = 1; i < num_integration_points; ++i) {
                const auto& pt = rule_.get_point_coords(i);
                const LocalCoordView xv{pt.data(), pt.size()};
                result += f(xv) * rule_.get_point_weight(i);
            }
            return result;
        } else {
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

#endif // FALL_N_SIMPLEX_INTEGRATOR_HH
