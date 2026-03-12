#ifndef FALL_N_GAUSS_LEGENDRE_CELL_INTEGRATOR_HH
#define FALL_N_GAUSS_LEGENDRE_CELL_INTEGRATOR_HH

#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>

#include <Eigen/Dense>

#include "CellQuadrature.hh"

template <std::size_t... N>
class GaussLegendreCellIntegrator
{
    using CellQuadrature = GaussLegendre::CellQuadrature<N...>;
    using NaturalArray   = std::array<double, sizeof...(N)>;
    using LocalCoordView = std::span<const double>;

    static constexpr CellQuadrature integrator_{};

public:
    static constexpr std::size_t num_integration_points = CellQuadrature::num_points;

    static constexpr auto reference_integration_point(std::size_t i) noexcept {
        const auto& p = integrator_.get_point_coords(i);
        return LocalCoordView{p.data(), p.size()};
    }

    static constexpr auto weight(std::size_t i) noexcept {
        return integrator_.get_point_weight(i);
    }

    constexpr decltype(auto) operator()(std::invocable<LocalCoordView> auto&& f) const noexcept {
        using ReturnType = std::invoke_result_t<decltype(f), LocalCoordView>;

        if constexpr (std::is_base_of_v<Eigen::MatrixBase<ReturnType>, ReturnType>) {
            return integrator_([&](const NaturalArray& x) -> ReturnType {
                const LocalCoordView xv{x.data(), x.size()};
                return f(xv).eval();
            });
        } else {
            return integrator_([&](const NaturalArray& x) {
                const LocalCoordView xv{x.data(), x.size()};
                return f(xv);
            });
        }
    }

    constexpr GaussLegendreCellIntegrator() noexcept = default;
    constexpr ~GaussLegendreCellIntegrator() noexcept = default;
};

#endif // FALL_N_GAUSS_LEGENDRE_CELL_INTEGRATOR_HH
