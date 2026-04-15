#ifndef FALL_N_LINE_QUADRATURE_INTEGRATOR_HH
#define FALL_N_LINE_QUADRATURE_INTEGRATOR_HH

#include <array>
#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>

#include <Eigen/Dense>

template <typename RuleProvider, std::size_t n>
class LineQuadratureIntegrator
{
    using LocalCoordView = std::span<const double>;
    using NaturalArray   = std::array<double, 1>;
    using PointArray     = std::array<NaturalArray, n>;

    static constexpr auto nodes_   = RuleProvider::template evaluation_points<n>();
    static constexpr auto weights_ = RuleProvider::template weights<n>();

    static consteval PointArray make_points() noexcept {
        PointArray points{};
        for (std::size_t i = 0; i < n; ++i) {
            points[i][0] = nodes_[i];
        }
        return points;
    }

    static constexpr PointArray points_ = make_points();

public:
    static constexpr std::size_t num_integration_points = n;

    static constexpr auto reference_integration_point(std::size_t i) noexcept {
        const auto& point = points_[i];
        return LocalCoordView{point.data(), point.size()};
    }

    static constexpr auto weight(std::size_t i) noexcept {
        return weights_[i];
    }

    constexpr decltype(auto) operator()(std::invocable<LocalCoordView> auto&& f) const noexcept {
        using ReturnType = std::invoke_result_t<decltype(f), LocalCoordView>;

        if constexpr (std::is_base_of_v<Eigen::MatrixBase<ReturnType>, ReturnType>) {
            using MatrixType =
                Eigen::Matrix<double, ReturnType::RowsAtCompileTime, ReturnType::ColsAtCompileTime>;
            return [&]<std::size_t... I>(std::index_sequence<I...>) -> MatrixType {
                return ((f(reference_integration_point(I)).eval() * weights_[I]) + ...);
            }(std::make_index_sequence<n>{});
        } else {
            ReturnType value{};
            for (std::size_t i = 0; i < n; ++i) {
                value += weights_[i] * f(reference_integration_point(i));
            }
            return value;
        }
    }

    constexpr LineQuadratureIntegrator() noexcept = default;
    constexpr ~LineQuadratureIntegrator() noexcept = default;
};

#endif // FALL_N_LINE_QUADRATURE_INTEGRATOR_HH
