#ifndef FALL_N_QUADRATURE_STRATEGY_HH
#define FALL_N_QUADRATURE_STRATEGY_HH

#include <concepts>
#include <cstddef>
#include <span>

template <typename Q>
concept QuadratureStrategy = requires(std::size_t i) {
    { Q::num_integration_points } -> std::convertible_to<std::size_t>;
    { Q::reference_integration_point(i) } -> std::convertible_to<std::span<const double>>;
    { Q::weight(i) } -> std::convertible_to<double>;
};

#endif // FALL_N_QUADRATURE_STRATEGY_HH
