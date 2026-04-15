#ifndef FALL_N_BEAM_AXIS_QUADRATURE_HH
#define FALL_N_BEAM_AXIS_QUADRATURE_HH

#include <cstddef>

#include "GaussLegendreCellIntegrator.hh"
#include "GaussLobattoCellIntegrator.hh"
#include "GaussRadauCellIntegrator.hh"

enum class BeamAxisQuadratureFamily {
    GaussLegendre,
    GaussLobatto,
    GaussRadauLeft,
    GaussRadauRight
};

template <BeamAxisQuadratureFamily Family, std::size_t n>
struct BeamAxisQuadratureSelector;

template <std::size_t n>
struct BeamAxisQuadratureSelector<BeamAxisQuadratureFamily::GaussLegendre, n> {
    using type = GaussLegendreCellIntegrator<n>;
};

template <std::size_t n>
struct BeamAxisQuadratureSelector<BeamAxisQuadratureFamily::GaussLobatto, n> {
    using type = GaussLobattoCellIntegrator<n>;
};

template <std::size_t n>
struct BeamAxisQuadratureSelector<BeamAxisQuadratureFamily::GaussRadauLeft, n> {
    using type = GaussRadauCellIntegrator<n, GaussRadau::Endpoint::Left>;
};

template <std::size_t n>
struct BeamAxisQuadratureSelector<BeamAxisQuadratureFamily::GaussRadauRight, n> {
    using type = GaussRadauCellIntegrator<n, GaussRadau::Endpoint::Right>;
};

template <BeamAxisQuadratureFamily Family, std::size_t n>
using BeamAxisQuadratureT = typename BeamAxisQuadratureSelector<Family, n>::type;

template <BeamAxisQuadratureFamily Family, std::size_t n>
struct BeamAxisQuadratureTraits {
    static constexpr std::size_t num_points = n;
    static constexpr bool includes_left_endpoint =
        (Family == BeamAxisQuadratureFamily::GaussLobatto ||
         Family == BeamAxisQuadratureFamily::GaussRadauLeft) &&
        (n >= 2);
    static constexpr bool includes_right_endpoint =
        (Family == BeamAxisQuadratureFamily::GaussLobatto ||
         Family == BeamAxisQuadratureFamily::GaussRadauRight) &&
        (n >= 2);
    static constexpr int polynomial_exactness_degree = []() constexpr {
        if constexpr (n == 1) {
            return 1;
        } else if constexpr (Family == BeamAxisQuadratureFamily::GaussLegendre) {
            return static_cast<int>(2 * n - 1);
        } else if constexpr (Family == BeamAxisQuadratureFamily::GaussLobatto) {
            return static_cast<int>(2 * n - 3);
        } else {
            return static_cast<int>(2 * n - 2);
        }
    }();
};

template <BeamAxisQuadratureFamily Family>
constexpr const char* beam_axis_quadrature_family_name() noexcept {
    if constexpr (Family == BeamAxisQuadratureFamily::GaussLegendre) {
        return "Gauss-Legendre";
    } else if constexpr (Family == BeamAxisQuadratureFamily::GaussLobatto) {
        return "Gauss-Lobatto";
    } else if constexpr (Family == BeamAxisQuadratureFamily::GaussRadauLeft) {
        return "Gauss-Radau (left endpoint)";
    } else {
        return "Gauss-Radau (right endpoint)";
    }
}

#endif // FALL_N_BEAM_AXIS_QUADRATURE_HH
