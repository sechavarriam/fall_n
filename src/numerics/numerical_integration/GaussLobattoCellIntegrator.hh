#ifndef FALL_N_GAUSS_LOBATTO_CELL_INTEGRATOR_HH
#define FALL_N_GAUSS_LOBATTO_CELL_INTEGRATOR_HH

#include "GaussLobattoNodes.hh"
#include "GaussLobattoWeights.hh"
#include "LineQuadratureIntegrator.hh"

namespace quadrature_detail {

struct GaussLobattoRuleProvider {
    template <std::size_t n>
    static consteval auto evaluation_points() {
        return GaussLobatto::evaluation_points<n>();
    }

    template <std::size_t n>
    static consteval auto weights() {
        return GaussLobatto::weights<n>();
    }
};

} // namespace quadrature_detail

template <std::size_t n>
using GaussLobattoCellIntegrator =
    LineQuadratureIntegrator<quadrature_detail::GaussLobattoRuleProvider, n>;

#endif // FALL_N_GAUSS_LOBATTO_CELL_INTEGRATOR_HH
