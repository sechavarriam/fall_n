#ifndef FALL_N_GAUSS_RADAU_CELL_INTEGRATOR_HH
#define FALL_N_GAUSS_RADAU_CELL_INTEGRATOR_HH

#include "GaussRadauNodes.hh"
#include "GaussRadauWeights.hh"
#include "LineQuadratureIntegrator.hh"

namespace quadrature_detail {

template <GaussRadau::Endpoint Endpoint>
struct GaussRadauRuleProvider {
    template <std::size_t n>
    static consteval auto evaluation_points() {
        return GaussRadau::evaluation_points<n, Endpoint>();
    }

    template <std::size_t n>
    static consteval auto weights() {
        return GaussRadau::weights<n, Endpoint>();
    }
};

} // namespace quadrature_detail

template <std::size_t n, GaussRadau::Endpoint Endpoint = GaussRadau::Endpoint::Left>
using GaussRadauCellIntegrator =
    LineQuadratureIntegrator<quadrature_detail::GaussRadauRuleProvider<Endpoint>, n>;

#endif // FALL_N_GAUSS_RADAU_CELL_INTEGRATOR_HH
