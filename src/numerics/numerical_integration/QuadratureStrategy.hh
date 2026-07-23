#ifndef FALL_N_QUADRATURE_STRATEGY_HH
#define FALL_N_QUADRATURE_STRATEGY_HH

// =============================================================================
//  QuadratureStrategy — reference-cell quadrature contract
// =============================================================================
//
//  A model of this concept integrates over the REFERENCE cell as Σ_i w_i f(ξ_i)
//  at reference coordinates ξ_i.  Members:
//    - num_integration_points        : the rule size
//    - reference_integration_point(i): the i-th station ξ_i (reference coords)
//    - weight(i)                     : the i-th weight w_i
//
//  Invariant shared by every model (GaussLegendre/Lobatto/Radau line rules,
//  the tensor-product cell integrators, and SimplexIntegrator): the rule does
//  NOT multiply by the element Jacobian |J| — the element applies its own
//  differential measure.  The weights therefore sum to the reference-cell
//  measure (2 per axis on [−1,1] for the Gauss families, 1/D! for the
//  D-simplex).
//
// =============================================================================

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
