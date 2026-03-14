#ifndef FN_YIELD_FUNCTION_HH
#define FN_YIELD_FUNCTION_HH

#include "PlasticityConcepts.hh"

// =============================================================================
//  YieldFunctionPolicy<F, N, Y, H>
// =============================================================================
//
//  Separates the scalar yield-function evaluation from:
//    - the yield-surface geometry              (YieldCriterion)
//    - the evolution of the admissible radius  (HardeningLaw)
//
//  In the current rate-independent plasticity kernel, the local return mapping
//  still assumes a standard overstress form driven by:
//
//      f = q(trial) - sigma_y(alpha)
//
//  where q(trial) is supplied by the YieldCriterion and sigma_y(alpha) by the
//  HardeningLaw.  Making this scalar map explicit is useful for two reasons:
//
//    1. it gives the architecture an unambiguous compile-time customization
//       point called "YieldFunction", distinct from the yield-surface geometry;
//    2. it allows alternative but still compatible scalar laws to be injected
//       without reworking PlasticityRelation itself.
//
//  The customization point is intentionally narrow.  It returns only the
//  scalar overstress used by the current return-mapping algorithm.  More
//  general inelastic kernels may later enlarge this contract.
//
// =============================================================================

template <typename F, std::size_t N, typename Y, typename H>
concept YieldFunctionPolicy =
    YieldCriterion<Y, N> &&
    HardeningLaw<H> &&
    requires(const F& f,
             const Y& yield,
             const TrialState<N>& trial,
             const H& hardening,
             const typename H::StateT& state)
{
    { f.value(yield, trial, hardening, state) } -> std::convertible_to<double>;
};


// =============================================================================
//  StandardYieldFunction
// =============================================================================
//
//  Default scalar yield function used throughout the existing return-mapping
//  implementation:
//
//      f = q(trial) - sigma_y(alpha)
//
//  with q supplied by the yield criterion and sigma_y by the hardening law.
//
// =============================================================================

struct StandardYieldFunction {
    template <std::size_t N, typename YieldCriterionT, typename HardeningT>
    [[nodiscard]] constexpr double value(
        const YieldCriterionT& yield,
        const TrialState<N>& trial,
        const HardeningT& hardening,
        const typename HardeningT::StateT& state) const
    {
        return yield.equivalent_stress(trial) - hardening.yield_stress(state);
    }
};

#endif // FN_YIELD_FUNCTION_HH
