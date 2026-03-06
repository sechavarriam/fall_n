#ifndef FN_ASSOCIATED_FLOW_HH
#define FN_ASSOCIATED_FLOW_HH

#include "PlasticityConcepts.hh"

// =============================================================================
//  AssociatedFlow — m̂ = ∂f/∂σ  (flow direction ≡ yield surface normal)
// =============================================================================
//
//  In associative plasticity, the plastic potential g(σ) coincides with the
//  yield function f(σ):  g ≡ f.  Consequently, the plastic flow direction
//  equals the yield surface gradient:
//
//      m̂ = ∂g/∂σ = ∂f/∂σ = yield.gradient(trial)
//
//  This guarantees:
//    - Maximum plastic dissipation (Hill's principle)
//    - Drucker's stability postulate
//    - Symmetric algorithmic consistent tangent C_ep
//    - Uniqueness of solutions under standard conditions
//    - Normality rule (plastic strain rate normal to yield surface)
//
//  Implementation: delegates to YieldCriterion::gradient() via the yield
//  reference passed by PlasticityRelation. This avoids storing a pointer
//  to the yield criterion (no self-referential struct issues during copy).
//
//  This struct is STATELESS.
//
//  ─── Non-associated flow (future) ──────────────────────────────────────
//
//  For non-associated models (e.g., Drucker-Prager with different friction
//  and dilation angles), a NonAssociatedFlow<PotentialF> will store its own
//  plastic potential function and ignore the yield argument:
//
//    template <typename PotentialF>
//    struct NonAssociatedFlow {
//        PotentialF potential_;
//        template <std::size_t N, typename YieldF>
//        auto direction(const YieldF&, const TrialState<N>& trial) const
//            -> Eigen::Vector<double, N>
//        {
//            return potential_.gradient(trial);  // uses own potential
//        }
//    };
//
// =============================================================================

struct AssociatedFlow {

    template <std::size_t N, typename YieldF>
    [[nodiscard]] Eigen::Vector<double, N> direction(
        const YieldF& yield,
        const TrialState<N>& trial) const
    {
        return yield.gradient(trial);
    }
};


// ─── Concept verification ────────────────────────────────────────────────────
//
// Note: FlowRule<AssociatedFlow, N, Y> can only be verified when both N and Y
// are known.  A full check is placed in PlasticityRelation.hh after all
// includes are resolved.


#endif // FN_ASSOCIATED_FLOW_HH
