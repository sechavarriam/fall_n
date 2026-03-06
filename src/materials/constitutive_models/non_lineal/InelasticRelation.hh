#ifndef INELASTIC_RELATION_HH
#define INELASTIC_RELATION_HH

// =============================================================================
//  InelasticRelation.hh — Backward-compatibility header
// =============================================================================
//
//  This file originally contained the monolithic J2PlasticityRelation<Policy>.
//
//  That implementation has been decomposed into orthogonal building blocks:
//
//    PlasticityConcepts.hh      — TrialState, PlasticInternalVariables,
//                                 concepts: YieldCriterion, HardeningLaw, FlowRule
//    VonMises.hh                — J₂ yield criterion (stateless)
//    IsotropicHardening.hh      — LinearIsotropicHardening (σ_y = σ_y0 + H·ε̄^p)
//    AssociatedFlow.hh          — Associated flow rule (m̂ = ∂f/∂σ)
//    PlasticityRelation.hh      — Generic template <Policy, YieldF, Hardening, Flow>
//
//  This header provides backward-compatible aliases so that existing code
//  using J2PlasticityRelation<Policy> and J2InternalVariables<N> continues
//  to compile without changes.
//
// =============================================================================

#include "PlasticityRelation.hh"


// =============================================================================
//  Backward-compatible aliases
// =============================================================================

// J2PlasticityRelation<Policy>  ≡  PlasticityRelation<Policy, VonMises,
//                                      LinearIsotropicHardening, AssociatedFlow>
//
// Preserves the old 4-arg constructor:  J2PlasticityRelation<3D>{E, ν, σ_y0, H}

template <class MaterialPolicy>
using J2PlasticityRelation = PlasticityRelation<
    MaterialPolicy, VonMises, LinearIsotropicHardening, AssociatedFlow>;


// J2InternalVariables<N>  ≡  PlasticInternalVariables<N, IsotropicHardeningState>
//
// Access patterns that still work:
//   alpha.plastic_strain                         (direct member)
//   alpha.eps_p()                                (named accessor)
//   alpha.eps_bar_p()                            (constrained accessor → hardening_state)
//   alpha.hardening_state.equivalent_plastic_strain  (full path)

template <std::size_t N>
using J2InternalVariables = PlasticInternalVariables<N, IsotropicHardeningState>;


// =============================================================================
//  Static concept verification — backward-compat types
// =============================================================================

static_assert(
    ConstitutiveRelation<J2PlasticityRelation<ThreeDimensionalMaterial>>,
    "J2PlasticityRelation<3D> must satisfy ConstitutiveRelation");

static_assert(
    InelasticConstitutiveRelation<J2PlasticityRelation<ThreeDimensionalMaterial>>,
    "J2PlasticityRelation<3D> must satisfy InelasticConstitutiveRelation");


#endif // INELASTIC_RELATION_HH