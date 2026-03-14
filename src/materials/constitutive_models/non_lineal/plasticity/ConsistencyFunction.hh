#ifndef FN_CONSISTENCY_FUNCTION_HH
#define FN_CONSISTENCY_FUNCTION_HH

#include <concepts>
#include <cmath>

// =============================================================================
//  ConsistencyFunction.hh
// =============================================================================
//
//  This header introduces an explicit compile-time customization point for the
//  local scalar nonlinear problem solved inside rate-independent return mapping.
//
//  In the current small-strain J2 path, the local unknown is the plastic
//  multiplier increment Δγ and the classical consistency condition is:
//
//      r(Δγ) = 0.
//
//  The important architectural point is that the scalar equation itself and
//  the algorithm used to solve it are not the same object:
//
//    - YieldFunction      : defines admissibility / overstress
//    - ConsistencyResidual: defines the local scalar equation to solve
//    - ConsistencyJacobian: defines its derivative / linearization
//    - ReturnAlgorithm    : defines how that scalar nonlinear problem is solved
//
//  This split is valuable even before full finite-strain plasticity arrives,
//  because finite-strain local updates will also be expressed as local
//  residual/Jacobian solves.  The current scalar contract therefore acts as
//  the first reusable seam toward that broader architecture.
//
// =============================================================================

template <typename R, typename Relation>
concept ConsistencyResidualPolicy = requires(
    const R& residual,
    const Relation& relation,
    const typename Relation::TrialStateT& trial,
    const typename Relation::InternalVariablesT& alpha,
    double delta_gamma,
    const typename Relation::StrainVectorT& flow_direction)
{
    { residual.value(relation, trial, alpha, delta_gamma, flow_direction) }
        -> std::convertible_to<double>;
};

template <typename J, typename Residual, typename Relation>
concept ConsistencyJacobianPolicy = requires(
    const J& jacobian,
    const Residual& residual,
    const Relation& relation,
    const typename Relation::TrialStateT& trial,
    const typename Relation::InternalVariablesT& alpha,
    double delta_gamma,
    const typename Relation::StrainVectorT& flow_direction)
{
    { jacobian.value(relation, residual, trial, alpha, delta_gamma, flow_direction) }
        -> std::convertible_to<double>;
};


// =============================================================================
//  StandardConsistencyResidual
// =============================================================================
//
//  Default residual used by the current radial-return family:
//
//      r(Δγ) = f(σ(Δγ), α(Δγ))
//
//  where the corrected stress and the evolved algorithmic state are rebuilt
//  through the constitutive kernel exposed by PlasticityRelation.
//
//  The implementation intentionally re-evaluates the yield function on the
//  corrected state.  That keeps the residual semantically honest and makes it
//  usable as a default even when the closed-form denominator used by the
//  classical radial return is no longer available.
//
// =============================================================================

struct StandardConsistencyResidual {
    template <typename Relation>
    [[nodiscard]] double value(
        const Relation& relation,
        const typename Relation::TrialStateT& trial,
        const typename Relation::InternalVariablesT& alpha,
        double delta_gamma,
        const typename Relation::StrainVectorT& flow_direction) const
    {
        const auto corrected_stress =
            relation.corrected_stress(trial, delta_gamma, flow_direction);
        const auto alpha_new =
            relation.evolve_internal_variables(alpha, delta_gamma, flow_direction);
        const auto corrected_trial =
            relation.trial_state_from_stress(corrected_stress);
        const auto effective_trial =
            relation.effective_trial(corrected_trial, alpha_new);
        return relation.evaluate_yield_function(effective_trial, alpha_new);
    }
};


// =============================================================================
//  StandardConsistencyJacobian
// =============================================================================
//
//  Closed-form derivative used by the existing J2 radial return family.
//
//  This is the fast default for the current associated isotropic path.  More
//  general models can swap it for a different analytic linearization or for a
//  finite-difference Jacobian without changing the return algorithm type.
//
// =============================================================================

struct StandardConsistencyJacobian {
    template <typename Relation, typename ResidualPolicyT>
    [[nodiscard]] double value(
        const Relation& relation,
        const ResidualPolicyT&,
        const typename Relation::TrialStateT&,
        const typename Relation::InternalVariablesT& alpha,
        double delta_gamma,
        const typename Relation::StrainVectorT& flow_direction) const
    {
        const auto alpha_new =
            relation.evolve_internal_variables(alpha, delta_gamma, flow_direction);
        return -std::sqrt(2.0 / 3.0) *
               (3.0 * relation.shear_modulus() + relation.hardening_modulus(alpha_new));
    }
};


// =============================================================================
//  FiniteDifferenceConsistencyJacobian
// =============================================================================
//
//  Generic numerical derivative for local scalar solves:
//
//      r'(Δγ) ≈ [r(Δγ + h) - r(Δγ - h)] / (2h)
//
//  This is not the preferred production path for the hot local loop, but it is
//  a useful correctness baseline and an important escape hatch when developing
//  new models whose analytic Jacobian is not ready yet.
//
// =============================================================================

template <int StepScale = 1000000>
struct FiniteDifferenceConsistencyJacobian {
    static_assert(StepScale > 0);

    template <typename Relation, typename ResidualPolicyT>
    [[nodiscard]] double value(
        const Relation& relation,
        const ResidualPolicyT& residual,
        const typename Relation::TrialStateT& trial,
        const typename Relation::InternalVariablesT& alpha,
        double delta_gamma,
        const typename Relation::StrainVectorT& flow_direction) const
    {
        const double h = 1.0 / static_cast<double>(StepScale);
        const double rp =
            residual.value(relation, trial, alpha, delta_gamma + h, flow_direction);
        const double rm =
            residual.value(relation, trial, alpha, delta_gamma - h, flow_direction);
        return (rp - rm) / (2.0 * h);
    }
};

#endif // FN_CONSISTENCY_FUNCTION_HH
