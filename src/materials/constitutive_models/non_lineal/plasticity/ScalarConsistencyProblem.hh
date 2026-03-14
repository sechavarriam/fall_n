#ifndef FALL_N_SCALAR_CONSISTENCY_PROBLEM_HH
#define FALL_N_SCALAR_CONSISTENCY_PROBLEM_HH

#include <cmath>

#include "../../../local_problem/LocalNonlinearProblem.hh"
#include "ConsistencyFunction.hh"

// =============================================================================
//  ScalarConsistencyProblem.hh
// =============================================================================
//
//  Adapter that turns the current scalar plastic consistency equation into a
//  first-class LocalNonlinearProblem.  This is the compatibility bridge between
//  the legacy J2 return-map and the more general local-problem architecture.
//
//  The local unknown is:
//
//      x = Δγ,
//
//  but the important part is the surrounding contract.  Once finite-strain or
//  damage models need a vector of local unknowns, they can define a different
//  Problem type and reuse the same NewtonLocalSolver without changing
//  PlasticityRelation or Material<>.
//
// =============================================================================

template <typename Relation>
struct ScalarConsistencyContext {
    using TrialStateT = typename Relation::TrialStateT;
    using InternalVariablesT = typename Relation::InternalVariablesT;
    using StrainVectorT = typename Relation::StrainVectorT;

    TrialStateT trial{};
    TrialStateT effective_trial{};
    InternalVariablesT alpha{};
    StrainVectorT flow_direction = StrainVectorT::Zero();
    double trial_overstress{0.0};
};

template <
    typename ResidualPolicyT = StandardConsistencyResidual,
    typename JacobianPolicyT = StandardConsistencyJacobian
>
struct ScalarConsistencyProblem {
    using UnknownT = double;
    using ResidualT = double;
    using JacobianT = double;

    [[no_unique_address]] ResidualPolicyT residual_{};
    [[no_unique_address]] JacobianPolicyT jacobian_{};

    template <typename Relation>
    using ContextT = ScalarConsistencyContext<Relation>;

    template <typename Relation>
        requires ConsistencyResidualPolicy<ResidualPolicyT, Relation> &&
                 ConsistencyJacobianPolicy<JacobianPolicyT, ResidualPolicyT, Relation>
    [[nodiscard]] double initial_guess(
        const Relation& relation,
        const ContextT<Relation>& context) const
    {
        return relation.consistency_increment(
            context.trial_overstress,
            relation.hardening_modulus(context.alpha));
    }

    [[nodiscard]] double residual_norm(double residual) const noexcept
    {
        return std::abs(residual);
    }

    template <typename Relation>
        requires ConsistencyResidualPolicy<ResidualPolicyT, Relation>
    [[nodiscard]] double residual(
        const Relation& relation,
        const ContextT<Relation>& context,
        double delta_gamma) const
    {
        return residual_.value(
            relation,
            context.trial,
            context.alpha,
            delta_gamma,
            context.flow_direction);
    }

    template <typename Relation>
        requires ConsistencyJacobianPolicy<JacobianPolicyT, ResidualPolicyT, Relation>
    [[nodiscard]] double jacobian(
        const Relation& relation,
        const ContextT<Relation>& context,
        double delta_gamma) const
    {
        return jacobian_.value(
            relation,
            residual_,
            context.trial,
            context.alpha,
            delta_gamma,
            context.flow_direction);
    }

    // The plastic multiplier must remain admissible for classical
    // rate-independent return-mapping.  More general local problems can choose
    // a different projection or none at all.
    void project_iterate(double& delta_gamma) const noexcept
    {
        if (delta_gamma < 0.0) {
            delta_gamma = 0.0;
        }
    }
};

#endif // FALL_N_SCALAR_CONSISTENCY_PROBLEM_HH
