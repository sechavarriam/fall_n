#ifndef FN_RETURN_ALGORITHM_HH
#define FN_RETURN_ALGORITHM_HH

#include <concepts>
#include <cmath>

#include "../../../local_problem/NewtonLocalSolver.hh"
#include "PlasticityConcepts.hh"
#include "ConsistencyFunction.hh"
#include "ScalarConsistencyProblem.hh"

// =============================================================================
//  ReturnAlgorithmPolicy<A, Relation>
// =============================================================================
//
//  Separates the local constitutive integration algorithm from the constitutive
//  ingredients that define the plasticity model itself:
//
//    - YieldCriterion   : geometry of the admissible surface
//    - HardeningLaw     : evolution of admissibility
//    - FlowRule         : plastic flow direction
//    - YieldFunction    : scalar overstress evaluation
//    - ReturnAlgorithm  : local nonlinear solve / correction strategy
//
//  The policy acts on a concrete PlasticityRelation kernel and an explicit
//  algorithmic state alpha.  This keeps the hot path fully static and avoids
//  virtual dispatch in local constitutive updates.
//
// =============================================================================

template <typename A, typename Relation>
concept ReturnAlgorithmPolicy = requires(
    const A& algorithm,
    const Relation& relation,
    const typename Relation::StrainVectorT& total_strain,
    const typename Relation::InternalVariablesT& alpha)
{
    { algorithm.integrate(relation, total_strain, alpha) }
        -> std::same_as<typename Relation::ReturnMapResultT>;
};


// =============================================================================
//  StandardRadialReturnAlgorithm
// =============================================================================
//
//  Default backward-Euler radial return used by the existing J2 path.  The
//  algorithm is now factored out of PlasticityRelation so that alternative local
//  schemes such as substepping, cutting-plane, viscoplastic updates, or capped
//  corrections can be added as compile-time policies without changing the law.
//
// =============================================================================

struct StandardRadialReturnAlgorithm {
    template <typename Relation>
    [[nodiscard]] auto integrate(
        const Relation& relation,
        const typename Relation::StrainVectorT& total_strain,
        const typename Relation::InternalVariablesT& alpha) const
        -> typename Relation::ReturnMapResultT
    {
        auto trial = relation.elastic_predictor(total_strain, alpha);
        auto eff = relation.effective_trial(trial, alpha);
        const double f_trial = relation.evaluate_yield_function(eff, alpha);

        if (f_trial <= 0.0) {
            return relation.make_elastic_result(trial, alpha);
        }

        const double H_mod = relation.hardening_modulus(alpha);
        const double delta_gamma = relation.consistency_increment(f_trial, H_mod);
        const auto n_hat = relation.flow_direction(eff);

        typename Relation::ReturnMapResultT out{};
        out.stress = relation.corrected_stress(trial, delta_gamma, n_hat);
        out.tangent = relation.consistent_tangent(eff, delta_gamma, H_mod, n_hat);
        out.alpha_new = relation.evolve_internal_variables(alpha, delta_gamma, n_hat);
        out.plastic = true;
        return out;
    }
};


// =============================================================================
//  LocalNonlinearReturnAlgorithm
// =============================================================================
//
//  Generic bridge between a constitutive return-mapping context and an abstract
//  local nonlinear problem.  The current scalar consistency update is one
//  instance through ScalarConsistencyProblem<...>, but the same return wrapper
//  is meant to survive when the local unknown becomes vector-valued in
//  finite-strain plasticity or damage.
//
// =============================================================================

template <
    typename LocalProblemT = ScalarConsistencyProblem<>,
    typename LocalSolverT = NewtonLocalSolver<>
>
struct LocalNonlinearReturnAlgorithm {
    [[no_unique_address]] LocalProblemT local_problem_{};
    [[no_unique_address]] LocalSolverT local_solver_{};

    template <typename Relation>
        requires LocalNonlinearProblem<
                     LocalProblemT,
                     Relation,
                     typename LocalProblemT::template ContextT<Relation>> &&
                 LocalNonlinearSolverPolicy<
                     LocalSolverT,
                     LocalProblemT,
                     Relation,
                     typename LocalProblemT::template ContextT<Relation>>
    [[nodiscard]] auto integrate(
        const Relation& relation,
        const typename Relation::StrainVectorT& total_strain,
        const typename Relation::InternalVariablesT& alpha) const
        -> typename Relation::ReturnMapResultT
    {
        auto trial = relation.elastic_predictor(total_strain, alpha);
        auto eff = relation.effective_trial(trial, alpha);
        const double f_trial = relation.evaluate_yield_function(eff, alpha);

        if (f_trial <= 0.0) {
            return relation.make_elastic_result(trial, alpha);
        }

        const auto n_hat = relation.flow_direction(eff);
        typename LocalProblemT::template ContextT<Relation> context{
            trial, eff, alpha, n_hat, f_trial
        };

        const auto solve_result =
            local_solver_.solve(local_problem_, relation, context);
        const double delta_gamma = solve_result.solution;
        const auto alpha_new =
            relation.evolve_internal_variables(alpha, delta_gamma, n_hat);
        const double H_mod = relation.hardening_modulus(alpha_new);

        typename Relation::ReturnMapResultT out{};
        out.stress = relation.corrected_stress(trial, delta_gamma, n_hat);
        out.tangent = relation.consistent_tangent(eff, delta_gamma, H_mod, n_hat);
        out.alpha_new = alpha_new;
        out.plastic = true;
        return out;
    }
};


// =============================================================================
//  NewtonConsistencyReturnAlgorithm
// =============================================================================
//
//  Generic Newton solver for the local scalar consistency equation:
//
//      r(Δγ) = 0
//
//  This algorithm is intentionally separate from the residual and Jacobian
//  definitions.  That is the architectural point of this file: new material
//  models should be free to keep the same nonlinear solver while changing the
//  local equation, or to keep the same equation while changing the solver.
//
//  The default initial guess reuses the closed-form estimate from the current
//  J2 radial-return path.  This gives good behaviour for the existing models
//  while leaving the surrounding contract open for more general future laws.
//
// =============================================================================

template <
    typename ResidualPolicyT = StandardConsistencyResidual,
    typename JacobianPolicyT = StandardConsistencyJacobian,
    int MaxIterations = 20
>
struct NewtonConsistencyReturnAlgorithm {
    using LocalProblemT =
        ScalarConsistencyProblem<ResidualPolicyT, JacobianPolicyT>;
    using LocalSolverT = NewtonLocalSolver<MaxIterations>;

    [[no_unique_address]] LocalProblemT local_problem_{};
    [[no_unique_address]] LocalSolverT local_solver_{};

    template <typename Relation>
    [[nodiscard]] auto integrate(
        const Relation& relation,
        const typename Relation::StrainVectorT& total_strain,
        const typename Relation::InternalVariablesT& alpha) const
        -> typename Relation::ReturnMapResultT
    {
        return LocalNonlinearReturnAlgorithm<LocalProblemT, LocalSolverT>{
            local_problem_, local_solver_
        }.integrate(relation, total_strain, alpha);
    }
};

#endif // FN_RETURN_ALGORITHM_HH
