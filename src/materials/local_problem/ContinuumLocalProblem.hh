#ifndef FALL_N_CONTINUUM_LOCAL_PROBLEM_HH
#define FALL_N_CONTINUUM_LOCAL_PROBLEM_HH

#include <concepts>
#include <type_traits>
#include <variant>

#include "../../continuum/ConstitutiveKinematics.hh"
#include "../ConstitutiveRelation.hh"
#include "LocalNonlinearProblem.hh"
#include "NewtonLocalSolver.hh"

// =============================================================================
//  ContinuumLocalProblem.hh
// =============================================================================
//
//  General local-problem layer for continuum constitutive updates driven by the
//  rich continuum carrier `ConstitutiveKinematics<dim>`.
//
//  The scalar small-strain plasticity path already proved that a local
//  nonlinear problem R(x)=0 is the right abstraction.  The next step toward
//  finite-strain inelasticity is to make that abstraction continuum-aware
//  without hard-coding any particular constitutive model.
//
//  This header therefore adds:
//
//    1. a reusable context carrier that stores
//         - the continuum kinematics,
//         - the committed algorithmic state,
//         - optional trial / auxiliary data;
//
//    2. a reusable result carrier for constitutive local updates;
//
//    3. a concept that constrains continuum-local problems;
//
//    4. a generic integration algorithm that delegates the nonlinear solve to
//       any injected `LocalNonlinearSolverPolicy`.
//
//  The architectural point is that finite-strain plasticity, damage, or mixed
//  constitutive updates should only specialize:
//    - the local unknown x,
//    - the residual R(x),
//    - the Jacobian K(x),
//    - the map from the solved unknown back to stress/tangent/state.
//
//  The nonlinear solve itself remains a separately injected policy.
//
// =============================================================================

namespace continuum_local_problem {

template <
    typename Relation,
    typename TrialStateT = std::monostate,
    typename AuxiliaryT = std::monostate
>
struct Context {
    static constexpr std::size_t dim = Relation::MaterialPolicyT::dim;

    using KinematicsT = continuum::ConstitutiveKinematics<dim>;
    using AlgorithmicStateT = typename Relation::InternalVariablesT;
    using TrialStateType = TrialStateT;
    using AuxiliaryType = AuxiliaryT;

    KinematicsT kinematics{};
    AlgorithmicStateT committed_state{};
    [[no_unique_address]] TrialStateT trial_state{};
    [[no_unique_address]] AuxiliaryT auxiliary{};
};

template <typename Relation>
struct UpdateResult {
    using ConjugateT = typename Relation::ConjugateT;
    using TangentT = typename Relation::TangentT;
    using AlgorithmicStateT = typename Relation::InternalVariablesT;

    ConjugateT response{};
    TangentT tangent{};
    AlgorithmicStateT algorithmic_state{};
    bool inelastic{false};
};

} // namespace continuum_local_problem

template <typename Problem, typename Relation>
concept ContinuumLocalProblemPolicy =
    continuum::ExternallyStateDrivenContinuumRelation<Relation> &&
    requires {
        typename Problem::UnknownT;
        typename Problem::ResidualT;
        typename Problem::JacobianT;
        typename Problem::template ContextT<Relation>;
        typename Problem::template ResultT<Relation>;
    } &&
    LocalNonlinearProblem<Problem, Relation, typename Problem::template ContextT<Relation>> &&
    requires(const typename Problem::template ContextT<Relation>& context) {
        requires std::same_as<
            std::remove_cvref_t<decltype(context.kinematics)>,
            continuum::ConstitutiveKinematics<Relation::MaterialPolicyT::dim>>;
        requires std::same_as<
            std::remove_cvref_t<decltype(context.committed_state)>,
            typename Relation::InternalVariablesT>;
    } &&
    requires(
        const Problem& problem,
        const Relation& relation,
        const typename Problem::template ContextT<Relation>& context,
        const typename Problem::UnknownT& unknown,
        const continuum::ConstitutiveKinematics<Relation::MaterialPolicyT::dim>& kin,
        const typename Relation::InternalVariablesT& alpha)
    {
        { problem.make_context(relation, kin, alpha) }
            -> std::same_as<typename Problem::template ContextT<Relation>>;
        { problem.is_inelastic(relation, context) } -> std::convertible_to<bool>;
        { problem.elastic_result(relation, context) }
            -> std::same_as<typename Problem::template ResultT<Relation>>;
        { problem.finalize(relation, context, unknown) }
            -> std::same_as<typename Problem::template ResultT<Relation>>;
    };

template <
    typename LocalProblemT,
    typename LocalSolverT = NewtonLocalSolver<>
>
struct ContinuumLocalIntegrationAlgorithm {
    [[no_unique_address]] LocalProblemT local_problem_{};
    [[no_unique_address]] LocalSolverT local_solver_{};

    template <typename Relation>
        requires ContinuumLocalProblemPolicy<LocalProblemT, Relation> &&
                 LocalNonlinearSolverPolicy<
                     LocalSolverT,
                     LocalProblemT,
                     Relation,
                     typename LocalProblemT::template ContextT<Relation>>
    [[nodiscard]] auto integrate(
        const Relation& relation,
        const continuum::ConstitutiveKinematics<Relation::MaterialPolicyT::dim>& kin,
        const typename Relation::InternalVariablesT& alpha) const
        -> typename LocalProblemT::template ResultT<Relation>
    {
        const auto context = local_problem_.make_context(relation, kin, alpha);
        if (!local_problem_.is_inelastic(relation, context)) {
            return local_problem_.elastic_result(relation, context);
        }

        const auto solve_result =
            local_solver_.solve(local_problem_, relation, context);
        return local_problem_.finalize(relation, context, solve_result.solution);
    }
};

#endif // FALL_N_CONTINUUM_LOCAL_PROBLEM_HH
