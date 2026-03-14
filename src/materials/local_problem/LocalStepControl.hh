#ifndef FALL_N_LOCAL_STEP_CONTROL_HH
#define FALL_N_LOCAL_STEP_CONTROL_HH

#include <concepts>

#include "LocalNonlinearProblem.hh"

// =============================================================================
//  LocalStepControl.hh
// =============================================================================
//
//  Step-control customization point for local nonlinear solvers.
//
//  Newton is only one way to generate a candidate increment Δx.  Whether that
//  full step is accepted, damped, line-searched, or projected is a different
//  concern.  Keeping this policy separate allows:
//    - full Newton for well-behaved local updates,
//    - fixed damping for robustness experiments,
//    - residual-based backtracking,
//    - future trust-region or PETSc/SNES line-search adapters.
//
// =============================================================================

template <typename Policy, typename Problem, typename Relation, typename ContextT>
concept LocalStepControlPolicy = requires(
    const Policy& policy,
    const Problem& problem,
    const Relation& relation,
    const ContextT& context,
    const typename Problem::UnknownT& unknown,
    const typename Problem::UnknownT& delta_unknown,
    double residual_norm)
{
    { policy.compute(problem, relation, context, unknown, delta_unknown, residual_norm) }
        -> std::same_as<typename Problem::UnknownT>;
};

struct FullStepControl {
    template <typename Problem, typename Relation, typename ContextT>
    [[nodiscard]] auto compute(
        const Problem&,
        const Relation&,
        const ContextT&,
        const typename Problem::UnknownT&,
        const typename Problem::UnknownT& delta_unknown,
        double) const
        -> typename Problem::UnknownT
    {
        return delta_unknown;
    }
};

template <int MaxBacktracks = 8>
struct BacktrackingResidualStepControl {
    static_assert(MaxBacktracks > 0);

    double contraction_{0.5};

    template <typename Problem, typename Relation, typename ContextT>
        requires LocalNonlinearProblem<Problem, Relation, ContextT>
    [[nodiscard]] auto compute(
        const Problem& problem,
        const Relation& relation,
        const ContextT& context,
        const typename Problem::UnknownT& unknown,
        const typename Problem::UnknownT& delta_unknown,
        double current_residual_norm) const
        -> typename Problem::UnknownT
    {
        double scale = 1.0;
        for (int i = 0; i < MaxBacktracks; ++i) {
            auto candidate_unknown = unknown + scale * delta_unknown;
            local_nonlinear_problem::project_iterate(problem, candidate_unknown);
            const auto candidate_residual =
                problem.residual(relation, context, candidate_unknown);
            const double candidate_norm =
                problem.residual_norm(candidate_residual);
            if (candidate_norm < current_residual_norm) {
                return scale * delta_unknown;
            }
            scale *= contraction_;
        }
        return scale * delta_unknown;
    }
};

#endif // FALL_N_LOCAL_STEP_CONTROL_HH
