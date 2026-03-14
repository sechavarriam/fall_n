#ifndef FALL_N_NEWTON_LOCAL_SOLVER_HH
#define FALL_N_NEWTON_LOCAL_SOLVER_HH

#include <concepts>

#include "LocalLinearSolver.hh"
#include "LocalNonlinearProblem.hh"
#include "LocalStepControl.hh"

// =============================================================================
//  NewtonLocalSolver.hh
// =============================================================================
//
//  Generic Newton solver for local constitutive problems.  It is intentionally
//  agnostic about the physical meaning of the unknown vector x.  The only
//  requirement is that a LocalNonlinearProblem provides:
//
//    - an initial guess,
//    - a residual R(x),
//    - a Jacobian K(x),
//    - a residual norm.
//
//  This keeps the nonlinear algorithm reusable for:
//    - scalar consistency equations,
//    - multi-variable finite-strain return maps,
//    - local damage equations,
//    - mixed stress/internal-variable updates.
//
// =============================================================================

template <typename Solver, typename Problem, typename Relation, typename ContextT>
concept LocalNonlinearSolverPolicy = requires(
    const Solver& solver,
    const Problem& problem,
    const Relation& relation,
    const ContextT& context)
{
    typename Problem::UnknownT;

    requires std::same_as<
        std::remove_cvref_t<decltype(solver.solve(problem, relation, context).solution)>,
        typename Problem::UnknownT>;
};

template <typename UnknownT>
struct LocalNewtonSolveResult {
    UnknownT solution{};
    bool converged{false};
    int iterations{0};
    double residual_norm{0.0};
};

template <
    int MaxIterations = 20,
    typename LinearSolvePolicyT = DefaultLocalLinearSolver,
    typename StepControlPolicyT = FullStepControl
>
struct NewtonLocalSolver {
    static_assert(MaxIterations > 0);

    [[no_unique_address]] LinearSolvePolicyT linear_solver_{};
    [[no_unique_address]] StepControlPolicyT step_control_{};
    double tolerance_{1e-12};

    template <typename Problem, typename Relation, typename ContextT>
        requires LocalNonlinearProblem<Problem, Relation, ContextT> &&
                 LocalLinearSolvePolicy<
                     LinearSolvePolicyT,
                     typename Problem::UnknownT,
                     typename Problem::JacobianT,
                     typename Problem::ResidualT> &&
                 LocalStepControlPolicy<
                     StepControlPolicyT,
                     Problem,
                     Relation,
                     ContextT>
    [[nodiscard]] auto solve(
        const Problem& problem,
        const Relation& relation,
        const ContextT& context) const
        -> LocalNewtonSolveResult<typename Problem::UnknownT>
    {
        using UnknownT = typename Problem::UnknownT;

        UnknownT unknown = problem.initial_guess(relation, context);
        LocalNewtonSolveResult<UnknownT> out{};
        out.solution = unknown;

        for (int iter = 0; iter < MaxIterations; ++iter) {
            const auto residual = problem.residual(relation, context, unknown);
            const double residual_norm = problem.residual_norm(residual);

            out.iterations = iter;
            out.residual_norm = residual_norm;
            if (residual_norm < tolerance_) {
                out.converged = true;
                out.solution = unknown;
                return out;
            }

            const auto jacobian = problem.jacobian(relation, context, unknown);
            const auto delta_unknown =
                linear_solver_.solve(jacobian, residual);
            const auto controlled_delta =
                step_control_.compute(
                    problem, relation, context, unknown, delta_unknown, residual_norm);
            local_nonlinear_problem::add_update(unknown, controlled_delta);
            local_nonlinear_problem::project_iterate(problem, unknown);
        }

        const auto residual = problem.residual(relation, context, unknown);
        out.solution = unknown;
        out.residual_norm = problem.residual_norm(residual);
        out.converged = out.residual_norm < tolerance_;
        out.iterations = MaxIterations;
        return out;
    }
};

#endif // FALL_N_NEWTON_LOCAL_SOLVER_HH
