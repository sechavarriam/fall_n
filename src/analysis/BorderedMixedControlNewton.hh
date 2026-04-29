#ifndef FALL_N_SRC_ANALYSIS_BORDERED_MIXED_CONTROL_NEWTON_HH
#define FALL_N_SRC_ANALYSIS_BORDERED_MIXED_CONTROL_NEWTON_HH

// =============================================================================
//  BorderedMixedControlNewton
// =============================================================================
//
//  This is the second-generation mixed-control kernel behind the lightweight
//  observable arc-length wrapper.  Instead of accepting/rejecting an already
//  converged Dirichlet step, it solves the augmented Newton system
//
//      [ K   r_l ] [du] = -[R]
//      [ g^T c_l ] [dl]   [c]
//
//  where R(u, lambda) is the equilibrium residual and c(u, lambda) is a scalar
//  mixed-control constraint.  The caller owns the physics: c may be a true
//  Crisfield/Riks arc constraint, a displacement/reaction mixed constraint, or
//  a local-model continuation observable.  This header owns only the typed
//  algebra, convergence bookkeeping, and an optional merit line search.
//
//  Keeping this kernel independent of PETSc makes it cheap to test and keeps
//  the future PETSc integration honest: the same callbacks can later be backed
//  by Mat/Vec/KSP without changing the mathematical contract.
//
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string_view>
#include <vector>

#include <Eigen/Dense>

namespace fall_n {

enum class BorderedMixedControlNewtonStatus {
    converged,
    max_iterations,
    invalid_evaluation,
    singular_augmented_system,
    line_search_failed
};

[[nodiscard]] constexpr std::string_view
to_string(BorderedMixedControlNewtonStatus status) noexcept
{
    switch (status) {
        case BorderedMixedControlNewtonStatus::converged:
            return "converged";
        case BorderedMixedControlNewtonStatus::max_iterations:
            return "max_iterations";
        case BorderedMixedControlNewtonStatus::invalid_evaluation:
            return "invalid_evaluation";
        case BorderedMixedControlNewtonStatus::singular_augmented_system:
            return "singular_augmented_system";
        case BorderedMixedControlNewtonStatus::line_search_failed:
            return "line_search_failed";
    }
    return "unknown_bordered_mixed_control_newton_status";
}

struct BorderedMixedControlState {
    Eigen::VectorXd unknowns{};
    double load_parameter{0.0};
};

struct BorderedMixedControlEvaluation {
    Eigen::VectorXd residual{};
    Eigen::MatrixXd tangent{};
    Eigen::VectorXd load_column{};
    Eigen::VectorXd constraint_gradient{};
    double constraint{0.0};
    double constraint_load_derivative{0.0};
};

struct BorderedMixedControlNewtonSettings {
    int max_iterations{30};
    double residual_tolerance{1.0e-10};
    double constraint_tolerance{1.0e-10};
    double correction_tolerance{1.0e-12};
    bool line_search_enabled{true};
    int max_line_search_cutbacks{10};
    double line_search_cutback_factor{0.5};
    double line_search_min_alpha{1.0e-4};
    double merit_decrease{1.0e-4};
};

struct BorderedMixedControlNewtonIterationRecord {
    int iteration{0};
    double residual_norm{0.0};
    double constraint_abs{0.0};
    double correction_norm{0.0};
    double load_correction_abs{0.0};
    double line_search_alpha{1.0};
};

struct BorderedMixedControlNewtonResult {
    BorderedMixedControlNewtonStatus status{
        BorderedMixedControlNewtonStatus::invalid_evaluation};
    BorderedMixedControlState state{};
    int iterations{0};
    double residual_norm{0.0};
    double constraint_abs{0.0};
    double correction_norm{0.0};
    std::vector<BorderedMixedControlNewtonIterationRecord> records{};

    [[nodiscard]] bool converged() const noexcept
    {
        return status == BorderedMixedControlNewtonStatus::converged;
    }
};

namespace detail {

[[nodiscard]] inline bool bordered_eval_has_consistent_dimensions(
    const BorderedMixedControlEvaluation& e,
    Eigen::Index n)
{
    return n > 0 &&
           e.residual.size() == n &&
           e.tangent.rows() == n &&
           e.tangent.cols() == n &&
           e.load_column.size() == n &&
           e.constraint_gradient.size() == n &&
           std::isfinite(e.constraint) &&
           std::isfinite(e.constraint_load_derivative);
}

[[nodiscard]] inline double bordered_merit(
    const BorderedMixedControlEvaluation& e)
{
    return e.residual.squaredNorm() + e.constraint * e.constraint;
}

} // namespace detail

template <typename EvaluatorT>
[[nodiscard]] BorderedMixedControlNewtonResult
solve_bordered_mixed_control_newton(
    BorderedMixedControlState initial_state,
    EvaluatorT&& evaluator,
    BorderedMixedControlNewtonSettings settings = {})
{
    BorderedMixedControlNewtonResult result;
    result.state = std::move(initial_state);
    settings.max_iterations = std::max(settings.max_iterations, 1);
    settings.max_line_search_cutbacks =
        std::max(settings.max_line_search_cutbacks, 0);
    settings.line_search_cutback_factor =
        std::clamp(settings.line_search_cutback_factor, 0.05, 0.95);
    settings.line_search_min_alpha =
        std::clamp(settings.line_search_min_alpha, 1.0e-12, 1.0);
    settings.merit_decrease = std::clamp(settings.merit_decrease, 0.0, 0.5);

    const auto n = result.state.unknowns.size();
    if (n <= 0) {
        result.status = BorderedMixedControlNewtonStatus::invalid_evaluation;
        return result;
    }

    for (int iter = 0; iter < settings.max_iterations; ++iter) {
        auto eval = evaluator(result.state);
        if (!detail::bordered_eval_has_consistent_dimensions(eval, n)) {
            result.status =
                BorderedMixedControlNewtonStatus::invalid_evaluation;
            return result;
        }

        const double residual_norm = eval.residual.norm();
        const double constraint_abs = std::abs(eval.constraint);
        if (residual_norm <= settings.residual_tolerance &&
            constraint_abs <= settings.constraint_tolerance)
        {
            result.status = BorderedMixedControlNewtonStatus::converged;
            result.iterations = iter;
            result.residual_norm = residual_norm;
            result.constraint_abs = constraint_abs;
            result.correction_norm = 0.0;
            return result;
        }

        Eigen::MatrixXd augmented =
            Eigen::MatrixXd::Zero(n + 1, n + 1);
        augmented.topLeftCorner(n, n) = eval.tangent;
        augmented.topRightCorner(n, 1) = eval.load_column;
        augmented.bottomLeftCorner(1, n) =
            eval.constraint_gradient.transpose();
        augmented(n, n) = eval.constraint_load_derivative;

        Eigen::VectorXd rhs(n + 1);
        rhs.head(n) = -eval.residual;
        rhs[n] = -eval.constraint;

        const Eigen::FullPivLU<Eigen::MatrixXd> lu(augmented);
        if (!lu.isInvertible()) {
            result.status =
                BorderedMixedControlNewtonStatus::singular_augmented_system;
            result.iterations = iter;
            result.residual_norm = residual_norm;
            result.constraint_abs = constraint_abs;
            return result;
        }

        const Eigen::VectorXd step = lu.solve(rhs);
        const Eigen::VectorXd du = step.head(n);
        const double dlambda = step[n];
        const double correction_norm =
            std::hypot(du.norm(), std::abs(dlambda));

        double alpha = 1.0;
        if (settings.line_search_enabled) {
            const double old_merit = detail::bordered_merit(eval);
            bool accepted = false;
            for (int cutback = 0;
                 cutback <= settings.max_line_search_cutbacks;
                 ++cutback)
            {
                BorderedMixedControlState trial = result.state;
                trial.unknowns += alpha * du;
                trial.load_parameter += alpha * dlambda;
                const auto trial_eval = evaluator(trial);
                if (detail::bordered_eval_has_consistent_dimensions(
                        trial_eval, n))
                {
                    const double trial_merit =
                        detail::bordered_merit(trial_eval);
                    const double target =
                        (1.0 - settings.merit_decrease * alpha) *
                        old_merit;
                    if (trial_merit <= target || trial_merit <= old_merit) {
                        accepted = true;
                        break;
                    }
                }

                alpha *= settings.line_search_cutback_factor;
                if (alpha < settings.line_search_min_alpha) {
                    break;
                }
            }

            if (!accepted) {
                result.status =
                    BorderedMixedControlNewtonStatus::line_search_failed;
                result.iterations = iter;
                result.residual_norm = residual_norm;
                result.constraint_abs = constraint_abs;
                result.correction_norm = correction_norm;
                return result;
            }
        }

        result.state.unknowns += alpha * du;
        result.state.load_parameter += alpha * dlambda;
        result.records.push_back(
            BorderedMixedControlNewtonIterationRecord{
                .iteration = iter + 1,
                .residual_norm = residual_norm,
                .constraint_abs = constraint_abs,
                .correction_norm = correction_norm,
                .load_correction_abs = std::abs(dlambda),
                .line_search_alpha = alpha});
        result.iterations = iter + 1;
        result.residual_norm = residual_norm;
        result.constraint_abs = constraint_abs;
        result.correction_norm = correction_norm;

        if (correction_norm <= settings.correction_tolerance &&
            residual_norm <= 10.0 * settings.residual_tolerance &&
            constraint_abs <= 10.0 * settings.constraint_tolerance)
        {
            result.status = BorderedMixedControlNewtonStatus::converged;
            return result;
        }
    }

    const auto eval = evaluator(result.state);
    if (detail::bordered_eval_has_consistent_dimensions(
            eval, result.state.unknowns.size()))
    {
        result.residual_norm = eval.residual.norm();
        result.constraint_abs = std::abs(eval.constraint);
        if (result.residual_norm <= settings.residual_tolerance &&
            result.constraint_abs <= settings.constraint_tolerance)
        {
            result.status = BorderedMixedControlNewtonStatus::converged;
            return result;
        }
    }
    result.status = BorderedMixedControlNewtonStatus::max_iterations;
    return result;
}

} // namespace fall_n

#endif // FALL_N_SRC_ANALYSIS_BORDERED_MIXED_CONTROL_NEWTON_HH
