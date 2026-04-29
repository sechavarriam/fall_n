#ifndef FALL_N_SRC_ANALYSIS_PETSC_NONLINEAR_ANALYSIS_BORDERED_ADAPTER_HH
#define FALL_N_SRC_ANALYSIS_PETSC_NONLINEAR_ANALYSIS_BORDERED_ADAPTER_HH

// =============================================================================
//  PetscNonlinearAnalysisBorderedAdapter
// =============================================================================
//
//  Thin bridge between NonlinearAnalysis and the PETSc bordered mixed-control
//  Newton backend.  The important design choice is that this adapter does not
//  reassemble physics on its own.  It asks NonlinearAnalysis to:
//
//      1. apply the already-registered incremental control law at p,
//      2. evaluate the same residual R(u,p) used by SNES,
//      3. evaluate the same tangent K(u,p) used by SNES,
//      4. approximate r_lambda = dR/dp by a finite-difference perturbation.
//
//  The finite-difference control column is intentionally explicit.  It is the
//  correct validation seam while the XFEM benchmark is being stabilized; a
//  future production implementation can replace only the load/control-column
//  provider with an analytic one without changing the bordered Newton algebra.
//
// =============================================================================

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "NLAnalysis.hh"
#include "PetscBorderedMixedControlNewton.hh"

namespace fall_n {

struct PetscNonlinearAnalysisBorderedAdapterSettings {
    double control_column_step{1.0e-6};
    double minimum_control_column_step{1.0e-10};
    bool use_central_difference{true};
};

namespace detail {

[[nodiscard]] inline double bounded_control_perturbation(
    double p,
    PetscNonlinearAnalysisBorderedAdapterSettings settings)
{
    const double scale = std::max({1.0, std::abs(p), settings.control_column_step});
    const double h = std::max(
        std::abs(settings.control_column_step) * scale,
        std::abs(settings.minimum_control_column_step));
    if (!std::isfinite(h) || h <= 0.0) {
        throw std::invalid_argument(
            "Invalid finite-difference control perturbation for bordered adapter");
    }
    return h;
}

template <typename AnalysisT>
[[nodiscard]] petsc::OwnedVec finite_difference_control_column(
    AnalysisT& analysis,
    Vec unknowns,
    Vec residual_at_p,
    double p,
    PetscNonlinearAnalysisBorderedAdapterSettings settings)
{
    const double h = bounded_control_perturbation(p, settings);
    auto column = analysis.create_global_vector();
    auto r_plus = analysis.create_global_vector();

    if (settings.use_central_difference) {
        auto r_minus = analysis.create_global_vector();
        analysis.revert_trial_state();
        analysis.apply_incremental_control_parameter(p + h);
        analysis.evaluate_residual_at(unknowns, r_plus.get());
        analysis.revert_trial_state();
        analysis.apply_incremental_control_parameter(p - h);
        analysis.evaluate_residual_at(unknowns, r_minus.get());
        FALL_N_PETSC_CHECK(VecCopy(r_plus.get(), column.get()));
        FALL_N_PETSC_CHECK(VecAXPY(column.get(), -1.0, r_minus.get()));
        FALL_N_PETSC_CHECK(VecScale(column.get(), 0.5 / h));
    } else {
        analysis.revert_trial_state();
        analysis.apply_incremental_control_parameter(p + h);
        analysis.evaluate_residual_at(unknowns, r_plus.get());
        FALL_N_PETSC_CHECK(VecCopy(r_plus.get(), column.get()));
        FALL_N_PETSC_CHECK(VecAXPY(column.get(), -1.0, residual_at_p));
        FALL_N_PETSC_CHECK(VecScale(column.get(), 1.0 / h));
    }

    analysis.revert_trial_state();
    analysis.apply_incremental_control_parameter(p);
    return column;
}

} // namespace detail

template <typename AnalysisT>
[[nodiscard]] PetscBorderedMixedControlEvaluation
make_fixed_control_petsc_bordered_evaluation(
    AnalysisT& analysis,
    const PetscBorderedMixedControlState& state,
    double target_control_parameter,
    PetscNonlinearAnalysisBorderedAdapterSettings settings = {})
{
    PetscBorderedMixedControlEvaluation eval;
    eval.residual = analysis.create_global_vector();
    eval.tangent = analysis.create_tangent_matrix();
    eval.constraint_gradient = analysis.create_global_vector();

    analysis.revert_trial_state();
    analysis.apply_incremental_control_parameter(state.load_parameter);
    analysis.evaluate_residual_at(state.unknowns, eval.residual.get());
    analysis.evaluate_tangent_at(state.unknowns, eval.tangent.get());
    eval.load_column = detail::finite_difference_control_column(
        analysis,
        state.unknowns,
        eval.residual.get(),
        state.load_parameter,
        settings);

    // Fixed-control continuation is the minimal integration check for the
    // bordered algebra: c(lambda) = lambda - lambda_target.
    FALL_N_PETSC_CHECK(VecSet(eval.constraint_gradient.get(), 0.0));
    eval.constraint = state.load_parameter - target_control_parameter;
    eval.constraint_load_derivative = 1.0;
    return eval;
}

} // namespace fall_n

#endif // FALL_N_SRC_ANALYSIS_PETSC_NONLINEAR_ANALYSIS_BORDERED_ADAPTER_HH
