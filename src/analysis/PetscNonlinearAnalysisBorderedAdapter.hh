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
    // Orden de la diferencia finita de la columna de carga dR/dλ:
    //  1 = adelantada O(h); 2 = central O(h^2); 4 = central de 5 puntos O(h^4).
    //  Un orden mayor da un dR/dλ más fiel -> mejor sistema bordered y mejor
    //  predictor tangente, a costa de más evaluaciones de residuo por columna.
    int control_column_order{2};
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

    // Orden efectivo: control_column_order manda; si quedó en el legado (2) se
    //  respeta use_central_difference (false -> orden 1 adelantada).
    int order = settings.control_column_order;
    if (order != 1 && order != 2 && order != 4) order = 2;
    if (order == 2 && !settings.use_central_difference) order = 1;

    auto eval_at = [&](double pp, Vec out) {
        analysis.revert_trial_state();
        analysis.apply_incremental_control_parameter(pp);
        analysis.evaluate_residual_at(unknowns, out);
    };

    if (order == 4) {
        // Central de 5 puntos O(h^4):
        //  (-R(p+2h) + 8R(p+h) - 8R(p-h) + R(p-2h)) / (12h)
        auto rpp = analysis.create_global_vector();
        auto rp = analysis.create_global_vector();
        auto rm = analysis.create_global_vector();
        auto rmm = analysis.create_global_vector();
        eval_at(p + 2.0 * h, rpp.get());
        eval_at(p + h, rp.get());
        eval_at(p - h, rm.get());
        eval_at(p - 2.0 * h, rmm.get());
        FALL_N_PETSC_CHECK(VecSet(column.get(), 0.0));
        FALL_N_PETSC_CHECK(VecAXPY(column.get(), -1.0, rpp.get()));
        FALL_N_PETSC_CHECK(VecAXPY(column.get(), 8.0, rp.get()));
        FALL_N_PETSC_CHECK(VecAXPY(column.get(), -8.0, rm.get()));
        FALL_N_PETSC_CHECK(VecAXPY(column.get(), 1.0, rmm.get()));
        FALL_N_PETSC_CHECK(VecScale(column.get(), 1.0 / (12.0 * h)));
    } else if (order == 2) {
        auto r_plus = analysis.create_global_vector();
        auto r_minus = analysis.create_global_vector();
        eval_at(p + h, r_plus.get());
        eval_at(p - h, r_minus.get());
        FALL_N_PETSC_CHECK(VecCopy(r_plus.get(), column.get()));
        FALL_N_PETSC_CHECK(VecAXPY(column.get(), -1.0, r_minus.get()));
        FALL_N_PETSC_CHECK(VecScale(column.get(), 0.5 / h));
    } else {  // order == 1, adelantada
        auto r_plus = analysis.create_global_vector();
        eval_at(p + h, r_plus.get());
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

// Spherical (Crisfield) ARC-LENGTH continuation evaluation.
//
//  Unlike fixed-control (which degenerates to standard Newton and stalls at
//  limit points), the arc-length constraint lets the control parameter lambda
//  ADJUST along the equilibrium path, so the continuation traverses limit points
//  and snapbacks while staying on the physical branch.  About the last converged
//  point (u0, lambda0):
//     c      = ||u - u0||^2 + psi^2 (lambda - lambda0)^2 - delta_s^2
//     dc/du  = 2 (u - u0)
//     dc/dl  = 2 psi^2 (lambda - lambda0)
//  psi (load_scaling) weights the load term; psi=0 gives cylindrical arc-length.
template <typename AnalysisT>
[[nodiscard]] PetscBorderedMixedControlEvaluation
make_arc_length_petsc_bordered_evaluation(
    AnalysisT& analysis,
    const PetscBorderedMixedControlState& state,
    Vec reference_unknowns,
    double reference_load,
    double arc_length,
    double load_scaling,
    PetscNonlinearAnalysisBorderedAdapterSettings settings = {},
    double regularization_mu_frac = 0.0)
{
    PetscBorderedMixedControlEvaluation eval;
    eval.residual = analysis.create_global_vector();
    eval.tangent = analysis.create_tangent_matrix();
    eval.constraint_gradient = analysis.create_global_vector();

    analysis.revert_trial_state();
    analysis.apply_incremental_control_parameter(state.load_parameter);
    analysis.evaluate_residual_at(state.unknowns, eval.residual.get());
    analysis.evaluate_tangent_at(state.unknowns, eval.tangent.get());
    // Regularización Levenberg-Marquardt del bloque K del sistema bordered:
    //  K <- K + mu*I con mu = mu_frac*||diag(K)||.  Vuelve resoluble el corrector
    //  bordered en el punto límite (donde K es casi-singular y el arc-length
    //  puro se atasca), combinando el trazado paramétrico del arc-length con la
    //  regularización que cruza el punto límite.
    if (regularization_mu_frac > 0.0) {
        petsc::OwnedVec diag = analysis.create_global_vector();
        FALL_N_PETSC_CHECK(MatGetDiagonal(eval.tangent.get(), diag.get()));
        PetscReal dn = 0.0;
        FALL_N_PETSC_CHECK(VecNorm(diag.get(), NORM_2, &dn));
        const double mu =
            regularization_mu_frac * std::max<double>(static_cast<double>(dn),
                                                      1.0e-30);
        FALL_N_PETSC_CHECK(MatShift(eval.tangent.get(), mu));
    }
    eval.load_column = detail::finite_difference_control_column(
        analysis,
        state.unknowns,
        eval.residual.get(),
        state.load_parameter,
        settings);

    // du = u - u0  (stored in constraint_gradient, then scaled by 2)
    FALL_N_PETSC_CHECK(
        VecWAXPY(eval.constraint_gradient.get(), -1.0,
                 reference_unknowns, state.unknowns));
    PetscReal du_norm = 0.0;
    FALL_N_PETSC_CHECK(
        VecNorm(eval.constraint_gradient.get(), NORM_2, &du_norm));
    const double dl = state.load_parameter - reference_load;
    eval.constraint = static_cast<double>(du_norm) * static_cast<double>(du_norm)
                      + load_scaling * load_scaling * dl * dl
                      - arc_length * arc_length;
    FALL_N_PETSC_CHECK(VecScale(eval.constraint_gradient.get(), 2.0));
    eval.constraint_load_derivative = 2.0 * load_scaling * load_scaling * dl;
    return eval;
}

} // namespace fall_n

#endif // FALL_N_SRC_ANALYSIS_PETSC_NONLINEAR_ANALYSIS_BORDERED_ADAPTER_HH
