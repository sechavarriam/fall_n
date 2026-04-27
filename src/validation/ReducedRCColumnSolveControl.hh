#ifndef FALL_N_REDUCED_RC_COLUMN_SOLVE_CONTROL_HH
#define FALL_N_REDUCED_RC_COLUMN_SOLVE_CONTROL_HH

#include "src/analysis/NonlinearSolvePolicy.hh"

#include <petscksp.h>
#include <petscpc.h>

#include <vector>

namespace fall_n::validation_reboot {

[[nodiscard]] inline fall_n::NonlinearSmallResidualAcceptancePolicy
make_reduced_rc_small_residual_linesearch_acceptance()
{
    // Keep this band tight: it exists to recover micro-step preload slices
    // where PETSc's safeguard exits slightly above the formal atol even though
    // the final residual is already in the 1e-8 range and the step is
    // physically negligible. We intentionally do not widen this enough to hide
    // materially unresolved nonlinear states.
    return {
        .absolute_function_norm_threshold = 0.0,
        .profile_atol_multiplier = 256.0,
        .accept_diverged_line_search = true,
        .accept_diverged_tr_delta = false,
        .accept_diverged_dtol = false,
        .accept_diverged_max_it = true};
}

[[nodiscard]] inline fall_n::NonlinearSmallResidualAcceptancePolicy
make_reduced_rc_small_residual_trust_region_acceptance()
{
    return {
        .absolute_function_norm_threshold = 0.0,
        .profile_atol_multiplier = 256.0,
        .accept_diverged_line_search = false,
        .accept_diverged_tr_delta = true,
        .accept_diverged_dtol = false,
        .accept_diverged_max_it = true};
}

enum class ReducedRCColumnContinuationKind {
    monolithic_incremental_displacement_control,
    segmented_incremental_displacement_control,
    reversal_guarded_incremental_displacement_control,
    arc_length_continuation_candidate
};

[[nodiscard]] constexpr const char*
to_string(ReducedRCColumnContinuationKind kind) noexcept
{
    switch (kind) {
        case ReducedRCColumnContinuationKind::
            monolithic_incremental_displacement_control:
            return "monolithic_incremental_displacement_control";
        case ReducedRCColumnContinuationKind::
            segmented_incremental_displacement_control:
            return "segmented_incremental_displacement_control";
        case ReducedRCColumnContinuationKind::
            reversal_guarded_incremental_displacement_control:
            return "reversal_guarded_incremental_displacement_control";
        case ReducedRCColumnContinuationKind::arc_length_continuation_candidate:
            return "arc_length_continuation_candidate";
    }
    return "unknown_reduced_rc_column_continuation_kind";
}

enum class ReducedRCColumnSolverPolicyKind {
    canonical_newton_profile_cascade,
    newton_basic_only,
    newton_backtracking_only,
    newton_l2_only,
    newton_l2_lu_symbolic_reuse_only,
    newton_l2_gmres_ilu1_only,
    newton_trust_region_only,
    newton_trust_region_dogleg_only,
    quasi_newton_only,
    nonlinear_gmres_only,
    nonlinear_conjugate_gradient_only,
    anderson_acceleration_only,
    nonlinear_richardson_only
};

[[nodiscard]] constexpr const char*
to_string(ReducedRCColumnSolverPolicyKind kind) noexcept
{
    switch (kind) {
        case ReducedRCColumnSolverPolicyKind::canonical_newton_profile_cascade:
            return "canonical_newton_profile_cascade";
        case ReducedRCColumnSolverPolicyKind::newton_basic_only:
            return "newton_basic_only";
        case ReducedRCColumnSolverPolicyKind::newton_backtracking_only:
            return "newton_backtracking_only";
        case ReducedRCColumnSolverPolicyKind::newton_l2_only:
            return "newton_l2_only";
        case ReducedRCColumnSolverPolicyKind::newton_l2_lu_symbolic_reuse_only:
            return "newton_l2_lu_symbolic_reuse_only";
        case ReducedRCColumnSolverPolicyKind::newton_l2_gmres_ilu1_only:
            return "newton_l2_gmres_ilu1_only";
        case ReducedRCColumnSolverPolicyKind::newton_trust_region_only:
            return "newton_trust_region_only";
        case ReducedRCColumnSolverPolicyKind::newton_trust_region_dogleg_only:
            return "newton_trust_region_dogleg_only";
        case ReducedRCColumnSolverPolicyKind::quasi_newton_only:
            return "quasi_newton_only";
        case ReducedRCColumnSolverPolicyKind::nonlinear_gmres_only:
            return "nonlinear_gmres_only";
        case ReducedRCColumnSolverPolicyKind::nonlinear_conjugate_gradient_only:
            return "nonlinear_conjugate_gradient_only";
        case ReducedRCColumnSolverPolicyKind::anderson_acceleration_only:
            return "anderson_acceleration_only";
        case ReducedRCColumnSolverPolicyKind::nonlinear_richardson_only:
            return "nonlinear_richardson_only";
    }
    return "unknown_reduced_rc_column_solver_policy_kind";
}

enum class ReducedRCColumnPredictorPolicyKind {
    current_state_only,
    secant_extrapolation,
    adaptive_secant_extrapolation,
    linearized_equilibrium_seed,
    secant_with_linearized_fallback
};

[[nodiscard]] constexpr const char*
to_string(ReducedRCColumnPredictorPolicyKind kind) noexcept
{
    switch (kind) {
        case ReducedRCColumnPredictorPolicyKind::current_state_only:
            return "current_state_only";
        case ReducedRCColumnPredictorPolicyKind::secant_extrapolation:
            return "secant_extrapolation";
        case ReducedRCColumnPredictorPolicyKind::adaptive_secant_extrapolation:
            return "adaptive_secant_extrapolation";
        case ReducedRCColumnPredictorPolicyKind::linearized_equilibrium_seed:
            return "linearized_equilibrium_seed";
        case ReducedRCColumnPredictorPolicyKind::secant_with_linearized_fallback:
            return "secant_with_linearized_fallback";
    }
    return "unknown_reduced_rc_column_predictor_policy_kind";
}

[[nodiscard]] inline fall_n::NonlinearSolveProfile
make_reduced_rc_newton_basic_profile()
{
    auto profile = fall_n::make_newton_basic_profile();
    profile.rtol = 1.0e-8;
    profile.atol = 1.0e-10;
    profile.stol = 1.0e-12;
    profile.max_iterations = 100;
    profile.max_function_evaluations = PETSC_DEFAULT;
    profile.ksp_type = KSPPREONLY;
    profile.pc_type = PCLU;
    profile.small_residual_acceptance =
        make_reduced_rc_small_residual_linesearch_acceptance();
    return profile;
}

[[nodiscard]] inline fall_n::NonlinearSolveProfile
make_reduced_rc_newton_backtracking_profile()
{
    auto profile = fall_n::make_newton_backtracking_profile();
    profile.rtol = 1.0e-8;
    profile.atol = 1.0e-10;
    profile.stol = 1.0e-12;
    profile.max_iterations = 100;
    profile.max_function_evaluations = PETSC_DEFAULT;
    profile.ksp_type = KSPPREONLY;
    profile.pc_type = PCLU;
    profile.small_residual_acceptance =
        make_reduced_rc_small_residual_linesearch_acceptance();
    return profile;
}

[[nodiscard]] inline fall_n::NonlinearSolveProfile
make_reduced_rc_newton_l2_profile()
{
    auto profile = fall_n::make_newton_l2_profile();
    profile.rtol = 1.0e-8;
    profile.atol = 1.0e-10;
    profile.stol = 1.0e-12;
    profile.max_iterations = 120;
    profile.max_function_evaluations = PETSC_DEFAULT;
    profile.ksp_type = KSPPREONLY;
    profile.pc_type = PCLU;
    profile.small_residual_acceptance =
        make_reduced_rc_small_residual_linesearch_acceptance();
    return profile;
}

[[nodiscard]] inline fall_n::NonlinearSolveProfile
make_reduced_rc_newton_l2_lu_symbolic_reuse_profile()
{
    auto profile = make_reduced_rc_newton_l2_profile();
    profile.label = "newton_l2_lu_symbolic_reuse";
    profile.linear_tuning.factor_mat_ordering_type = MATORDERINGRCM;
    profile.linear_tuning.factor_reuse_ordering = true;
    profile.linear_tuning.factor_reuse_fill = true;
    return profile;
}

[[nodiscard]] inline fall_n::NonlinearSolveProfile
make_reduced_rc_newton_l2_gmres_ilu1_profile()
{
    auto profile = make_reduced_rc_newton_l2_profile();
    profile.label = "newton_l2_gmres_ilu1";
    profile.ksp_type = KSPGMRES;
    profile.pc_type = PCILU;
    profile.linear_tuning.ksp_rtol = 1.0e-6;
    profile.linear_tuning.ksp_atol = PETSC_DETERMINE;
    // The penalty-coupled continuum tangent can be badly scaled during
    // cracking/preload micro-steps.  Do not let the linear residual growth
    // test terminate GMRES before the nonlinear globalization decides whether
    // the step is physically acceptable.
    profile.linear_tuning.ksp_dtol = PETSC_UNLIMITED;
    profile.linear_tuning.ksp_max_iterations = 1000;
    profile.linear_tuning.factor_mat_ordering_type = MATORDERINGRCM;
    profile.linear_tuning.factor_levels = 1;
    profile.linear_tuning.factor_reuse_ordering = true;
    profile.linear_tuning.factor_reuse_fill = true;
    return profile;
}

[[nodiscard]] inline fall_n::NonlinearSolveProfile
make_reduced_rc_newton_trust_region_profile()
{
    auto profile = fall_n::make_newton_trust_region_profile();
    profile.rtol = 1.0e-8;
    profile.atol = 1.0e-10;
    profile.stol = 1.0e-12;
    profile.max_iterations = 140;
    profile.max_function_evaluations = PETSC_DEFAULT;
    profile.ksp_type = KSPPREONLY;
    profile.pc_type = PCLU;
    profile.small_residual_acceptance =
        make_reduced_rc_small_residual_trust_region_acceptance();
    return profile;
}

[[nodiscard]] inline fall_n::NonlinearSolveProfile
make_reduced_rc_newton_trust_region_dogleg_profile()
{
    auto profile = fall_n::make_newton_trust_region_dogleg_profile();
    profile.rtol = 1.0e-8;
    profile.atol = 1.0e-10;
    profile.stol = 1.0e-12;
    profile.max_iterations = 160;
    profile.max_function_evaluations = PETSC_DEFAULT;
    profile.ksp_type = KSPPREONLY;
    profile.pc_type = PCLU;
    profile.small_residual_acceptance =
        make_reduced_rc_small_residual_trust_region_acceptance();
    return profile;
}

[[nodiscard]] inline fall_n::NonlinearSolveProfile
make_reduced_rc_quasi_newton_profile()
{
    auto profile = fall_n::make_quasi_newton_profile();
    profile.rtol = 1.0e-8;
    profile.atol = 1.0e-10;
    profile.stol = 1.0e-12;
    profile.max_iterations = 180;
    profile.max_function_evaluations = PETSC_DEFAULT;
    profile.ksp_type = KSPPREONLY;
    profile.pc_type = PCLU;
    return profile;
}

[[nodiscard]] inline fall_n::NonlinearSolveProfile
make_reduced_rc_nonlinear_gmres_profile()
{
    auto profile = fall_n::make_nonlinear_gmres_profile();
    profile.rtol = 1.0e-8;
    profile.atol = 1.0e-10;
    profile.stol = 1.0e-12;
    profile.max_iterations = 220;
    profile.max_function_evaluations = PETSC_DEFAULT;
    profile.ksp_type = KSPPREONLY;
    profile.pc_type = PCLU;
    return profile;
}

[[nodiscard]] inline fall_n::NonlinearSolveProfile
make_reduced_rc_nonlinear_conjugate_gradient_profile()
{
    auto profile = fall_n::make_nonlinear_conjugate_gradient_profile();
    profile.rtol = 1.0e-8;
    profile.atol = 1.0e-10;
    profile.stol = 1.0e-12;
    profile.max_iterations = 220;
    profile.max_function_evaluations = PETSC_DEFAULT;
    profile.ksp_type = KSPPREONLY;
    profile.pc_type = PCLU;
    return profile;
}

[[nodiscard]] inline fall_n::NonlinearSolveProfile
make_reduced_rc_anderson_profile()
{
    auto profile = fall_n::make_anderson_profile();
    profile.rtol = 1.0e-8;
    profile.atol = 1.0e-10;
    profile.stol = 1.0e-12;
    profile.max_iterations = 220;
    profile.max_function_evaluations = PETSC_DEFAULT;
    profile.ksp_type = KSPPREONLY;
    profile.pc_type = PCLU;
    return profile;
}

[[nodiscard]] inline fall_n::NonlinearSolveProfile
make_reduced_rc_nonlinear_richardson_profile()
{
    auto profile = fall_n::make_nonlinear_richardson_profile();
    profile.rtol = 1.0e-8;
    profile.atol = 1.0e-10;
    profile.stol = 1.0e-12;
    profile.max_iterations = 240;
    profile.max_function_evaluations = PETSC_DEFAULT;
    profile.ksp_type = KSPPREONLY;
    profile.pc_type = PCLU;
    return profile;
}

[[nodiscard]] inline std::vector<fall_n::NonlinearSolveProfile>
make_reduced_rc_validation_solve_profiles(
    ReducedRCColumnSolverPolicyKind policy_kind)
{
    switch (policy_kind) {
        case ReducedRCColumnSolverPolicyKind::canonical_newton_profile_cascade:
            return {
                make_reduced_rc_newton_basic_profile(),
                make_reduced_rc_newton_backtracking_profile(),
                make_reduced_rc_newton_l2_profile(),
                make_reduced_rc_newton_trust_region_profile()};
        case ReducedRCColumnSolverPolicyKind::newton_basic_only:
            return {make_reduced_rc_newton_basic_profile()};
        case ReducedRCColumnSolverPolicyKind::newton_backtracking_only:
            return {make_reduced_rc_newton_backtracking_profile()};
        case ReducedRCColumnSolverPolicyKind::newton_l2_only:
            return {make_reduced_rc_newton_l2_profile()};
        case ReducedRCColumnSolverPolicyKind::
            newton_l2_lu_symbolic_reuse_only:
            return {make_reduced_rc_newton_l2_lu_symbolic_reuse_profile()};
        case ReducedRCColumnSolverPolicyKind::newton_l2_gmres_ilu1_only:
            return {make_reduced_rc_newton_l2_gmres_ilu1_profile()};
        case ReducedRCColumnSolverPolicyKind::newton_trust_region_only:
            return {make_reduced_rc_newton_trust_region_profile()};
        case ReducedRCColumnSolverPolicyKind::newton_trust_region_dogleg_only:
            return {make_reduced_rc_newton_trust_region_dogleg_profile()};
        case ReducedRCColumnSolverPolicyKind::quasi_newton_only:
            return {make_reduced_rc_quasi_newton_profile()};
        case ReducedRCColumnSolverPolicyKind::nonlinear_gmres_only:
            return {make_reduced_rc_nonlinear_gmres_profile()};
        case ReducedRCColumnSolverPolicyKind::
            nonlinear_conjugate_gradient_only:
            return {make_reduced_rc_nonlinear_conjugate_gradient_profile()};
        case ReducedRCColumnSolverPolicyKind::anderson_acceleration_only:
            return {make_reduced_rc_anderson_profile()};
        case ReducedRCColumnSolverPolicyKind::nonlinear_richardson_only:
            return {make_reduced_rc_nonlinear_richardson_profile()};
    }

    return {make_reduced_rc_newton_backtracking_profile()};
}

[[nodiscard]] inline std::vector<fall_n::NonlinearSolveProfile>
override_reduced_rc_divergence_tolerance(
    std::vector<fall_n::NonlinearSolveProfile> profiles,
    double divergence_tolerance)
{
    for (auto& profile : profiles) {
        profile.divergence_tolerance = divergence_tolerance;
    }
    return profiles;
}

[[nodiscard]] inline fall_n::IncrementPredictorSettings
make_reduced_rc_increment_predictor_settings(
    ReducedRCColumnPredictorPolicyKind policy_kind)
{
    using Settings = fall_n::IncrementPredictorSettings;
    using Kind = fall_n::IncrementPredictorKind;

    switch (policy_kind) {
        case ReducedRCColumnPredictorPolicyKind::current_state_only:
            return Settings{
                .enabled = false,
                .kind = Kind::current_state,
            };
        case ReducedRCColumnPredictorPolicyKind::secant_extrapolation:
            return Settings{
                .enabled = true,
                .kind = Kind::secant_extrapolation,
                .max_scale_factor = 1.5,
                .max_relative_increment_norm = 1.5,
                .difficult_newton_iterations = 12,
                .disable_during_bisection = false,
                .disable_after_cutback = false,
            };
        case ReducedRCColumnPredictorPolicyKind::adaptive_secant_extrapolation:
            return Settings{
                .enabled = true,
                .kind = Kind::adaptive_secant_extrapolation,
                .max_scale_factor = 1.25,
                .max_relative_increment_norm = 1.25,
                .difficult_newton_iterations = 10,
                .disable_during_bisection = true,
                .disable_after_cutback = true,
            };
        case ReducedRCColumnPredictorPolicyKind::linearized_equilibrium_seed:
            return Settings{
                .enabled = true,
                .kind = Kind::linearized_equilibrium_seed,
                .max_scale_factor = 1.0,
                .max_relative_increment_norm = 1.0,
                .difficult_newton_iterations = 10,
                .disable_during_bisection = false,
                .disable_after_cutback = false,
            };
        case ReducedRCColumnPredictorPolicyKind::secant_with_linearized_fallback:
            return Settings{
                .enabled = true,
                .kind = Kind::secant_with_linearized_fallback,
                .max_scale_factor = 1.5,
                .max_relative_increment_norm = 1.5,
                .difficult_newton_iterations = 12,
                .disable_during_bisection = false,
                .disable_after_cutback = false,
            };
    }

    return Settings{};
}

} // namespace fall_n::validation_reboot

#endif // FALL_N_REDUCED_RC_COLUMN_SOLVE_CONTROL_HH
