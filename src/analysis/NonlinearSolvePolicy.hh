#ifndef FALL_N_SRC_ANALYSIS_NONLINEAR_SOLVE_POLICY_HH
#define FALL_N_SRC_ANALYSIS_NONLINEAR_SOLVE_POLICY_HH

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <petscsnes.h>

#include "../petsc/PetscRaii.hh"

namespace fall_n {

// Shared nonlinear-solver declaration surface for both static incremental
// analyses and PETSc-TS-driven dynamic solves.
//
// The enum is intentionally broader than the currently promoted baselines:
// methods that benchmark poorly today still belong here if they are real,
// audited extension routes. This keeps future integrations such as
// TAO/Levenberg-Marquardt-style damped least-squares solves on the same typed
// declaration seam instead of forcing driver-specific ad hoc switches.

enum class NonlinearSolveMethodKind {
    newton_line_search,
    newton_trust_region,
    newton_trust_region_dogleg,
    quasi_newton,
    nonlinear_gmres,
    nonlinear_conjugate_gradient,
    anderson_acceleration,
    nonlinear_richardson
};

enum class NonlinearLineSearchKind {
    none,
    basic,
    backtracking,
    l2,
    secant,
    ncg_linear,
    critical_point,
    error_oriented
};

[[nodiscard]] constexpr std::string_view
to_string(NonlinearSolveMethodKind kind) noexcept
{
    switch (kind) {
        case NonlinearSolveMethodKind::newton_line_search:
            return "newton_line_search";
        case NonlinearSolveMethodKind::newton_trust_region:
            return "newton_trust_region";
        case NonlinearSolveMethodKind::newton_trust_region_dogleg:
            return "newton_trust_region_dogleg";
        case NonlinearSolveMethodKind::quasi_newton:
            return "quasi_newton";
        case NonlinearSolveMethodKind::nonlinear_gmres:
            return "nonlinear_gmres";
        case NonlinearSolveMethodKind::nonlinear_conjugate_gradient:
            return "nonlinear_conjugate_gradient";
        case NonlinearSolveMethodKind::anderson_acceleration:
            return "anderson_acceleration";
        case NonlinearSolveMethodKind::nonlinear_richardson:
            return "nonlinear_richardson";
    }
    return "unknown_nonlinear_solve_method_kind";
}

[[nodiscard]] constexpr std::string_view
to_string(NonlinearLineSearchKind kind) noexcept
{
    switch (kind) {
        case NonlinearLineSearchKind::none:
            return "none";
        case NonlinearLineSearchKind::basic:
            return "basic";
        case NonlinearLineSearchKind::backtracking:
            return "backtracking";
        case NonlinearLineSearchKind::l2:
            return "l2";
        case NonlinearLineSearchKind::secant:
            return "secant";
        case NonlinearLineSearchKind::ncg_linear:
            return "ncg_linear";
        case NonlinearLineSearchKind::critical_point:
            return "critical_point";
        case NonlinearLineSearchKind::error_oriented:
            return "error_oriented";
    }
    return "unknown_nonlinear_line_search_kind";
}

[[nodiscard]] constexpr const char*
to_petsc_snes_type(NonlinearSolveMethodKind kind) noexcept
{
    switch (kind) {
        case NonlinearSolveMethodKind::newton_line_search:
            return SNESNEWTONLS;
        case NonlinearSolveMethodKind::newton_trust_region:
            return SNESNEWTONTR;
        case NonlinearSolveMethodKind::newton_trust_region_dogleg:
            return SNESNEWTONTRDC;
        case NonlinearSolveMethodKind::quasi_newton:
            return SNESQN;
        case NonlinearSolveMethodKind::nonlinear_gmres:
            return SNESNGMRES;
        case NonlinearSolveMethodKind::nonlinear_conjugate_gradient:
            return SNESNCG;
        case NonlinearSolveMethodKind::anderson_acceleration:
            return SNESANDERSON;
        case NonlinearSolveMethodKind::nonlinear_richardson:
            return SNESNRICHARDSON;
    }
    return SNESNEWTONLS;
}

[[nodiscard]] constexpr const char*
to_petsc_linesearch_type(NonlinearLineSearchKind kind) noexcept
{
    switch (kind) {
        case NonlinearLineSearchKind::none:
            return "";
        case NonlinearLineSearchKind::basic:
            return SNESLINESEARCHBASIC;
        case NonlinearLineSearchKind::backtracking:
            return SNESLINESEARCHBT;
        case NonlinearLineSearchKind::l2:
            return SNESLINESEARCHL2;
        case NonlinearLineSearchKind::secant:
            return SNESLINESEARCHSECANT;
        case NonlinearLineSearchKind::ncg_linear:
            return SNESLINESEARCHNCGLINEAR;
        case NonlinearLineSearchKind::critical_point:
            return SNESLINESEARCHCP;
        case NonlinearLineSearchKind::error_oriented:
            return SNESLINESEARCHNLEQERR;
    }
    return "";
}

enum class IncrementPredictorKind {
    current_state,
    secant_extrapolation,
    adaptive_secant_extrapolation,
    linearized_equilibrium_seed,
    secant_with_linearized_fallback
};

struct IncrementPredictorSettings {
    bool enabled{false};
    IncrementPredictorKind kind{IncrementPredictorKind::current_state};
    double max_scale_factor{1.5};
    double max_relative_increment_norm{1.5};
    int difficult_newton_iterations{12};
    bool disable_during_bisection{true};
    bool disable_after_cutback{true};
};

struct NonlinearSmallResidualAcceptancePolicy {
    // Optional post-SNES acceptance seam for cases where PETSc exits with a
    // negative reason (typically line-search/trust-region safeguards) even
    // though the final nonlinear residual is already negligible for the
    // declared benchmark tolerances.
    //
    // This keeps the decision typed, auditable, and reusable across
    // validation slices instead of hiding it in ad hoc options-database
    // tweaks or benchmark-specific "if residual < ..." branches.
    double absolute_function_norm_threshold{0.0};
    double profile_atol_multiplier{0.0};
    bool accept_diverged_line_search{false};
    bool accept_diverged_tr_delta{false};
    bool accept_diverged_dtol{false};
    bool accept_diverged_max_it{false};

    [[nodiscard]] bool enabled() const noexcept
    {
        return absolute_function_norm_threshold > 0.0 ||
               profile_atol_multiplier > 0.0;
    }

    [[nodiscard]] bool accepts_reason(SNESConvergedReason reason) const noexcept
    {
        switch (reason) {
            case SNES_DIVERGED_LINE_SEARCH:
                return accept_diverged_line_search;
            case SNES_DIVERGED_TR_DELTA:
                return accept_diverged_tr_delta;
            case SNES_DIVERGED_DTOL:
                return accept_diverged_dtol;
            case SNES_DIVERGED_MAX_IT:
                return accept_diverged_max_it;
            default:
                return false;
        }
    }
};

struct NonlinearSolveProfile {
    std::string label{"newton_backtracking"};
    NonlinearSolveMethodKind method_kind{
        NonlinearSolveMethodKind::newton_line_search};
    NonlinearLineSearchKind linesearch_kind{
        NonlinearLineSearchKind::backtracking};
    std::string snes_type_override{};
    std::string linesearch_type_override{};
    double rtol{1.0e-8};
    double atol{1.0e-10};
    double stol{1.0e-12};
    // PETSc stops with SNES_DIVERGED_DTOL when ||F_n|| grows above
    // divergence_tolerance * ||F_0||. Keeping this on the typed profile makes
    // tiny-load continuation audits reproducible instead of hiding them behind
    // ad hoc options-database tweaks.
    double divergence_tolerance{PETSC_DETERMINE};
    int max_iterations{100};
    int max_function_evaluations{PETSC_DEFAULT};
    std::string ksp_type{KSPPREONLY};
    std::string pc_type{PCLU};
    std::string factor_solver_type{};
    struct PetscLinearSolverTuning {
        // Factorization tuning is intentionally attached to the typed profile
        // rather than to PETSc's global options database.  Reusing symbolic
        // information is safe for fixed finite-element sparsity patterns; it
        // does not reuse stale numeric factors when the tangent values change.
        double ksp_rtol{PETSC_DETERMINE};
        double ksp_atol{PETSC_DETERMINE};
        double ksp_dtol{PETSC_DETERMINE};
        int ksp_max_iterations{PETSC_DETERMINE};
        std::string factor_mat_ordering_type{};
        int factor_levels{-1};
        bool factor_reuse_ordering{false};
        bool factor_reuse_fill{false};
        bool ksp_reuse_preconditioner{false};
        // Zero means "leave PETSc's default". PETSc documents positive values
        // as rebuild periods and negative values as special one-shot modes.
        int snes_lag_preconditioner{0};
        int snes_lag_jacobian{0};
    } linear_tuning{};
    NonlinearSmallResidualAcceptancePolicy small_residual_acceptance{};

    [[nodiscard]] std::string resolved_snes_type() const
    {
        if (!snes_type_override.empty()) {
            return snes_type_override;
        }
        return std::string{to_petsc_snes_type(method_kind)};
    }

    [[nodiscard]] std::string resolved_linesearch_type() const
    {
        if (!linesearch_type_override.empty()) {
            return linesearch_type_override;
        }
        return std::string{to_petsc_linesearch_type(linesearch_kind)};
    }
};

[[nodiscard]] inline bool profile_uses_factor_preconditioner(
    const NonlinearSolveProfile& profile) noexcept
{
    return profile.pc_type == PCLU ||
           profile.pc_type == PCCHOLESKY ||
           profile.pc_type == PCILU ||
           profile.pc_type == PCICC;
}

struct NonlinearSolveAttemptAssessment {
    bool accepted{false};
    bool accepted_by_small_residual_policy{false};
    double accepted_function_norm_threshold{0.0};
};

[[nodiscard]] inline double resolve_small_residual_acceptance_threshold(
    const NonlinearSolveProfile& profile) noexcept
{
    const auto& policy = profile.small_residual_acceptance;
    if (!policy.enabled()) {
        return 0.0;
    }

    double threshold = policy.absolute_function_norm_threshold;
    if (policy.profile_atol_multiplier > 0.0 &&
        std::isfinite(profile.atol) &&
        profile.atol > 0.0)
    {
        threshold = std::max(
            threshold,
            profile.atol * policy.profile_atol_multiplier);
    }

    return threshold;
}

[[nodiscard]] inline NonlinearSolveAttemptAssessment
assess_nonlinear_solve_attempt(
    const NonlinearSolveProfile& profile,
    SNESConvergedReason reason,
    double function_norm) noexcept
{
    if (reason > 0) {
        return {
            .accepted = true,
            .accepted_by_small_residual_policy = false,
            .accepted_function_norm_threshold = 0.0};
    }

    const auto threshold = resolve_small_residual_acceptance_threshold(profile);
    if (!profile.small_residual_acceptance.enabled() ||
        threshold <= 0.0 ||
        !profile.small_residual_acceptance.accepts_reason(reason))
    {
        return {};
    }

    if (std::isfinite(function_norm) && function_norm <= threshold) {
        return {
            .accepted = true,
            .accepted_by_small_residual_policy = true,
            .accepted_function_norm_threshold = threshold};
    }

    return {};
}

[[nodiscard]] inline NonlinearSolveProfile make_newton_backtracking_profile(
    std::string label = "newton_backtracking")
{
    return NonlinearSolveProfile{
        .label = std::move(label),
        .method_kind = NonlinearSolveMethodKind::newton_line_search,
        .linesearch_kind = NonlinearLineSearchKind::backtracking,
    };
}

[[nodiscard]] inline NonlinearSolveProfile make_newton_basic_profile(
    std::string label = "newton_basic")
{
    return NonlinearSolveProfile{
        .label = std::move(label),
        .method_kind = NonlinearSolveMethodKind::newton_line_search,
        .linesearch_kind = NonlinearLineSearchKind::basic,
    };
}

[[nodiscard]] inline NonlinearSolveProfile make_newton_l2_profile(
    std::string label = "newton_l2")
{
    return NonlinearSolveProfile{
        .label = std::move(label),
        .method_kind = NonlinearSolveMethodKind::newton_line_search,
        .linesearch_kind = NonlinearLineSearchKind::l2,
    };
}

[[nodiscard]] inline NonlinearSolveProfile make_newton_trust_region_profile(
    std::string label = "newton_trust_region")
{
    return NonlinearSolveProfile{
        .label = std::move(label),
        .method_kind = NonlinearSolveMethodKind::newton_trust_region,
        .linesearch_kind = NonlinearLineSearchKind::none,
    };
}

[[nodiscard]] inline NonlinearSolveProfile
make_newton_trust_region_dogleg_profile(
    std::string label = "newton_trust_region_dogleg")
{
    return NonlinearSolveProfile{
        .label = std::move(label),
        .method_kind = NonlinearSolveMethodKind::newton_trust_region_dogleg,
        .linesearch_kind = NonlinearLineSearchKind::none,
    };
}

[[nodiscard]] inline NonlinearSolveProfile
make_nonlinear_gmres_profile(std::string label = "nonlinear_gmres")
{
    return NonlinearSolveProfile{
        .label = std::move(label),
        .method_kind = NonlinearSolveMethodKind::nonlinear_gmres,
        .linesearch_kind = NonlinearLineSearchKind::secant,
    };
}

[[nodiscard]] inline NonlinearSolveProfile make_quasi_newton_profile(
    std::string label = "quasi_newton")
{
    return NonlinearSolveProfile{
        .label = std::move(label),
        .method_kind = NonlinearSolveMethodKind::quasi_newton,
        .linesearch_kind = NonlinearLineSearchKind::critical_point,
    };
}

[[nodiscard]] inline NonlinearSolveProfile
make_nonlinear_conjugate_gradient_profile(
    std::string label = "nonlinear_conjugate_gradient")
{
    return NonlinearSolveProfile{
        .label = std::move(label),
        .method_kind =
            NonlinearSolveMethodKind::nonlinear_conjugate_gradient,
        .linesearch_kind = NonlinearLineSearchKind::ncg_linear,
    };
}

[[nodiscard]] inline NonlinearSolveProfile make_anderson_profile(
    std::string label = "anderson_acceleration")
{
    return NonlinearSolveProfile{
        .label = std::move(label),
        .method_kind = NonlinearSolveMethodKind::anderson_acceleration,
        .linesearch_kind = NonlinearLineSearchKind::basic,
    };
}

[[nodiscard]] inline NonlinearSolveProfile make_nonlinear_richardson_profile(
    std::string label = "nonlinear_richardson")
{
    return NonlinearSolveProfile{
        .label = std::move(label),
        .method_kind = NonlinearSolveMethodKind::nonlinear_richardson,
        .linesearch_kind = NonlinearLineSearchKind::backtracking,
    };
}

inline const std::vector<NonlinearSolveProfile>&
canonical_default_nonlinear_solve_profiles()
{
    static const std::vector<NonlinearSolveProfile> defaults{
        make_newton_backtracking_profile()};
    return defaults;
}

inline const std::vector<NonlinearSolveProfile>&
active_nonlinear_solve_profiles(
    const std::vector<NonlinearSolveProfile>& profiles)
{
    return profiles.empty() ? canonical_default_nonlinear_solve_profiles()
                            : profiles;
}

inline const NonlinearSolveProfile&
select_nonlinear_solve_profile(
    const std::vector<NonlinearSolveProfile>& profiles,
    std::size_t requested_index)
{
    const auto& active = active_nonlinear_solve_profiles(profiles);
    const auto index =
        std::min(requested_index, active.empty() ? std::size_t{0}
                                                 : active.size() - 1);
    return active[index];
}

inline void apply_nonlinear_solve_profile(SNES snes,
                                          const NonlinearSolveProfile& profile);

inline void apply_linear_solver_profile(KSP ksp,
                                        const NonlinearSolveProfile& profile)
{
    FALL_N_PETSC_CHECK(KSPSetType(ksp, profile.ksp_type.c_str()));
    FALL_N_PETSC_CHECK(KSPSetTolerances(
        ksp,
        profile.linear_tuning.ksp_rtol,
        profile.linear_tuning.ksp_atol,
        profile.linear_tuning.ksp_dtol,
        profile.linear_tuning.ksp_max_iterations));

    PC pc{nullptr};
    FALL_N_PETSC_CHECK(KSPGetPC(ksp, &pc));
    FALL_N_PETSC_CHECK(PCSetType(pc, profile.pc_type.c_str()));

    if (!profile.factor_solver_type.empty() &&
        (profile.pc_type == PCLU || profile.pc_type == PCCHOLESKY))
    {
        FALL_N_PETSC_CHECK(
            PCFactorSetMatSolverType(pc, profile.factor_solver_type.c_str()));
    }

    if (profile_uses_factor_preconditioner(profile)) {
        const auto& tuning = profile.linear_tuning;
        if (!tuning.factor_mat_ordering_type.empty()) {
            FALL_N_PETSC_CHECK(PCFactorSetMatOrderingType(
                pc, tuning.factor_mat_ordering_type.c_str()));
        }
        if (tuning.factor_levels >= 0) {
            FALL_N_PETSC_CHECK(PCFactorSetLevels(
                pc, tuning.factor_levels));
        }
        if (tuning.factor_reuse_ordering) {
            FALL_N_PETSC_CHECK(PCFactorSetReuseOrdering(pc, PETSC_TRUE));
        }
        if (tuning.factor_reuse_fill) {
            FALL_N_PETSC_CHECK(PCFactorSetReuseFill(pc, PETSC_TRUE));
        }
    }

    if (profile.linear_tuning.ksp_reuse_preconditioner) {
        FALL_N_PETSC_CHECK(KSPSetReusePreconditioner(ksp, PETSC_TRUE));
    }
}

inline void apply_nonlinear_solve_profile(SNES snes,
                                          const NonlinearSolveProfile& profile)
{
    const auto snes_type = profile.resolved_snes_type();
    const auto linesearch_type = profile.resolved_linesearch_type();

    FALL_N_PETSC_CHECK(SNESSetType(snes, snes_type.c_str()));
    FALL_N_PETSC_CHECK(SNESSetTolerances(snes,
                                        profile.atol,
                                        profile.rtol,
                                        profile.stol,
                                        profile.max_iterations,
                                        profile.max_function_evaluations));
    FALL_N_PETSC_CHECK(
        SNESSetDivergenceTolerance(snes, profile.divergence_tolerance));

    if (profile.linear_tuning.snes_lag_preconditioner != 0) {
        FALL_N_PETSC_CHECK(SNESSetLagPreconditioner(
            snes, profile.linear_tuning.snes_lag_preconditioner));
    }
    if (profile.linear_tuning.snes_lag_jacobian != 0) {
        FALL_N_PETSC_CHECK(SNESSetLagJacobian(
            snes, profile.linear_tuning.snes_lag_jacobian));
    }

    KSP ksp{nullptr};
    FALL_N_PETSC_CHECK(SNESGetKSP(snes, &ksp));
    apply_linear_solver_profile(ksp, profile);

    if (!linesearch_type.empty()) {
        SNESLineSearch line_search{nullptr};
        FALL_N_PETSC_CHECK(SNESGetLineSearch(snes, &line_search));
        FALL_N_PETSC_CHECK(
            SNESLineSearchSetType(line_search, linesearch_type.c_str()));
    }
}

} // namespace fall_n

#endif // FALL_N_SRC_ANALYSIS_NONLINEAR_SOLVE_POLICY_HH
