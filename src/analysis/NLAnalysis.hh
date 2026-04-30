#ifndef FALL_N_SRC_ANALYSIS_NLANALYSIS_HH
#define FALL_N_SRC_ANALYSIS_NLANALYSIS_HH

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <petscsnes.h>

#include "../model/Model.hh"
#include "../petsc/PetscRaii.hh"
#include "../utils/Benchmark.hh"
#include "AnalysisRouteAudit.hh"
#include "AnalysisObserver.hh"
#include "IncrementalControl.hh"
#include "NonlinearSolvePolicy.hh"
#include "StepDirector.hh"
#include "SteppableSolver.hh"

// =============================================================================
//  NonlinearAnalysis — PETSc SNES-driven Newton-Raphson solver
// =============================================================================
//
//  Solves the nonlinear equilibrium equation:
//
//      R(u) = f_int(u) − f_ext = 0
//
//  using PETSc's SNES (Scalable Nonlinear Equations Solver).
//
//  SNES handles:
//    - Newton with line-search   (-snes_type newtonls)   [default]
//    - Newton with trust-region  (-snes_type newtontr)
//    - Convergence monitoring    (-snes_monitor -snes_converged_reason)
//    - Jacobian verification     (-snes_test_jacobian)
//    - Finite-difference Jacobian (-snes_fd)   [debugging only]
//
//  This class provides:
//    - FormResidual:  R(u) = f_int(u) − f_ext
//    - FormJacobian:  J(u) = K_t(u)  (tangent stiffness)
//    - Incremental load stepping with convergence checking and
//      automatic bisection on diverged steps
//
//  The constitutive response flows through the Strategy injected in
//  each Material<>:
//    element → material_point → Material<> → Strategy → relation
//
//  CONVERGENCE SAFETY:
//    - solve() and solve_incremental() NEVER commit material state
//      when SNES diverges.  This prevents corrupted internal variables
//      from propagating to subsequent load steps.
//    - solve_incremental() implements automatic bisection: when a load
//      step diverges, the step is split into two half-steps and retried,
//      up to a configurable maximum bisection depth.
//    - Both functions return bool (true = all converged).
//
// =============================================================================

template <typename MaterialPolicy,
          typename KinematicPolicy = continuum::SmallStrain,
          std::size_t ndofs = MaterialPolicy::dim,
          typename ElemPolicy = SingleElementPolicy<ContinuumElement<MaterialPolicy, ndofs, KinematicPolicy>>>
class NonlinearAnalysis {
public:
    using analysis_route_tag =
        fall_n::AnalysisRouteTag<fall_n::AnalysisRouteKind::nonlinear_incremental_newton>;
    static constexpr fall_n::AnalysisRouteKind analysis_route_kind =
        fall_n::AnalysisRouteKind::nonlinear_incremental_newton;
    static constexpr fall_n::AnalysisRouteAuditScope analysis_route_audit_scope =
        fall_n::canonical_analysis_route_audit_scope(analysis_route_kind);

private:
    using ModelT = Model<MaterialPolicy, KinematicPolicy, ndofs, ElemPolicy>;
    using ElementT = typename ModelT::element_type;
    static constexpr auto dim = MaterialPolicy::dim;

public:
    using model_type = ModelT;
    using element_type = ElementT;
    using NonlinearSolveProfile = fall_n::NonlinearSolveProfile;
    using IncrementPredictorKind = fall_n::IncrementPredictorKind;
    using IncrementPredictorSettings = fall_n::IncrementPredictorSettings;

private:

    static constexpr bool has_explicit_local_nonlinear_api =
        requires (ElementT& elem, Vec u_local, const Eigen::VectorXd& u_e) {
            { elem.extract_element_dofs(u_local) } -> std::same_as<Eigen::VectorXd>;
            { elem.compute_internal_force_vector(u_e) } -> std::same_as<Eigen::VectorXd>;
            { elem.compute_tangent_stiffness_matrix(u_e) } -> std::same_as<Eigen::MatrixXd>;
            elem.get_dof_indices();
        };

    ModelT* model_{nullptr};
    petsc::OwnedSNES snes_{};

    petsc::OwnedVec U{};       // Global solution
    petsc::OwnedVec R_vec{};   // Residual vector
    petsc::OwnedVec f_ext{};   // External forces (current load level)
    petsc::OwnedMat J{};       // Jacobian (tangent stiffness)

    bool is_setup_{false};

    // ─── Performance timing ───────────────────────────────────────
    AnalysisTimer timer_;

public:

    // ─── Step callback (optional) ─────────────────────────────────
    //  Invoked after each converged load step in solve_incremental().
    //  Arguments: (step_number, lambda, model_ref)
    struct IncrementStepDiagnostics {
        double p_start{0.0};
        double p_target{0.0};
        double last_attempt_p_start{0.0};
        double last_attempt_p_target{0.0};
        int accepted_substep_count{0};
        int max_bisection_level{0};
        int total_newton_iterations{0};
        int failed_attempt_count{0};
        int solver_profile_attempt_count{0};
        int last_snes_reason{0};
        double last_function_norm{0.0};
        std::string last_solver_profile_label{};
        std::string last_solver_snes_type{};
        std::string last_solver_linesearch_type{};
        std::string last_solver_ksp_type{};
        std::string last_solver_pc_type{};
        double last_solver_ksp_rtol{PETSC_DETERMINE};
        double last_solver_ksp_atol{PETSC_DETERMINE};
        double last_solver_ksp_dtol{PETSC_DETERMINE};
        int last_solver_ksp_max_iterations{PETSC_DETERMINE};
        int last_solver_ksp_reason{0};
        int last_solver_ksp_iterations{0};
        std::string last_solver_factor_mat_ordering_type{};
        int last_solver_factor_levels{-1};
        bool last_solver_factor_reuse_ordering{false};
        bool last_solver_factor_reuse_fill{false};
        int last_solver_pc_asm_overlap{-1};
        std::string last_solver_pc_asm_type{};
        std::string last_solver_petsc_options_prefix{};
        std::string last_solver_pc_sub_ksp_type{};
        std::string last_solver_pc_sub_pc_type{};
        bool last_solver_ksp_reuse_preconditioner{false};
        int last_solver_snes_lag_preconditioner{0};
        int last_solver_snes_lag_jacobian{0};
        bool converged{false};
        bool accepted_by_small_residual_policy{false};
        double accepted_function_norm_threshold{0.0};
    };

    struct IncrementAdaptationSettings {
        bool enabled{false};
        double min_increment_size{0.0};
        double max_increment_size{0.0};
        double cutback_factor{0.5};
        double growth_factor{1.25};
        int max_cutbacks_per_step{8};
        int easy_newton_iterations{6};
        int difficult_newton_iterations{12};
        int easy_steps_before_growth{2};
    };

private:

    using StepCallback = std::function<void(int, double, const ModelT&)>;
    StepCallback step_callback_{};
    using FailedAttemptCallback =
        std::function<void(const ModelT&, const IncrementStepDiagnostics&)>;
    FailedAttemptCallback failed_attempt_callback_{};

    // ─── Observer protocol (optional) ─────────────────────────────
    //  Structured alternative to StepCallback, supporting start/step/end
    //  lifecycle events via the AnalysisObserver protocol.
    fall_n::ObserverCallback<ModelT> observer_{};

    // ─── Single-step (incremental) state ──────────────────────────
    //  Persisted between begin_incremental() and subsequent step() calls.
    using ApplyFn = std::function<void(double, Vec, Vec, ModelT*)>;
    ApplyFn       apply_fn_{};
    petsc::OwnedVec f_full_{};
    double        p_done_{0.0};
    int           step_count_{0};
    double        dp_{0.0};
    int           max_bisections_{4};
    bool          incremental_active_{false};
    bool          auto_commit_{true};
    bool          incremental_logging_enabled_{true};
    IncrementAdaptationSettings increment_adaptation_{};
    double        initial_dp_{0.0};
    int           easy_step_streak_{0};
    fall_n::StepDirector<ModelT> director_{};
    std::vector<NonlinearSolveProfile> solve_profiles_{};
    IncrementPredictorSettings increment_predictor_{};
    petsc::OwnedVec previous_converged_u_{};
    petsc::OwnedVec predictor_delta_u_{};
    petsc::OwnedVec predictor_linear_rhs_{};
    petsc::OwnedVec predictor_linear_step_{};
    petsc::OwnedKSP predictor_linear_ksp_{};
    double previous_converged_p_{0.0};
    bool has_previous_converged_u_{false};
    IncrementStepDiagnostics previous_requested_step_diagnostics_{};

public:
    struct SolverCheckpoint {
        typename ModelT::checkpoint_type model{};
        petsc::OwnedVec displacement{};
        petsc::OwnedVec external_force{};
        petsc::OwnedVec previous_converged_displacement{};
        double p_done{0.0};
        double previous_converged_p{0.0};
        int    step_count{0};
        double dp{0.0};
        int    max_bisections{0};
        bool   incremental_active{false};
        bool   has_previous_converged_displacement{false};
        IncrementStepDiagnostics previous_requested_step_diagnostics{};
    };

    using checkpoint_type = SolverCheckpoint;

private:

    // ─── Penalty coupling hooks (optional) ────────────────────────
    //  Called after standard element assembly in FormResidual / FormJacobian
    //  to inject additional coupling terms (e.g. penalty rebar coupling).
    //    residual_hook_(u_local, f_int_local, dm)
    //    jacobian_hook_(u_local, J_mat, dm)
    using ResidualHook = std::function<void(Vec, Vec, DM)>;
    using GlobalResidualHook = std::function<void(Vec, Vec, DM)>;
    using JacobianHook = std::function<void(Vec, Mat, DM)>;
    ResidualHook residual_hook_{};
    GlobalResidualHook global_residual_hook_{};
    JacobianHook jacobian_hook_{};
    IncrementStepDiagnostics last_increment_step_diagnostics_{};

    template <typename... Args>
    void incremental_printf_(const char* format, Args&&... args) const
    {
        if (incremental_logging_enabled_) {
            PetscPrintf(
                PETSC_COMM_WORLD,
                format,
                std::forward<Args>(args)...);
        }
    }

    [[nodiscard]] double minimum_increment_size_() const noexcept
    {
        if (!increment_adaptation_.enabled) {
            return 0.0;
        }
        if (increment_adaptation_.min_increment_size > 0.0) {
            return increment_adaptation_.min_increment_size;
        }
        const auto bisection_floor =
            std::ldexp(initial_dp_ > 0.0 ? initial_dp_ : dp_,
                       -(std::max(max_bisections_, 0) + 3));
        return std::max(bisection_floor, 1.0e-12);
    }

    [[nodiscard]] double maximum_increment_size_() const noexcept
    {
        if (!increment_adaptation_.enabled) {
            return 1.0;
        }
        if (increment_adaptation_.max_increment_size > 0.0) {
            return increment_adaptation_.max_increment_size;
        }
        return initial_dp_ > 0.0 ? initial_dp_ : 1.0;
    }

    [[nodiscard]] double clamp_increment_size_(double value) const noexcept
    {
        if (!increment_adaptation_.enabled) {
            return value;
        }
        return std::clamp(
            value,
            minimum_increment_size_(),
            maximum_increment_size_());
    }

    void reset_increment_predictor_history_()
    {
        previous_converged_u_ = petsc::OwnedVec{};
        predictor_delta_u_ = petsc::OwnedVec{};
        predictor_linear_rhs_ = petsc::OwnedVec{};
        predictor_linear_step_ = petsc::OwnedVec{};
        predictor_linear_ksp_ = petsc::OwnedKSP{};
        previous_converged_p_ = 0.0;
        has_previous_converged_u_ = false;
        previous_requested_step_diagnostics_ = IncrementStepDiagnostics{};
    }

    void capture_previous_converged_state_(Vec u_previous, double p_previous)
    {
        if (!u_previous) {
            return;
        }
        if (!previous_converged_u_) {
            FALL_N_PETSC_CHECK(VecDuplicate(u_previous, previous_converged_u_.ptr()));
        }
        FALL_N_PETSC_CHECK(VecCopy(u_previous, previous_converged_u_.get()));
        previous_converged_p_ = p_previous;
        has_previous_converged_u_ = true;
    }

    bool apply_linearized_equilibrium_predictor_(
        Vec current_state,
        const NonlinearSolveProfile& profile)
    {
        if (!current_state) {
            return false;
        }

        if (!predictor_linear_rhs_) {
            FALL_N_PETSC_CHECK(VecDuplicate(current_state, predictor_linear_rhs_.ptr()));
        }
        if (!predictor_linear_step_) {
            FALL_N_PETSC_CHECK(VecDuplicate(current_state, predictor_linear_step_.ptr()));
        }
        if (!predictor_linear_ksp_) {
            FALL_N_PETSC_CHECK(KSPCreate(PETSC_COMM_WORLD, predictor_linear_ksp_.ptr()));
        }

        FALL_N_PETSC_CHECK(VecCopy(current_state, U));
        FALL_N_PETSC_CHECK(SNESComputeFunction(snes_.get(), U.get(), predictor_linear_rhs_.get()));

        PetscReal residual_norm = 0.0;
        FALL_N_PETSC_CHECK(VecNorm(predictor_linear_rhs_.get(), NORM_2, &residual_norm));
        if (!(residual_norm > 0.0)) {
            return false;
        }

        FALL_N_PETSC_CHECK(
            SNESComputeJacobian(snes_.get(), U.get(), J.get(), J.get()));
        fall_n::apply_linear_solver_profile(predictor_linear_ksp_.get(), profile);
        FALL_N_PETSC_CHECK(
            KSPSetOperators(predictor_linear_ksp_.get(), J.get(), J.get()));

        FALL_N_PETSC_CHECK(VecCopy(predictor_linear_rhs_.get(), predictor_linear_step_.get()));
        FALL_N_PETSC_CHECK(VecScale(predictor_linear_step_.get(), -1.0));
        FALL_N_PETSC_CHECK(VecSet(predictor_linear_rhs_.get(), 0.0));
        FALL_N_PETSC_CHECK(
            KSPSolve(
                predictor_linear_ksp_.get(),
                predictor_linear_step_.get(),
                predictor_linear_rhs_.get()));

        KSPConvergedReason ksp_reason{KSP_CONVERGED_ITERATING};
        FALL_N_PETSC_CHECK(KSPGetConvergedReason(predictor_linear_ksp_.get(), &ksp_reason));
        if (ksp_reason <= 0) {
            FALL_N_PETSC_CHECK(VecCopy(current_state, U));
            return false;
        }

        PetscReal correction_norm = 0.0;
        FALL_N_PETSC_CHECK(VecNorm(predictor_linear_rhs_.get(), NORM_2, &correction_norm));
        if (!(correction_norm > 0.0)) {
            FALL_N_PETSC_CHECK(VecCopy(current_state, U));
            return false;
        }

        FALL_N_PETSC_CHECK(VecAXPY(U, 1.0, predictor_linear_rhs_.get()));
        return true;
    }

    bool apply_increment_predictor_(Vec current_state,
                                    double p_current,
                                    double p_target,
                                    int bisection_level,
                                    const NonlinearSolveProfile& profile)
    {
        FALL_N_PETSC_CHECK(VecCopy(current_state, U));

        if (!increment_predictor_.enabled ||
            increment_predictor_.kind == IncrementPredictorKind::current_state) {
            return false;
        }

        if (increment_predictor_.kind ==
            IncrementPredictorKind::linearized_equilibrium_seed) {
            if (increment_predictor_.disable_during_bisection &&
                bisection_level > 0) {
                return false;
            }
            return apply_linearized_equilibrium_predictor_(current_state, profile);
        }

        if (increment_predictor_.kind ==
                IncrementPredictorKind::secant_with_linearized_fallback &&
            !has_previous_converged_u_) {
            if (increment_predictor_.disable_during_bisection &&
                bisection_level > 0) {
                return false;
            }
            return apply_linearized_equilibrium_predictor_(current_state, profile);
        }

        if (!has_previous_converged_u_) {
            return false;
        }

        if (increment_predictor_.disable_during_bisection &&
            bisection_level > 0) {
            return false;
        }

        if (increment_predictor_.kind ==
                IncrementPredictorKind::adaptive_secant_extrapolation &&
            increment_predictor_.disable_after_cutback &&
            (previous_requested_step_diagnostics_.failed_attempt_count > 0 ||
             previous_requested_step_diagnostics_.max_bisection_level > 0)) {
            return false;
        }

        const double previous_increment = p_current - previous_converged_p_;
        const double requested_increment = p_target - p_current;
        if (previous_increment <= 1.0e-14 || requested_increment <= 1.0e-14) {
            return false;
        }

        if (increment_predictor_.kind ==
            IncrementPredictorKind::adaptive_secant_extrapolation)
        {
            const auto accepted_substeps = std::max(
                previous_requested_step_diagnostics_.accepted_substep_count,
                1);
            const auto average_newton_iterations =
                static_cast<double>(
                    previous_requested_step_diagnostics_.total_newton_iterations) /
                static_cast<double>(accepted_substeps);
            if (average_newton_iterations >
                static_cast<double>(
                    increment_predictor_.difficult_newton_iterations)) {
                return false;
            }
        }

        double scale = requested_increment / previous_increment;
        scale = std::clamp(
            scale,
            0.0,
            std::max(0.0, increment_predictor_.max_scale_factor));
        scale = std::min(scale, increment_predictor_.max_relative_increment_norm);

        if (!(scale > 0.0)) {
            return false;
        }

        if (!predictor_delta_u_) {
            FALL_N_PETSC_CHECK(VecDuplicate(current_state, predictor_delta_u_.ptr()));
        }
        FALL_N_PETSC_CHECK(VecCopy(current_state, predictor_delta_u_.get()));
        FALL_N_PETSC_CHECK(
            VecAXPY(predictor_delta_u_.get(), -1.0, previous_converged_u_.get()));

        PetscReal last_increment_norm = 0.0;
        FALL_N_PETSC_CHECK(
            VecNorm(predictor_delta_u_.get(), NORM_2, &last_increment_norm));
        if (!(last_increment_norm > 0.0)) {
            return false;
        }

        FALL_N_PETSC_CHECK(VecAXPY(U, scale, predictor_delta_u_.get()));
        return true;
    }

    void adapt_next_increment_after_requested_step_(
        double final_substep_size,
        bool used_cutback)
    {
        if (!increment_adaptation_.enabled) {
            return;
        }

        const auto accepted_substeps = std::max(
            last_increment_step_diagnostics_.accepted_substep_count,
            1);
        const auto average_newton_iterations =
            static_cast<double>(
                last_increment_step_diagnostics_.total_newton_iterations) /
            static_cast<double>(accepted_substeps);

        if (used_cutback ||
            last_increment_step_diagnostics_.max_bisection_level > 0 ||
            average_newton_iterations >=
                static_cast<double>(
                    increment_adaptation_.difficult_newton_iterations)) {
            dp_ = clamp_increment_size_(final_substep_size);
            easy_step_streak_ = 0;
            return;
        }

        if (average_newton_iterations <=
            static_cast<double>(
                increment_adaptation_.easy_newton_iterations)) {
            ++easy_step_streak_;
            if (easy_step_streak_ >=
                std::max(increment_adaptation_.easy_steps_before_growth, 1)) {
                dp_ = clamp_increment_size_(
                    std::max(dp_, final_substep_size) *
                    increment_adaptation_.growth_factor);
                easy_step_streak_ = 0;
                return;
            }
        } else {
            easy_step_streak_ = 0;
        }

        dp_ = clamp_increment_size_(std::max(dp_, final_substep_size));
    }

    [[nodiscard]] const std::vector<NonlinearSolveProfile>&
    active_solve_profiles_() const noexcept
    {
        return fall_n::active_nonlinear_solve_profiles(solve_profiles_);
    }

    void apply_solve_profile_(const NonlinearSolveProfile& profile)
    {
        fall_n::apply_nonlinear_solve_profile(snes_.get(), profile);
    }

    void record_solver_profile_diagnostics_(
        const NonlinearSolveProfile& profile,
        SNESConvergedReason reason,
        double residual_norm,
        const fall_n::NonlinearSolveAttemptAssessment& acceptance)
    {
        auto& diag = last_increment_step_diagnostics_;
        diag.solver_profile_attempt_count += 1;
        diag.last_solver_profile_label = profile.label;
        diag.last_solver_snes_type = profile.resolved_snes_type();
        diag.last_solver_linesearch_type = profile.resolved_linesearch_type();
        diag.last_solver_ksp_type = profile.ksp_type;
        diag.last_solver_pc_type = profile.pc_type;
        diag.last_solver_ksp_rtol = profile.linear_tuning.ksp_rtol;
        diag.last_solver_ksp_atol = profile.linear_tuning.ksp_atol;
        diag.last_solver_ksp_dtol = profile.linear_tuning.ksp_dtol;
        diag.last_solver_ksp_max_iterations =
            profile.linear_tuning.ksp_max_iterations;
        KSP ksp{nullptr};
        FALL_N_PETSC_CHECK(SNESGetKSP(snes_.get(), &ksp));
        KSPConvergedReason ksp_reason{KSP_CONVERGED_ITERATING};
        PetscInt ksp_iterations{0};
        if (ksp != nullptr) {
            FALL_N_PETSC_CHECK(KSPGetConvergedReason(ksp, &ksp_reason));
            FALL_N_PETSC_CHECK(KSPGetIterationNumber(ksp, &ksp_iterations));
        }
        diag.last_solver_ksp_reason = static_cast<int>(ksp_reason);
        diag.last_solver_ksp_iterations = static_cast<int>(ksp_iterations);
        diag.last_solver_factor_mat_ordering_type =
            profile.linear_tuning.factor_mat_ordering_type;
        diag.last_solver_factor_levels = profile.linear_tuning.factor_levels;
        diag.last_solver_factor_reuse_ordering =
            profile.linear_tuning.factor_reuse_ordering;
        diag.last_solver_factor_reuse_fill =
            profile.linear_tuning.factor_reuse_fill;
        diag.last_solver_pc_asm_overlap =
            profile.linear_tuning.pc_asm_overlap;
        diag.last_solver_pc_asm_type =
            profile.linear_tuning.pc_asm_type_enabled
                ? std::string{
                      fall_n::to_string(profile.linear_tuning.pc_asm_type)}
                : std::string{};
        diag.last_solver_petsc_options_prefix =
            profile.linear_tuning.petsc_options_prefix;
        diag.last_solver_pc_sub_ksp_type =
            profile.linear_tuning.pc_sub_ksp_type;
        diag.last_solver_pc_sub_pc_type =
            profile.linear_tuning.pc_sub_pc_type;
        diag.last_solver_ksp_reuse_preconditioner =
            profile.linear_tuning.ksp_reuse_preconditioner;
        diag.last_solver_snes_lag_preconditioner =
            profile.linear_tuning.snes_lag_preconditioner;
        diag.last_solver_snes_lag_jacobian =
            profile.linear_tuning.snes_lag_jacobian;
        diag.last_snes_reason = static_cast<int>(reason);
        diag.last_function_norm = residual_norm;
        diag.accepted_by_small_residual_policy =
            acceptance.accepted_by_small_residual_policy;
        diag.accepted_function_norm_threshold =
            acceptance.accepted_function_norm_threshold;
    }

    // ─── SNES callback context ────────────────────────────────────

    struct Context {
        ModelT* model;
        Vec     f_ext;
        ResidualHook* residual_hook;
        GlobalResidualHook* global_residual_hook;
        JacobianHook* jacobian_hook;
    } ctx_{};

    // ─── SNES callback: Residual  R(u) = f_int(u) − f_ext ────────
    //
    //  Called by SNES at each Newton iteration to evaluate the residual.
    //  Internal forces f_int are assembled element-by-element, where
    //  each element evaluates σ(ε) through the material's Strategy.
    //
    //  Assembly is parallelised in two phases:
    //    Phase 1 — Extract element DOFs + compute f_e  (parallel, thread-safe)
    //    Phase 2 — Inject f_e into PETSc local vector  (sequential, PETSc API)

    static PetscErrorCode FormResidual(
        SNES /*snes*/, Vec u_global, Vec R_out, void* ctx_ptr)
    {
        PetscFunctionBeginUser;

        auto* ctx   = static_cast<Context*>(ctx_ptr);
        auto* model = ctx->model;
        DM    dm    = model->get_plex();

        // Scatter global → local (+ add imposed BCs)
        Vec u_local;
        DMGetLocalVector(dm, &u_local);
        VecSet(u_local, 0.0);
        DMGlobalToLocal(dm, u_global, INSERT_VALUES, u_local);
        VecAXPY(u_local, 1.0, model->imposed_solution());

        Vec f_int_local;
        DMGetLocalVector(dm, &f_int_local);
        VecSet(f_int_local, 0.0);

        if constexpr (has_explicit_local_nonlinear_api) {
            const auto num_elems = model->elements().size();

            // Fast path for elements exposing local-vector assembly kernels.
            std::vector<Eigen::VectorXd> elem_dofs(num_elems);
            for (std::size_t e = 0; e < num_elems; ++e) {
                elem_dofs[e] = model->elements()[e].extract_element_dofs(u_local);
            }

            std::vector<Eigen::VectorXd> elem_f(num_elems);

            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (std::size_t e = 0; e < num_elems; ++e) {
                elem_f[e] = model->elements()[e].compute_internal_force_vector(elem_dofs[e]);
            }

            for (std::size_t e = 0; e < num_elems; ++e) {
                const auto& dofs = model->elements()[e].get_dof_indices();
                VecSetValues(f_int_local, static_cast<PetscInt>(dofs.size()),
                             dofs.data(), elem_f[e].data(), ADD_VALUES);
            }
        } else {
            // Generic path for structural type-erased elements: assemble through
            // the FiniteElement interface without assuming access to local DOF
            // extraction or explicit local stiffness vectors.
            for (auto& element : model->elements()) {
                element.compute_internal_forces(u_local, f_int_local);
            }
        }

          // Penalty coupling hook (e.g. embedded rebar)
          if (ctx->residual_hook && *ctx->residual_hook)
              (*ctx->residual_hook)(u_local, f_int_local, dm);

          // Scatter local f_int → global residual
          VecSet(R_out, 0.0);
          DMLocalToGlobal(dm, f_int_local, ADD_VALUES, R_out);

          // Global residual hook for contributions whose constrained scatter
          // is better represented directly in the reduced algebraic space.
          if (ctx->global_residual_hook && *ctx->global_residual_hook)
              (*ctx->global_residual_hook)(u_local, R_out, dm);

          // The global hook injects ADD_VALUES directly into the reduced
          // algebraic vector, so it must be explicitly assembled before PETSc
          // can interpret the residual consistently inside SNES.
          VecAssemblyBegin(R_out);
          VecAssemblyEnd(R_out);

          // R = f_int − f_ext
          VecAXPY(R_out, -1.0, ctx->f_ext);

        DMRestoreLocalVector(dm, &u_local);
        DMRestoreLocalVector(dm, &f_int_local);

        PetscFunctionReturn(0);
    }

    // ─── SNES callback: Jacobian  J(u) = K_t(u) ──────────────────
    //
    //  Called by SNES to assemble the tangent stiffness matrix.
    //  Each element evaluates C_t(ε) through the material's Strategy:
    //    - ElasticUpdate: C_t = C_e (constant)
    //    - InelasticUpdate: C_t = C_ep (algorithmic consistent tangent)
    //
    //  Assembly is parallelised in two phases:
    //    Phase 1 — Extract DOFs + compute K_e  (parallel, thread-safe)
    //    Phase 2 — Inject K_e into PETSc Mat   (sequential, PETSc API)

    static PetscErrorCode FormJacobian(
        SNES /*snes*/, Vec u_global, Mat J_mat, Mat /*P*/, void* ctx_ptr)
    {
        PetscFunctionBeginUser;

        auto* ctx   = static_cast<Context*>(ctx_ptr);
        auto* model = ctx->model;
        DM    dm    = model->get_plex();

        MatZeroEntries(J_mat);

        // Scatter global → local (+ imposed BCs)
        Vec u_local;
        DMGetLocalVector(dm, &u_local);
        VecSet(u_local, 0.0);
        DMGlobalToLocal(dm, u_global, INSERT_VALUES, u_local);
        VecAXPY(u_local, 1.0, model->imposed_solution());

        if constexpr (has_explicit_local_nonlinear_api) {
            const auto num_elems = model->elements().size();

            std::vector<Eigen::VectorXd> elem_dofs(num_elems);
            for (std::size_t e = 0; e < num_elems; ++e) {
                elem_dofs[e] = model->elements()[e].extract_element_dofs(u_local);
            }

            std::vector<Eigen::MatrixXd> elem_K(num_elems);

            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (std::size_t e = 0; e < num_elems; ++e) {
                elem_K[e] = model->elements()[e].compute_tangent_stiffness_matrix(elem_dofs[e]);
            }

            for (std::size_t e = 0; e < num_elems; ++e) {
                const auto& dofs = model->elements()[e].get_dof_indices();
                const auto n = static_cast<PetscInt>(dofs.size());
                MatSetValuesLocal(J_mat, n, dofs.data(), n, dofs.data(),
                                  elem_K[e].data(), ADD_VALUES);
            }
        } else {
            for (auto& element : model->elements()) {
                element.inject_tangent_stiffness(u_local, J_mat);
            }
        }

        // Penalty coupling hook (e.g. embedded rebar)
        if (ctx->jacobian_hook && *ctx->jacobian_hook)
            (*ctx->jacobian_hook)(u_local, J_mat, dm);

        MatAssemblyBegin(J_mat, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(J_mat, MAT_FINAL_ASSEMBLY);

        DMRestoreLocalVector(dm, &u_local);

        PetscFunctionReturn(0);
    }

    // ─── Commit material state after global convergence ─────────

    void sync_model_state_from_solution_() {
        DM dm = model_->get_plex();

        Vec u_local;
        DMGetLocalVector(dm, &u_local);
        VecSet(u_local, 0.0);
        DMGlobalToLocal(dm, U, INSERT_VALUES, u_local);
        VecAXPY(u_local, 1.0, model_->imposed_solution());

        VecSet(model_->state_vector(), 0.0);
        VecCopy(u_local, model_->state_vector());

        DMRestoreLocalVector(dm, &u_local);
    }

    void commit_state() {
        DM dm = model_->get_plex();

        Vec u_local;
        DMGetLocalVector(dm, &u_local);
        VecSet(u_local, 0.0);
        DMGlobalToLocal(dm, U, INSERT_VALUES, u_local);
        VecAXPY(u_local, 1.0, model_->imposed_solution());

        for (auto& element : model_->elements()) {
            element.commit_material_state(u_local);
        }

        VecSet(model_->state_vector(), 0.0);
        VecCopy(u_local, model_->state_vector());
        DMRestoreLocalVector(dm, &u_local);
    }

    // ─── Revert material state after diverged step ───────────────
    //
    //  Explicitly reverts all element material states to the last
    //  committed values.  Called when the bisection engine detects
    //  SNES divergence.  This is a safety measure: constitutive
    //  evaluations during failed Newton iterations may leave
    //  trial-state buffers in an inconsistent state; revert
    //  guarantees a clean slate for the next attempt.

    void revert_state() {
        for (auto& element : model_->elements()) {
            element.revert_material_state();
        }
    }

public:

    auto get_model() const { return model_; }
    void set_auto_commit(bool enabled) { auto_commit_ = enabled; }
    [[nodiscard]] bool auto_commit() const noexcept { return auto_commit_; }

    void commit_trial_state() {
        commit_state();
        model_->update_elements_state();
    }

    [[nodiscard]] checkpoint_type capture_checkpoint() const {
        checkpoint_type checkpoint;

        if (U) {
            FALL_N_PETSC_CHECK(VecDuplicate(U.get(),
                                            checkpoint.displacement.ptr()));
            FALL_N_PETSC_CHECK(VecCopy(U.get(),
                                       checkpoint.displacement.get()));
        }

        if (f_ext) {
            FALL_N_PETSC_CHECK(VecDuplicate(f_ext.get(),
                                            checkpoint.external_force.ptr()));
            FALL_N_PETSC_CHECK(VecCopy(f_ext.get(),
                                       checkpoint.external_force.get()));
        }
        if (has_previous_converged_u_ && previous_converged_u_) {
            FALL_N_PETSC_CHECK(VecDuplicate(
                previous_converged_u_.get(),
                checkpoint.previous_converged_displacement.ptr()));
            FALL_N_PETSC_CHECK(VecCopy(
                previous_converged_u_.get(),
                checkpoint.previous_converged_displacement.get()));
        }

        checkpoint.model = model_->capture_checkpoint();
        checkpoint.p_done = p_done_;
        checkpoint.previous_converged_p = previous_converged_p_;
        checkpoint.step_count = step_count_;
        checkpoint.dp = dp_;
        checkpoint.max_bisections = max_bisections_;
        checkpoint.incremental_active = incremental_active_;
        checkpoint.has_previous_converged_displacement =
            has_previous_converged_u_;
        checkpoint.previous_requested_step_diagnostics =
            previous_requested_step_diagnostics_;
        return checkpoint;
    }

    void restore_checkpoint(const checkpoint_type& checkpoint) {
        setup();
        revert_state();

        if (checkpoint.displacement && U) {
            FALL_N_PETSC_CHECK(VecCopy(checkpoint.displacement.get(), U.get()));
        }

        if (checkpoint.external_force && f_ext) {
            FALL_N_PETSC_CHECK(VecCopy(checkpoint.external_force.get(),
                                       f_ext.get()));
        }

        ctx_.f_ext = f_ext;
        p_done_ = checkpoint.p_done;
        previous_converged_p_ = checkpoint.previous_converged_p;
        step_count_ = checkpoint.step_count;
        dp_ = checkpoint.dp;
        max_bisections_ = checkpoint.max_bisections;
        incremental_active_ = checkpoint.incremental_active;
        has_previous_converged_u_ =
            checkpoint.has_previous_converged_displacement;
        previous_requested_step_diagnostics_ =
            checkpoint.previous_requested_step_diagnostics;
        previous_converged_u_ = petsc::OwnedVec{};
        if (checkpoint.previous_converged_displacement) {
            FALL_N_PETSC_CHECK(VecDuplicate(
                checkpoint.previous_converged_displacement.get(),
                previous_converged_u_.ptr()));
            FALL_N_PETSC_CHECK(VecCopy(
                checkpoint.previous_converged_displacement.get(),
                previous_converged_u_.get()));
        }
        model_->restore_checkpoint(checkpoint.model);
    }

    /// Register a callback invoked after each converged load step.
    /// Signature: void(int step, double lambda, const ModelT& model).
    void set_step_callback(StepCallback cb) { step_callback_ = std::move(cb); }
    void set_failed_attempt_callback(FailedAttemptCallback cb) {
        failed_attempt_callback_ = std::move(cb);
    }

    /// Register penalty coupling hooks called after standard element assembly.
    void set_residual_hook(ResidualHook hook) { residual_hook_ = std::move(hook); }
    void set_global_residual_hook(GlobalResidualHook hook) {
        global_residual_hook_ = std::move(hook);
    }
    void set_jacobian_hook(JacobianHook hook) { jacobian_hook_ = std::move(hook); }

    /// Register a structured observer (start/step/end lifecycle).
    /// Accepts any observer-like object (CompositeObserver, DynamicObserverList, etc.)
    template <typename Obs>
    void set_observer(Obs& obs) {
        observer_ = fall_n::make_observer_callback<ModelT>(obs);
    }

    /// Register an ObserverCallback directly.
    void set_observer(fall_n::ObserverCallback<ModelT> cb) {
        observer_ = std::move(cb);
    }

    /// Query SNES convergence reason after solve (positive = converged).
    SNESConvergedReason converged_reason() const {
        SNESConvergedReason reason;
        SNESGetConvergedReason(snes_, &reason);
        return reason;
    }

    /// Number of SNES iterations from the last solve.
    PetscInt num_iterations() const {
        PetscInt its;
        SNESGetIterationNumber(snes_, &its);
        return its;
    }

    /// Euclidean norm of the last SNES residual vector, when available.
    double function_norm() const {
        if (!snes_ || !R_vec) {
            return 0.0;
        }
        PetscReal norm = 0.0;
        Vec residual = nullptr;
        SNESGetFunction(snes_, &residual, nullptr, nullptr);
        if (!residual) {
            residual = R_vec.get();
        }
        if (residual) {
            VecNorm(residual, NORM_2, &norm);
        }
        return static_cast<double>(norm);
    }

    /// Borrowed view of the current global unknown vector.
    ///
    /// This is a solver-extension seam: second-generation continuation
    /// policies can seed their own algebra from the exact vector that SNES
    /// would otherwise update. The returned handle is non-owning and remains
    /// valid only while this analysis object owns its PETSc storage.
    [[nodiscard]] Vec solution_vector()
    {
        setup();
        return U.get();
    }

    /// Borrowed view of the current external-force vector in global algebra.
    [[nodiscard]] Vec external_force_vector()
    {
        setup();
        return f_ext.get();
    }

    /// Borrowed view of the reusable tangent matrix used by SNES.
    [[nodiscard]] Mat tangent_matrix()
    {
        setup();
        return J.get();
    }

    /// Clone the current global unknown vector.
    [[nodiscard]] petsc::OwnedVec clone_solution_vector()
    {
        setup();
        petsc::OwnedVec out;
        FALL_N_PETSC_CHECK(VecDuplicate(U.get(), out.ptr()));
        FALL_N_PETSC_CHECK(VecCopy(U.get(), out.get()));
        return out;
    }

    /// Allocate a global vector compatible with the SNES residual/unknowns.
    [[nodiscard]] petsc::OwnedVec create_global_vector()
    {
        setup();
        petsc::OwnedVec out;
        FALL_N_PETSC_CHECK(VecDuplicate(U.get(), out.ptr()));
        FALL_N_PETSC_CHECK(VecSet(out.get(), 0.0));
        return out;
    }

    /// Allocate a tangent matrix compatible with the active DMPlex layout.
    [[nodiscard]] petsc::OwnedMat create_tangent_matrix()
    {
        setup();
        petsc::OwnedMat out;
        FALL_N_PETSC_CHECK(DMCreateMatrix(model_->get_plex(), out.ptr()));
        FALL_N_PETSC_CHECK(MatSetOption(
            out.get(), MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
        return out;
    }

    /// Replace the current global unknown vector without committing material
    /// state. This is useful when an external continuation kernel accepts a
    /// correction and wants the standard analysis/model state to observe it.
    void set_solution_vector(Vec source)
    {
        setup();
        FALL_N_PETSC_CHECK(VecCopy(source, U.get()));
        sync_model_state_from_solution_();
    }

    /// Apply the stored incremental control law at a trial parameter value
    /// without advancing p_done_ or committing constitutive history.
    ///
    /// The control law is the same absolute law installed by
    /// begin_incremental(...). For the XFEM reduced-column benchmark this means
    /// the top-face Dirichlet drift and axial force split are evaluated exactly
    /// as in the SNES path. This method deliberately does not call SNESSolve().
    void apply_incremental_control_parameter(double p)
    {
        setup();
        if (!incremental_active_ || !apply_fn_ || !f_full_) {
            throw std::logic_error(
                "NonlinearAnalysis::apply_incremental_control_parameter "
                "requires begin_incremental(...) first");
        }
        apply_fn_(p, f_full_, f_ext, model_);
        ctx_.f_ext = f_ext;
    }

    /// Evaluate R(u, p) with the same assembly path used by PETSc SNES.
    ///
    /// The caller controls the current load/control parameter through
    /// apply_incremental_control_parameter(...). No material state is committed
    /// here; callers implementing trial continuation should checkpoint/rollback
    /// around rejected states just as solve_incremental() does internally.
    void evaluate_residual_at(Vec u_global, Vec residual_out)
    {
        setup();
        FALL_N_PETSC_CHECK(
            FormResidual(nullptr, u_global, residual_out, &ctx_));
    }

    /// Evaluate K_t(u, p) with the same tangent assembly path used by SNES.
    void evaluate_tangent_at(Vec u_global, Mat tangent_out)
    {
        setup();
        FALL_N_PETSC_CHECK(
            FormJacobian(nullptr, u_global, tangent_out, tangent_out, &ctx_));
    }

    /// Expose rollback of non-committed trial material buffers to external
    /// continuation drivers. This mirrors the safety step used after failed
    /// SNES attempts.
    void revert_trial_state()
    {
        revert_state();
    }

    /// Accept a solution produced by an external nonlinear kernel.
    ///
    /// This is the commit seam used by bordered/mixed-control solvers that
    /// reuse NonlinearAnalysis assembly but do not call SNESSolve(). The caller
    /// is responsible for having applied the corresponding incremental control
    /// parameter before acceptance.
    void accept_external_solution_step(
        Vec accepted_solution,
        double accepted_p,
        IncrementStepDiagnostics diagnostics = {})
    {
        setup();
        FALL_N_PETSC_CHECK(VecCopy(accepted_solution, U.get()));
        sync_model_state_from_solution_();
        if (auto_commit_) {
            commit_state();
        }
        p_done_ = accepted_p;
        ++step_count_;
        diagnostics.p_target = accepted_p;
        diagnostics.last_attempt_p_target = accepted_p;
        diagnostics.accepted_substep_count =
            std::max(diagnostics.accepted_substep_count, 1);
        diagnostics.converged = true;
        last_increment_step_diagnostics_ = diagnostics;
        previous_requested_step_diagnostics_ = diagnostics;
        model_->update_elements_state();
        emit_requested_step_callbacks_();
    }

    // ─── Setup (call before solve, or called automatically) ──────

    void setup() {
        if (is_setup_) return;

        DM dm = model_->get_plex();

        FALL_N_PETSC_CHECK(DMCreateGlobalVector(dm, U.ptr()));
        FALL_N_PETSC_CHECK(VecDuplicate(U, R_vec.ptr()));
        FALL_N_PETSC_CHECK(VecDuplicate(U, f_ext.ptr()));
        FALL_N_PETSC_CHECK(DMCreateMatrix(dm, J.ptr()));
        FALL_N_PETSC_CHECK(MatSetOption(J, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

        VecSet(U, 0.0);
        VecSet(R_vec, 0.0);

        // Assemble global external forces
        VecSet(f_ext, 0.0);
        DMLocalToGlobal(dm, model_->force_vector(), ADD_VALUES, f_ext);

          ctx_ = {model_, f_ext, &residual_hook_, &global_residual_hook_, &jacobian_hook_};

        SNESSetFunction(snes_, R_vec, FormResidual, &ctx_);
        SNESSetJacobian(snes_, J, J, FormJacobian, &ctx_);

        is_setup_ = true;
    }

    // ─── Single load step solve ─────────────────────────────────
    //
    //  Solves the nonlinear equilibrium R(u) = f_int(u) - f_ext = 0
    //  in a single step from u = 0.
    //
    //  Returns:
    //    true  — SNES converged; material state committed, model updated.
    //    false — SNES diverged; u is left at the SNES output but material
    //            state is NOT committed (prevents corruption).
    //
    //  After calling solve(), you can always query converged_reason()
    //  and num_iterations() for diagnostics.

    [[nodiscard]] const IncrementStepDiagnostics&
    last_increment_step_diagnostics() const noexcept
    {
        return last_increment_step_diagnostics_;
    }

    void set_solve_profiles(std::vector<NonlinearSolveProfile> profiles)
    {
        solve_profiles_ = std::move(profiles);
    }

    [[nodiscard]] const std::vector<NonlinearSolveProfile>&
    solve_profiles() const noexcept
    {
        return solve_profiles_;
    }

    void set_increment_predictor(IncrementPredictorSettings settings)
    {
        increment_predictor_ = settings;
    }

    [[nodiscard]] const IncrementPredictorSettings&
    increment_predictor() const noexcept
    {
        return increment_predictor_;
    }

    /// Access the performance timer (read timing data after solve).
    const AnalysisTimer& timer() const { return timer_; }
          AnalysisTimer& timer()       { return timer_; }

    bool solve() {
        timer_.start("setup");
        setup();
        timer_.stop("setup");

        last_increment_step_diagnostics_ = IncrementStepDiagnostics{};
        VecSet(U, 0.0);

        timer_.start("solve");
        SNESConvergedReason reason{SNES_CONVERGED_ITERATING};
        double residual_norm = 0.0;
        fall_n::NonlinearSolveAttemptAssessment acceptance{};
        for (const auto& profile : active_solve_profiles_()) {
            apply_solve_profile_(profile);
            SNESSolve(snes_, nullptr, U);
            SNESGetConvergedReason(snes_, &reason);
            residual_norm = function_norm();
            acceptance =
                fall_n::assess_nonlinear_solve_attempt(
                    profile, reason, residual_norm);
            record_solver_profile_diagnostics_(
                profile, reason, residual_norm, acceptance);
            if (acceptance.accepted) {
                break;
            }
        }
        timer_.stop("solve");

        if (acceptance.accepted) {
            // Converged — safe to commit
            timer_.start("commit");
            if (auto_commit_)
                commit_state();
            else
                sync_model_state_from_solution_();
            timer_.stop("commit");
            last_increment_step_diagnostics_.converged = true;
            if (acceptance.accepted_by_small_residual_policy) {
                incremental_printf_(
                    "  NonlinearAnalysis::solve() accepted small-residual "
                    "policy (reason=%d, ||F||=%.6e <= %.6e)\n",
                    static_cast<int>(reason),
                    residual_norm,
                    acceptance.accepted_function_norm_threshold);
            }
            model_->update_elements_state();
            return true;
        }

        // Diverged — revert material state and do NOT commit
        revert_state();
        incremental_printf_(
            "  *** NonlinearAnalysis::solve() DIVERGED (reason=%d) ***\n",
            static_cast<int>(reason));
        model_->update_elements_state();  // update for post-processing (u may be garbage)
        return false;
    }

    // ─── Incremental stepping with automatic bisection ─────────
    //
    //  Advances the control parameter p from 0 to 1 in N equal steps.
    //  The actual system state at each step is determined by the
    //  injected IncrementalControlPolicy (default: LoadControl).
    //
    //    step k:  p = k/N      k = 1, ..., N
    //             scheme.apply(p, f_full, f_ext, model)
    //
    //  At each step, SNES solves R(u) = f_int(u) - f_ext = 0 with the
    //  previous converged u as the initial guess.
    //
    //  AUTOMATIC BISECTION (adaptive step-cutting):
    //    When a step [p_prev → p_target] diverges, the algorithm:
    //      1. Reverts u to the last converged state.
    //      2. Restores the scheme to p_prev (absolute apply).
    //      3. Splits the step into two half-steps:
    //            [p_prev → p_mid]  then  [p_mid → p_target]
    //      4. Retries each half, recursively up to max_bisections levels.
    //
    //  Returns:
    //    true  — all steps converged (possibly after bisection).
    //    false — at least one step failed even after max bisections.
    //
    //  Usage:
    //    nl.solve_incremental(10);                     // LoadControl (default)
    //    nl.solve_incremental(20, 4, LoadControl{});   // explicit
    //    nl.solve_incremental(20, 4, DisplacementControl{42, 0, 0.01});
    //    nl.solve_incremental(10, 4, make_control(my_lambda));

    /// Backward-compatible overload (pure load control).
    bool solve_incremental(int num_steps, int max_bisections = 4) {
        return solve_incremental(num_steps, max_bisections, LoadControl{});
    }

    /// Incremental solve with injectable control scheme.
    ///
    /// The control parameter p advances from 0 to 1 in num_steps equal
    /// increments.  At each sub-step, scheme.apply(p, f_full, f_ext, model)
    /// sets the system state (external forces, imposed displacements, etc.).
    ///
    /// CS must satisfy IncrementalControlPolicy<CS, ModelT>.
    template <typename CS>
    bool solve_incremental(int num_steps, int max_bisections, CS scheme) {
        static_assert(IncrementalControlPolicy<CS, ModelT>,
            "CS must satisfy IncrementalControlPolicy<CS, ModelT>");

        begin_incremental(num_steps, max_bisections, std::move(scheme));
        bool all_ok = true;

        incremental_printf_(
            "  Incremental solve: %d steps, max bisection depth = %d\n",
            num_steps, max_bisections);

        if (observer_.on_start) observer_.on_start(*model_);

        for (int step = 1; step <= num_steps; ++step) {
            const double p_target = static_cast<double>(step) / num_steps;

            incremental_printf_(
                "\n  ── Step %d/%d: p = %.4f → %.4f ──\n",
                step, num_steps, p_done_, p_target);

            last_increment_step_diagnostics_ = IncrementStepDiagnostics{
                .p_start = p_done_,
                .p_target = p_target,
            };

            if (!advance_requested_step_(p_target)) {
                all_ok = false;
                incremental_printf_(
                    "\n  *** ABORT at step %d/%d (p=%.4f) — "
                    "bisection exhausted after %d levels ***\n",
                    step, num_steps, p_target, max_bisections);
                break;
            }
        }

        incremental_printf_(
            "  Incremental solve %s at p = %.4f\n",
            all_ok ? "COMPLETED" : "ABORTED", p_done_);

        if (observer_.on_end) observer_.on_end(*model_);

        return all_ok;
    }

private:

    // ─── Bisection engine (recursive, policy-driven) ──────────
    //
    //  Attempts to advance the solution from p_current to p_target
    //  using the injected control scheme.
    //
    //  Algorithm:
    //    1. Snapshot u.
    //    2. scheme.apply(p_target, ...) → set system state, SNESSolve.
    //    3. Converged → commit, return true.
    //    4. Diverged → revert u, revert material, scheme.apply(p_current).
    //       Bisect: p_mid = (p_current + p_target) / 2, recurse.
    //    5. budget == 0 → give up, return false.
    //
    //  The scheme's apply() is ABSOLUTE: no save/restore is needed
    //  in the scheme itself — only p determines the system state.

    template <typename CS>
    bool advance_to_p_(CS& scheme, Vec f_full,
                       double p_current, double p_target,
                       int bisections_left)
    {
        // ── 1. Snapshot current displacement for rollback ─────────
        petsc::OwnedVec U_snap{};
        VecDuplicate(U, U_snap.ptr());
        VecCopy(U, U_snap);

        // ── 2. Apply control scheme and solve ────────────────────
        scheme.apply(p_target, f_full, f_ext, model_);
        ctx_.f_ext = f_ext;

        SNESSolve(snes_, nullptr, U);

        SNESConvergedReason reason;
        SNESGetConvergedReason(snes_, &reason);
        PetscInt its;
        SNESGetIterationNumber(snes_, &its);
        const double residual_norm = function_norm();
        const auto& profile =
            fall_n::select_nonlinear_solve_profile(active_solve_profiles_(), 0);
        const auto acceptance =
            fall_n::assess_nonlinear_solve_attempt(
                profile, reason, residual_norm);

        // ── 3. Converged? Commit and return ──────────────────────
        if (acceptance.accepted) {
            if (auto_commit_)
                commit_state();
            else
                sync_model_state_from_solution_();

            incremental_printf_(
                "    p=%.6f  converged  (reason=%d, %d iterations%s)\n",
                p_target,
                static_cast<int>(reason),
                static_cast<int>(its),
                acceptance.accepted_by_small_residual_policy
                    ? ", accepted-by-small-residual-policy"
                    : "");
            return true;
        }

        // ── 4. Diverged — revert to pre-step state ──────────────
        VecCopy(U_snap, U);
        revert_state();

        // Restore the scheme to the last converged control parameter
        // (absolute apply — no incremental state to unwind).
        scheme.apply(p_current, f_full, f_ext, model_);
        ctx_.f_ext = f_ext;

        incremental_printf_(
            "    p=%.6f  DIVERGED   (reason=%d, %d iterations)\n",
            p_target,
            static_cast<int>(reason),
            static_cast<int>(its));

        // ── 5. Bisection budget exhausted? ───────────────────────
        if (bisections_left <= 0) {
            incremental_printf_(
                "    No bisection budget remaining at p=%.6f.\n",
                p_target);
            return false;
        }

        // ── 6. Bisect: split [p_current → p_target] into halves ──
        double p_mid = 0.5 * (p_current + p_target);

        incremental_printf_(
            "    Bisecting: p = %.6f → %.6f → %.6f  "
            "(depth remaining: %d)\n",
            p_current, p_mid, p_target,
            bisections_left - 1);

        // First half: [p_current → p_mid]
        if (!advance_to_p_(scheme, f_full, p_current, p_mid,
                           bisections_left - 1))
            return false;

        // Second half: [p_mid → p_target]
        if (!advance_to_p_(scheme, f_full, p_mid, p_target,
                           bisections_left - 1))
            return false;

        return true;
    }

    // ─── Bisection engine (type-erased, for single-step API) ─────
    //
    //  Same algorithm as advance_to_p_ but uses the stored apply_fn_
    //  and f_full_ members instead of template parameters.

    bool advance_step_impl_(double p_current, double p_target,
                            int bisections_left,
                            int bisection_level = 0)
    {
        last_increment_step_diagnostics_.last_attempt_p_start = p_current;
        last_increment_step_diagnostics_.last_attempt_p_target = p_target;

        petsc::OwnedVec U_snap{};
        VecDuplicate(U, U_snap.ptr());
        VecCopy(U, U_snap);

        SNESConvergedReason reason{SNES_CONVERGED_ITERATING};
        PetscInt its{0};
        double residual_norm = 0.0;

        for (const auto& profile : active_solve_profiles_()) {
            revert_state();
            apply_fn_(p_target, f_full_, f_ext, model_);
            ctx_.f_ext = f_ext;
            const bool used_predictor = apply_increment_predictor_(
                U_snap.get(), p_current, p_target, bisection_level, profile);
            apply_solve_profile_(profile);

            incremental_printf_(
                "    trying solver profile '%s' at p=%.6f (%s guess)\n",
                profile.label.c_str(),
                p_target,
                used_predictor ? "predicted" : "current-state");

            SNESSolve(snes_, nullptr, U);

            SNESGetConvergedReason(snes_, &reason);
            SNESGetIterationNumber(snes_, &its);
            residual_norm = function_norm();
            const auto acceptance =
                fall_n::assess_nonlinear_solve_attempt(
                    profile, reason, residual_norm);
            record_solver_profile_diagnostics_(
                profile, reason, residual_norm, acceptance);
            last_increment_step_diagnostics_.max_bisection_level =
                std::max(last_increment_step_diagnostics_.max_bisection_level,
                         bisection_level);

            if (acceptance.accepted) {
                if (auto_commit_)
                    commit_state();
                else
                    sync_model_state_from_solution_();
                capture_previous_converged_state_(U_snap.get(), p_current);
                last_increment_step_diagnostics_.accepted_substep_count += 1;
                last_increment_step_diagnostics_.total_newton_iterations +=
                    static_cast<int>(its);
                last_increment_step_diagnostics_.converged = true;
                incremental_printf_(
                    "    p=%.6f  converged  (profile=%s, reason=%d, %d iterations%s)\n",
                    p_target,
                    profile.label.c_str(),
                    static_cast<int>(reason),
                    static_cast<int>(its),
                    acceptance.accepted_by_small_residual_policy
                        ? ", accepted-by-small-residual-policy"
                        : "");
                return true;
            }
        }

        // Capture the failed trial state before rollback so validation audits
        // can inspect the exact section/fiber configuration that exhausted
        // the current continuation slice.
        sync_model_state_from_solution_();
        if (failed_attempt_callback_) {
            failed_attempt_callback_(*model_, last_increment_step_diagnostics_);
        }

        VecCopy(U_snap, U);
        revert_state();

        apply_fn_(p_current, f_full_, f_ext, model_);
        ctx_.f_ext = f_ext;
        sync_model_state_from_solution_();

        incremental_printf_(
            "    p=%.6f  DIVERGED   (profile=%s, reason=%d, %d iterations)\n",
            p_target,
            last_increment_step_diagnostics_.last_solver_profile_label.c_str(),
            static_cast<int>(reason),
            static_cast<int>(its));
        last_increment_step_diagnostics_.failed_attempt_count += 1;
        last_increment_step_diagnostics_.converged = false;

        if (bisections_left <= 0) {
            incremental_printf_(
                "    No bisection budget remaining at p=%.6f.\n",
                p_target);
            return false;
        }

        double p_mid = 0.5 * (p_current + p_target);

        incremental_printf_(
            "    Bisecting: p = %.6f → %.6f → %.6f  "
            "(depth remaining: %d)\n",
            p_current, p_mid, p_target,
            bisections_left - 1);

        if (!advance_step_impl_(p_current, p_mid,
                                bisections_left - 1,
                                bisection_level + 1))
            return false;

        if (!advance_step_impl_(p_mid, p_target,
                                bisections_left - 1,
                                bisection_level + 1))
            return false;

        return true;
    }

    void emit_requested_step_callbacks_()
    {
        if (auto_commit_ && step_callback_) {
            step_callback_(step_count_, p_done_, *model_);
        }
        if (auto_commit_ && observer_.on_step) {
            fall_n::StepEvent ev{
                static_cast<PetscInt>(step_count_),
                p_done_,
                U.get(),
                nullptr};
            observer_.on_step(ev, *model_);
        }
    }

    bool advance_requested_step_(double requested_target)
    {
        constexpr double tol = 1.0e-14;

        const double outer_target = std::clamp(requested_target, p_done_, 1.0);
        if (outer_target <= p_done_ + tol) {
            return false;
        }

        const auto outer_checkpoint = capture_checkpoint();
        double local_increment = outer_target - p_done_;
        if (increment_adaptation_.enabled) {
            local_increment = std::min(
                clamp_increment_size_(std::max(dp_, minimum_increment_size_())),
                local_increment);
        }

        bool used_cutback = false;
        int cutbacks_used = 0;

        while (p_done_ < outer_target - tol) {
            local_increment = std::min(local_increment, outer_target - p_done_);
            const double p_subtarget = p_done_ + local_increment;

            if (advance_step_impl_(p_done_, p_subtarget, max_bisections_, 0)) {
                p_done_ = p_subtarget;
                continue;
            }

            if (!increment_adaptation_.enabled ||
                cutbacks_used >= increment_adaptation_.max_cutbacks_per_step) {
                restore_checkpoint(outer_checkpoint);
                last_increment_step_diagnostics_.converged = false;
                model_->update_elements_state();
                return false;
            }

            const double next_increment =
                clamp_increment_size_(
                    local_increment * increment_adaptation_.cutback_factor);
            if (next_increment >= local_increment - tol ||
                next_increment < minimum_increment_size_() - tol) {
                restore_checkpoint(outer_checkpoint);
                last_increment_step_diagnostics_.converged = false;
                model_->update_elements_state();
                return false;
            }

            used_cutback = true;
            ++cutbacks_used;
            incremental_printf_(
                "    Requested-step cutback: Δp = %.6f → %.6f  "
                "(remaining to target: %.6f, cutback %d/%d)\n",
                local_increment,
                next_increment,
                outer_target - p_done_,
                cutbacks_used,
                increment_adaptation_.max_cutbacks_per_step);
            local_increment = next_increment;
        }

        ++step_count_;
        adapt_next_increment_after_requested_step_(local_increment, used_cutback);
        previous_requested_step_diagnostics_ = last_increment_step_diagnostics_;
        emit_requested_step_callbacks_();
        model_->update_elements_state();
        return true;
    }

public:

    // =================================================================
    //  Single-Step API (incremental)
    // =================================================================

    /// Initialize the incremental stepping state.
    ///
    /// Must be called before step() / step_to() / step_n().
    /// Sets up the SNES solver, zeroes the solution, captures the
    /// reference force vector, and stores the control scheme.
    ///
    /// After begin_incremental(), call step() or step_to(p) to advance.
    void begin_incremental(int num_steps, int max_bisections = 4) {
        begin_incremental(num_steps, max_bisections, LoadControl{});
    }

    /// Initialize incremental stepping with an injectable control scheme.
    template <typename CS>
    void begin_incremental(int num_steps, int max_bisections, CS scheme) {
        static_assert(IncrementalControlPolicy<CS, ModelT>,
            "CS must satisfy IncrementalControlPolicy<CS, ModelT>");

        setup();
        VecSet(U, 0.0);

        // Capture reference force at p = 1.0
        f_full_ = petsc::OwnedVec{};
        VecDuplicate(f_ext, f_full_.ptr());
        VecCopy(f_ext, f_full_);

        dp_              = 1.0 / num_steps;
        initial_dp_      = dp_;
        max_bisections_  = max_bisections;
        p_done_          = 0.0;
        step_count_      = 0;
        easy_step_streak_ = 0;
        last_increment_step_diagnostics_ = IncrementStepDiagnostics{};
        reset_increment_predictor_history_();

        // Type-erase the scheme into a std::function
        apply_fn_ = [s = std::move(scheme)](
            double p, Vec ff, Vec fe, ModelT* m) mutable {
            s.apply(p, ff, fe, m);
        };

        incremental_active_ = true;

        incremental_printf_(
            "  begin_incremental: %d steps (dp=%.6f), "
            "max bisection depth = %d\n",
            num_steps, dp_, max_bisections_);
    }

    /// Advance exactly one load increment.
    ///
    /// Requires a prior call to begin_incremental().
    /// Returns true if the step converged (possibly after bisection).
    bool step() {
        assert(incremental_active_ &&
            "call begin_incremental() before step()");

        if (p_done_ >= 1.0 - 1.0e-14) {
            return false;
        }

        const double p_target = std::min(p_done_ + dp_, 1.0);

        incremental_printf_(
            "\n  ── Step %d: p = %.4f → %.4f ──\n",
            step_count_ + 1, p_done_, p_target);

        last_increment_step_diagnostics_ = IncrementStepDiagnostics{
            .p_start = p_done_,
            .p_target = p_target,
        };

        return advance_requested_step_(p_target);
    }

    /// Advance until p >= p_target, respecting the StepDirector.
    ///
    /// Returns the StepVerdict that caused the loop to exit:
    ///   - Continue — reached p_target normally
    ///   - Pause    — a director condition fired before p_target
    ///   - Stop     — divergence or director requested termination
    fall_n::StepVerdict step_to(double p_target) {
        return step_to(p_target, director_);
    }

    /// Advance until p >= p_target with an explicit director.
    fall_n::StepVerdict step_to(double p_target,
                                const fall_n::StepDirector<ModelT>& director)
    {
        assert(incremental_active_ &&
            "call begin_incremental() before step_to()");

        while (p_done_ < p_target - 1e-14) {
            const double step_target =
                std::min(p_done_ + std::max(dp_, 0.0), p_target);

            last_increment_step_diagnostics_ = IncrementStepDiagnostics{
                .p_start = p_done_,
                .p_target = step_target,
            };

            if (!advance_requested_step_(step_target)) {
                return fall_n::StepVerdict::Stop;
            }

            if (director) {
                fall_n::StepEvent ev{
                    static_cast<PetscInt>(step_count_),
                    p_done_,
                    U,
                    nullptr   // no velocity in static analysis
                };
                auto verdict = director(ev, *model_);
                if (verdict != fall_n::StepVerdict::Continue) {
                    return verdict;
                }
            }
        }
        return fall_n::StepVerdict::Continue;
    }

    /// Advance exactly n load increments, respecting the StepDirector.
    fall_n::StepVerdict step_n(int n) {
        return step_n(n, director_);
    }

    /// Advance exactly n load increments with an explicit director.
    fall_n::StepVerdict step_n(int n,
                               const fall_n::StepDirector<ModelT>& director)
    {
        assert(incremental_active_ &&
            "call begin_incremental() before step_n()");

        for (int i = 0; i < n; ++i) {
            if (!step()) {
                return fall_n::StepVerdict::Stop;
            }

            if (director) {
                fall_n::StepEvent ev{
                    static_cast<PetscInt>(step_count_),
                    p_done_,
                    U,
                    nullptr
                };
                auto verdict = director(ev, *model_);
                if (verdict != fall_n::StepVerdict::Continue) {
                    return verdict;
                }
            }
        }
        return fall_n::StepVerdict::Continue;
    }


    // =================================================================
    //  Accessors for SteppableSolver concept
    // =================================================================

    /// Current control parameter (0 = unloaded, 1 = full load).
    double current_time() const { return p_done_; }

    /// Current step count since begin_incremental().
    PetscInt current_step() const { return static_cast<PetscInt>(step_count_); }


    // =================================================================
    //  Runtime reconfiguration
    // =================================================================

    /// Set a persistent StepDirector for step_to() / step_n() calls.
    void set_director(fall_n::StepDirector<ModelT> dir) {
        director_ = std::move(dir);
    }

    /// Get the current increment size dp.
    double get_increment_size() const { return dp_; }

    /// Change the increment size at runtime.
    void set_increment_size(double dp) { dp_ = dp; }

    void set_increment_adaptation(IncrementAdaptationSettings settings)
    {
        increment_adaptation_ = std::move(settings);
    }

    [[nodiscard]] const IncrementAdaptationSettings&
    increment_adaptation() const noexcept
    {
        return increment_adaptation_;
    }

    /// Enable or silence the incremental-step trace emitted by this solver.
    ///
    /// This only affects the solver's own progress messages; higher-level
    /// observers and validation summaries remain under their own control.
    void set_incremental_logging(bool enabled) noexcept
    {
        incremental_logging_enabled_ = enabled;
    }


    // =================================================================
    //  State transfer
    // =================================================================

    /// Get a snapshot of the current analysis state.
    /// The returned Vecs are borrowed — valid only while this object lives.
    fall_n::AnalysisState get_analysis_state() const {
        return fall_n::AnalysisState{
            .displacement = U,
            .velocity     = nullptr,
            .time         = p_done_,
            .step         = static_cast<PetscInt>(step_count_)
        };
    }


    // ─── Constructor / Destructor ─────────────────────────────────

    explicit NonlinearAnalysis(ModelT* model) : model_{model} {
        FALL_N_PETSC_CHECK(SNESCreate(PETSC_COMM_WORLD, snes_.ptr()));
        FALL_N_PETSC_CHECK(SNESSetFromOptions(snes_.get()));
        FALL_N_PETSC_CHECK(SNESSetDM(snes_.get(), model_->get_plex()));
    }

    NonlinearAnalysis() = default;

    ~NonlinearAnalysis() = default;

    // Non-copyable (owns PETSc objects)
    NonlinearAnalysis(const NonlinearAnalysis&)            = delete;
    NonlinearAnalysis& operator=(const NonlinearAnalysis&) = delete;
    NonlinearAnalysis(NonlinearAnalysis&&)                 = default;
    NonlinearAnalysis& operator=(NonlinearAnalysis&&)      = default;
};


#endif // FALL_N_SRC_ANALYSIS_NLANALYSIS_HH
