#ifndef FALL_N_SRC_ANALYSIS_NLANALYSIS_HH
#define FALL_N_SRC_ANALYSIS_NLANALYSIS_HH

#include <cstddef>
#include <functional>
#include <petscsnes.h>

#include "../model/Model.hh"
#include "../petsc/PetscRaii.hh"
#include "../utils/Benchmark.hh"
#include "AnalysisRouteAudit.hh"
#include "AnalysisObserver.hh"
#include "IncrementalControl.hh"
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

    // ─── Step callback (optional) ─────────────────────────────────
    //  Invoked after each converged load step in solve_incremental().
    //  Arguments: (step_number, lambda, model_ref)
    using StepCallback = std::function<void(int, double, const ModelT&)>;
    StepCallback step_callback_{};

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
    fall_n::StepDirector<ModelT> director_{};

public:
    struct SolverCheckpoint {
        typename ModelT::checkpoint_type model{};
        petsc::OwnedVec displacement{};
        petsc::OwnedVec external_force{};
        double p_done{0.0};
        int    step_count{0};
        double dp{0.0};
        int    max_bisections{0};
        bool   incremental_active{false};
    };

    using checkpoint_type = SolverCheckpoint;

private:

    // ─── Penalty coupling hooks (optional) ────────────────────────
    //  Called after standard element assembly in FormResidual / FormJacobian
    //  to inject additional coupling terms (e.g. penalty rebar coupling).
    //    residual_hook_(u_local, f_int_local, dm)
    //    jacobian_hook_(u_local, J_mat, dm)
    using ResidualHook = std::function<void(Vec, Vec, DM)>;
    using JacobianHook = std::function<void(Vec, Mat, DM)>;
    ResidualHook residual_hook_{};
    JacobianHook jacobian_hook_{};

    // ─── SNES callback context ────────────────────────────────────

    struct Context {
        ModelT* model;
        Vec     f_ext;
        ResidualHook* residual_hook;
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

        checkpoint.model = model_->capture_checkpoint();
        checkpoint.p_done = p_done_;
        checkpoint.step_count = step_count_;
        checkpoint.dp = dp_;
        checkpoint.max_bisections = max_bisections_;
        checkpoint.incremental_active = incremental_active_;
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
        step_count_ = checkpoint.step_count;
        dp_ = checkpoint.dp;
        max_bisections_ = checkpoint.max_bisections;
        incremental_active_ = checkpoint.incremental_active;
        model_->restore_checkpoint(checkpoint.model);
    }

    /// Register a callback invoked after each converged load step.
    /// Signature: void(int step, double lambda, const ModelT& model).
    void set_step_callback(StepCallback cb) { step_callback_ = std::move(cb); }

    /// Register penalty coupling hooks called after standard element assembly.
    void set_residual_hook(ResidualHook hook) { residual_hook_ = std::move(hook); }
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

        ctx_ = {model_, f_ext, &residual_hook_, &jacobian_hook_};

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

    /// Access the performance timer (read timing data after solve).
    const AnalysisTimer& timer() const { return timer_; }
          AnalysisTimer& timer()       { return timer_; }

    bool solve() {
        timer_.start("setup");
        setup();
        timer_.stop("setup");

        VecSet(U, 0.0);

        timer_.start("solve");
        SNESSolve(snes_, nullptr, U);
        timer_.stop("solve");

        SNESConvergedReason reason;
        SNESGetConvergedReason(snes_, &reason);

        if (reason > 0) {
            // Converged — safe to commit
            timer_.start("commit");
            if (auto_commit_)
                commit_state();
            else
                sync_model_state_from_solution_();
            timer_.stop("commit");
            model_->update_elements_state();
            return true;
        }

        // Diverged — revert material state and do NOT commit
        revert_state();
        PetscPrintf(PETSC_COMM_WORLD,
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

        setup();
        VecSet(U, 0.0);

        // Save a copy of the full (un-scaled) external force vector.
        // This is the reference load at p = 1.0.
        petsc::OwnedVec f_full{};
        VecDuplicate(f_ext, f_full.ptr());
        VecCopy(f_ext, f_full);

        double p_done = 0.0;   // last converged control parameter
        bool   all_ok = true;
        int    steps_done = 0;

        PetscPrintf(PETSC_COMM_WORLD,
            "  Incremental solve: %d steps, max bisection depth = %d\n",
            num_steps, max_bisections);

        if (observer_.on_start) observer_.on_start(*model_);

        for (int step = 1; step <= num_steps; ++step) {
            double p_target = static_cast<double>(step) / num_steps;

            PetscPrintf(PETSC_COMM_WORLD,
                "\n  ── Step %d/%d: p = %.4f → %.4f ──\n",
                step, num_steps, p_done, p_target);

            bool ok = advance_to_p_(scheme, f_full.get(), p_done, p_target,
                                    max_bisections);

            if (ok) {
                p_done = p_target;
                ++steps_done;
                if (auto_commit_ && step_callback_)
                    step_callback_(step, p_done, *model_);
                if (auto_commit_ && observer_.on_step) {
                    fall_n::StepEvent ev{step, p_done, U.get(), nullptr};
                    observer_.on_step(ev, *model_);
                }
            } else {
                all_ok = false;
                PetscPrintf(PETSC_COMM_WORLD,
                    "\n  *** ABORT at step %d/%d (p=%.4f) — "
                    "bisection exhausted after %d levels ***\n",
                    step, num_steps, p_target, max_bisections);
                break;
            }
        }

        model_->update_elements_state();

        // Synchronise single-step state so get_analysis_state()
        // reflects the terminal position even when called after
        // the batch solve_incremental() path.
        p_done_     = p_done;
        step_count_ = steps_done;

        PetscPrintf(PETSC_COMM_WORLD,
            "  Incremental solve %s at p = %.4f\n",
            all_ok ? "COMPLETED" : "ABORTED", p_done);

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

        // ── 3. Converged? Commit and return ──────────────────────
        if (reason > 0) {
            if (auto_commit_)
                commit_state();
            else
                sync_model_state_from_solution_();

            PetscPrintf(PETSC_COMM_WORLD,
                "    p=%.6f  converged  (reason=%d, %d iterations)\n",
                p_target,
                static_cast<int>(reason),
                static_cast<int>(its));
            return true;
        }

        // ── 4. Diverged — revert to pre-step state ──────────────
        VecCopy(U_snap, U);
        revert_state();

        // Restore the scheme to the last converged control parameter
        // (absolute apply — no incremental state to unwind).
        scheme.apply(p_current, f_full, f_ext, model_);
        ctx_.f_ext = f_ext;

        PetscPrintf(PETSC_COMM_WORLD,
            "    p=%.6f  DIVERGED   (reason=%d, %d iterations)\n",
            p_target,
            static_cast<int>(reason),
            static_cast<int>(its));

        // ── 5. Bisection budget exhausted? ───────────────────────
        if (bisections_left <= 0) {
            PetscPrintf(PETSC_COMM_WORLD,
                "    No bisection budget remaining at p=%.6f.\n",
                p_target);
            return false;
        }

        // ── 6. Bisect: split [p_current → p_target] into halves ──
        double p_mid = 0.5 * (p_current + p_target);

        PetscPrintf(PETSC_COMM_WORLD,
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
                            int bisections_left)
    {
        petsc::OwnedVec U_snap{};
        VecDuplicate(U, U_snap.ptr());
        VecCopy(U, U_snap);

        apply_fn_(p_target, f_full_, f_ext, model_);
        ctx_.f_ext = f_ext;

        SNESSolve(snes_, nullptr, U);

        SNESConvergedReason reason;
        SNESGetConvergedReason(snes_, &reason);
        PetscInt its;
        SNESGetIterationNumber(snes_, &its);

        if (reason > 0) {
            if (auto_commit_)
                commit_state();
            else
                sync_model_state_from_solution_();
            PetscPrintf(PETSC_COMM_WORLD,
                "    p=%.6f  converged  (reason=%d, %d iterations)\n",
                p_target,
                static_cast<int>(reason),
                static_cast<int>(its));
            return true;
        }

        VecCopy(U_snap, U);
        revert_state();

        apply_fn_(p_current, f_full_, f_ext, model_);
        ctx_.f_ext = f_ext;

        PetscPrintf(PETSC_COMM_WORLD,
            "    p=%.6f  DIVERGED   (reason=%d, %d iterations)\n",
            p_target,
            static_cast<int>(reason),
            static_cast<int>(its));

        if (bisections_left <= 0) {
            PetscPrintf(PETSC_COMM_WORLD,
                "    No bisection budget remaining at p=%.6f.\n",
                p_target);
            return false;
        }

        double p_mid = 0.5 * (p_current + p_target);

        PetscPrintf(PETSC_COMM_WORLD,
            "    Bisecting: p = %.6f → %.6f → %.6f  "
            "(depth remaining: %d)\n",
            p_current, p_mid, p_target,
            bisections_left - 1);

        if (!advance_step_impl_(p_current, p_mid,
                                bisections_left - 1))
            return false;

        if (!advance_step_impl_(p_mid, p_target,
                                bisections_left - 1))
            return false;

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
        max_bisections_  = max_bisections;
        p_done_          = 0.0;
        step_count_      = 0;

        // Type-erase the scheme into a std::function
        apply_fn_ = [s = std::move(scheme)](
            double p, Vec ff, Vec fe, ModelT* m) mutable {
            s.apply(p, ff, fe, m);
        };

        incremental_active_ = true;

        PetscPrintf(PETSC_COMM_WORLD,
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

        double p_target = p_done_ + dp_;
        if (p_target > 1.0 + 1e-14) return false;  // already done

        PetscPrintf(PETSC_COMM_WORLD,
            "\n  ── Step %d: p = %.4f → %.4f ──\n",
            step_count_ + 1, p_done_, p_target);

        bool ok = advance_step_impl_(p_done_, p_target, max_bisections_);

        if (ok) {
            p_done_ = p_target;
            ++step_count_;
            if (auto_commit_ && step_callback_)
                step_callback_(step_count_, p_done_, *model_);
            if (auto_commit_ && observer_.on_step) {
                fall_n::StepEvent ev{step_count_, p_done_, U.get(), nullptr};
                observer_.on_step(ev, *model_);
            }
        }

        model_->update_elements_state();
        return ok;
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
            if (!step()) {
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
