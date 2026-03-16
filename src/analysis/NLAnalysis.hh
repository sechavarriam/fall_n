#ifndef FALL_N_SRC_ANALYSIS_NLANALYSIS_HH
#define FALL_N_SRC_ANALYSIS_NLANALYSIS_HH

#include <cstddef>
#include <functional>
#include <petscsnes.h>

#include "../model/Model.hh"
#include "../petsc/PetscRaii.hh"
#include "../utils/Benchmark.hh"

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
    using ModelT = Model<MaterialPolicy, KinematicPolicy, ndofs, ElemPolicy>;
    using ElementT = typename ModelT::element_type;
    static constexpr auto dim = MaterialPolicy::dim;

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

    // ─── SNES callback context ────────────────────────────────────

    struct Context {
        ModelT* model;
        Vec     f_ext;
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

        MatAssemblyBegin(J_mat, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(J_mat, MAT_FINAL_ASSEMBLY);

        DMRestoreLocalVector(dm, &u_local);

        PetscFunctionReturn(0);
    }

    // ─── Commit material state after global convergence ─────────

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

        DMRestoreLocalVector(dm, &u_local);

        // Update model's current_state for post-processing
        DMGlobalToLocal(dm, U, INSERT_VALUES, model_->state_vector());
        VecAXPY(model_->state_vector(), 1.0, model_->imposed_solution());
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

    /// Register a callback invoked after each converged load step.
    /// Signature: void(int step, double lambda, const ModelT& model).
    void set_step_callback(StepCallback cb) { step_callback_ = std::move(cb); }

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

        ctx_ = {model_, f_ext};

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
            commit_state();
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

    // ─── Incremental loading with automatic bisection ────────────
    //
    //  Applies the external load f_ext in N equal increments:
    //
    //      step k:  f_applied = (k/N) · f_ext      k = 1, ..., N
    //
    //  At each step, SNES solves R(u) = f_int(u) - λ·f_ext = 0 with the
    //  previous converged u as the initial guess.
    //
    //  AUTOMATIC BISECTION (adaptive step-cutting):
    //    When a load step [λ_prev → λ_target] diverges, the algorithm:
    //      1. Reverts u to the last converged state (before this step).
    //      2. Splits the step into two half-steps:
    //            [λ_prev → λ_mid]  then  [λ_mid → λ_target]
    //      3. Retries each half-step, recursively bisecting up to
    //         max_bisections levels deep.
    //
    //    This is critical for:
    //      - Neo-Hookean under large deformation (stiffness varies rapidly)
    //      - Plasticity near yield surfaces (tangent discontinuity)
    //      - Any load level where the Newton radius of convergence
    //        is smaller than the load increment
    //
    //    Example: with num_steps=5 and max_bisections=3, a step that
    //    diverges will be refined to Δλ/2, Δλ/4, or Δλ/8 before giving up.
    //
    //  Returns:
    //    true  — all steps converged (possibly after bisection).
    //    false — at least one step failed even after max bisections.
    //            In this case, u and material state reflect the last
    //            SUCCESSFULLY converged step (safe for post-processing).
    //
    //  Usage:
    //    bool ok = analysis.solve_incremental(10);      // 10 steps, 4 bisections
    //    bool ok = analysis.solve_incremental(5, 6);    // 5 steps, 6 bisections
    //
    //  PETSc runtime options apply at each SNES solve:
    //    -snes_monitor -snes_converged_reason -ksp_type preonly
    //    -pc_type lu -pc_factor_mat_solver_type mumps

    bool solve_incremental(int num_steps, int max_bisections = 4) {
        setup();
        VecSet(U, 0.0);

        // Save a copy of the full (un-scaled) external force vector.
        // This is the target load at λ = 1.0.
        petsc::OwnedVec f_full{};
        VecDuplicate(f_ext, f_full.ptr());
        VecCopy(f_ext, f_full);

        double lambda_done = 0.0;   // last converged load level
        bool   all_ok      = true;

        PetscPrintf(PETSC_COMM_WORLD,
            "  Incremental solve: %d steps, max bisection depth = %d\n",
            num_steps, max_bisections);

        for (int step = 1; step <= num_steps; ++step) {
            double lambda_target = static_cast<double>(step) / num_steps;

            PetscPrintf(PETSC_COMM_WORLD,
                "\n  ── Load step %d/%d: λ = %.4f → %.4f ──\n",
                step, num_steps, lambda_done, lambda_target);

            bool ok = advance_to_lambda_(f_full.get(), lambda_done, lambda_target,
                                         max_bisections);

            if (ok) {
                lambda_done = lambda_target;
                if (step_callback_) step_callback_(step, lambda_done, *model_);
            } else {
                all_ok = false;
                PetscPrintf(PETSC_COMM_WORLD,
                    "\n  *** ABORT at step %d/%d (λ=%.4f) — "
                    "bisection exhausted after %d levels ***\n",
                    step, num_steps, lambda_target, max_bisections);
                break;
            }
        }

        model_->update_elements_state();

        PetscPrintf(PETSC_COMM_WORLD,
            "  Incremental solve %s at λ = %.4f\n",
            all_ok ? "COMPLETED" : "ABORTED", lambda_done);

        return all_ok;
    }

private:

    // ─── Bisection engine (recursive) ───────────────────────────
    //
    //  Attempts to advance the solution from the current committed state
    //  to k·f_full (where lambda_target = k).
    //
    //  Algorithm:
    //    1. Snapshot u (in case we need to revert on divergence).
    //    2. Set f_ext = lambda_target · f_full, call SNESSolve.
    //    3. If converged → commit material state, return true.
    //    4. If diverged → revert u to snapshot, bisect:
    //         λ_mid = (lambda_current + lambda_target) / 2
    //         advance_to_lambda_(f_full, lambda_current, λ_mid, depth-1)
    //         advance_to_lambda_(f_full, λ_mid, lambda_target, depth-1)
    //    5. If bisections_left == 0, give up and return false.
    //
    //  Each recursion level allocates one temporary Vec for the snapshot.
    //  With max_bisections ≤ 6, this is at most 6 extra vectors — negligible
    //  compared to the global stiffness matrix.
    //
    //  IMPORTANT: This function never commits state on failure.  On success,
    //  state is committed at each sub-step so the material history is correct
    //  for path-dependent materials (plasticity, damage).

    bool advance_to_lambda_(Vec f_full,
                            double lambda_current, double lambda_target,
                            int bisections_left)
    {
        // ── 1. Snapshot current displacement for rollback ─────────
        petsc::OwnedVec U_snap{};
        VecDuplicate(U, U_snap.ptr());
        VecCopy(U, U_snap);

        // ── 2. Set load level and solve ──────────────────────────
        VecCopy(f_full, f_ext);
        VecScale(f_ext, lambda_target);
        ctx_.f_ext = f_ext;

        SNESSolve(snes_, nullptr, U);

        SNESConvergedReason reason;
        SNESGetConvergedReason(snes_, &reason);
        PetscInt its;
        SNESGetIterationNumber(snes_, &its);

        // ── 3. Converged? Commit and return ──────────────────────
        if (reason > 0) {
            commit_state();

            PetscPrintf(PETSC_COMM_WORLD,
                "    λ=%.6f  converged  (reason=%d, %d iterations)\n",
                lambda_target,
                static_cast<int>(reason),
                static_cast<int>(its));
            return true;
        }

        // ── 4. Diverged — revert to pre-step state ──────────────
        //
        //  Revert displacements to the snapshot and explicitly revert
        //  material state.  While commit was never called (so committed
        //  state is safe), trial buffers may hold residuals from the
        //  failed Newton iterates; explicit revert guarantees a clean
        //  constitutive state for the next solve attempt.
        VecCopy(U_snap, U);
        revert_state();

        PetscPrintf(PETSC_COMM_WORLD,
            "    λ=%.6f  DIVERGED   (reason=%d, %d iterations)\n",
            lambda_target,
            static_cast<int>(reason),
            static_cast<int>(its));

        // ── 5. Bisection budget exhausted? ───────────────────────
        if (bisections_left <= 0) {
            PetscPrintf(PETSC_COMM_WORLD,
                "    No bisection budget remaining at λ=%.6f.\n",
                lambda_target);
            return false;
        }

        // ── 6. Bisect: split [λ_current → λ_target] into halves ─
        double lambda_mid = 0.5 * (lambda_current + lambda_target);

        PetscPrintf(PETSC_COMM_WORLD,
            "    Bisecting: λ = %.6f → %.6f → %.6f  "
            "(depth remaining: %d)\n",
            lambda_current, lambda_mid, lambda_target,
            bisections_left - 1);

        // First half: [λ_current → λ_mid]
        if (!advance_to_lambda_(f_full, lambda_current, lambda_mid,
                                bisections_left - 1))
            return false;

        // Second half: [λ_mid → λ_target]
        // Note: U and material state are now at λ_mid (committed above).
        if (!advance_to_lambda_(f_full, lambda_mid, lambda_target,
                                bisections_left - 1))
            return false;

        return true;
    }

public:

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
