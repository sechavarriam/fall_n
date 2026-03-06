#ifndef FALL_N_SRC_ANALYSIS_NLANALYSIS_HH
#define FALL_N_SRC_ANALYSIS_NLANALYSIS_HH

#include <cstddef>
#include <petscsnes.h>

#include "../model/Model.hh"

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
//    - Incremental load stepping with material state commitment
//
//  The constitutive response flows through the Strategy injected in
//  each Material<>:
//    element → material_point → Material<> → Strategy → relation
//
// =============================================================================

template <typename MaterialPolicy, std::size_t ndofs = MaterialPolicy::dim>
class NonlinearAnalysis {
    using ModelT = Model<MaterialPolicy, ndofs>;
    static constexpr auto dim = MaterialPolicy::dim;

    ModelT* model_{nullptr};
    SNES    snes_{nullptr};

    Vec U{nullptr};       // Global solution
    Vec R_vec{nullptr};   // Residual vector
    Vec f_ext{nullptr};   // External forces (current load level)
    Mat J{nullptr};       // Jacobian (tangent stiffness)

    bool is_setup_{false};

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
        VecAXPY(u_local, 1.0, model->global_imposed_solution);

        // Compute f_int element-by-element
        Vec f_int_local;
        DMGetLocalVector(dm, &f_int_local);
        VecSet(f_int_local, 0.0);

        for (auto& element : model->elements) {
            element.compute_internal_forces(u_local, f_int_local);
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
        VecAXPY(u_local, 1.0, model->global_imposed_solution);

        for (auto& element : model->elements) {
            element.inject_tangent_stiffness(u_local, J_mat);
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
        VecAXPY(u_local, 1.0, model_->global_imposed_solution);

        for (auto& element : model_->elements) {
            element.commit_material_state(u_local);
        }

        DMRestoreLocalVector(dm, &u_local);

        // Update model's current_state for post-processing
        DMGlobalToLocal(dm, U, INSERT_VALUES, model_->current_state);
        VecAXPY(model_->current_state, 1.0, model_->global_imposed_solution);
    }

public:

    auto get_model() const { return model_; }

    void record_solution(VTKDataContainer& recorder) {
        double* u;
        VecGetArray(model_->current_state, &u);
        recorder.template load_vector_field<dim>(
            "displacement", u, model_->get_domain().num_nodes());
        VecRestoreArray(model_->current_state, &u);
    }

    // ─── Setup (call before solve, or called automatically) ──────

    void setup() {
        if (is_setup_) return;

        DM dm = model_->get_plex();

        DMCreateGlobalVector(dm, &U);
        VecDuplicate(U, &R_vec);
        VecDuplicate(U, &f_ext);
        DMCreateMatrix(dm, &J);

        VecSet(U, 0.0);
        VecSet(R_vec, 0.0);

        // Assemble global external forces
        VecSet(f_ext, 0.0);
        DMLocalToGlobal(dm, model_->nodal_forces, ADD_VALUES, f_ext);

        ctx_ = {model_, f_ext};

        SNESSetFunction(snes_, R_vec, FormResidual, &ctx_);
        SNESSetJacobian(snes_, J, J, FormJacobian, &ctx_);

        is_setup_ = true;
    }

    // ─── Single load step solve ─────────────────────────────────

    void solve() {
        setup();
        VecSet(U, 0.0);

        SNESSolve(snes_, nullptr, U);

        commit_state();
        model_->update_elements_state();
    }

    // ─── Incremental loading (N load steps) ──────────────────────
    //
    //  Applies the load in N equal increments, solving a nonlinear
    //  system at each step and committing the material state.
    //
    //  Usage:
    //    analysis.solve_incremental(10);  // 10 load steps
    //
    //  PETSc runtime options apply at each step:
    //    -snes_monitor -snes_converged_reason -ksp_type preonly
    //    -pc_type lu -pc_factor_mat_solver_type mumps

    void solve_incremental(int num_steps) {
        setup();
        VecSet(U, 0.0);

        Vec f_full;
        VecDuplicate(f_ext, &f_full);
        VecCopy(f_ext, f_full);

        for (int step = 1; step <= num_steps; ++step) {
            double lambda = static_cast<double>(step) / num_steps;

            // Scale external forces for this increment
            VecCopy(f_full, f_ext);
            VecScale(f_ext, lambda);
            ctx_.f_ext = f_ext;

            SNESSolve(snes_, nullptr, U);

            // Commit material state (evolve internal variables)
            commit_state();

            PetscPrintf(PETSC_COMM_WORLD,
                "  Load step %d/%d (lambda=%.4f) converged\n",
                step, num_steps, lambda);
        }

        model_->update_elements_state();

        VecDestroy(&f_full);
    }

    // ─── Constructor / Destructor ─────────────────────────────────

    explicit NonlinearAnalysis(ModelT* model) : model_{model} {
        SNESCreate(PETSC_COMM_WORLD, &snes_);
        SNESSetFromOptions(snes_);
        SNESSetDM(snes_, model_->get_plex());
    }

    NonlinearAnalysis() = default;

    ~NonlinearAnalysis() {
        if (snes_) SNESDestroy(&snes_);
        if (U)     VecDestroy(&U);
        if (R_vec) VecDestroy(&R_vec);
        if (f_ext) VecDestroy(&f_ext);
        if (J)     MatDestroy(&J);
    }

    // Non-copyable (owns PETSc objects)
    NonlinearAnalysis(const NonlinearAnalysis&)            = delete;
    NonlinearAnalysis& operator=(const NonlinearAnalysis&) = delete;
    NonlinearAnalysis(NonlinearAnalysis&&)                 = default;
    NonlinearAnalysis& operator=(NonlinearAnalysis&&)      = default;
};


#endif // FALL_N_SRC_ANALYSIS_NLANALYSIS_HH