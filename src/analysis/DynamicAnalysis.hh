#ifndef FALL_N_SRC_ANALYSIS_DYNAMIC_ANALYSIS_HH
#define FALL_N_SRC_ANALYSIS_DYNAMIC_ANALYSIS_HH

// =============================================================================
//  DynamicAnalysis — PETSc TS-driven structural dynamic solver
// =============================================================================
//
//  Solves the second-order structural dynamics equation:
//
//      M·ü + C·u̇ + f_int(u) = f_ext(t)
//
//  using PETSc's TS (Time Stepping) framework with the I2Function interface
//  for second-order implicit systems:
//
//      F(t, u, u̇, ü) = M·ü + C·u̇ + f_int(u) − f_ext(t) = 0
//
//  Jacobian:
//
//      J = ∂F/∂u + v·∂F/∂u̇ + a·∂F/∂ü = K_t(u) + v·C + a·M
//
//  where v and a are shift parameters provided by PETSc.
//
//  ─── Supported time-stepping algorithms ─────────────────────────────────
//
//  PETSc TS provides a rich set of methods via command-line or API:
//
//    IMPLICIT (recommended for structural dynamics):
//      -ts_type alpha2   : Generalized-α (Chung-Hulbert, 1993)
//                          Second-order accurate, A-stable, controllable
//                          high-frequency dissipation via ρ∞ ∈ [0,1].
//                          Use TSAlpha2SetRadius(ts, rho_inf) to control.
//
//      -ts_type newmark   : Newmark-β method (HHT variant available)
//                          Classical choice:  β=1/4, γ=1/2 (trapezoidal)
//                          or β=1/4, γ=1/2 with ρ∞ for numerical damping.
//
//    EXPLICIT (with lumped mass):
//      -ts_type euler     : Forward Euler (conditionally stable, Δt < Δt_cr)
//      -ts_type rk        : Runge-Kutta (various orders)
//
//    The default is alpha2 (generalized-α) with ρ∞ = 1.0 (no dissipation).
//
//  ─── Material & geometric nonlinearity ──────────────────────────────────
//
//  Both are supported transparently:
//
//    Material NL:  ContinuumElement calls material.compute_response(ε) and
//                  material.tangent(ε) through the Strategy pattern.
//                  Inelastic materials with MemoryState are automatically
//                  committed after each converged time step.
//
//    Geometric NL: Controlled by the KinematicPolicy template parameter:
//                  SmallStrain      → linear kinematics, no K_σ
//                  TotalLagrangian  → Green-Lagrange E, 2nd Piola S, + K_σ
//                  UpdatedLagrangian → co-rotational update
//
//  ─── Parallel assembly ──────────────────────────────────────────────────
//
//  Internal force and tangent stiffness use the same 3-phase parallel
//  pattern as NonlinearAnalysis:
//    1. Extract element DOFs (sequential PETSc read)
//    2. Compute element contributions (OpenMP parallel)
//    3. Inject into PETSc vectors/matrices (sequential)
//
//  ─── Ground motion input ────────────────────────────────────────────────
//
//  For earthquake analysis, provide a GroundMotionBC.  The pseudo-force
//  −M·ĝ_dir·a_g(t) is computed during each I2Function evaluation using
//  the global influence vector and mass matrix.
//
//  ─── VTK time series output ─────────────────────────────────────────────
//
//  Register a snapshot callback to write VTK files at regular intervals.
//  The callback receives (step, time, displacement, velocity) and can
//  use VTKModelExporter to write .vtu snapshots.
//
// =============================================================================

#include <cstddef>
#include <cstdio>
#include <functional>
#include <limits>
#include <vector>

#include <petscts.h>

#include "../model/Model.hh"
#include "../model/BoundaryCondition.hh"
#include "../petsc/PetscRaii.hh"
#include "../utils/Benchmark.hh"
#include "AnalysisRouteAudit.hh"
#include "Damping.hh"
#include "AnalysisObserver.hh"
#include "NonlinearSolvePolicy.hh"
#include "StepDirector.hh"
#include "SteppableSolver.hh"


template <typename MaterialPolicy,
          typename KinematicPolicy = continuum::SmallStrain,
          std::size_t ndofs = MaterialPolicy::dim,
          typename ElemPolicy = SingleElementPolicy<ContinuumElement<MaterialPolicy, ndofs, KinematicPolicy>>>
class DynamicAnalysis {
public:
    using analysis_route_tag =
        fall_n::AnalysisRouteTag<fall_n::AnalysisRouteKind::implicit_second_order_dynamics>;
    static constexpr fall_n::AnalysisRouteKind analysis_route_kind =
        fall_n::AnalysisRouteKind::implicit_second_order_dynamics;
    static constexpr fall_n::AnalysisRouteAuditScope analysis_route_audit_scope =
        fall_n::canonical_analysis_route_audit_scope(analysis_route_kind);

private:
    using ModelT = Model<MaterialPolicy, KinematicPolicy, ndofs, ElemPolicy>;
    static constexpr auto dim = MaterialPolicy::dim;

public:
    using model_type = ModelT;
    using element_type = typename ModelT::element_type;

private:

    ModelT* model_{nullptr};
    petsc::OwnedTS ts_{};

    // ── System matrices ────────────────────────────────────────────────
    petsc::OwnedMat M_{};    // Mass matrix (assembled once at setup, constant)
    petsc::OwnedMat C_{};    // Damping matrix (optional, assembled at setup)
    petsc::OwnedMat J_{};    // Jacobian work matrix: a·M + v·C + K_t(u)
    petsc::OwnedMat K0_{};   // Initial stiffness (for Rayleigh damping)

    // ── System vectors ─────────────────────────────────────────────────
    petsc::OwnedVec U_{};       // Global displacement
    petsc::OwnedVec V_{};       // Global velocity
    petsc::OwnedVec F_work_{};  // Work vector for residual assembly
    petsc::OwnedVec f_ext_work_{}; // Pre-allocated work vector for f_ext

    bool is_setup_{false};

    // ─── Performance timing ───────────────────────────────────────
    AnalysisTimer timer_;

    // ── Configuration ────────────────────────────────────────────────

    // External force evaluator: fills a GLOBAL Vec with f_ext(t)
    using ForceEvaluator = std::function<void(double t, Vec f_ext_global)>;
    ForceEvaluator force_evaluator_;

    // Prescribed displacement evaluator: fills the model's imposed_solution
    using PrescribedEvaluator = std::function<void(double t)>;
    PrescribedEvaluator prescribed_evaluator_;

    // Damping assembler
    DampingAssembler damping_assembler_;

    // Post-step monitor callback (legacy interface)
    using MonitorCallback = std::function<void(PetscInt step, double t, Vec u, Vec v)>;
    MonitorCallback monitor_callback_;

    // Observer pipeline (new interface — replaces monitor_callback_ when set)
    fall_n::ObserverCallback<ModelT> observer_callback_;

    // Step director for condition-based breakpoints (optional)
    fall_n::StepDirector<ModelT> director_;

    // Deduplicate post-step processing when PETSc monitor and manual stepping
    // both report the same converged step.
    PetscInt last_post_step_{-1};
    double   last_post_time_{-std::numeric_limits<double>::infinity()};
    bool     auto_commit_{true};
    bool     observer_notifications_enabled_{true};
    std::vector<fall_n::NonlinearSolveProfile> nonlinear_solve_profiles_{};
    std::size_t active_nonlinear_solve_profile_index_{0};

    // Ground motion influence vectors (one per direction with ground motion)
    struct GroundMotionInfo {
        std::size_t direction;
        TimeFunction accel_fn;
        petsc::OwnedVec influence;   // M·ĝ_dir (precomputed)
    };
    std::vector<GroundMotionInfo> ground_motions_;

public:
    struct SolverCheckpoint {
        typename ModelT::checkpoint_type model{};
        petsc::OwnedVec displacement{};
        petsc::OwnedVec velocity{};
        double   time{0.0};
        PetscInt step{0};
        double   time_step{0.0};
    };

    using checkpoint_type = SolverCheckpoint;

private:


    // ── TS callback context ──────────────────────────────────────────

    struct Context {
        DynamicAnalysis* self;
        ModelT*          model;

        // Reusable scratch buffers for I2Function / I2Jacobian element
        // assembly. Resized lazily; capacity preserved across TS steps to
        // avoid heap thrash inside the time-integration hot path.
        std::vector<Eigen::VectorXd> elem_dofs_scratch{};
        std::vector<Eigen::VectorXd> elem_f_scratch{};
        std::vector<Eigen::MatrixXd> elem_K_scratch{};
    };
    Context ctx_{};

    [[nodiscard]] const std::vector<fall_n::NonlinearSolveProfile>&
    active_nonlinear_solve_profiles_() const noexcept
    {
        return fall_n::active_nonlinear_solve_profiles(nonlinear_solve_profiles_);
    }

    [[nodiscard]] const fall_n::NonlinearSolveProfile&
    active_nonlinear_solve_profile_() const noexcept
    {
        return fall_n::select_nonlinear_solve_profile(
            nonlinear_solve_profiles_,
            active_nonlinear_solve_profile_index_);
    }

    void apply_active_nonlinear_solve_profile_()
    {
        SNES snes{nullptr};
        FALL_N_PETSC_CHECK(TSGetSNES(ts_, &snes));
        if (snes != nullptr) {
            fall_n::apply_nonlinear_solve_profile(
                snes,
                active_nonlinear_solve_profile_());
        }
    }


    // =================================================================
    //  I2Function:  F(t, u, u̇, ü) = M·ü + C·u̇ + f_int(u) − f_ext(t)
    // =================================================================

    static PetscErrorCode I2Function(
        TS /*ts*/, PetscReal t,
        Vec U, Vec Udot, Vec Uddot,
        Vec F, void* ctx_ptr)
    {
        PetscFunctionBeginUser;

        auto* ctx   = static_cast<Context*>(ctx_ptr);
        auto* self  = ctx->self;
        auto* model = ctx->model;
        DM    dm    = model->get_plex();

        // ── 1. Compute f_int(u) ─────────────────────────────────────

        Vec u_local;
        DMGetLocalVector(dm, &u_local);
        VecSet(u_local, 0.0);
        FALL_N_PETSC_CHECK(DMGlobalToLocalBegin(dm, U, INSERT_VALUES, u_local));
        FALL_N_PETSC_CHECK(DMGlobalToLocalEnd(dm, U, INSERT_VALUES, u_local));

        // Add prescribed displacements if applicable
        if (self->prescribed_evaluator_) {
            self->prescribed_evaluator_(t);
        }
        VecAXPY(u_local, 1.0, model->imposed_solution());

        const auto num_elems = model->elements().size();

        // Reuse persistent scratch buffers (capacity preserved across calls)
        auto& elem_dofs = ctx->elem_dofs_scratch;
        auto& elem_f    = ctx->elem_f_scratch;
        elem_dofs.resize(num_elems);
        elem_f   .resize(num_elems);

        // Phase 1: Extract element DOFs (sequential)
        for (std::size_t e = 0; e < num_elems; ++e) {
            elem_dofs[e] = model->elements()[e].extract_element_dofs(u_local);
        }

        // Phase 2: Compute element internal forces (parallel)
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (std::size_t e = 0; e < num_elems; ++e) {
            elem_f[e] = model->elements()[e].compute_internal_force_vector(elem_dofs[e]);
        }

        // Phase 3: Inject into local vector (sequential)
        Vec f_int_local;
        DMGetLocalVector(dm, &f_int_local);
        VecSet(f_int_local, 0.0);

        for (std::size_t e = 0; e < num_elems; ++e) {
            const auto& dofs = model->elements()[e].get_dof_indices();
            VecSetValues(f_int_local, static_cast<PetscInt>(dofs.size()),
                         dofs.data(), elem_f[e].data(), ADD_VALUES);
        }

        // Scatter f_int to global → F = f_int
        VecSet(F, 0.0);
        DMLocalToGlobal(dm, f_int_local, ADD_VALUES, F);

        DMRestoreLocalVector(dm, &u_local);
        DMRestoreLocalVector(dm, &f_int_local);

        // ── 2. F += M·ü ─────────────────────────────────────────────
        MatMultAdd(self->M_, Uddot, F, F);

        // ── 3. F += C·u̇  (if damping exists) ────────────────────────
        if (self->C_) {
            MatMultAdd(self->C_, Udot, F, F);
        }

        // ── 4. F −= f_ext(t) ────────────────────────────────────────
        //  Uses pre-allocated work vector to avoid per-step allocation.
        if (self->force_evaluator_) {
            VecSet(self->f_ext_work_, 0.0);
            self->force_evaluator_(t, self->f_ext_work_);
            VecAXPY(F, -1.0, self->f_ext_work_);
        }

        // ── 5. F −= ground motion pseudo-forces: −M·ĝ·a_g(t) ───────
        for (const auto& gm : self->ground_motions_) {
            double ag = gm.accel_fn(t);
            // f_pseudo = M · influence_vector (precomputed)
            // F += ag * M·ĝ  (note: influence = M·ĝ already)
            VecAXPY(F, ag, gm.influence);
        }

        PetscFunctionReturn(PETSC_SUCCESS);
    }


    // =================================================================
    //  I2Jacobian:  J = K_t(u) + v·C + a·M
    // =================================================================

    static PetscErrorCode I2Jacobian(
        TS /*ts*/, PetscReal t,
        Vec U, Vec /*Udot*/, Vec /*Uddot*/,
        PetscReal v, PetscReal a,
        Mat J_mat, Mat /*P*/, void* ctx_ptr)
    {
        PetscFunctionBeginUser;

        auto* ctx   = static_cast<Context*>(ctx_ptr);
        auto* self  = ctx->self;
        auto* model = ctx->model;
        DM    dm    = model->get_plex();

        // ── 1. Assemble tangent stiffness K_t(u) ────────────────────

        MatZeroEntries(J_mat);

        Vec u_local;
        DMGetLocalVector(dm, &u_local);
        VecSet(u_local, 0.0);
        FALL_N_PETSC_CHECK(DMGlobalToLocalBegin(dm, U, INSERT_VALUES, u_local));
        FALL_N_PETSC_CHECK(DMGlobalToLocalEnd(dm, U, INSERT_VALUES, u_local));

        if (self->prescribed_evaluator_) {
            self->prescribed_evaluator_(t);
        }
        VecAXPY(u_local, 1.0, model->imposed_solution());

        const auto num_elems = model->elements().size();

        // Reuse persistent scratch buffers (capacity preserved across calls)
        auto& elem_dofs = ctx->elem_dofs_scratch;
        auto& elem_K    = ctx->elem_K_scratch;
        elem_dofs.resize(num_elems);
        elem_K   .resize(num_elems);

        // Phase 1: Extract element DOFs (sequential)
        for (std::size_t e = 0; e < num_elems; ++e) {
            elem_dofs[e] = model->elements()[e].extract_element_dofs(u_local);
        }

        // Phase 2: Compute element tangent stiffness (parallel)
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (std::size_t e = 0; e < num_elems; ++e) {
            elem_K[e] = model->elements()[e].compute_tangent_stiffness_matrix(elem_dofs[e]);
        }

        // Phase 3: Inject into PETSc Mat (sequential)
        for (std::size_t e = 0; e < num_elems; ++e) {
            const auto& dofs = model->elements()[e].get_dof_indices();
            const auto n = static_cast<PetscInt>(dofs.size());
            MatSetValuesLocal(J_mat, n, dofs.data(), n, dofs.data(),
                              elem_K[e].data(), ADD_VALUES);
        }

        MatAssemblyBegin(J_mat, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(J_mat, MAT_FINAL_ASSEMBLY);

        DMRestoreLocalVector(dm, &u_local);

        // ── 2. J += a·M ─────────────────────────────────────────────
        MatAXPY(J_mat, a, self->M_, SAME_NONZERO_PATTERN);

        // ── 3. J += v·C ─────────────────────────────────────────────
        if (self->C_) {
            MatAXPY(J_mat, v, self->C_, SAME_NONZERO_PATTERN);
        }

        PetscFunctionReturn(PETSC_SUCCESS);
    }


    // =================================================================
    //  Post-step monitor — commit material state + user callback
    // =================================================================

    static PetscErrorCode Monitor(
        TS ts, PetscInt step, PetscReal t, Vec U, void* ctx_ptr)
    {
        PetscFunctionBeginUser;

        auto* ctx  = static_cast<Context*>(ctx_ptr);
        auto* self = ctx->self;

        Vec V;
        TS2GetSolution(ts, &U, &V);
        self->post_step_(step, t, U, V);

        PetscFunctionReturn(PETSC_SUCCESS);
    }

    void sync_model_state_from_solution_(Vec U) {
        DM dm = model_->get_plex();

        Vec scratch;
        DMGetLocalVector(dm, &scratch);
        VecSet(scratch, 0.0);
        FALL_N_PETSC_CHECK(DMGlobalToLocalBegin(dm, U, INSERT_VALUES, scratch));
        FALL_N_PETSC_CHECK(DMGlobalToLocalEnd  (dm, U, INSERT_VALUES, scratch));

        VecAXPY(scratch, 1.0, model_->imposed_solution());
        VecCopy(scratch, model_->state_vector());
        DMRestoreLocalVector(dm, &scratch);
    }

    void commit_material_state_from_solution_(Vec U) {
        sync_model_state_from_solution_(U);
        for (auto& element : model_->elements()) {
            element.commit_material_state(model_->state_vector());
        }
    }

    void revert_state_() {
        for (auto& element : model_->elements()) {
            element.revert_material_state();
        }
    }


    // =================================================================
    //  post_step_ — extracted core of the post-step pipeline
    // =================================================================
    //
    //  Called after every converged time step (from Monitor or from
    //  the manual step() API).  Performs:
    //    1. Material state commit
    //    2. Model state vector update
    //    3. Observer notification
    //    4. Legacy monitor callback

    void post_step_(PetscInt step, double t, Vec U, Vec V) {

        if (step == last_post_step_ && std::abs(t - last_post_time_) < 1e-14) {
            return;
        }

        last_post_step_ = step;
        last_post_time_ = t;

        sync_model_state_from_solution_(U);

        if (auto_commit_) {
            commit_material_state_from_solution_(U);
        }

        if (observer_notifications_enabled_ && observer_callback_.on_step) {
            fall_n::StepEvent ev{step, t, U, V};
            observer_callback_.on_step(ev, *model_);
        }

        if (observer_notifications_enabled_ && monitor_callback_) {
            monitor_callback_(step, t, U, V);
        }

        return;

        DM dm = model_->get_plex();

        // ── 1. Update model state vector ─────────────────────────────
        Vec scratch;
        DMGetLocalVector(dm, &scratch);
        VecSet(scratch, 0.0);
        FALL_N_PETSC_CHECK(DMGlobalToLocalBegin(dm, U, INSERT_VALUES, scratch));
        FALL_N_PETSC_CHECK(DMGlobalToLocalEnd  (dm, U, INSERT_VALUES, scratch));

        VecAXPY(scratch, 1.0, model_->imposed_solution());
        VecCopy(scratch, model_->state_vector());
        DMRestoreLocalVector(dm, &scratch);

        // ── 2. Commit material state from the same localized vector ──
        for (auto& element : model_->elements()) {
            element.commit_material_state(model_->state_vector());
        }

        // ── 3. Observer pipeline ─────────────────────────────────────
        if (observer_callback_.on_step) {
            fall_n::StepEvent ev{step, t, U, V};
            observer_callback_.on_step(ev, *model_);
        }

        // ── 4. Legacy monitor callback ───────────────────────────────
        if (monitor_callback_) {
            monitor_callback_(step, t, U, V);
        }
    }


    // =================================================================
    //  Assemble initial stiffness (for Rayleigh damping)
    // =================================================================

    void assemble_initial_stiffness() {
        FALL_N_PETSC_CHECK(DMCreateMatrix(model_->get_plex(), K0_.ptr()));
        FALL_N_PETSC_CHECK(MatSetOption(K0_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
        FALL_N_PETSC_CHECK(MatZeroEntries(K0_));

        // For initial stiffness, evaluate at u = 0
        model_->inject_K(K0_);  // uses linear K() method
    }


    // =================================================================
    //  Setup ground motion influence vectors
    // =================================================================

    void setup_ground_motion_influence() {
        DM dm = model_->get_plex();

        for (auto& gm : ground_motions_) {
            // Create unit direction vector:  ĝ[i] = 1 if dof % dim == direction
            petsc::OwnedVec g_dir;
            FALL_N_PETSC_CHECK(DMCreateGlobalVector(dm, g_dir.ptr()));
            FALL_N_PETSC_CHECK(VecSet(g_dir, 0.0));

            // Build ĝ in local space
            Vec g_local;
            DMGetLocalVector(dm, &g_local);
            FALL_N_PETSC_CHECK(VecSet(g_local, 0.0));

            for (const auto& node : model_->get_domain().nodes()) {
                if (node.num_dof() > gm.direction) {
                    PetscInt dof_idx = node.dof_index()[gm.direction];
                    PetscScalar one = 1.0;
                    FALL_N_PETSC_CHECK(VecSetValueLocal(g_local, dof_idx, one, INSERT_VALUES));
                }
            }
            FALL_N_PETSC_CHECK(VecAssemblyBegin(g_local));
            FALL_N_PETSC_CHECK(VecAssemblyEnd(g_local));

            FALL_N_PETSC_CHECK(DMLocalToGlobal(dm, g_local, INSERT_VALUES, g_dir));
            DMRestoreLocalVector(dm, &g_local);

            // influence = M · ĝ
            FALL_N_PETSC_CHECK(DMCreateGlobalVector(dm, gm.influence.ptr()));
            FALL_N_PETSC_CHECK(MatMult(M_, g_dir, gm.influence));
        }
    }


public:

    // =================================================================
    //  Configuration API
    // =================================================================

    /// Get the underlying model.
    auto get_model() const { return model_; }

    /// Get the PETSc TS handle (for advanced configuration).
    TS get_ts() const { return ts_; }

    /// Declare the nonlinear solve profiles used by the SNES embedded in TS.
    ///
    /// The dynamic route currently activates one declared profile at a time,
    /// which keeps the extension point explicit without yet coupling stepwise
    /// retry logic to PETSc TS internals.  The shared profile representation
    /// is the same one used by NonlinearAnalysis, so future quasi-Newton,
    /// trust-region, or least-squares-style strategies can be introduced in a
    /// consistent way across quasi-static and dynamic paths.
    void set_nonlinear_solve_profiles(
        std::vector<fall_n::NonlinearSolveProfile> profiles)
    {
        nonlinear_solve_profiles_ = std::move(profiles);
        if (is_setup_) {
            apply_active_nonlinear_solve_profile_();
        }
    }

    [[nodiscard]] const std::vector<fall_n::NonlinearSolveProfile>&
    nonlinear_solve_profiles() const noexcept
    {
        return active_nonlinear_solve_profiles_();
    }

    void set_active_nonlinear_solve_profile_index(std::size_t index)
    {
        active_nonlinear_solve_profile_index_ = index;
        if (is_setup_) {
            apply_active_nonlinear_solve_profile_();
        }
    }

    [[nodiscard]] std::size_t
    active_nonlinear_solve_profile_index() const noexcept
    {
        const auto& profiles = active_nonlinear_solve_profiles_();
        return std::min(active_nonlinear_solve_profile_index_,
                        profiles.empty() ? std::size_t{0}
                                         : profiles.size() - 1);
    }

    [[nodiscard]] const fall_n::NonlinearSolveProfile&
    active_nonlinear_solve_profile() const noexcept
    {
        return active_nonlinear_solve_profile_();
    }

    /// Set density uniformly on all elements.
    void set_density(double rho) {
        model_->set_density(rho);
    }

    /// Set density per physical group.
    void set_density(const std::map<std::string, double>& density_map,
                     double default_density = 0.0)
    {
        model_->set_density(density_map, default_density);
    }

    /// Set Rayleigh damping:  C = α·M + β·K₀
    void set_rayleigh_damping(double alpha_M, double beta_K) {
        damping_assembler_ = damping::rayleigh(alpha_M, beta_K);
    }

    /// Set damping from two frequencies and a single ratio.
    void set_rayleigh_damping(double omega1, double omega2, double xi) {
        damping_assembler_ = damping::rayleigh_from_single_ratio(omega1, omega2, xi);
    }

    /// Set custom damping assembler.
    void set_damping(DampingAssembler da) {
        damping_assembler_ = std::move(da);
    }

    /// Set external force evaluator.
    ///
    /// The function receives (t, f_ext_global) and must fill f_ext_global
    /// with the external force vector at time t.
    void set_force_function(ForceEvaluator fe) {
        force_evaluator_ = std::move(fe);
    }

    /// Set prescribed displacement evaluator.
    ///
    /// The function receives (t) and must update the model's
    /// imposed_solution vector with u_D(t).
    void set_prescribed_function(PrescribedEvaluator pe) {
        prescribed_evaluator_ = std::move(pe);
    }

    /// Set external forces from a BoundaryConditionSet.
    ///
    /// Convenience method that creates force and prescribed evaluators
    /// from the BC set.
    void set_boundary_conditions(const BoundaryConditionSet<dim>& bcs) {
        // Force evaluator: evaluate all NodalForceBCs at time t
        if (!bcs.forces().empty()) {
            force_evaluator_ = [this, &bcs](double t, Vec f_ext_global) {
                DM dm = model_->get_plex();
                Vec f_local;
                DMGetLocalVector(dm, &f_local);
                bcs.evaluate_forces(t, f_local, *model_);
                DMLocalToGlobal(dm, f_local, ADD_VALUES, f_ext_global);
                DMRestoreLocalVector(dm, &f_local);
            };
        }

        // Prescribed displacement evaluator
        if (bcs.has_prescribed()) {
            prescribed_evaluator_ = [this, &bcs](double t) {
                bcs.evaluate_prescribed(t, model_->imposed_solution(), *model_);
            };
        }

        // Ground motions — store for setup_ground_motion_influence()
        for (const auto& gm : bcs.ground_motions()) {
            ground_motions_.push_back({gm.direction, gm.acceleration, {}});
        }
    }

    /// Register a monitor callback (legacy interface).
    ///
    /// Called after each converged time step with (step, time, U, V).
    /// Prefer set_observer() for the composable observer pipeline.
    void set_monitor(MonitorCallback mc) {
        monitor_callback_ = std::move(mc);
    }

    /// Register an observer pipeline.
    ///
    /// The observer receives (StepEvent, ModelT&) on every converged step,
    /// plus lifecycle hooks (on_analysis_start, on_analysis_end).
    /// Any observer-like object (CompositeObserver, DynamicObserverList, or
    /// any class with the three methods) can be passed via make_observer_callback.
    void set_observer(fall_n::ObserverCallback<ModelT> obs) {
        observer_callback_ = std::move(obs);
    }

    /// Convenience: register any observer-like object directly.
    /// The object must outlive the analysis.
    template <typename Obs>
    void set_observer(Obs& obs) {
        observer_callback_ = fall_n::make_observer_callback<ModelT>(obs);
    }


    // =================================================================
    //  Setup
    // =================================================================

    void setup() {
        if (is_setup_) return;
        timer_.start("setup");

        DM dm = model_->get_plex();

        // ── Create vectors (if not already set by set_initial_conditions) ──
        if (!U_) {
            FALL_N_PETSC_CHECK(DMCreateGlobalVector(dm, U_.ptr()));
            FALL_N_PETSC_CHECK(VecSet(U_, 0.0));
        }
        if (!V_) {
            FALL_N_PETSC_CHECK(DMCreateGlobalVector(dm, V_.ptr()));
            FALL_N_PETSC_CHECK(VecSet(V_, 0.0));
        }
        FALL_N_PETSC_CHECK(VecDuplicate(U_, F_work_.ptr()));
        FALL_N_PETSC_CHECK(VecDuplicate(U_, f_ext_work_.ptr()));

        // ── Assemble mass matrix ─────────────────────────────────────
        FALL_N_PETSC_CHECK(DMCreateMatrix(dm, M_.ptr()));
        FALL_N_PETSC_CHECK(MatSetOption(M_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
        model_->assemble_mass_matrix(M_);

        // ── Assemble damping matrix (if damping is configured) ───────
        if (damping_assembler_) {
            // Need initial stiffness for Rayleigh damping
            if (!K0_) assemble_initial_stiffness();

            FALL_N_PETSC_CHECK(DMCreateMatrix(dm, C_.ptr()));
            FALL_N_PETSC_CHECK(MatSetOption(C_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
            damping_assembler_(M_, K0_, C_);
        }

        // ── Setup ground motion pseudo-force vectors ─────────────────
        if (!ground_motions_.empty()) {
            setup_ground_motion_influence();
        }

        // ── Create Jacobian matrix ───────────────────────────────────
        FALL_N_PETSC_CHECK(DMCreateMatrix(dm, J_.ptr()));
        FALL_N_PETSC_CHECK(MatSetOption(J_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

        // ── Register TS callbacks ────────────────────────────────────
        ctx_ = {this, model_};

        FALL_N_PETSC_CHECK(TSSetI2Function(ts_, F_work_, I2Function, &ctx_));
        FALL_N_PETSC_CHECK(TSSetI2Jacobian(ts_, J_, J_, I2Jacobian, &ctx_));
        FALL_N_PETSC_CHECK(TSMonitorSet(ts_, Monitor, &ctx_, nullptr));

        // ── Set initial solution ─────────────────────────────────────
        FALL_N_PETSC_CHECK(TS2SetSolution(ts_, U_, V_));

        // ── Apply TS options from command line ───────────────────────
        FALL_N_PETSC_CHECK(TSSetFromOptions(ts_));
        apply_active_nonlinear_solve_profile_();

        is_setup_ = true;
        timer_.stop("setup");
    }


    // =================================================================
    //  Set initial conditions
    // =================================================================

    /// Set initial displacement (global vector).
    void set_initial_displacement(Vec u0_global) {
        if (!U_) {
            DM dm = model_->get_plex();
            FALL_N_PETSC_CHECK(DMCreateGlobalVector(dm, U_.ptr()));
        }
        FALL_N_PETSC_CHECK(VecCopy(u0_global, U_));
    }

    /// Set initial velocity (global vector).
    void set_initial_velocity(Vec v0_global) {
        if (!V_) {
            DM dm = model_->get_plex();
            FALL_N_PETSC_CHECK(DMCreateGlobalVector(dm, V_.ptr()));
        }
        FALL_N_PETSC_CHECK(VecCopy(v0_global, V_));
    }

    /// Set initial conditions from a BoundaryConditionSet.
    void set_initial_conditions(const BoundaryConditionSet<dim>& bcs) {
        DM dm = model_->get_plex();

        if (!U_) FALL_N_PETSC_CHECK(DMCreateGlobalVector(dm, U_.ptr()));
        if (!V_) FALL_N_PETSC_CHECK(DMCreateGlobalVector(dm, V_.ptr()));
        FALL_N_PETSC_CHECK(VecSet(U_, 0.0));
        FALL_N_PETSC_CHECK(VecSet(V_, 0.0));

        // Apply ICs in local space, then scatter to global
        Vec u_local, v_local;
        DMGetLocalVector(dm, &u_local);
        DMGetLocalVector(dm, &v_local);
        VecSet(u_local, 0.0);
        VecSet(v_local, 0.0);

        bcs.apply_initial_displacement(u_local, *model_);
        bcs.apply_initial_velocity(v_local, *model_);

        DMLocalToGlobal(dm, u_local, INSERT_VALUES, U_);
        DMLocalToGlobal(dm, v_local, INSERT_VALUES, V_);

        DMRestoreLocalVector(dm, &u_local);
        DMRestoreLocalVector(dm, &v_local);
    }


    // =================================================================
    //  Single-Step API
    // =================================================================

    /// Advance exactly one time step.
    ///
    /// Calls TSStep(ts_) to advance the TS integrator by one step.
    /// Then explicitly runs the post-step pipeline so step(), step_to(),
    /// and step_n() all behave consistently even when PETSc monitors are not
    /// invoked for manual stepping.
    ///
    /// Returns true if the step converged, false if it diverged.
    bool step() {
        setup();

        // TSStep requires a step budget and won't advance if the TS
        // is already in a "converged" state (e.g. from a previous step).
        // Reset to ITERATING and allow one additional step.
        // Also ensure max_time is large enough to not block the step.
        FALL_N_PETSC_CHECK(TSSetConvergedReason(ts_, TS_CONVERGED_ITERATING));
        FALL_N_PETSC_CHECK(TSSetMaxSteps(ts_, current_step() + 1));
        FALL_N_PETSC_CHECK(TSSetMaxTime(ts_, current_time() + 100.0 * get_time_step()));
        FALL_N_PETSC_CHECK(TSStep(ts_));

        PetscInt  sn;
        PetscReal t;
        Vec U_cur, V_cur;
        FALL_N_PETSC_CHECK(TSGetStepNumber(ts_, &sn));
        FALL_N_PETSC_CHECK(TSGetTime(ts_, &t));
        FALL_N_PETSC_CHECK(TS2GetSolution(ts_, &U_cur, &V_cur));
        post_step_(sn, static_cast<double>(t), U_cur, V_cur);

        TSConvergedReason reason;
        FALL_N_PETSC_CHECK(TSGetConvergedReason(ts_, &reason));

        // TS_CONVERGED_ITS means "reached max steps" — that's expected
        // here since we set max_steps = current+1.  Only a negative
        // reason indicates actual divergence.
        return (reason >= 0);
    }

    /// Advance until t >= t_target, respecting the StepDirector if set.
    ///
    /// Returns the StepVerdict that caused the loop to exit:
    ///   - Continue  → reached t_target normally
    ///   - Pause     → a director condition fired before t_target
    ///   - Stop      → a director condition requested termination
    ///
    /// After Pause, call step_to() again (same or new target) to resume.
    fall_n::StepVerdict step_to(double t_target) {
        return step_to(t_target, director_);
    }

    /// Advance until t >= t_target with an explicit director.
    fall_n::StepVerdict step_to(double t_target,
                                const fall_n::StepDirector<ModelT>& director)
    {
        setup();

        FALL_N_PETSC_CHECK(TSSetMaxTime(ts_, t_target));
        FALL_N_PETSC_CHECK(TSSetExactFinalTime(ts_, TS_EXACTFINALTIME_MATCHSTEP));

        while (current_time() < t_target - 1e-14) {
            if (!step()) {
                return fall_n::StepVerdict::Stop;
            }

            PetscInt  sn;
            PetscReal t;
            Vec U_cur, V_cur;
            FALL_N_PETSC_CHECK(TSGetStepNumber(ts_, &sn));
            FALL_N_PETSC_CHECK(TSGetTime(ts_, &t));
            FALL_N_PETSC_CHECK(TS2GetSolution(ts_, &U_cur, &V_cur));

            // Evaluate director after step
            if (director) {
                fall_n::StepEvent ev{sn, static_cast<double>(t), U_cur, V_cur};
                auto verdict = director(ev, *model_);
                if (verdict != fall_n::StepVerdict::Continue) {
                    return verdict;
                }
            }
        }
        return fall_n::StepVerdict::Continue;
    }

    /// Advance exactly n time steps, respecting the StepDirector if set.
    ///
    /// Returns the StepVerdict (Continue if all n steps completed normally).
    fall_n::StepVerdict step_n(int n) {
        return step_n(n, director_);
    }

    /// Advance exactly n time steps with an explicit director.
    fall_n::StepVerdict step_n(int n,
                               const fall_n::StepDirector<ModelT>& director)
    {
        setup();

        for (int i = 0; i < n; ++i) {
            if (!step()) {
                return fall_n::StepVerdict::Stop;
            }

            PetscInt  sn;
            PetscReal t;
            Vec U_cur, V_cur;
            FALL_N_PETSC_CHECK(TSGetStepNumber(ts_, &sn));
            FALL_N_PETSC_CHECK(TSGetTime(ts_, &t));
            FALL_N_PETSC_CHECK(TS2GetSolution(ts_, &U_cur, &V_cur));

            if (director) {
                fall_n::StepEvent ev{sn, static_cast<double>(t), U_cur, V_cur};
                auto verdict = director(ev, *model_);
                if (verdict != fall_n::StepVerdict::Continue) {
                    return verdict;
                }
            }
        }
        return fall_n::StepVerdict::Continue;
    }


    // =================================================================
    //  Runtime Reconfiguration
    // =================================================================

    /// Change the time step size at runtime.
    void set_time_step(double dt) {
        FALL_N_PETSC_CHECK(TSSetTimeStep(ts_, dt));
    }

    /// Get the current time step size.
    double get_time_step() const {
        PetscReal dt;
        TSGetTimeStep(ts_, &dt);
        return dt;
    }

    /// Change the integration method at runtime (e.g. TSALPHA2, TSNEWMARK,
    /// TSEULER, TSRK).  The TS must be re-setup after changing the type.
    void set_integration_method(TSType type) {
        FALL_N_PETSC_CHECK(TSSetType(ts_, type));
    }

    /// Set a persistent StepDirector for step_to() / step_n() calls.
    void set_director(fall_n::StepDirector<ModelT> dir) {
        director_ = std::move(dir);
    }

    void set_auto_commit(bool enabled) { auto_commit_ = enabled; }
    [[nodiscard]] bool auto_commit() const noexcept { return auto_commit_; }

    void set_observer_notifications(bool enabled) {
        observer_notifications_enabled_ = enabled;
    }

    void commit_trial_state() {
        Vec U_cur, V_cur;
        FALL_N_PETSC_CHECK(TS2GetSolution(ts_, &U_cur, &V_cur));
        sync_model_state_from_solution_(U_cur);
        commit_material_state_from_solution_(U_cur);
        model_->update_elements_state();
    }

    [[nodiscard]] checkpoint_type capture_checkpoint() const {
        checkpoint_type checkpoint;

        if (U_) {
            FALL_N_PETSC_CHECK(VecDuplicate(U_.get(),
                                            checkpoint.displacement.ptr()));
            FALL_N_PETSC_CHECK(VecCopy(U_.get(),
                                       checkpoint.displacement.get()));
        }

        if (V_) {
            FALL_N_PETSC_CHECK(VecDuplicate(V_.get(),
                                            checkpoint.velocity.ptr()));
            FALL_N_PETSC_CHECK(VecCopy(V_.get(),
                                       checkpoint.velocity.get()));
        }

        checkpoint.model = model_->capture_checkpoint();
        checkpoint.time = current_time();
        checkpoint.step = current_step();
        checkpoint.time_step = get_time_step();
        return checkpoint;
    }

    void restore_checkpoint(const checkpoint_type& checkpoint) {
        setup();
        revert_state_();
        model_->restore_checkpoint(checkpoint.model);

        if (checkpoint.displacement && U_) {
            FALL_N_PETSC_CHECK(VecCopy(checkpoint.displacement.get(), U_.get()));
        }

        if (checkpoint.velocity && V_) {
            FALL_N_PETSC_CHECK(VecCopy(checkpoint.velocity.get(), V_.get()));
        }

        FALL_N_PETSC_CHECK(TS2SetSolution(ts_, U_, V_));
        FALL_N_PETSC_CHECK(TSSetTime(ts_, checkpoint.time));
        FALL_N_PETSC_CHECK(TSSetStepNumber(ts_, checkpoint.step));
        FALL_N_PETSC_CHECK(TSSetTimeStep(ts_, checkpoint.time_step));
        FALL_N_PETSC_CHECK(TSSetConvergedReason(ts_, TS_CONVERGED_ITERATING));
        FALL_N_PETSC_CHECK(TSRestartStep(ts_));
        last_post_step_ = checkpoint.step - 1;
        last_post_time_ = checkpoint.time - checkpoint.time_step;
    }


    // =================================================================
    //  Solve (batch API — backward compatible)
    // =================================================================

    /// Solve from t = 0 to t = t_final with time step dt.
    ///
    /// Returns true if the solve completed successfully.
    bool solve(double t_final, double dt) {
        setup();

        FALL_N_PETSC_CHECK(TSSetTimeStep(ts_, dt));
        FALL_N_PETSC_CHECK(TSSetMaxTime(ts_, t_final));
        FALL_N_PETSC_CHECK(TSSetExactFinalTime(ts_, TS_EXACTFINALTIME_MATCHSTEP));

        PetscPrintf(PETSC_COMM_WORLD,
            "\n  ══════════════════════════════════════════════════════════\n"
            "  DynamicAnalysis: solving from t=0 to t=%.6f, dt=%.6e\n"
            "  ══════════════════════════════════════════════════════════\n\n",
            t_final, dt);

        // ── Observer: on_analysis_start ──────────────────────────────
        if (observer_callback_.on_start)
            observer_callback_.on_start(*model_);

        timer_.start("solve");
        FALL_N_PETSC_CHECK(TSSolve(ts_, U_));
        timer_.stop("solve");

        TSConvergedReason reason;
        FALL_N_PETSC_CHECK(TSGetConvergedReason(ts_, &reason));

        PetscInt steps;
        PetscReal final_time;
        FALL_N_PETSC_CHECK(TSGetStepNumber(ts_, &steps));
        FALL_N_PETSC_CHECK(TSGetTime(ts_, &final_time));

        PetscPrintf(PETSC_COMM_WORLD,
            "\n  DynamicAnalysis: %s at t=%.6f after %d steps (reason=%d)\n",
            (reason >= 0) ? "COMPLETED" : "DIVERGED",
            final_time, static_cast<int>(steps), static_cast<int>(reason));

        // Update model state after solve
        timer_.start("post");
        {
            DM dm = model_->get_plex();
            VecSet(model_->state_vector(), 0.0);
            FALL_N_PETSC_CHECK(DMGlobalToLocalBegin(dm, U_, INSERT_VALUES, model_->state_vector()));
            FALL_N_PETSC_CHECK(DMGlobalToLocalEnd(dm, U_, INSERT_VALUES, model_->state_vector()));
            VecAXPY(model_->state_vector(), 1.0, model_->imposed_solution());
        }
        timer_.stop("post");

        // ── Observer: on_analysis_end ────────────────────────────────
        if (observer_callback_.on_end)
            observer_callback_.on_end(*model_);

        return (reason >= 0);
    }

    /// Solve between arbitrary times [t_start, t_final].
    bool solve(double t_start, double t_final, double dt) {
        setup();

        FALL_N_PETSC_CHECK(TSSetTime(ts_, t_start));
        FALL_N_PETSC_CHECK(TSSetTimeStep(ts_, dt));
        FALL_N_PETSC_CHECK(TSSetMaxTime(ts_, t_final));
        FALL_N_PETSC_CHECK(TSSetExactFinalTime(ts_, TS_EXACTFINALTIME_MATCHSTEP));

        // ── Observer: on_analysis_start ──────────────────────────────
        if (observer_callback_.on_start)
            observer_callback_.on_start(*model_);

        FALL_N_PETSC_CHECK(TSSolve(ts_, U_));

        TSConvergedReason reason;
        FALL_N_PETSC_CHECK(TSGetConvergedReason(ts_, &reason));

        // Update model state
        {
            DM dm = model_->get_plex();
            VecSet(model_->state_vector(), 0.0);
            FALL_N_PETSC_CHECK(DMGlobalToLocalBegin(dm, U_, INSERT_VALUES, model_->state_vector()));
            FALL_N_PETSC_CHECK(DMGlobalToLocalEnd(dm, U_, INSERT_VALUES, model_->state_vector()));
            VecAXPY(model_->state_vector(), 1.0, model_->imposed_solution());
        }

        // ── Observer: on_analysis_end ────────────────────────────────
        if (observer_callback_.on_end)
            observer_callback_.on_end(*model_);

        return (reason >= 0);
    }


    // =================================================================
    //  Post-solve access
    // =================================================================

    Vec displacement()  const { return U_; }
    Vec velocity()      const { return V_; }
    Mat mass_matrix()   const { return M_; }
    Mat damping_matrix() const { return C_; }

    const AnalysisTimer& timer() const { return timer_; }
          AnalysisTimer& timer()       { return timer_; }

    /// Get current time from TS.
    double current_time() const {
        PetscReal t;
        TSGetTime(ts_, &t);
        return t;
    }

    /// Get current step number from TS.
    PetscInt current_step() const {
        PetscInt s;
        TSGetStepNumber(ts_, &s);
        return s;
    }

    /// Get the displacement at a specific node (from current global U_).
    std::vector<double> get_nodal_displacement(std::size_t node_id) const {
        DM dm = model_->get_plex();
        Vec u_local;
        DMGetLocalVector(dm, &u_local);
        VecSet(u_local, 0.0);
        FALL_N_PETSC_CHECK(DMGlobalToLocalBegin(dm, U_, INSERT_VALUES, u_local));
        FALL_N_PETSC_CHECK(DMGlobalToLocalEnd(dm, U_, INSERT_VALUES, u_local));
        VecAXPY(u_local, 1.0, model_->imposed_solution());

        const auto& node = model_->get_domain().node(node_id);
        auto num_dofs = node.num_dof();
        std::vector<double> u(num_dofs);
        std::vector<PetscInt> idx(node.dof_index().begin(), node.dof_index().end());
        VecGetValues(u_local, static_cast<PetscInt>(num_dofs), idx.data(), u.data());

        DMRestoreLocalVector(dm, &u_local);
        return u;
    }

    /// Get a snapshot of the current analysis state.
    /// The returned Vecs are borrowed — valid only while this object lives.
    fall_n::AnalysisState get_analysis_state() const {
        return fall_n::AnalysisState{
            .displacement = U_,
            .velocity     = V_,
            .time         = current_time(),
            .step         = current_step()
        };
    }


    // =================================================================
    //  Constructors / Destructor
    // =================================================================

    explicit DynamicAnalysis(ModelT* model) : model_{model} {
        FALL_N_PETSC_CHECK(TSCreate(PETSC_COMM_WORLD, ts_.ptr()));
        FALL_N_PETSC_CHECK(TSSetDM(ts_, model_->get_plex()));

        // Default: generalized-α (optimal for structural dynamics)
        FALL_N_PETSC_CHECK(TSSetType(ts_, TSALPHA2));
    }

    DynamicAnalysis() = default;

    ~DynamicAnalysis() = default;

    DynamicAnalysis(const DynamicAnalysis&)            = delete;
    DynamicAnalysis& operator=(const DynamicAnalysis&) = delete;
    DynamicAnalysis(DynamicAnalysis&&)                 = default;
    DynamicAnalysis& operator=(DynamicAnalysis&&)      = default;
};


#endif // FALL_N_SRC_ANALYSIS_DYNAMIC_ANALYSIS_HH
