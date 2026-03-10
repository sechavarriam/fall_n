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
#include <functional>
#include <vector>

#include <petscts.h>

#include "../model/Model.hh"
#include "../model/BoundaryCondition.hh"
#include "../utils/Benchmark.hh"
#include "Damping.hh"


template <typename MaterialPolicy,
          typename KinematicPolicy = continuum::SmallStrain,
          std::size_t ndofs = MaterialPolicy::dim,
          typename ElemPolicy = SingleElementPolicy<ContinuumElement<MaterialPolicy, ndofs, KinematicPolicy>>>
class DynamicAnalysis {

    using ModelT = Model<MaterialPolicy, KinematicPolicy, ndofs, ElemPolicy>;
    static constexpr auto dim = MaterialPolicy::dim;

    ModelT* model_{nullptr};
    TS      ts_{nullptr};

    // ── System matrices ──────────────────────────────────────────────
    Mat M_{nullptr};    // Mass matrix (assembled once at setup, constant)
    Mat C_{nullptr};    // Damping matrix (optional, assembled at setup)
    Mat J_{nullptr};    // Jacobian work matrix: a·M + v·C + K_t(u)
    Mat K0_{nullptr};   // Initial stiffness (for Rayleigh damping)

    // ── System vectors ───────────────────────────────────────────────
    Vec U_{nullptr};       // Global displacement
    Vec V_{nullptr};       // Global velocity
    Vec F_work_{nullptr};  // Work vector for residual assembly
    Vec f_ext_work_{nullptr}; // Pre-allocated work vector for f_ext (avoid per-step alloc)

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

    // Post-step monitor callback
    using MonitorCallback = std::function<void(PetscInt step, double t, Vec u, Vec v)>;
    MonitorCallback monitor_callback_;

    // Ground motion influence vectors (one per direction with ground motion)
    struct GroundMotionInfo {
        std::size_t direction;
        TimeFunction accel_fn;
        Vec influence;   // M·ĝ_dir (precomputed)
    };
    std::vector<GroundMotionInfo> ground_motions_;


    // ── TS callback context ──────────────────────────────────────────

    struct Context {
        DynamicAnalysis* self;
        ModelT*          model;
    };
    Context ctx_{};


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
        DMGlobalToLocal(dm, U, INSERT_VALUES, u_local);

        // Add prescribed displacements if applicable
        if (self->prescribed_evaluator_) {
            self->prescribed_evaluator_(t);
        }
        VecAXPY(u_local, 1.0, model->imposed_solution());

        const auto num_elems = model->elements().size();

        // Phase 1: Extract element DOFs (sequential)
        std::vector<Eigen::VectorXd> elem_dofs(num_elems);
        for (std::size_t e = 0; e < num_elems; ++e) {
            elem_dofs[e] = model->elements()[e].extract_element_dofs(u_local);
        }

        // Phase 2: Compute element internal forces (parallel)
        std::vector<Eigen::VectorXd> elem_f(num_elems);

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
        DMGlobalToLocal(dm, U, INSERT_VALUES, u_local);

        if (self->prescribed_evaluator_) {
            self->prescribed_evaluator_(t);
        }
        VecAXPY(u_local, 1.0, model->imposed_solution());

        const auto num_elems = model->elements().size();

        // Phase 1: Extract element DOFs (sequential)
        std::vector<Eigen::VectorXd> elem_dofs(num_elems);
        for (std::size_t e = 0; e < num_elems; ++e) {
            elem_dofs[e] = model->elements()[e].extract_element_dofs(u_local);
        }

        // Phase 2: Compute element tangent stiffness (parallel)
        std::vector<Eigen::MatrixXd> elem_K(num_elems);

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

        auto* ctx   = static_cast<Context*>(ctx_ptr);
        auto* self  = ctx->self;
        auto* model = ctx->model;
        DM    dm    = model->get_plex();

        // ── Commit material state after each converged time step ─────
        {
            Vec u_local;
            DMGetLocalVector(dm, &u_local);
            VecSet(u_local, 0.0);
            DMGlobalToLocal(dm, U, INSERT_VALUES, u_local);
            VecAXPY(u_local, 1.0, model->imposed_solution());

            for (auto& element : model->elements()) {
                element.commit_material_state(u_local);
            }

            DMRestoreLocalVector(dm, &u_local);

            // Update model state vector
            DMGlobalToLocal(dm, U, INSERT_VALUES, model->state_vector());
            VecAXPY(model->state_vector(), 1.0, model->imposed_solution());
        }

        // ── User monitor callback ─────────────────────────────────────
        if (self->monitor_callback_) {
            Vec V;
            TS2GetSolution(ts, &U, &V);
            self->monitor_callback_(step, t, U, V);
        }

        PetscFunctionReturn(PETSC_SUCCESS);
    }


    // =================================================================
    //  Assemble initial stiffness (for Rayleigh damping)
    // =================================================================

    void assemble_initial_stiffness() {
        DMCreateMatrix(model_->get_plex(), &K0_);
        MatZeroEntries(K0_);

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
            Vec g_dir;
            DMCreateGlobalVector(dm, &g_dir);
            VecSet(g_dir, 0.0);

            // Build ĝ in local space
            Vec g_local;
            DMGetLocalVector(dm, &g_local);
            VecSet(g_local, 0.0);

            for (const auto& node : model_->get_domain().nodes()) {
                if (node.num_dof() > gm.direction) {
                    PetscInt dof_idx = node.dof_index()[gm.direction];
                    PetscScalar one = 1.0;
                    VecSetValueLocal(g_local, dof_idx, one, INSERT_VALUES);
                }
            }
            VecAssemblyBegin(g_local);
            VecAssemblyEnd(g_local);

            DMLocalToGlobal(dm, g_local, INSERT_VALUES, g_dir);
            DMRestoreLocalVector(dm, &g_local);

            // influence = M · ĝ
            DMCreateGlobalVector(dm, &gm.influence);
            MatMult(M_, g_dir, gm.influence);

            VecDestroy(&g_dir);
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
            ground_motions_.push_back({gm.direction, gm.acceleration, nullptr});
        }
    }

    /// Register a monitor callback.
    ///
    /// Called after each converged time step with (step, time, U, V).
    /// Use for VTK snapshots, energy monitoring, etc.
    void set_monitor(MonitorCallback mc) {
        monitor_callback_ = std::move(mc);
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
            DMCreateGlobalVector(dm, &U_);
            VecSet(U_, 0.0);
        }
        if (!V_) {
            DMCreateGlobalVector(dm, &V_);
            VecSet(V_, 0.0);
        }
        VecDuplicate(U_, &F_work_);
        VecDuplicate(U_, &f_ext_work_);

        // ── Assemble mass matrix ─────────────────────────────────────
        DMCreateMatrix(dm, &M_);
        model_->assemble_mass_matrix(M_);

        // ── Assemble damping matrix (if damping is configured) ───────
        if (damping_assembler_) {
            // Need initial stiffness for Rayleigh damping
            if (!K0_) assemble_initial_stiffness();

            DMCreateMatrix(dm, &C_);
            damping_assembler_(M_, K0_, C_);
        }

        // ── Setup ground motion pseudo-force vectors ─────────────────
        if (!ground_motions_.empty()) {
            setup_ground_motion_influence();
        }

        // ── Create Jacobian matrix ───────────────────────────────────
        DMCreateMatrix(dm, &J_);

        // ── Register TS callbacks ────────────────────────────────────
        ctx_ = {this, model_};

        TSSetI2Function(ts_, F_work_, I2Function, &ctx_);
        TSSetI2Jacobian(ts_, J_, J_, I2Jacobian, &ctx_);
        TSMonitorSet(ts_, Monitor, &ctx_, nullptr);

        // ── Set initial solution ─────────────────────────────────────
        TS2SetSolution(ts_, U_, V_);

        // ── Apply TS options from command line ───────────────────────
        TSSetFromOptions(ts_);

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
            DMCreateGlobalVector(dm, &U_);
        }
        VecCopy(u0_global, U_);
    }

    /// Set initial velocity (global vector).
    void set_initial_velocity(Vec v0_global) {
        if (!V_) {
            DM dm = model_->get_plex();
            DMCreateGlobalVector(dm, &V_);
        }
        VecCopy(v0_global, V_);
    }

    /// Set initial conditions from a BoundaryConditionSet.
    void set_initial_conditions(const BoundaryConditionSet<dim>& bcs) {
        DM dm = model_->get_plex();

        if (!U_) DMCreateGlobalVector(dm, &U_);
        if (!V_) DMCreateGlobalVector(dm, &V_);
        VecSet(U_, 0.0);
        VecSet(V_, 0.0);

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
    //  Solve
    // =================================================================

    /// Solve from t = 0 to t = t_final with time step dt.
    ///
    /// Returns true if the solve completed successfully.
    bool solve(double t_final, double dt) {
        setup();

        TSSetTimeStep(ts_, dt);
        TSSetMaxTime(ts_, t_final);
        TSSetExactFinalTime(ts_, TS_EXACTFINALTIME_MATCHSTEP);

        PetscPrintf(PETSC_COMM_WORLD,
            "\n  ══════════════════════════════════════════════════════════\n"
            "  DynamicAnalysis: solving from t=0 to t=%.6f, dt=%.6e\n"
            "  ══════════════════════════════════════════════════════════\n\n",
            t_final, dt);

        timer_.start("solve");
        TSSolve(ts_, U_);
        timer_.stop("solve");

        TSConvergedReason reason;
        TSGetConvergedReason(ts_, &reason);

        PetscInt steps;
        PetscReal final_time;
        TSGetStepNumber(ts_, &steps);
        TSGetTime(ts_, &final_time);

        PetscPrintf(PETSC_COMM_WORLD,
            "\n  DynamicAnalysis: %s at t=%.6f after %d steps (reason=%d)\n",
            (reason >= 0) ? "COMPLETED" : "DIVERGED",
            final_time, static_cast<int>(steps), static_cast<int>(reason));

        // Update model state after solve
        timer_.start("post");
        {
            DM dm = model_->get_plex();
            DMGlobalToLocal(dm, U_, INSERT_VALUES, model_->state_vector());
            VecAXPY(model_->state_vector(), 1.0, model_->imposed_solution());
        }
        timer_.stop("post");

        return (reason >= 0);
    }

    /// Solve between arbitrary times [t_start, t_final].
    bool solve(double t_start, double t_final, double dt) {
        setup();

        TSSetTime(ts_, t_start);
        TSSetTimeStep(ts_, dt);
        TSSetMaxTime(ts_, t_final);
        TSSetExactFinalTime(ts_, TS_EXACTFINALTIME_MATCHSTEP);

        TSSolve(ts_, U_);

        TSConvergedReason reason;
        TSGetConvergedReason(ts_, &reason);

        // Update model state
        {
            DM dm = model_->get_plex();
            DMGlobalToLocal(dm, U_, INSERT_VALUES, model_->state_vector());
            VecAXPY(model_->state_vector(), 1.0, model_->imposed_solution());
        }

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
        DMGlobalToLocal(dm, U_, INSERT_VALUES, u_local);
        VecAXPY(u_local, 1.0, model_->imposed_solution());

        const auto& node = model_->get_domain().node(node_id);
        auto num_dofs = node.num_dof();
        std::vector<double> u(num_dofs);
        std::vector<PetscInt> idx(node.dof_index().begin(), node.dof_index().end());
        VecGetValues(u_local, static_cast<PetscInt>(num_dofs), idx.data(), u.data());

        DMRestoreLocalVector(dm, &u_local);
        return u;
    }


    // =================================================================
    //  Constructors / Destructor
    // =================================================================

    explicit DynamicAnalysis(ModelT* model) : model_{model} {
        TSCreate(PETSC_COMM_WORLD, &ts_);
        TSSetDM(ts_, model_->get_plex());

        // Default: generalized-α (optimal for structural dynamics)
        TSSetType(ts_, TSALPHA2);
    }

    DynamicAnalysis() = default;

    ~DynamicAnalysis() {
        if (ts_)         TSDestroy(&ts_);
        if (U_)          VecDestroy(&U_);
        if (V_)          VecDestroy(&V_);
        if (F_work_)     VecDestroy(&F_work_);
        if (f_ext_work_) VecDestroy(&f_ext_work_);
        if (M_)      MatDestroy(&M_);
        if (C_)      MatDestroy(&C_);
        if (J_)      MatDestroy(&J_);
        if (K0_)     MatDestroy(&K0_);

        for (auto& gm : ground_motions_) {
            if (gm.influence) VecDestroy(&gm.influence);
        }
    }

    DynamicAnalysis(const DynamicAnalysis&)            = delete;
    DynamicAnalysis& operator=(const DynamicAnalysis&) = delete;
    DynamicAnalysis(DynamicAnalysis&&)                 = default;
    DynamicAnalysis& operator=(DynamicAnalysis&&)      = default;
};


#endif // FALL_N_SRC_ANALYSIS_DYNAMIC_ANALYSIS_HH
