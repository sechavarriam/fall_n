#ifndef FALL_N_SRC_ANALYSIS_ARC_LENGTH_SOLVER_HH
#define FALL_N_SRC_ANALYSIS_ARC_LENGTH_SOLVER_HH

// =============================================================================
//  ArcLengthSolver — SNES-based arc-length (Riks/Crisfield) solver
// =============================================================================
//
//  Implements the arc-length method for tracing equilibrium paths through
//  limit points (snap-through) and turning points (snap-back).
//
//  The standard Newton iteration solves:
//      K_t · δu = −R(u, λ)
//
//  The arc-length method augments this with a constraint:
//      g(Δu, Δλ) = 0
//
//  Each iteration performs the Sherman-Morrison decomposition:
//
//      K_t · δu_R = −R                 (residual correction)
//      K_t · δu_f =  f_ref             (load-sensitivity direction)
//      δu = δu_R + δλ · δu_f           (total correction)
//
//  Then δλ is determined from the constraint equation g.
//
//  This solver manages its own PETSc objects (KSP for the tangent solves,
//  no SNES — the arc-length iteration loop replaces SNES's Newton loop).
//
//  ─── Template parameters ────────────────────────────────────────────────
//
//    MaterialPolicy   — the constitutive policy (e.g. ThreeDimensionalMaterial)
//    KinematicPolicy  — strain measure (SmallStrain, UpdatedLagrangian, etc.)
//    ndofs            — DOFs per node
//    ElemPolicy       — SingleElementPolicy<> or MultiElementPolicy
//
// =============================================================================

#include <cmath>
#include <cstddef>
#include <functional>
#include <print>

#include <petscksp.h>
#include <petscmat.h>
#include <petscvec.h>

#include "../model/Model.hh"
#include "IncrementalControl.hh"


namespace fall_n {


// =============================================================================
//  ArcLengthResult — output of one arc-length step
// =============================================================================

struct ArcLengthResult {
    bool   converged{false};
    double lambda{0.0};         ///< final load parameter
    double delta_lambda{0.0};   ///< load increment for this step
    double arc_length{0.0};     ///< actual arc distance traversed
    int    iterations{0};       ///< Newton iterations used
    double residual_norm{0.0};  ///< final residual norm
};


// =============================================================================
//  ArcLengthSolver
// =============================================================================

template <typename MaterialPolicy,
          typename KinematicPolicy = continuum::SmallStrain,
          std::size_t ndofs = MaterialPolicy::dim,
          typename ElemPolicy = SingleElementPolicy<
              ContinuumElement<MaterialPolicy, ndofs, KinematicPolicy>>>
class ArcLengthSolver {
public:
    using ModelT   = Model<MaterialPolicy, KinematicPolicy, ndofs, ElemPolicy>;
    using ElementT = typename ModelT::element_type;

private:
    ModelT* model_{nullptr};

    // PETSc objects (owned)
    Vec  U_{nullptr};           ///< global displacement (free DOFs)
    Vec  R_{nullptr};           ///< residual work vector
    Vec  f_ref_{nullptr};       ///< reference external force (unscaled)
    Vec  f_ext_{nullptr};       ///< current external force = λ·f_ref
    Mat  K_{nullptr};           ///< tangent stiffness matrix
    KSP  ksp_{nullptr};         ///< linear solver (for K_t · x = b)

    // Arc-length work vectors
    Vec  delta_u_R_{nullptr};   ///< residual correction (K_t⁻¹ · (−R))
    Vec  delta_u_f_{nullptr};   ///< load sensitivity (K_t⁻¹ · f_ref)
    Vec  delta_u_step_{nullptr};///< accumulated Δu within current step
    Vec  u_trial_{nullptr};     ///< trial displacement

    // Arc-length state
    double lambda_{0.0};            ///< current load parameter
    double delta_lambda_step_{0.0}; ///< accumulated Δλ within current step
    double delta_lambda_prev_{0.0}; ///< previous step's Δλ (for sign)

    // Configuration
    ArcLengthVariant variant_{ArcLengthVariant::Cylindrical};
    double delta_ell_{1.0};         ///< arc-length increment
    double psi_{1.0};               ///< load scaling factor
    double f_ref_norm_sq_{0.0};     ///< ‖f_ref‖² (cached)

    int    max_iter_{50};           ///< max Newton iterations per step
    double rtol_{1e-6};             ///< relative residual tolerance
    double atol_{1e-10};            ///< absolute residual tolerance

    bool is_setup_{false};

    // ─── Internal: assemble residual R(u, λ) = f_int(u) − λ·f_ref ───

    void assemble_residual() {
        DM dm = model_->get_plex();

        Vec u_local;
        DMGetLocalVector(dm, &u_local);
        VecSet(u_local, 0.0);
        DMGlobalToLocal(dm, U_, INSERT_VALUES, u_local);
        VecAXPY(u_local, 1.0, model_->imposed_solution());

        Vec f_int_local;
        DMGetLocalVector(dm, &f_int_local);
        VecSet(f_int_local, 0.0);

        for (auto& elem : model_->elements())
            elem.compute_internal_forces(u_local, f_int_local);

        VecSet(R_, 0.0);
        DMLocalToGlobal(dm, f_int_local, ADD_VALUES, R_);
        VecAXPY(R_, -1.0, f_ext_);   // R = f_int − λ·f_ref

        DMRestoreLocalVector(dm, &u_local);
        DMRestoreLocalVector(dm, &f_int_local);
    }

    // ─── Internal: assemble tangent K_t(u) ──────────────────────────

    void assemble_tangent() {
        DM dm = model_->get_plex();
        MatZeroEntries(K_);

        Vec u_local;
        DMGetLocalVector(dm, &u_local);
        VecSet(u_local, 0.0);
        DMGlobalToLocal(dm, U_, INSERT_VALUES, u_local);
        VecAXPY(u_local, 1.0, model_->imposed_solution());

        for (auto& elem : model_->elements())
            elem.inject_tangent_stiffness(u_local, K_);

        MatAssemblyBegin(K_, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(K_, MAT_FINAL_ASSEMBLY);

        DMRestoreLocalVector(dm, &u_local);
    }

    // ─── Internal: commit material state ────────────────────────────

    void commit_state() {
        DM dm = model_->get_plex();

        Vec u_local;
        DMGetLocalVector(dm, &u_local);
        VecSet(u_local, 0.0);
        DMGlobalToLocal(dm, U_, INSERT_VALUES, u_local);
        VecAXPY(u_local, 1.0, model_->imposed_solution());

        for (auto& elem : model_->elements())
            elem.commit_material_state(u_local);

        VecCopy(u_local, model_->state_vector());
        DMRestoreLocalVector(dm, &u_local);
    }

    void revert_state() {
        for (auto& elem : model_->elements())
            elem.revert_material_state();
    }

    // ─── Internal: destroy PETSc objects ────────────────────────────

    void destroy() {
        auto safe_destroy = [](auto& obj, auto fn) {
            if (obj) { fn(&obj); obj = nullptr; }
        };
        safe_destroy(U_,            VecDestroy);
        safe_destroy(R_,            VecDestroy);
        safe_destroy(f_ref_,        VecDestroy);
        safe_destroy(f_ext_,        VecDestroy);
        safe_destroy(delta_u_R_,    VecDestroy);
        safe_destroy(delta_u_f_,    VecDestroy);
        safe_destroy(delta_u_step_, VecDestroy);
        safe_destroy(u_trial_,      VecDestroy);
        safe_destroy(K_,            MatDestroy);
        safe_destroy(ksp_,          KSPDestroy);
    }

public:

    // ─── Constructor / destructor ──────────────────────────────────────

    explicit ArcLengthSolver(ModelT* model) : model_{model} {}

    ~ArcLengthSolver() { destroy(); }

    // Non-copyable (owns PETSc objects)
    ArcLengthSolver(const ArcLengthSolver&) = delete;
    ArcLengthSolver& operator=(const ArcLengthSolver&) = delete;

    ArcLengthSolver(ArcLengthSolver&& o) noexcept
        : model_{o.model_}
        , U_{std::exchange(o.U_, nullptr)}
        , R_{std::exchange(o.R_, nullptr)}
        , f_ref_{std::exchange(o.f_ref_, nullptr)}
        , f_ext_{std::exchange(o.f_ext_, nullptr)}
        , K_{std::exchange(o.K_, nullptr)}
        , ksp_{std::exchange(o.ksp_, nullptr)}
        , delta_u_R_{std::exchange(o.delta_u_R_, nullptr)}
        , delta_u_f_{std::exchange(o.delta_u_f_, nullptr)}
        , delta_u_step_{std::exchange(o.delta_u_step_, nullptr)}
        , u_trial_{std::exchange(o.u_trial_, nullptr)}
        , lambda_{o.lambda_}
        , delta_lambda_step_{o.delta_lambda_step_}
        , delta_lambda_prev_{o.delta_lambda_prev_}
        , variant_{o.variant_}, delta_ell_{o.delta_ell_}
        , psi_{o.psi_}, f_ref_norm_sq_{o.f_ref_norm_sq_}
        , max_iter_{o.max_iter_}, rtol_{o.rtol_}, atol_{o.atol_}
        , is_setup_{o.is_setup_}
    { o.is_setup_ = false; }

    // ─── Configuration ─────────────────────────────────────────────────

    void set_arc_length(double dell)     { delta_ell_ = dell; }
    void set_psi(double p)               { psi_ = p; }
    void set_variant(ArcLengthVariant v) { variant_ = v; }
    void set_max_iterations(int n)       { max_iter_ = n; }
    void set_tolerances(double rtol, double atol) { rtol_ = rtol; atol_ = atol; }

    [[nodiscard]] double lambda() const noexcept { return lambda_; }

    // ─── Setup ─────────────────────────────────────────────────────────

    void setup() {
        if (is_setup_) return;

        DM dm = model_->get_plex();

        DMCreateGlobalVector(dm, &U_);
        VecDuplicate(U_, &R_);
        VecDuplicate(U_, &f_ref_);
        VecDuplicate(U_, &f_ext_);
        VecDuplicate(U_, &delta_u_R_);
        VecDuplicate(U_, &delta_u_f_);
        VecDuplicate(U_, &delta_u_step_);
        VecDuplicate(U_, &u_trial_);

        DMCreateMatrix(dm, &K_);
        MatSetOption(K_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

        VecSet(U_, 0.0);

        // Copy reference force from model's force vector
        VecSet(f_ref_, 0.0);
        DMLocalToGlobal(dm, model_->force_vector(), ADD_VALUES, f_ref_);
        VecDot(f_ref_, f_ref_, &f_ref_norm_sq_);

        // Set up KSP (direct LU)
        KSPCreate(PETSC_COMM_SELF, &ksp_);
        KSPSetType(ksp_, KSPPREONLY);
        PC pc;
        KSPGetPC(ksp_, &pc);
        PCSetType(pc, PCLU);
        KSPSetFromOptions(ksp_);

        is_setup_ = true;
    }

    // ─── Set reference force explicitly (for sub-models, etc.) ─────

    void set_reference_force(Vec f_ref) {
        if (f_ref_) VecCopy(f_ref, f_ref_);
        VecDot(f_ref_, f_ref_, &f_ref_norm_sq_);
    }

    // ─── Solve one arc-length step ─────────────────────────────────
    //
    //  Advances the solution by one arc-length increment Δℓ:
    //
    //    1. Predictor: extrapolate from tangent direction
    //    2. Corrector: iterate with arc-length constraint until convergence
    //
    //  The current U_ and lambda_ are updated in-place.  Material state
    //  is committed only on convergence.

    ArcLengthResult solve_step() {
        setup();

        ArcLengthResult result;

        // ── 1. Assemble tangent at current state ──────────────────
        VecCopy(f_ref_, f_ext_);
        VecScale(f_ext_, lambda_);
        assemble_tangent();
        KSPSetOperators(ksp_, K_, K_);

        // ── 2. Solve for load-sensitivity direction ───────────────
        //       K_t · δu_f = f_ref
        KSPSolve(ksp_, f_ref_, delta_u_f_);

        // ── 3. Predictor ──────────────────────────────────────────
        double duf_dot_duf;
        VecDot(delta_u_f_, delta_u_f_, &duf_dot_duf);

        double denom = duf_dot_duf + psi_ * psi_ * f_ref_norm_sq_;
        double abs_dlam = (denom > 1e-30)
                              ? delta_ell_ / std::sqrt(denom)
                              : delta_ell_;

        // Sign determination
        double sign = 1.0;
        if (delta_lambda_prev_ != 0.0) {
            if (delta_u_step_) {
                double dot;
                VecDot(delta_u_step_, delta_u_f_, &dot);
                sign = (dot >= 0.0) ? 1.0 : -1.0;
            } else {
                sign = (delta_lambda_prev_ > 0.0) ? 1.0 : -1.0;
            }
        }

        double delta_lambda_pred = sign * abs_dlam;

        // Save snapshot for rollback
        VecCopy(U_, u_trial_);
        double lambda_snapshot = lambda_;

        // Apply predictor
        VecSet(delta_u_step_, 0.0);
        VecAXPY(delta_u_step_, delta_lambda_pred, delta_u_f_);
        VecAXPY(U_, 1.0, delta_u_step_);
        delta_lambda_step_ = delta_lambda_pred;
        lambda_ += delta_lambda_pred;

        // Update external force
        VecCopy(f_ref_, f_ext_);
        VecScale(f_ext_, lambda_);

        // ── 4. Corrector iterations ───────────────────────────────
        bool converged = false;
        double R_norm0 = 0.0;

        for (int iter = 0; iter < max_iter_; ++iter) {
            // Assemble residual and tangent
            assemble_residual();
            assemble_tangent();
            KSPSetOperators(ksp_, K_, K_);

            // Check convergence
            double R_norm;
            VecNorm(R_, NORM_2, &R_norm);
            if (iter == 0) R_norm0 = R_norm;

            result.residual_norm = R_norm;
            result.iterations = iter + 1;

            if (R_norm < atol_ || (R_norm0 > 0.0 && R_norm / R_norm0 < rtol_)) {
                converged = true;
                break;
            }

            // Solve two linear systems:
            //   K_t · δu_R = −R
            //   K_t · δu_f =  f_ref
            VecScale(R_, -1.0);
            KSPSolve(ksp_, R_, delta_u_R_);
            KSPSolve(ksp_, f_ref_, delta_u_f_);

            // Compute δλ from constraint
            double delta_lambda_corr = 0.0;

            if (variant_ == ArcLengthVariant::Cylindrical) {
                double dU_dot_duR, dU_dot_duf;
                VecDot(delta_u_step_, delta_u_R_, &dU_dot_duR);
                VecDot(delta_u_step_, delta_u_f_, &dU_dot_duf);

                double den = dU_dot_duf + psi_ * psi_ * delta_lambda_step_ * f_ref_norm_sq_;
                if (std::abs(den) > 1e-30)
                    delta_lambda_corr = -dU_dot_duR / den;
            } else {
                // Spherical (Crisfield) — quadratic in δλ
                VecDot(delta_u_f_, delta_u_f_, &duf_dot_duf);

                // v = Δu + δu_R
                Vec v;
                VecDuplicate(delta_u_R_, &v);
                VecCopy(delta_u_step_, v);
                VecAXPY(v, 1.0, delta_u_R_);

                double v_dot_v, v_dot_duf;
                VecDot(v, v, &v_dot_v);
                VecDot(v, delta_u_f_, &v_dot_duf);
                VecDestroy(&v);

                double a = duf_dot_duf + psi_ * psi_ * f_ref_norm_sq_;
                double b = 2.0 * v_dot_duf
                         + 2.0 * psi_ * psi_ * delta_lambda_step_ * f_ref_norm_sq_;
                double c = v_dot_v
                         + psi_ * psi_ * delta_lambda_step_ * delta_lambda_step_ * f_ref_norm_sq_
                         - delta_ell_ * delta_ell_;

                double disc = b * b - 4.0 * a * c;
                if (disc < 0.0) disc = 0.0;
                double sq = std::sqrt(disc);
                double s1 = (-b + sq) / (2.0 * a);
                double s2 = (-b - sq) / (2.0 * a);
                delta_lambda_corr = (std::abs(s1) < std::abs(s2)) ? s1 : s2;
            }

            // Update displacement and load parameter
            // δu = δu_R + δλ·δu_f
            VecAXPY(U_, 1.0, delta_u_R_);
            VecAXPY(U_, delta_lambda_corr, delta_u_f_);

            VecAXPY(delta_u_step_, 1.0, delta_u_R_);
            VecAXPY(delta_u_step_, delta_lambda_corr, delta_u_f_);

            delta_lambda_step_ += delta_lambda_corr;
            lambda_ += delta_lambda_corr;

            // Update external force for new λ
            VecCopy(f_ref_, f_ext_);
            VecScale(f_ext_, lambda_);
        }

        // ── 5. Post-process ───────────────────────────────────────
        if (converged) {
            commit_state();
            delta_lambda_prev_ = delta_lambda_step_;
        } else {
            // Rollback
            VecCopy(u_trial_, U_);
            lambda_ = lambda_snapshot;
            delta_lambda_step_ = 0.0;
            revert_state();
        }

        result.converged    = converged;
        result.lambda       = lambda_;
        result.delta_lambda = delta_lambda_step_;

        // Compute actual arc distance
        double du_norm;
        VecNorm(delta_u_step_, NORM_2, &du_norm);
        result.arc_length = std::sqrt(du_norm * du_norm
                                      + psi_ * psi_ * delta_lambda_step_ * delta_lambda_step_
                                        * f_ref_norm_sq_);

        return result;
    }

    // ─── Solve N arc-length steps ──────────────────────────────────

    std::vector<ArcLengthResult> solve_n_steps(int n_steps) {
        setup();
        std::vector<ArcLengthResult> results;
        results.reserve(n_steps);

        for (int step = 0; step < n_steps; ++step) {
            auto r = solve_step();
            results.push_back(r);

            std::println("  Arc-length step {}/{}: λ={:.6f}  Δλ={:.3e}  "
                         "iter={}  converged={}",
                         step + 1, n_steps, r.lambda, r.delta_lambda,
                         r.iterations, r.converged ? "yes" : "no");

            if (!r.converged) {
                std::println("  [!] Arc-length diverged at step {} — stopping.", step + 1);
                break;
            }
        }
        return results;
    }

    // ─── Access to displacement and load parameter ─────────────────

    [[nodiscard]] Vec displacement_vector() const noexcept { return U_; }
    [[nodiscard]] Vec reference_force()     const noexcept { return f_ref_; }
};


} // namespace fall_n

#endif // FALL_N_SRC_ANALYSIS_ARC_LENGTH_SOLVER_HH
