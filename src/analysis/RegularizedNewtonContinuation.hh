#ifndef FALL_N_SRC_ANALYSIS_REGULARIZED_NEWTON_CONTINUATION_HH
#define FALL_N_SRC_ANALYSIS_REGULARIZED_NEWTON_CONTINUATION_HH
// =============================================================================
//  RegularizedNewtonContinuation
// -----------------------------------------------------------------------------
//  Levenberg-Marquardt (pseudo-transient) regularized Newton continuation for a
//  displacement-controlled equilibrium path with LIMIT POINTS that are
//  convergence failures (near-singular tangent) rather than geometric snapbacks.
//
//  Solving  (K + mu*I) du = -R  with an adaptive mu:
//    - mu > 0 makes a near-singular tangent RESOLVABLE, so the continuation
//      crosses the limit point that pure Newton (mu=0) stalls at;
//    - mu -> 0 on well-conditioned branches (e.g. elastic unloading) gives a
//      LOCAL Newton step, which warm-started from the previous converged state
//      stays on the physically continuous branch (no jump to a spurious
//      higher-force equilibrium at load reversals).
//    - mu is capped at a fraction of ||diag(K)|| so it never explodes and
//      freezes the step at a limit point; a stagnation cutoff accepts the best
//      regularized solution when the residual floor (e.g. penalty coupling) is
//      reached.
//
//  Design: the assembly is injected at COMPILE TIME through the
//  ContinuationBackend concept (an adapter over NonlinearAnalysis satisfies it),
//  and the regularization schedule through the RegularizationPolicy concept.
//  Neither the physics nor PETSc SNES are re-implemented here; the solver only
//  drives assemble/solve/accept.
// =============================================================================

#include <petscksp.h>
#include <petscmat.h>
#include <petscvec.h>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <utility>

#include "src/petsc/PetscRaii.hh"
#include "src/petsc/check.hh"

namespace fall_n {

// ── Regularization policy (compile-time) ─────────────────────────────────────
//  Schedules mu across Newton iterations.  initial() seeds it; on_accept()
//  relaxes it after a residual decrease; on_reject() tightens it (capped).
template <typename P>
concept RegularizationPolicy =
    requires(const P p, double mu, double mu_max) {
        { p.initial() } -> std::convertible_to<double>;
        { p.on_accept(mu) } -> std::convertible_to<double>;
        { p.on_reject(mu, mu_max) } -> std::convertible_to<double>;
    };

// Classic Levenberg-Marquardt schedule.
struct LevenbergMarquardt {
    double mu0{1.0e-2};
    double grow{4.0};
    double drop{0.3};
    double mu_min{0.0};
    [[nodiscard]] constexpr double initial() const noexcept { return mu0; }
    [[nodiscard]] constexpr double on_accept(double mu) const noexcept {
        return std::max(mu_min, mu * drop);
    }
    [[nodiscard]] constexpr double on_reject(double mu,
                                             double mu_max) const noexcept {
        return std::min(mu * grow, mu_max);
    }
};

// Pure Newton (no regularization) — for well-conditioned continuation.
struct PureNewton {
    [[nodiscard]] static constexpr double initial() noexcept { return 0.0; }
    [[nodiscard]] static constexpr double on_accept(double) noexcept {
        return 0.0;
    }
    [[nodiscard]] static constexpr double on_reject(double, double) noexcept {
        return 0.0;
    }
};

static_assert(RegularizationPolicy<LevenbergMarquardt>);
static_assert(RegularizationPolicy<PureNewton>);

// ── Assembly backend (compile-time) ──────────────────────────────────────────
//  Provides the physics: vector/matrix factories, control application, residual
//  and tangent assembly, and the commit seam.  An adapter over NonlinearAnalysis
//  (see NLAnalysisContinuationBackend below) satisfies this.
template <typename B>
concept ContinuationBackend =
    requires(B& b, Vec u, Vec r, Mat k, double p) {
        { b.create_vector() } -> std::convertible_to<petsc::OwnedVec>;
        { b.create_matrix() } -> std::convertible_to<petsc::OwnedMat>;
        b.apply_control(p);
        b.residual(u, r);
        b.tangent(u, k);
        b.accept(u, p);
    };

struct RegularizedNewtonConfig {
    int    max_newton{120};
    double tol_abs{1.0e-6};
    double tol_rel{1.0e-6};
    double mu_max_frac{1.0e-1};   // mu_max = mu_max_frac * ||diag(K)||
    int    stag_max{12};          // iters without residual improvement -> accept
    double accept_floor{1.0e-5};  // regularized solution below this = accepted
    // Secant predictor: warm-start each step by extrapolating the last two
    //  accepted states along the control parameter,
    //    u_pred = u_k + s (u_k - u_{k-1}),  s = (p - p_k)/(p_k - p_{k-1}).
    //  At a LOAD REVERSAL the plain warm-start (u_k) sits on the softened
    //  extreme and LM can relax onto a spurious HIGH-force equilibrium (the
    //  reversal "spike").  The secant guess already points down the elastic
    //  UNLOADING branch, so LM converges there instead.  Off by default to
    //  reproduce the plain warm-start continuation; the driver enables it.
    bool   secant_predictor{false};
    double secant_max_scale{4.0};  // clamp |s| so a tiny dp cannot blow up u_pred
};

struct RegularizedNewtonStepResult {
    bool   converged{false};
    int    iterations{0};
    double residual{0.0};
    double mu_final{0.0};
};

// ── Solver ───────────────────────────────────────────────────────────────────
template <ContinuationBackend Backend,
          RegularizationPolicy Reg = LevenbergMarquardt>
class RegularizedNewtonContinuation {
public:
    explicit RegularizedNewtonContinuation(Backend backend,
                                           RegularizedNewtonConfig cfg = {},
                                           Reg reg = {})
        : backend_(std::move(backend)), cfg_(cfg), reg_(reg)
    {
        u_      = backend_.create_vector();
        R_      = backend_.create_vector();
        du_     = backend_.create_vector();
        ut_     = backend_.create_vector();
        dg_     = backend_.create_vector();
        u_prev_ = backend_.create_vector();  // second-last accepted (secant)
        u_hold_ = backend_.create_vector();  // this step's warm-start origin
        K_      = backend_.create_matrix();
        FALL_N_PETSC_CHECK(KSPCreate(
            PetscObjectComm(reinterpret_cast<PetscObject>(u_.get())),
            ksp_.ptr()));
        FALL_N_PETSC_CHECK(KSPSetType(ksp_.get(), KSPPREONLY));
        PC pc{};
        FALL_N_PETSC_CHECK(KSPGetPC(ksp_.get(), &pc));
        FALL_N_PETSC_CHECK(PCSetType(pc, PCLU));
        FALL_N_PETSC_CHECK(VecSet(u_.get(), 0.0));
    }

    // Advance the equilibrium path to control parameter p, warm-started from the
    // current solution.  Commits the accepted state through the backend.
    RegularizedNewtonStepResult advance_to(double p)
    {
        backend_.apply_control(p);

        // Remember the state we warm-start FROM (the last accepted u_k); it
        //  becomes u_{k-1} for the NEXT secant, and lets us recover if the
        //  predicted guess is worse than the plain warm-start.
        FALL_N_PETSC_CHECK(VecCopy(u_.get(), u_hold_.get()));

        // Secant predictor: extrapolate u along the control parameter so the
        //  warm-start already lies on the branch the path is heading to (kills
        //  the load-reversal spike).  Guarded by two accepted states and a
        //  bounded scale.  If the extrapolation raises the residual over the
        //  plain warm-start, fall back to u_k (never worse than no predictor).
        if (cfg_.secant_predictor && accepted_ >= 2 &&
            std::abs(p_curr_ - p_prev_) > 1.0e-30) {
            double s = (p - p_curr_) / (p_curr_ - p_prev_);
            s = std::clamp(s, -cfg_.secant_max_scale, cfg_.secant_max_scale);
            FALL_N_PETSC_CHECK(
                VecWAXPY(du_.get(), -1.0, u_prev_.get(), u_.get()));  // u_k-u_{k-1}
            backend_.residual(u_.get(), R_.get());
            const double r_warm = norm_(R_.get());
            FALL_N_PETSC_CHECK(VecAXPY(u_.get(), s, du_.get()));  // u_k + s*(Δ)
            backend_.residual(u_.get(), R_.get());
            if (norm_(R_.get()) > r_warm) {              // predictor no ayudó
                FALL_N_PETSC_CHECK(VecCopy(u_hold_.get(), u_.get()));
            }
        }

        // mu_max from the stiffness scale ||diag(K)|| at the warm-start.
        backend_.tangent(u_.get(), K_.get());
        FALL_N_PETSC_CHECK(MatGetDiagonal(K_.get(), dg_.get()));
        PetscReal diagn = 0.0;
        FALL_N_PETSC_CHECK(VecNorm(dg_.get(), NORM_2, &diagn));
        const double mu_max = cfg_.mu_max_frac * std::max<double>(diagn, 1.0e-30);

        backend_.residual(u_.get(), R_.get());
        const double r0 = std::max(norm_(R_.get()), 1.0e-30);
        double r_best = r0;
        double mu = reg_.initial();
        int stag = 0;
        int it = 0;
        bool converged = false;
        for (; it < cfg_.max_newton; ++it) {
            const double rn = norm_(R_.get());
            if (rn <= cfg_.tol_abs || rn <= cfg_.tol_rel * r0) {
                converged = true;
                break;
            }
            if (rn < r_best * 0.9999) { r_best = rn; stag = 0; }
            else if (++stag > cfg_.stag_max) break;  // accept best regularized

            backend_.tangent(u_.get(), K_.get());
            mu = std::min(mu, mu_max);
            if (mu > 0.0) FALL_N_PETSC_CHECK(MatShift(K_.get(), mu));  // K + mu I
            FALL_N_PETSC_CHECK(KSPSetOperators(ksp_.get(), K_.get(), K_.get()));
            FALL_N_PETSC_CHECK(VecScale(R_.get(), -1.0));               // -R
            FALL_N_PETSC_CHECK(KSPSolve(ksp_.get(), R_.get(), du_.get()));
            FALL_N_PETSC_CHECK(
                VecWAXPY(ut_.get(), 1.0, du_.get(), u_.get()));         // u+du
            backend_.residual(ut_.get(), R_.get());
            const double rnew = norm_(R_.get());
            if (rnew < rn) {
                FALL_N_PETSC_CHECK(VecCopy(ut_.get(), u_.get()));
                mu = reg_.on_accept(mu);
            } else {
                mu = reg_.on_reject(mu, mu_max);
                backend_.residual(u_.get(), R_.get());  // R back at u
            }
        }
        const double rfinal = norm_(R_.get());
        if (!converged && rfinal <= cfg_.accept_floor) converged = true;

        backend_.accept(u_.get(), p);

        // Shift the secant history: the state we started from (u_hold_ = u_k)
        //  becomes u_{k-1}; the just-accepted u_ is the new u_k at p.
        FALL_N_PETSC_CHECK(VecCopy(u_hold_.get(), u_prev_.get()));
        p_prev_ = p_curr_;
        p_curr_ = p;
        if (accepted_ < 2) ++accepted_;

        return {converged, it, rfinal, mu};
    }

    [[nodiscard]] Vec solution() const noexcept { return u_.get(); }
    [[nodiscard]] Backend& backend() noexcept { return backend_; }

private:
    [[nodiscard]] double norm_(Vec v) const
    {
        PetscReal n = 0.0;
        VecNorm(v, NORM_2, &n);
        return static_cast<double>(n);
    }

    Backend backend_;
    RegularizedNewtonConfig cfg_;
    Reg reg_;
    petsc::OwnedVec u_{}, R_{}, du_{}, ut_{}, dg_{}, u_prev_{}, u_hold_{};
    petsc::OwnedMat K_{};
    petsc::OwnedKSP ksp_{};
    double p_curr_{0.0};   // control of the last accepted state (secant u_k)
    double p_prev_{0.0};   // control of the state before that (secant u_{k-1})
    int    accepted_{0};   // number of accepted steps (capped at 2)
};

// ── Adapter: NonlinearAnalysis -> ContinuationBackend ────────────────────────
//  Thin, header-only bridge.  Templated on the analysis type so it composes with
//  any NonlinearAnalysis instantiation without a virtual layer.
template <typename AnalysisT>
class NLAnalysisContinuationBackend {
public:
    explicit NLAnalysisContinuationBackend(AnalysisT& analysis) noexcept
        : analysis_(&analysis) {}

    [[nodiscard]] petsc::OwnedVec create_vector() const
    {
        return analysis_->create_global_vector();
    }
    [[nodiscard]] petsc::OwnedMat create_matrix() const
    {
        return analysis_->create_tangent_matrix();
    }
    void apply_control(double p) const
    {
        analysis_->apply_incremental_control_parameter(p);
    }
    void residual(Vec u, Vec r) const
    {
        analysis_->evaluate_residual_at(u, r);
    }
    void tangent(Vec u, Mat k) const
    {
        analysis_->evaluate_tangent_at(u, k);
    }
    void accept(Vec u, double p) const
    {
        analysis_->accept_external_solution_step(u, p);
    }

private:
    AnalysisT* analysis_;
};

}  // namespace fall_n

#endif  // FALL_N_SRC_ANALYSIS_REGULARIZED_NEWTON_CONTINUATION_HH
