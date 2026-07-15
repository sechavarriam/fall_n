#ifndef FALL_N_SRC_ANALYSIS_TAO_ENERGY_CONTINUATION_HH
#define FALL_N_SRC_ANALYSIS_TAO_ENERGY_CONTINUATION_HH
// =============================================================================
//  TaoEnergyContinuation
// -----------------------------------------------------------------------------
//  Continuation by ENERGY MINIMISATION instead of residual root-finding. At a
//  load reversal the softening branch makes the equilibrium residual R(u)=0 have
//  several solutions, and a Newton/LM step can land on a spurious HIGH-force one
//  (which is a stationary point of the incremental potential, typically a saddle
//  or maximum along the reversal, not a minimum). A trust-region optimiser driven
//  by the incremental energy Pi descends to a MINIMISER, so it selects the
//  physical branch by construction.
//
//  The backend supplies Pi through incremental_energy(u, u_ref); its gradient is
//  the existing residual R = grad Pi and its Hessian the tangent K = Hess Pi
//  (valid within a step with the material state frozen). PETSc TAO (BNTR by
//  default) does the optimisation; no physics or SNES is re-implemented here.
// =============================================================================

#include <petsctao.h>
#include <petscmat.h>
#include <petscvec.h>

#include <concepts>
#include <string>
#include <utility>

#include "src/analysis/RegularizedNewtonContinuation.hh"  // ContinuationBackend, StepResult
#include "src/petsc/PetscRaii.hh"
#include "src/petsc/check.hh"

namespace fall_n {

// ── Energy backend (compile-time) ────────────────────────────────────────────
//  A ContinuationBackend that can also report the incremental potential between
//  the current iterate and the reference (last accepted) state.
template <typename B>
concept EnergyContinuationBackend =
    ContinuationBackend<B> &&
    requires(B& b, Vec u, Vec u_ref) {
        { b.incremental_energy(u, u_ref) } -> std::convertible_to<double>;
    };

struct TaoEnergyConfig {
    std::string tao_type{"bntr"};   // trust-region Newton (TAOBNTR)
    int    max_it{200};
    double gatol{1.0e-8};           // absolute gradient tolerance
    double grtol{1.0e-8};           // relative gradient tolerance
    double accept_floor{1.0e-6};    // ||R|| below this => step accepted
};

template <EnergyContinuationBackend Backend>
class TaoEnergyContinuation {
    struct Ctx {
        Backend* backend;
        Vec      u_ref;
    };

    static PetscErrorCode obj_cb(Tao, Vec x, PetscReal* f, void* ctx) {
        auto* c = static_cast<Ctx*>(ctx);
        *f = static_cast<PetscReal>(c->backend->incremental_energy(x, c->u_ref));
        return PETSC_SUCCESS;
    }
    static PetscErrorCode grad_cb(Tao, Vec x, Vec g, void* ctx) {
        auto* c = static_cast<Ctx*>(ctx);
        c->backend->residual(x, g);          // g = R = grad Pi
        return PETSC_SUCCESS;
    }
    static PetscErrorCode hess_cb(Tao, Vec x, Mat H, Mat /*Hpre*/, void* ctx) {
        auto* c = static_cast<Ctx*>(ctx);
        c->backend->tangent(x, H);           // H = K = Hess Pi
        return PETSC_SUCCESS;
    }

public:
    explicit TaoEnergyContinuation(Backend backend, TaoEnergyConfig cfg = {})
        : backend_(std::move(backend)), cfg_(std::move(cfg))
    {
        u_    = backend_.create_vector();
        uref_ = backend_.create_vector();
        R_    = backend_.create_vector();
        H_    = backend_.create_matrix();
        FALL_N_PETSC_CHECK(VecSet(u_.get(), 0.0));
    }

    // Warm-start handoff (e.g. from the LM continuation in a hybrid step).
    void set_initial_guess(Vec u) { FALL_N_PETSC_CHECK(VecCopy(u, u_.get())); }

    [[nodiscard]] Vec solution() const noexcept { return u_.get(); }
    [[nodiscard]] Backend& backend() noexcept { return backend_; }

    // Incremental energy Pi(u)-Pi(u_ref) of the LAST advance_to, measured
    //  before the commit (material state still frozen), so it is comparable
    //  across candidate branches of the SAME step.
    [[nodiscard]] double last_energy() const noexcept { return last_energy_; }

    RegularizedNewtonStepResult advance_to(double p)
    {
        backend_.apply_control(p);
        FALL_N_PETSC_CHECK(VecCopy(u_.get(), uref_.get()));  // reference state

        Ctx ctx{&backend_, uref_.get()};

        Tao tao = nullptr;
        FALL_N_PETSC_CHECK(TaoCreate(
            PetscObjectComm(reinterpret_cast<PetscObject>(u_.get())), &tao));
        FALL_N_PETSC_CHECK(TaoSetType(tao, cfg_.tao_type.c_str()));
        FALL_N_PETSC_CHECK(TaoSetSolution(tao, u_.get()));
        FALL_N_PETSC_CHECK(TaoSetObjective(tao, obj_cb, &ctx));
        FALL_N_PETSC_CHECK(TaoSetGradient(tao, R_.get(), grad_cb, &ctx));
        FALL_N_PETSC_CHECK(TaoSetHessian(tao, H_.get(), H_.get(), hess_cb, &ctx));
        FALL_N_PETSC_CHECK(TaoSetTolerances(tao, cfg_.gatol, cfg_.grtol, 0.0));
        FALL_N_PETSC_CHECK(TaoSetMaximumIterations(tao, cfg_.max_it));
        FALL_N_PETSC_CHECK(TaoSolve(tao));

        PetscInt it = 0;
        FALL_N_PETSC_CHECK(TaoGetIterationNumber(tao, &it));
        FALL_N_PETSC_CHECK(TaoDestroy(&tao));

        // Convergence is judged on the PURE residual norm, not the energy.
        backend_.residual(u_.get(), R_.get());
        PetscReal rn = 0.0;
        FALL_N_PETSC_CHECK(VecNorm(R_.get(), NORM_2, &rn));
        const bool converged = static_cast<double>(rn) <= cfg_.accept_floor;

        last_energy_ = backend_.incremental_energy(u_.get(), uref_.get());

        backend_.accept(u_.get(), p);
        return {converged, static_cast<int>(it), static_cast<double>(rn), 0.0};
    }

private:
    Backend         backend_;
    TaoEnergyConfig cfg_;
    petsc::OwnedVec u_{}, uref_{}, R_{};
    petsc::OwnedMat H_{};
    double          last_energy_{0.0};
};

// ── Adapter: NonlinearAnalysis -> EnergyContinuationBackend ──────────────────
//  Extends the residual/tangent adapter with the incremental potential through
//  a LINE QUADRATURE of the existing residual,
//    Pi(u) - Pi(u_ref) = \int_0^1 R(u_ref + s (u - u_ref))^T (u - u_ref) ds,
//  with a 3-point Gauss-Legendre rule on [0,1].  The integral is honest
//  because within a step the material state is FROZEN (evaluate_residual_at
//  does not advance the crack ratchet), so R = grad Pi and the line integral
//  is path-independent.  No material/element/Model change is needed.
template <typename AnalysisT>
class NLAnalysisEnergyBackend : public NLAnalysisContinuationBackend<AnalysisT> {
public:
    explicit NLAnalysisEnergyBackend(AnalysisT& analysis)
        : NLAnalysisContinuationBackend<AnalysisT>(analysis)
    {
        du_ = this->create_vector();
        us_ = this->create_vector();
        r_  = this->create_vector();
    }

    [[nodiscard]] double incremental_energy(Vec u, Vec u_ref)
    {
        FALL_N_PETSC_CHECK(VecWAXPY(du_.get(), -1.0, u_ref, u));  // u - u_ref
        static constexpr double half = 0.3872983346207417;        // sqrt(3/5)/2
        static constexpr double xs[3] = {0.5 - half, 0.5, 0.5 + half};
        static constexpr double ws[3] = {5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0};
        double e = 0.0;
        for (int q = 0; q < 3; ++q) {
            FALL_N_PETSC_CHECK(
                VecWAXPY(us_.get(), xs[q], du_.get(), u_ref));  // u_ref + s du
            this->residual(us_.get(), r_.get());
            PetscScalar dot = 0.0;
            FALL_N_PETSC_CHECK(VecDot(r_.get(), du_.get(), &dot));
            e += ws[q] * static_cast<double>(PetscRealPart(dot));
        }
        return e;
    }

private:
    petsc::OwnedVec du_{}, us_{}, r_{};
};

}  // namespace fall_n

#endif  // FALL_N_SRC_ANALYSIS_TAO_ENERGY_CONTINUATION_HH
