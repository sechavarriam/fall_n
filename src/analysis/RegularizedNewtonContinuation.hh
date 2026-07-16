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
#include <cstddef>
#include <utility>
#include <vector>

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

// ── Deflation policy (compile-time) ──────────────────────────────────────────
//  Branch selection at load reversals.  The softening + fixed smeared crack
//  admits multiple equilibria and plain LM can commit to the spurious HIGH-force
//  branch.  Deflation multiplies the Newton direction by a scalar that repels the
//  iterate from previously registered spurious roots u_i (Farrell, Birkisson &
//  Funke).  With eta(u) = prod_i ( ||u-u_i||^-power + shift ), the deflated
//  direction is du_G = tau du_N with
//    tau = 1 / (1 - (grad eta / eta) . du_N)      (rank-1 Sherman-Morrison).
//  NoDeflation is the default: the solver path is then COMPILED IDENTICALLY to
//  the un-deflated continuation (every deflation branch sits behind
//  `if constexpr (Defl::enabled)`).
template <typename D>
concept DeflationPolicy =
    requires(const D d, Vec u, Vec du, const std::vector<petsc::OwnedVec>& roots,
             Vec work) {
        { D::enabled } -> std::convertible_to<bool>;
        { d.factor(u, du, roots, work) } -> std::convertible_to<double>;
        { d.eta(u, roots, work) } -> std::convertible_to<double>;
    };

struct NoDeflation {
    static constexpr bool enabled = false;
    [[nodiscard]] double factor(Vec, Vec, const std::vector<petsc::OwnedVec>&,
                                Vec) const noexcept { return 1.0; }
    [[nodiscard]] double eta(Vec, const std::vector<petsc::OwnedVec>&,
                             Vec) const noexcept { return 1.0; }
};

struct ShermanMorrisonDeflation {
    static constexpr bool enabled = true;
    double power{2.0};
    double shift{1.0};
    double tau_max{50.0};

    // eta(u) = prod_i ( ||u-u_i||^-power + shift ).  Compared across a step
    //  (rnew*eta(ut) vs rn*eta(u)) so LM accepts the move that escapes a spurious
    //  basin; a plain product suffices for the few roots a reversal generates.
    [[nodiscard]] double eta(Vec u, const std::vector<petsc::OwnedVec>& roots,
                             Vec work) const {
        double e = 1.0;
        for (const auto& ri : roots) {
            FALL_N_PETSC_CHECK(VecWAXPY(work, -1.0, ri.get(), u));  // u - u_i
            PetscReal d = 0.0;
            FALL_N_PETSC_CHECK(VecNorm(work, NORM_2, &d));
            const double dd = std::max<double>(static_cast<double>(d), 1.0e-30);
            e *= std::pow(dd, -power) + shift;
        }
        return e;
    }

    // tau = 1 / (1 - (grad eta / eta) . du_N), clamped to +/- tau_max (sign kept).
    [[nodiscard]] double factor(Vec u, Vec du_N,
                                const std::vector<petsc::OwnedVec>& roots,
                                Vec work) const {
        if (roots.empty()) return 1.0;
        double s = 0.0;  // (grad eta / eta) . du_N
        for (const auto& ri : roots) {
            FALL_N_PETSC_CHECK(VecWAXPY(work, -1.0, ri.get(), u));  // u - u_i
            PetscReal d = 0.0;
            FALL_N_PETSC_CHECK(VecNorm(work, NORM_2, &d));
            const double dd = static_cast<double>(d);
            if (dd < 1.0e-30) continue;                 // at the root; skip
            const double m = std::pow(dd, -power) + shift;   // m_i
            PetscScalar dot = 0.0;
            FALL_N_PETSC_CHECK(VecDot(work, du_N, &dot));     // (u-u_i).du_N
            s += (-power * std::pow(dd, -power - 2.0) *
                  static_cast<double>(PetscRealPart(dot))) / m;
        }
        double denom = 1.0 - s;
        if (std::abs(denom) < 1.0e-30) denom = (denom < 0.0 ? -1.0e-30 : 1.0e-30);
        double tau = 1.0 / denom;
        if (std::abs(tau) > tau_max) tau = (tau < 0.0 ? -tau_max : tau_max);
        return tau;
    }
};

static_assert(DeflationPolicy<NoDeflation>);
static_assert(DeflationPolicy<ShermanMorrisonDeflation>);

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
    // Energy line search (only effective when the Backend also models the
    //  incremental energy, e.g. NLAnalysisEnergyBackend). The LM direction
    //  with K + mu I positive definite is a DESCENT direction of the
    //  incremental potential Pi (R = grad Pi), so a full step that RAISES Pi
    //  is jumping over an energy barrier into another basin: alpha is
    //  backtracked, and if no alpha lowers Pi (e.g. heading into a maximum of
    //  Pi that is still a residual root) the step is REJECTED so mu grows —
    //  the large-mu limit is gradient descent on Pi, which cannot stall at a
    //  maximum/saddle. Off by default: bit-identical to the plain LM.
    bool   energy_linesearch{false};
    int    linesearch_max_backtracks{4};
    // Continuación proximal (selección de rama por CONTINUIDAD): con
    //  proximal_frac > 0 el paso se resuelve en dos fases, primero el punto
    //  proximal  argmin Pi(u) + kappa/2 ||u-u_n||^2  con
    //  kappa = proximal_frac * K_rms, donde K_rms es el RMS del diagonal de
    //  K (escala POR grado de libertad: la norma-2 completa crece con
    //  sqrt(N) e inflaría kappa en un modelo real). Convexifica y elige la
    //  cuenca conexa con el último aceptado; el pulido con kappa = 0 lleva
    //  al equilibrio verdadero. Si el flujo proximal no alcanza
    //  estacionariedad dentro de su presupuesto, el paso se RESTAURA y se
    //  resuelve plano con presupuesto completo (nunca peor que el lazo
    //  original).  0 = apagado (lazo original intacto).
    double proximal_frac{0.0};
};

struct RegularizedNewtonStepResult {
    bool   converged{false};
    int    iterations{0};
    double residual{0.0};
    double mu_final{0.0};
};

// ── Solver ───────────────────────────────────────────────────────────────────
template <ContinuationBackend Backend,
          RegularizationPolicy Reg = LevenbergMarquardt,
          DeflationPolicy Defl = NoDeflation>
class RegularizedNewtonContinuation {
public:
    explicit RegularizedNewtonContinuation(Backend backend,
                                           RegularizedNewtonConfig cfg = {},
                                           Reg reg = {},
                                           Defl defl = {})
        : backend_(std::move(backend)), cfg_(cfg), reg_(reg), defl_(defl)
    {
        u_      = backend_.create_vector();
        R_      = backend_.create_vector();
        du_     = backend_.create_vector();
        ut_     = backend_.create_vector();
        dg_     = backend_.create_vector();
        u_prev_ = backend_.create_vector();  // second-last accepted (secant)
        u_hold_ = backend_.create_vector();  // this step's warm-start origin
        pv_     = backend_.create_vector();  // proximal work vector (u - a)
        pa_     = backend_.create_vector();  // proximal flow anchor a
        p0_     = backend_.create_vector();  // proximal restore point (step start)
        w_      = backend_.create_vector();  // deflation work vector
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
        double mu = reg_.initial();
        int it = 0;
        bool converged = false;

        //  Continuación PROXIMAL (opt-in, proximal_frac > 0): la fase A
        //  resuelve el paso proximal
        //      argmin_u  Pi(u) + kappa/2 ||u - u_n||^2
        //  (residuo R + kappa (u-u_n), tangente K + kappa I).  kappa
        //  convexifica el potencial incremental y sesga la solución hacia la
        //  rama CONEXA con el último estado aceptado: selección de rama por
        //  CONTINUIDAD (la versión por-paso del mu por-iteración del LM;
        //  pseudo-transient continuation en forma de punto proximal).  La
        //  fase B (kappa = 0) pule desde esa cuenca hasta el equilibrio
        //  verdadero.  Con proximal_frac = 0 la fase A no existe y el lazo
        //  es idéntico al original.
        //  kappa se escala con el K_ii representativo POR grado de libertad
        //  (RMS del diagonal): la norma-2 completa crece con sqrt(N) y en un
        //  modelo real inflaría kappa en órdenes de magnitud respecto de la
        //  rigidez de cada DOF (mu_max conserva su convención histórica).
        PetscInt prox_n = 1;
        FALL_N_PETSC_CHECK(VecGetSize(dg_.get(), &prox_n));
        const double diag_rms =
            diagn / std::sqrt(static_cast<double>(std::max<PetscInt>(prox_n, 1)));
        const double kappa_full =
            cfg_.proximal_frac > 0.0
                ? cfg_.proximal_frac * std::max<double>(diag_rms, 1.0e-30)
                : 0.0;
        //  El ancla del flujo proximal AVANZA con cada sub-paso convergido
        //  (backward-Euler del flujo de gradiente u' = -grad Pi con paso
        //  1/kappa): un único paso proximal anclado en u_n se queda a mitad
        //  de camino cuando la cuenca está lejos; el flujo la recorre.
        if (kappa_full > 0.0) {
            FALL_N_PETSC_CHECK(VecCopy(u_.get(), pa_.get()));
            FALL_N_PETSC_CHECK(VecCopy(u_.get(), p0_.get()));  // punto de restauración
        }
        const auto eval_residual = [&](Vec x, Vec r, double kappa) {
            backend_.residual(x, r);
            if (kappa > 0.0) {
                FALL_N_PETSC_CHECK(
                    VecWAXPY(pv_.get(), -1.0, pa_.get(), x));  // x - ancla
                FALL_N_PETSC_CHECK(VecAXPY(r, kappa, pv_.get()));
            }
        };
        const auto prox_energy_delta = [&](Vec xt, Vec x, double kappa) {
            //  kappa/2 (||xt - a||^2 - ||x - a||^2): el término proximal del
            //  mérito energético de la fase A.
            FALL_N_PETSC_CHECK(
                VecWAXPY(pv_.get(), -1.0, pa_.get(), xt));
            PetscReal nt = 0.0;
            FALL_N_PETSC_CHECK(VecNorm(pv_.get(), NORM_2, &nt));
            FALL_N_PETSC_CHECK(
                VecWAXPY(pv_.get(), -1.0, pa_.get(), x));
            PetscReal nx = 0.0;
            FALL_N_PETSC_CHECK(VecNorm(pv_.get(), NORM_2, &nx));
            return 0.5 * kappa *
                   (static_cast<double>(nt) * static_cast<double>(nt) -
                    static_cast<double>(nx) * static_cast<double>(nx));
        };

        for (int phase = (kappa_full > 0.0 ? 0 : 1); phase < 2; ++phase) {
            const double kappa = (phase == 0) ? kappa_full : 0.0;
            const int it_cap = (phase == 0)
                ? std::max(10, cfg_.max_newton / 2)
                : cfg_.max_newton;
            if (phase == 0) {
                eval_residual(u_.get(), R_.get(), kappa);
            } else if (kappa_full > 0.0) {
                eval_residual(u_.get(), R_.get(), 0.0);  // residuo verdadero
            }
            bool phase_conv = false;
            bool flow_stationary = false;
            for (;;) {   // lazo de FLUJO proximal (una vuelta si kappa = 0)
            double r_best = std::max(norm_(R_.get()), 1.0e-30);
            int stag = 0;
            phase_conv = false;
            for (; it < it_cap; ++it) {
                const double rn = norm_(R_.get());
                if (rn <= cfg_.tol_abs || rn <= cfg_.tol_rel * r0) {
                    phase_conv = true;
                    break;
                }
                if (rn < r_best * 0.9999) { r_best = rn; stag = 0; }
                else if (++stag > cfg_.stag_max) break;  // accept best regularized

                backend_.tangent(u_.get(), K_.get());
                mu = std::min(mu, mu_max);
                if (mu + kappa > 0.0) {
                    FALL_N_PETSC_CHECK(
                        MatShift(K_.get(), mu + kappa));  // K + (mu+kappa) I
                }
                FALL_N_PETSC_CHECK(
                    KSPSetOperators(ksp_.get(), K_.get(), K_.get()));
                FALL_N_PETSC_CHECK(VecScale(R_.get(), -1.0));           // -R
                FALL_N_PETSC_CHECK(KSPSolve(ksp_.get(), R_.get(), du_.get()));
                if constexpr (Defl::enabled) {
                    if (!roots_.empty()) {
                        const double tau =
                            defl_.factor(u_.get(), du_.get(), roots_, w_.get());
                        FALL_N_PETSC_CHECK(
                            VecScale(du_.get(), tau));  // du_G = tau du_N
                    }
                }
                FALL_N_PETSC_CHECK(
                    VecWAXPY(ut_.get(), 1.0, du_.get(), u_.get()));     // u+du
                eval_residual(ut_.get(), R_.get(), kappa);
                double rnew = norm_(R_.get());
                const auto residual_accept = [&](double rnew_) {
                    if constexpr (Defl::enabled) {
                        if (!roots_.empty()) {
                            // Compare the DEFLATED residual so LM keeps the
                            //  step that escapes a spurious basin even if
                            //  ||F|| ticks up locally.
                            const double eu =
                                defl_.eta(u_.get(), roots_, w_.get());
                            const double et =
                                defl_.eta(ut_.get(), roots_, w_.get());
                            return (rnew_ * et) < (rn * eu);
                        }
                        return rnew_ < rn;
                    } else {
                        return rnew_ < rn;
                    }
                };
                bool accept_step = residual_accept(rnew);
                if constexpr (requires(Backend& b, Vec x, Vec y) {
                                  { b.incremental_energy(x, y) }
                                      -> std::convertible_to<double>;
                              }) {
                    if (cfg_.energy_linesearch && accept_step) {
                        double alpha = 1.0;
                        double e =
                            backend_.incremental_energy(ut_.get(), u_.get());
                        if (kappa > 0.0) {
                            e += prox_energy_delta(ut_.get(), u_.get(), kappa);
                        }
                        int bt = 0;
                        while (e > 0.0 && bt < cfg_.linesearch_max_backtracks) {
                            alpha *= 0.5;
                            FALL_N_PETSC_CHECK(VecWAXPY(
                                ut_.get(), alpha, du_.get(), u_.get()));
                            eval_residual(ut_.get(), R_.get(), kappa);
                            rnew = norm_(R_.get());
                            ++bt;
                            e = backend_.incremental_energy(ut_.get(),
                                                            u_.get());
                            if (kappa > 0.0) {
                                e += prox_energy_delta(ut_.get(), u_.get(),
                                                       kappa);
                            }
                        }
                        accept_step = (e <= 0.0) && residual_accept(rnew);
                    }
                }
                if (accept_step) {
                    FALL_N_PETSC_CHECK(VecCopy(ut_.get(), u_.get()));
                    mu = reg_.on_accept(mu);
                } else {
                    mu = reg_.on_reject(mu, mu_max);
                    eval_residual(u_.get(), R_.get(), kappa);  // R back at u
                }
            }
            if (phase != 0 || !phase_conv || it >= it_cap) break;
            //  ¿Flujo estacionario?  Si el sub-paso proximal convergió con
            //  ||u - ancla|| ~ 0, entonces grad Pi ~ 0 en un punto ESTABLE
            //  del flujo (los máximos/sillas repelen la iteración); si no,
            //  el ancla avanza (backward-Euler) y se resuelve el siguiente
            //  sub-paso.
            {
                FALL_N_PETSC_CHECK(
                    VecWAXPY(pv_.get(), -1.0, pa_.get(), u_.get()));
                PetscReal step_n = 0.0, u_n = 0.0;
                FALL_N_PETSC_CHECK(VecNorm(pv_.get(), NORM_2, &step_n));
                FALL_N_PETSC_CHECK(VecNorm(u_.get(), NORM_2, &u_n));
                if (static_cast<double>(step_n) <=
                    1.0e-8 * std::max(1.0, static_cast<double>(u_n))) {
                    flow_stationary = true;
                    break;  // estacionario: el flujo llegó a un mínimo
                }
                FALL_N_PETSC_CHECK(VecCopy(u_.get(), pa_.get()));
                eval_residual(u_.get(), R_.get(), kappa);  // re-anclado
            }
            }
            if (phase == 0 && !flow_stationary) {
                //  SALVAGUARDA: el flujo se agotó (presupuesto o estancamiento)
                //  a mitad de camino. Ese estado intermedio está sesgado por
                //  kappa y NO debe llegar al accept(): se restaura el arranque
                //  del paso y la fase B corre como resolución plana con
                //  presupuesto completo (nunca peor que el lazo original).
                FALL_N_PETSC_CHECK(VecCopy(p0_.get(), u_.get()));
                it = 0;
                mu = reg_.initial();
                continue;
            }
            converged = phase_conv;
        }
        //  El piso de aceptación y el residuo reportado son SIEMPRE del
        //  residuo verdadero (la fase A converge sobre el proximal).
        if (kappa_full > 0.0) {
            backend_.residual(u_.get(), R_.get());
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

    // ── Branch-selection / retry API ─────────────────────────────────────────
    //  Drives the driver's detect-restore-retry loop across a reversal. All of
    //  these leave the default (NoDeflation) continuation behaviour untouched:
    //  registered roots only matter inside `if constexpr (Defl::enabled)`.
    void register_spurious_root(Vec r) {
        petsc::OwnedVec v = backend_.create_vector();
        FALL_N_PETSC_CHECK(VecCopy(r, v.get()));
        roots_.push_back(std::move(v));
    }
    void clear_spurious_roots() noexcept { roots_.clear(); }
    [[nodiscard]] std::size_t spurious_root_count() const noexcept {
        return roots_.size();
    }

    // Overwrite the current solution and reset the secant history, so the next
    //  advance_to warm-starts cleanly from the injected state.
    void set_solution(Vec u) {
        FALL_N_PETSC_CHECK(VecCopy(u, u_.get()));
        accepted_ = 0;
    }

    void set_config(const RegularizedNewtonConfig& cfg) noexcept { cfg_ = cfg; }
    void set_regularization(const Reg& reg) noexcept { reg_ = reg; }

    // Cheap checkpoint of the continuation state (solution + secant history) so a
    //  spurious step can be rolled back and retried.
    struct State {
        petsc::OwnedVec u{}, u_prev{};
        double p_curr{0.0};
        double p_prev{0.0};
        int    accepted{0};
    };
    [[nodiscard]] State capture_state() {
        State s;
        s.u      = backend_.create_vector();
        s.u_prev = backend_.create_vector();
        FALL_N_PETSC_CHECK(VecCopy(u_.get(),      s.u.get()));
        FALL_N_PETSC_CHECK(VecCopy(u_prev_.get(), s.u_prev.get()));
        s.p_curr   = p_curr_;
        s.p_prev   = p_prev_;
        s.accepted = accepted_;
        return s;
    }
    void restore_state(const State& s) {
        FALL_N_PETSC_CHECK(VecCopy(s.u.get(),      u_.get()));
        FALL_N_PETSC_CHECK(VecCopy(s.u_prev.get(), u_prev_.get()));
        p_curr_   = s.p_curr;
        p_prev_   = s.p_prev;
        accepted_ = s.accepted;
    }

private:
    [[nodiscard]] double norm_(Vec v) const
    {
        PetscReal n = 0.0;
        VecNorm(v, NORM_2, &n);
        return static_cast<double>(n);
    }

    Backend backend_;
    RegularizedNewtonConfig cfg_;
    Reg  reg_;
    Defl defl_;
    petsc::OwnedVec u_{}, R_{}, du_{}, ut_{}, dg_{}, u_prev_{}, u_hold_{}, w_{},
                    pv_{}, pa_{}, p0_{};
    petsc::OwnedMat K_{};
    petsc::OwnedKSP ksp_{};
    std::vector<petsc::OwnedVec> roots_{};  // registered spurious roots (deflation)
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
