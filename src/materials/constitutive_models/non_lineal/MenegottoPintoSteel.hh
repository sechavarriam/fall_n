#ifndef FN_MENEGOTTO_PINTO_STEEL_HH
#define FN_MENEGOTTO_PINTO_STEEL_HH

// =============================================================================
//  MenegottoPintoSteel — Uniaxial cyclic steel model
// =============================================================================
//
//  Implementation of the Menegotto-Pinto (1973) model with isotropic strain
//  hardening, as modified by Filippou, Popov & Bertero (1983).
//
//  ─── Constitutive law ───────────────────────────────────────────────────
//
//  The model describes the monotonic and cyclic stress-strain behavior of
//  reinforcing steel.  The stress-strain curve transitions smoothly between
//  two asymptotes (initial elastic and post-yield) using a curved transition
//  controlled by the parameter R:
//
//    σ* = b·ε* + (1 − b)·ε* / (1 + |ε*|^R)^(1/R)
//
//  where σ* and ε* are normalized stress and strain measured from the last
//  reversal point:
//
//    ε* = (ε − ε_r) / (ε_0 − ε_r)
//    σ* = (σ − σ_r) / (σ_0 − σ_r)
//
//  Here:
//    (ε_r, σ_r) = last reversal point (where loading direction changed)
//    (ε_0, σ_0) = intersection of the two asymptotes for the current branch
//    b           = strain hardening ratio = E_h / E_0
//    R           = curvature parameter (controls roundedness of the knee)
//
//  ─── R parameter evolution (Filippou et al. 1983) ───────────────────────
//
//  R decreases with accumulated plastic strain, simulating the Bauschinger
//  effect observed in cyclic tests:
//
//    R(ξ) = R_0 − cR₁·ξ / (cR₂ + ξ)
//
//  where ξ = |ε_0 − ε_r| is the "shift" parameter measuring the plastic
//  excursion since the last reversal.
//
//  Typical values for reinforcing steel:
//    R_0  = 20     (initial curvature — large means sharp knee)
//    cR₁  = 18.5   (calibration parameter 1)
//    cR₂  = 0.15   (calibration parameter 2)
//
//  ─── Isotropic hardening (optional) ─────────────────────────────────────
//
//  The yield stress can grow symmetrically with accumulated plastic strain:
//
//    σ_y,shifted = σ_y + a₁·ε_max · (a₂ + ε_max)^(-1)
//
//  where ε_max = max |ε_p| accumulated.  Parameters:
//    a₁ = isotropic hardening amplitude   (default = 0 → no isotropic hardening)
//    a₂ = isotropic hardening saturation parameter
//
//  ─── Satisfies ──────────────────────────────────────────────────────────
//
//    ConstitutiveRelation (Level 1):
//      KinematicT = Strain<1>,  ConjugateT = Stress<1>,  TangentT = Matrix<1,1>
//      compute_response(ε) → σ   [const]
//      tangent(ε)          → C_t [const]
//
//    InelasticConstitutiveRelation (Level 2b):
//      InternalVariablesT  → MenegottoPintoState
//      update(ε)           → commit internal variables
//      internal_state()    → const MenegottoPintoState&
//
//    ExternallyStateDrivenConstitutiveRelation (Level 3):
//      compute_response(ε, α) → σ
//      tangent(ε, α)          → C_t
//      commit(α, ε)           → evolve an explicit external α
//
//  ─── References ─────────────────────────────────────────────────────────
//
//  [1] Menegotto, M. and Pinto, P.E. (1973). "Method of analysis for
//      cyclically loaded reinforced concrete plane frames including changes
//      in geometry and non-elastic behavior of elements under combined
//      normal force and bending." Proc. IABSE Symposium, Lisbon.
//
//  [2] Filippou, F.C., Popov, E.P. and Bertero, V.V. (1983). "Effects of
//      bond deterioration on hysteretic behavior of reinforced concrete
//      joints." Report EERC 83-19, UC Berkeley.
//
// =============================================================================

#include <cmath>
#include <cstddef>
#include <iostream>
#include <algorithm>

#include <Eigen/Dense>

#include "../../MaterialPolicy.hh"
#include "../../ConstitutiveRelation.hh"


// =============================================================================
//  MenegottoPintoState — Internal variables for history tracking
// =============================================================================
//
//  Stores the full state needed to reconstruct the cyclic stress-strain
//  response across arbitrary loading histories.  Each Gauss point (or fiber)
//  owns an independent copy.
//
//  The state is organized into three groups:
//    1. Reversal point (ε_r, σ_r) — where the loading direction last changed
//    2. Target point (ε_0, σ_0)   — asymptote intersection for current branch
//    3. Accumulated damage (ε_max, ξ) — for R-evolution and isotropic hardening
//
// =============================================================================

struct MenegottoPintoState {
    // ── Reversal point ────────────────────────────────────────────────
    double eps_r{0.0};       // strain at last reversal
    double sig_r{0.0};       // stress at last reversal

    // ── Asymptote intersection for current branch ─────────────────────
    double eps_0{0.0};       // strain at asymptote intersection
    double sig_0{0.0};       // stress at asymptote intersection

    // ── Previous reversal points (needed for branch tracking) ─────────
    double eps_r_prev{0.0};  // reversal point from the branch before last
    double sig_r_prev{0.0};

    // ── Accumulated plastic excursion ─────────────────────────────────
    double eps_max{0.0};     // max absolute accumulated plastic strain
    double xi{0.0};          // |ε_0 − ε_r| shift parameter for R evolution

    // ── Loading direction ─────────────────────────────────────────────
    //   +1 = loading (tension)
    //   −1 = unloading (compression)
    //    0 = initial (virgin state)
    int    direction{0};

    // ── Committed strain (for reversal detection) ─────────────────────
    double eps_committed{0.0};
    double sig_committed{0.0};

    // ── Flag: has the material ever yielded? ──────────────────────────
    bool   yielded{false};
};


// =============================================================================
//  MenegottoPintoSteel
// =============================================================================

class MenegottoPintoSteel {

public:
    // ── Concept-required type aliases ─────────────────────────────────
    using MaterialPolicyT    = UniaxialMaterial;
    using KinematicT         = Strain<1>;
    using ConjugateT         = Stress<1>;
    using TangentT           = Eigen::Matrix<double, 1, 1>;
    using InternalVariablesT = MenegottoPintoState;

    static constexpr std::size_t N   = 1;
    static constexpr std::size_t dim = 1;

private:
    // ── Material parameters (immutable after construction) ────────────

    double E0_;    // Initial elastic modulus (Young's modulus)
    double fy_;    // Yield stress
    double b_;     // Strain hardening ratio = E_h / E_0

    // R-evolution parameters (Filippou et al.)
    double R0_;    // Initial R parameter
    double cR1_;   // Calibration parameter 1
    double cR2_;   // Calibration parameter 2

    // Isotropic hardening parameters (optional)
    double a1_;    // Amplitude
    double a2_;    // Saturation

    // Derived constants
    double ey_;    // Yield strain = fy / E0
    double Eh_;    // Hardening modulus = b * E0

    // ── Internal state (per material point) ───────────────────────────
    MenegottoPintoState state_{};

    // (Cache removed: const methods are now truly const and thread-safe.
    //  The 2-arg overloads used through MaterialInstance never needed it.)


    // =================================================================
    //  Core: evaluate stress and tangent at given strain
    // =================================================================
    //
    //  This is the heart of the model.  Given the current total strain ε
    //  and the internal state (reversal points, direction), it computes:
    //    σ  = stress
    //    Et = tangent modulus (dσ/dε)
    //
    //  The algorithm:
    //    1. Detect if a reversal has occurred (direction change)
    //    2. If reversal: update state (new reversal point, target point)
    //    3. Compute normalized strain ε* = (ε − ε_r) / (ε_0 − ε_r)
    //    4. Compute normalized stress σ* via the Menegotto-Pinto curve
    //    5. De-normalize to get physical σ and Et
    //
    //  This method is const and does NOT modify state_ — it uses a
    //  local copy of the state to simulate what would happen at ε.
    //  The actual state update happens in update(ε).

    void evaluate(double eps, double& sig, double& Et,
                  MenegottoPintoState local_state) const
    {
        // ── Virgin state: purely elastic ──────────────────────────────
        if (!local_state.yielded) {
            double sig_trial = E0_ * eps;
            double fy_eff = effective_yield_stress(local_state);

            if (std::abs(sig_trial) < fy_eff) {
                sig = sig_trial;
                Et  = E0_;
                return;
            }

            // First yield: initialize state
            local_state.yielded   = true;
            local_state.direction = (eps > 0.0) ? 1 : -1;
            local_state.eps_r     = 0.0;
            local_state.sig_r     = 0.0;

            // Target point: intersection of elastic and hardening asymptotes
            double sgn = static_cast<double>(local_state.direction);
            local_state.eps_0 = sgn * fy_eff / E0_;
            local_state.sig_0 = sgn * fy_eff;
        }

        // ── Check for reversal ────────────────────────────────────────
        int new_dir = (eps > local_state.eps_committed) ? 1 : -1;
        if (std::abs(eps - local_state.eps_committed) < 1e-20) {
            new_dir = local_state.direction;  // no movement
        }

        if (local_state.direction != 0 && new_dir != local_state.direction) {
            // Reversal detected: swap reversal points
            local_state.eps_r_prev = local_state.eps_r;
            local_state.sig_r_prev = local_state.sig_r;

            local_state.eps_r = local_state.eps_committed;
            local_state.sig_r = local_state.sig_committed;

            // New target point
            double d = static_cast<double>(new_dir);
            double fy_eff = effective_yield_stress(local_state);

            // The target point (ε_0, σ_0) is the intersection of:
            //   Line 1: σ = E0·(ε − ε_r) + σ_r       (elastic from reversal)
            //   Line 2: σ = d·fy + Eh·(ε − d·εy)      (hardening through ±yield)
            //
            // Solving:  ε_0 = [d·(1−b)·fy − σ_r + E0·ε_r] / (E0 − Eh)
            //           σ_0 = σ_r + E0·(ε_0 − ε_r)

            local_state.eps_0 = (d * (1.0 - b_) * fy_eff
                                - local_state.sig_r
                                + E0_ * local_state.eps_r) / (E0_ - Eh_);
            local_state.sig_0 = local_state.sig_r
                              + E0_ * (local_state.eps_0 - local_state.eps_r);

            local_state.direction = new_dir;

            // ξ for R-evolution
            local_state.xi = std::abs(local_state.eps_0 - local_state.eps_r);
        }

        // ── Compute R for current branch ──────────────────────────────
        double xi = local_state.xi;
        double R = R0_ - cR1_ * xi / (cR2_ + xi);
        R = std::max(R, 1.0);  // R must be > 0 (clamp to 1 for stability)

        // ── Compute normalized strain ε* ──────────────────────────────
        double denom = local_state.eps_0 - local_state.eps_r;
        if (std::abs(denom) < 1e-30) {
            // Degenerate case: reversal and target coincide
            sig = local_state.sig_r;
            Et  = E0_;
            return;
        }

        double eps_star = (eps - local_state.eps_r) / denom;

        // ── Menegotto-Pinto curve ─────────────────────────────────────
        //
        //   σ* = b·ε* + (1 − b)·ε* / (1 + |ε*|^R)^(1/R)
        //
        //   dσ*/dε* = b + (1 − b) / (1 + |ε*|^R)^(1 + 1/R)

        double abs_eps_star = std::abs(eps_star);
        double pow_term     = std::pow(abs_eps_star, R);
        double bracket      = 1.0 + pow_term;
        double bracket_inv  = std::pow(bracket, -1.0 / R);

        double sig_star = b_ * eps_star + (1.0 - b_) * eps_star * bracket_inv;

        // Derivative  dσ*/dε*
        double dsig_deps_star = b_ + (1.0 - b_) * bracket_inv / bracket;

        // ── De-normalize ──────────────────────────────────────────────
        double sig_range = local_state.sig_0 - local_state.sig_r;
        sig = sig_star * sig_range + local_state.sig_r;
        Et  = dsig_deps_star * sig_range / denom;
    }


    // =================================================================
    //  Effective yield stress (with isotropic hardening)
    // =================================================================

    double effective_yield_stress(const MenegottoPintoState& s) const {
        if (a1_ <= 0.0) return fy_;

        double eps_max = s.eps_max;
        double shift = a1_ * eps_max / (a2_ + eps_max);
        return fy_ * (1.0 + shift);
    }

    void commit_state(MenegottoPintoState& state, double eps, double sig) const {
        int new_dir = (eps > state.eps_committed) ? 1 : -1;
        if (std::abs(eps - state.eps_committed) < 1e-20) {
            new_dir = state.direction;
        }

        if (!state.yielded) {
            double fy_eff = effective_yield_stress(state);
            if (std::abs(sig) >= fy_eff) {
                state.yielded   = true;
                state.direction = (eps > 0.0) ? 1 : -1;
                state.eps_r     = 0.0;
                state.sig_r     = 0.0;
                double sgn = static_cast<double>(state.direction);
                state.eps_0 = sgn * fy_eff / E0_;
                state.sig_0 = sgn * fy_eff;
            }
        }

        if (state.yielded && state.direction != 0 && new_dir != state.direction) {
            state.eps_r_prev = state.eps_r;
            state.sig_r_prev = state.sig_r;

            state.eps_r = state.eps_committed;
            state.sig_r = state.sig_committed;

            double d = static_cast<double>(new_dir);
            double fy_eff = effective_yield_stress(state);
            state.eps_0 = (d * (1.0 - b_) * fy_eff
                         - state.sig_r
                         + E0_ * state.eps_r) / (E0_ - Eh_);
            state.sig_0 = state.sig_r
                        + E0_ * (state.eps_0 - state.eps_r);

            state.direction = new_dir;
            state.xi = std::abs(state.eps_0 - state.eps_r);
        }

        double eps_p = eps - sig / E0_;
        state.eps_max = std::max(state.eps_max, std::abs(eps_p));

        state.eps_committed = eps;
        state.sig_committed = sig;
    }

public:

    // =================================================================
    //  ConstitutiveRelation interface (Level 1) — const
    // =================================================================

    [[nodiscard]] ConjugateT compute_response(
        const KinematicT& strain,
        const InternalVariablesT& state) const
    {
        double eps = strain.components();
        double sig, Et;
        evaluate(eps, sig, Et, state);

        ConjugateT stress;
        stress.set_components(sig);
        return stress;
    }

    [[nodiscard]] TangentT tangent(
        const KinematicT& strain,
        const InternalVariablesT& state) const
    {
        double eps = strain.components();
        double sig, Et;
        evaluate(eps, sig, Et, state);

        TangentT C;
        C(0, 0) = Et;
        return C;
    }

    void commit(InternalVariablesT& state, const KinematicT& strain) const {
        double eps = strain.components();
        double sig, Et;
        evaluate(eps, sig, Et, state);
        commit_state(state, eps, sig);
    }

    [[nodiscard]] ConjugateT compute_response(const KinematicT& strain) const {
        double eps = strain.components();
        double sig, Et;
        evaluate(eps, sig, Et, state_);

        ConjugateT stress;
        stress.set_components(sig);
        return stress;
    }

    [[nodiscard]] TangentT tangent(const KinematicT& strain) const {
        double eps = strain.components();
        double sig, Et;
        evaluate(eps, sig, Et, state_);

        TangentT C;
        C(0, 0) = Et;
        return C;
    }

    // =================================================================
    //  InelasticConstitutiveRelation interface (Level 2b)
    // =================================================================

    void update(const KinematicT& strain) {
        double eps = strain.components();
        double sig, Et;
        evaluate(eps, sig, Et, state_);
        commit_state(state_, eps, sig);
    }

    [[nodiscard]] const InternalVariablesT& internal_state() const {
        return state_;
    }

    /// Replace the internal state wholesale (for FE² state injection).
    void set_internal_state(const InternalVariablesT& s) {
        state_ = s;
    }


    // =================================================================
    //  Parameter accessors
    // =================================================================

    [[nodiscard]] double young_modulus()    const noexcept { return E0_; }
    [[nodiscard]] double yield_stress()    const noexcept { return fy_; }
    [[nodiscard]] double hardening_ratio() const noexcept { return b_;  }
    [[nodiscard]] double R0()              const noexcept { return R0_; }
    [[nodiscard]] double cR1()             const noexcept { return cR1_; }
    [[nodiscard]] double cR2()             const noexcept { return cR2_; }


    // =================================================================
    //  Constructors
    // =================================================================

    /// Full constructor with all parameters.
    ///
    ///  @param E0    Initial elastic modulus
    ///  @param fy    Yield stress (positive)
    ///  @param b     Strain hardening ratio (E_h / E_0, typically 0.005–0.02)
    ///  @param R0    Initial curvature parameter (typically 15–25)
    ///  @param cR1   R-evolution parameter 1 (typically 18.5)
    ///  @param cR2   R-evolution parameter 2 (typically 0.15)
    ///  @param a1    Isotropic hardening amplitude (0 = none)
    ///  @param a2    Isotropic hardening saturation (0 = none)
    constexpr MenegottoPintoSteel(
        double E0, double fy, double b,
        double R0  = 20.0,
        double cR1 = 18.5,
        double cR2 = 0.15,
        double a1  = 0.0,
        double a2  = 1.0)
        : E0_{E0}, fy_{fy}, b_{b},
          R0_{R0}, cR1_{cR1}, cR2_{cR2},
          a1_{a1}, a2_{a2},
          ey_{fy / E0}, Eh_{b * E0}
    {}

    constexpr MenegottoPintoSteel() = default;
    ~MenegottoPintoSteel() = default;

    MenegottoPintoSteel(const MenegottoPintoSteel&)               = default;
    MenegottoPintoSteel(MenegottoPintoSteel&&) noexcept           = default;
    MenegottoPintoSteel& operator=(const MenegottoPintoSteel&)    = default;
    MenegottoPintoSteel& operator=(MenegottoPintoSteel&&) noexcept = default;


    // =================================================================
    //  Diagnostics
    // =================================================================

    void print_constitutive_parameters() const {
        std::cout << "=== Menegotto-Pinto Steel ===" << std::endl;
        std::cout << "E0 = " << E0_ << ",  fy = " << fy_ << std::endl;
        std::cout << "b  = " << b_  << ",  εy = " << ey_ << std::endl;
        std::cout << "R0 = " << R0_ << ",  cR1 = " << cR1_
                  << ",  cR2 = " << cR2_ << std::endl;
        std::cout << "a1 = " << a1_ << ",  a2 = " << a2_ << std::endl;
    }
};


// =============================================================================
//  Static concept verification
// =============================================================================

static_assert(
    ConstitutiveRelation<MenegottoPintoSteel>,
    "MenegottoPintoSteel must satisfy ConstitutiveRelation");

static_assert(
    InelasticConstitutiveRelation<MenegottoPintoSteel>,
    "MenegottoPintoSteel must satisfy InelasticConstitutiveRelation");

static_assert(
    ExternallyStateDrivenConstitutiveRelation<MenegottoPintoSteel>,
    "MenegottoPintoSteel must satisfy ExternallyStateDrivenConstitutiveRelation");

static_assert(
    ExternallyStateInjectable<MenegottoPintoSteel>,
    "MenegottoPintoSteel must satisfy ExternallyStateInjectable");


#endif // FN_MENEGOTTO_PINTO_STEEL_HH
