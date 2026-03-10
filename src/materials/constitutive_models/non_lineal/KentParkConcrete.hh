#ifndef FN_KENT_PARK_CONCRETE_HH
#define FN_KENT_PARK_CONCRETE_HH

// =============================================================================
//  KentParkConcrete — Uniaxial cyclic concrete model
// =============================================================================
//
//  Implementation of the modified Kent-Park model (Scott, Park & Priestley
//  1982), with tension cut-off and linear unloading/reloading.
//
//  This model captures the essential features of concrete behavior under
//  cyclic loading relevant for reinforced concrete fiber sections:
//    - Parabolic compression ascending branch
//    - Linear softening descending branch
//    - Linear unloading/reloading to the origin
//    - Optional tension capacity with linear descending
//    - Confinement effect via K parameter
//
//  ─── Compressive envelope ───────────────────────────────────────────────
//
//  The compression envelope (ε < 0 in sign convention) consists of two
//  regions. Using f'c  = peak compressive strength (positive value),
//  ε_0 = strain at peak (negative), ε_u = ultimate strain (negative):
//
//  Region 1: Ascending parabola (ε_0 ≤ ε ≤ 0)
//
//    σ = K·f'c · [ 2·(ε/ε_0) − (ε/ε_0)² ]
//
//  Region 2: Linear descending (ε < ε_0)
//
//    σ = K·f'c · [ 1 − Z·(ε − ε_0) ]  ≥  0.2·K·f'c
//
//  where Z is the slope of the descending branch:
//
//    Z = 0.5 / (ε_50u + ε_50h − ε_0)
//
//  Parameters K, ε_50u, ε_50h account for confinement:
//    K     = 1 + ρ_s · f_yh / f'c        (strength enhancement)
//    ε_0   = −0.002 · K                   (strain at peak)
//    ε_50u = (3 + 0.29·f'c) / (145·f'c − 1000)   (unconfined)
//    ε_50h = 0.75 · ρ_s · √(h'/s_h)              (confinement term)
//
//  For unconfined concrete (K = 1, ε_50h = 0):
//    ε_0   = −0.002
//    Z     = 0.5 / (ε_50u − ε_0)
//
//  ─── Tensile behavior ───────────────────────────────────────────────────
//
//  Linear elastic up to f_t = 0.1·f'c (default), then immediate drop
//  to zero (tension cut-off).  The tensile stiffness is the initial
//  compressive stiffness E_c = 2·K·f'c / ε_0.
//
//  ─── Cyclic behavior (simplified) ───────────────────────────────────────
//
//  Unloading from the compression envelope follows a straight line back
//  toward (ε_pl, 0), where ε_pl is the plastic strain:
//
//    ε_pl = ε_unload − σ_unload / E_c
//
//  Reloading follows the same line back to the envelope.  This simplified
//  rule captures the essential behavior without the complexity of Karsan-Jirsa
//  or Palermo-Vecchio rules.
//
//  ─── Sign convention ────────────────────────────────────────────────────
//
//  Following structural engineering convention:
//    Compression: ε < 0, σ < 0
//    Tension:     ε > 0, σ > 0
//
//  Note: f'c and f_t are stored as POSITIVE values internally.
//
//  ─── Satisfies ──────────────────────────────────────────────────────────
//
//    ConstitutiveRelation (Level 1)
//    InelasticConstitutiveRelation (Level 2b)
//
//  ─── References ─────────────────────────────────────────────────────────
//
//  [1] Kent, D.C. and Park, R. (1971). "Flexural members with confined
//      concrete." ASCE J. Struct. Div., 97(7), 1969-1990.
//
//  [2] Scott, B.D., Park, R. and Priestley, M.J.N. (1982). "Stress-strain
//      behavior of concrete confined by overlapping hoops at low and high
//      strain rates." ACI J., 79(1), 13-27.
//
//  [3] Park, R., Priestley, M.J.N. and Gill, W.D. (1982). "Ductility of
//      square-confined concrete columns." ASCE J. Struct. Div., 108(4).
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
//  KentParkState — Internal variables for concrete history
// =============================================================================

struct KentParkState {
    // ── Envelope tracking ─────────────────────────────────────────────
    double eps_min{0.0};       // minimum strain ever reached (compression = negative)
    double sig_at_eps_min{0.0}; // stress at eps_min (on envelope)

    // ── Unloading/reloading reference ─────────────────────────────────
    double eps_pl{0.0};         // current plastic strain (compression residual)
    double eps_unload{0.0};     // strain at start of unloading
    double sig_unload{0.0};     // stress at start of unloading

    // ── Loading state ─────────────────────────────────────────────────
    //   0 = virgin
    //   1 = on compression envelope
    //   2 = unloading from compression
    //   3 = reloading toward compression envelope
    //   4 = tension (cracked or uncracked)
    int    state{0};

    // ── Committed values ──────────────────────────────────────────────
    double eps_committed{0.0};
    double sig_committed{0.0};

    // ── Tension cracking ──────────────────────────────────────────────
    bool   cracked{false};      // has tension cracking occurred?
};


// =============================================================================
//  KentParkConcrete
// =============================================================================

class KentParkConcrete {

public:
    // ── Concept-required type aliases ─────────────────────────────────
    using MaterialPolicyT    = UniaxialMaterial;
    using KinematicT         = Strain<1>;
    using ConjugateT         = Stress<1>;
    using TangentT           = Eigen::Matrix<double, 1, 1>;
    using InternalVariablesT = KentParkState;

    static constexpr std::size_t N   = 1;
    static constexpr std::size_t dim = 1;

private:
    // ── Material parameters ───────────────────────────────────────────

    double fpc_;    // Peak compressive strength f'c (POSITIVE value, MPa)
    double eps0_;   // Strain at peak (NEGATIVE value)
    double Ec_;     // Initial tangent modulus = 2·K·f'c / |ε_0|
    double ft_;     // Tensile strength (POSITIVE value)
    double Zslope_; // Softening slope parameter Z
    double Kconf_;  // Confinement factor K (1.0 for unconfined)

    // Derived
    double fpc_residual_; // Residual compressive stress = 0.2·K·f'c
    double eps_u_;        // Ultimate strain (where stress = residual)

    // ── Internal state ────────────────────────────────────────────────
    KentParkState state_{};

    // ── Mutable cache ─────────────────────────────────────────────────
    mutable double last_eps_{0.0};
    mutable double last_sig_{0.0};
    mutable double last_Et_{0.0};
    mutable bool   cache_valid_{false};


    // =================================================================
    //  Compression envelope: σ = f(ε) for ε ≤ 0
    // =================================================================

    void compression_envelope(double eps, double& sig, double& Et) const {
        // eps is negative, eps0_ is negative
        double eta = eps / eps0_;  // ratio ε/ε_0  (positive in compression)

        if (eps >= eps0_) {
            // Region 1: Ascending parabola  σ = Kf'c · [2η − η²]
            double fpc_K = Kconf_ * fpc_;
            sig = -fpc_K * (2.0 * eta - eta * eta);
            Et  = -fpc_K * (2.0 - 2.0 * eta) / eps0_;
            // Note: sig < 0 (compression), Et > 0 (stiffness)
        } else {
            // Region 2: Linear descending  σ = Kf'c · [1 − Z·(ε_0 − ε)]
            // Note: (eps0_ - eps) > 0 since eps < eps0_ (more negative)
            double fpc_K = Kconf_ * fpc_;
            double descent = 1.0 - Zslope_ * (eps0_ - eps);
            double residual_ratio = 0.2;  // 20% residual strength

            if (descent >= residual_ratio) {
                sig = -fpc_K * descent;
                Et  = -fpc_K * Zslope_;  // negative (softening)
            } else {
                // Residual plateau
                sig = -fpc_K * residual_ratio;
                Et  = 1e-6 * Ec_;  // small but nonzero for numerical stability
            }
        }
    }


    // =================================================================
    //  Full evaluation: stress and tangent at given strain
    // =================================================================

    void evaluate(double eps, double& sig, double& Et,
                  KentParkState local_state) const
    {
        // ══════════════════════════════════════════════════════════════
        //  TENSION (ε > 0)
        // ══════════════════════════════════════════════════════════════

        if (eps > 0.0) {
            if (local_state.cracked) {
                // Cracked: zero tension
                sig = 0.0;
                Et  = 1e-6 * Ec_;
                return;
            }

            // Uncracked: linear elastic up to f_t
            sig = Ec_ * eps;
            Et  = Ec_;
            if (sig > ft_) {
                // Tension cut-off
                sig = 0.0;
                Et  = 1e-6 * Ec_;
            }
            return;
        }

        // ══════════════════════════════════════════════════════════════
        //  COMPRESSION (ε ≤ 0)
        // ══════════════════════════════════════════════════════════════

        // Check if on envelope or unloading/reloading
        if (eps <= local_state.eps_min) {
            // On or extending the envelope
            compression_envelope(eps, sig, Et);
            return;
        }

        // Unloading/reloading: linear between (eps_min, sig_at_eps_min) and (eps_pl, 0)
        double denom = local_state.eps_min - local_state.eps_pl;
        if (std::abs(denom) < 1e-30) {
            sig = 0.0;
            Et  = Ec_;
            return;
        }

        // If strain has passed the plastic strain point, stress is zero
        // (fully unloaded from compression)
        if (eps >= local_state.eps_pl) {
            sig = 0.0;
            Et  = 1e-6 * Ec_;
            return;
        }

        // Linear path: σ = sig_at_eps_min · (ε − ε_pl) / (ε_min − ε_pl)
        double ratio = (eps - local_state.eps_pl) / denom;
        sig = local_state.sig_at_eps_min * ratio;
        Et  = local_state.sig_at_eps_min / denom;
    }


public:

    // =================================================================
    //  ConstitutiveRelation interface (Level 1) — const
    // =================================================================

    [[nodiscard]] ConjugateT compute_response(const KinematicT& strain) const {
        double eps = strain.components();
        double sig, Et;
        evaluate(eps, sig, Et, state_);

        last_eps_ = eps;
        last_sig_ = sig;
        last_Et_  = Et;
        cache_valid_ = true;

        ConjugateT stress;
        stress.set_components(sig);
        return stress;
    }

    [[nodiscard]] TangentT tangent(const KinematicT& strain) const {
        double eps = strain.components();
        if (cache_valid_ && std::abs(eps - last_eps_) < 1e-30) {
            TangentT C;
            C(0, 0) = last_Et_;
            return C;
        }
        double sig, Et;
        evaluate(eps, sig, Et, state_);
        last_eps_ = eps;
        last_sig_ = sig;
        last_Et_  = Et;
        cache_valid_ = true;

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

        // ── Update envelope tracking ──────────────────────────────────
        if (eps < state_.eps_min) {
            state_.eps_min       = eps;
            compression_envelope(eps, state_.sig_at_eps_min, Et);
            // Compute plastic strain for unloading
            state_.eps_pl = eps - state_.sig_at_eps_min / Ec_;
            state_.state  = 1;  // on envelope
        } else if (eps <= 0.0) {
            // Unloading or reloading in compression
            state_.state = 2;
        }

        // ── Check tension cracking ────────────────────────────────────
        if (eps > 0.0 && !state_.cracked) {
            if (Ec_ * eps > ft_) {
                state_.cracked = true;
            }
            state_.state = 4;
        }

        // ── Commit ────────────────────────────────────────────────────
        state_.eps_committed = eps;
        state_.sig_committed = sig;
        cache_valid_ = false;
    }

    [[nodiscard]] const InternalVariablesT& internal_state() const {
        return state_;
    }


    // =================================================================
    //  Parameter accessors
    // =================================================================

    [[nodiscard]] double peak_compressive_strength()  const noexcept { return fpc_; }
    [[nodiscard]] double strain_at_peak()             const noexcept { return eps0_; }
    [[nodiscard]] double initial_modulus()             const noexcept { return Ec_; }
    [[nodiscard]] double tensile_strength()            const noexcept { return ft_; }
    [[nodiscard]] double confinement_factor()          const noexcept { return Kconf_; }
    [[nodiscard]] double softening_slope()             const noexcept { return Zslope_; }


    // =================================================================
    //  Constructors
    // =================================================================

    /// Unconfined concrete constructor.
    ///
    /// @param fpc  Peak compressive strength f'c (POSITIVE, MPa)
    /// @param ft   Tensile strength (POSITIVE, default = 0.1·f'c)
    ///
    /// For unconfined concrete: K=1, ε_0 = −0.002, Z from f'c.
    KentParkConcrete(double fpc, double ft = 0.0)
        : fpc_{fpc},
          eps0_{-0.002},
          ft_{(ft > 0.0) ? ft : 0.1 * fpc},
          Kconf_{1.0}
    {
        Ec_ = 2.0 * fpc_ / std::abs(eps0_);

        // Z for unconfined concrete (Scott et al. 1982)
        // ε_50u from Park et al. (f'c in MPa, formula from psi-based original):
        double eps_50u = (3.0 + 0.29 * fpc_) / (145.0 * fpc_ - 1000.0);
        if (eps_50u < 1e-6) eps_50u = 1e-6;
        // Z = 0.5 / (ε_50u − |ε_0|);  since eps0_ < 0: |ε_0| = -eps0_
        double denom_z = eps_50u + eps0_;  // = eps_50u - |eps0_|
        if (denom_z < 1e-6) denom_z = 1e-6;
        Zslope_ = 0.5 / denom_z;

        fpc_residual_ = 0.2 * Kconf_ * fpc_;
        eps_u_ = eps0_ - (1.0 - 0.2) / Zslope_;  // where σ reaches 0.2·f'c
    }

    /// Confined concrete constructor.
    ///
    /// @param fpc       Peak compressive strength of UNCONFINED concrete (POSITIVE, MPa)
    /// @param ft        Tensile strength (POSITIVE)
    /// @param rho_s     Ratio of volume of transverse reinforcement to volume of core
    /// @param fyh       Yield stress of transverse reinforcement (MPa)
    /// @param h_prime   Width of core (center-to-center of ties, mm)
    /// @param sh        Spacing of ties (mm)
    KentParkConcrete(double fpc, double ft,
                     double rho_s, double fyh,
                     double h_prime, double sh)
        : fpc_{fpc},
          ft_{(ft > 0.0) ? ft : 0.1 * fpc},
          Kconf_{1.0 + rho_s * fyh / fpc}
    {
        eps0_ = -0.002 * Kconf_;
        Ec_ = 2.0 * Kconf_ * fpc_ / std::abs(eps0_);

        double eps_50u = (3.0 + 0.29 * fpc_) / (145.0 * fpc_ - 1000.0);
        if (eps_50u < 1e-6) eps_50u = 1e-6;
        double eps_50h = 0.75 * rho_s * std::sqrt(h_prime / sh);
        // Z = 0.5 / (ε_50u + ε_50h − |ε_0|);  since eps0_ < 0: +eps0_ = -|eps0_|
        double denom_z = eps_50u + eps_50h + eps0_;
        if (denom_z < 1e-6) denom_z = 1e-6;
        Zslope_ = 0.5 / denom_z;

        fpc_residual_ = 0.2 * Kconf_ * fpc_;
        eps_u_ = eps0_ - (1.0 - 0.2) / Zslope_;
    }

    constexpr KentParkConcrete() : fpc_{0}, eps0_{-0.002}, Ec_{0}, ft_{0},
                                   Zslope_{0}, Kconf_{1.0}, fpc_residual_{0},
                                   eps_u_{-0.01} {}
    ~KentParkConcrete() = default;

    KentParkConcrete(const KentParkConcrete&)               = default;
    KentParkConcrete(KentParkConcrete&&) noexcept           = default;
    KentParkConcrete& operator=(const KentParkConcrete&)    = default;
    KentParkConcrete& operator=(KentParkConcrete&&) noexcept = default;


    // =================================================================
    //  Diagnostics
    // =================================================================

    void print_constitutive_parameters() const {
        std::cout << "=== Kent-Park Concrete ===" << std::endl;
        std::cout << "f'c  = " << fpc_   << " MPa (positive)" << std::endl;
        std::cout << "ε_0  = " << eps0_  << std::endl;
        std::cout << "E_c  = " << Ec_    << " MPa" << std::endl;
        std::cout << "f_t  = " << ft_    << " MPa" << std::endl;
        std::cout << "K    = " << Kconf_ << "  (confinement)" << std::endl;
        std::cout << "Z    = " << Zslope_ << std::endl;
    }
};


// =============================================================================
//  Static concept verification
// =============================================================================

static_assert(
    ConstitutiveRelation<KentParkConcrete>,
    "KentParkConcrete must satisfy ConstitutiveRelation");

static_assert(
    InelasticConstitutiveRelation<KentParkConcrete>,
    "KentParkConcrete must satisfy InelasticConstitutiveRelation");


#endif // FN_KENT_PARK_CONCRETE_HH
