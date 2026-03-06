#ifndef FN_ISOTROPIC_HARDENING_HH
#define FN_ISOTROPIC_HARDENING_HH

#include "PlasticityConcepts.hh"

// =============================================================================
//  IsotropicHardeningState — Scalar state for isotropic hardening
// =============================================================================
//
//  Stores the single scalar internal variable required by isotropic hardening
//  rules: the equivalent (accumulated) plastic strain  ε̄^p ≥ 0.
//
//  The yield surface expands uniformly in all directions by an amount
//  proportional to ε̄^p.  This is the simplest (and most common) hardening
//  assumption for monotonic loading.
//
//  For kinematic hardening (Bauschinger effect, hysteretic loops), a richer
//  state type carrying a backstress tensor β will be used.  For mixed
//  hardening, both scalar and tensor components are combined.
//
// =============================================================================

struct IsotropicHardeningState {
    double equivalent_plastic_strain = 0.0;

    // Named accessor (backward-compat with J2InternalVariables)
    [[nodiscard]] double eps_bar_p() const noexcept {
        return equivalent_plastic_strain;
    }

    // Trivially copyable — no need for explicit Rule of Five.
};


// =============================================================================
//  LinearIsotropicHardening — σ_y(ε̄^p) = σ_y0 + H · ε̄^p
// =============================================================================
//
//  The simplest isotropic hardening law: linear relationship between the
//  yield stress and the equivalent plastic strain.
//
//  Parameters:
//    σ_y0  — initial yield stress  (intercept of backbone curve)
//    H     — hardening modulus     (slope of backbone curve)
//
//  Special cases:
//    H = 0   → perfect plasticity  (no hardening)
//    H < 0   → strain softening    (mesh-dependent, use with localization
//                                    regularization)
//
//  ─── Backbone curve connection ──────────────────────────────────────────
//
//  For hysteretic models, the monotonic backbone is defined by σ_y(ε̄^p).
//  This linear law produces a bilinear backbone.  Future extensions:
//
//    NonlinearIsotropicHardening  — Voce:  σ_y = σ_∞ − (σ_∞−σ_y0)·exp(−δ·ε̄^p)
//    PowerLawHardening           — σ_y = σ_y0 · (1 + ε̄^p/ε₀)^n
//    RambergOsgoodHardening      — implicit backbone ε = σ/E + (σ/K)^(1/n)
//    PiecewiseLinearHardening    — tabulated multi-linear backbone curve
//
//  These define the backbone surface that unloading/reloading rules (Masing,
//  non-Masing, etc.) will reference in cyclic / time-history analysis.
//
// =============================================================================

class LinearIsotropicHardening {
public:
    using StateT = IsotropicHardeningState;

private:
    double sigma_y0_{0.0};   // Initial yield stress
    double H_{0.0};          // Hardening modulus

public:
    // Current yield stress:  σ_y = σ_y0 + H · ε̄^p
    [[nodiscard]] double yield_stress(const StateT& state) const noexcept {
        return sigma_y0_ + H_ * state.equivalent_plastic_strain;
    }

    // Hardening modulus:  dσ_y/d(ε̄^p) = H  (constant for linear hardening)
    [[nodiscard]] double modulus([[maybe_unused]] const StateT& state) const noexcept {
        return H_;
    }

    // Evolve state:  ε̄^p_{n+1} = ε̄^p_n + √(2/3) · Δγ
    //
    // The factor √(2/3) converts from the consistency parameter Δγ (which
    // lives in the n̂-direction in 6D Voigt space) to the equivalent plastic
    // strain increment Δε̄^p = √(2/3 · Δε^p : Δε^p).
    [[nodiscard]] StateT evolve(const StateT& state, double delta_gamma) const {
        return StateT{state.equivalent_plastic_strain
                      + std::sqrt(2.0 / 3.0) * delta_gamma};
    }

    // ─── Parameter accessors ─────────────────────────────────────────────

    [[nodiscard]] double initial_yield_stress() const noexcept { return sigma_y0_; }
    [[nodiscard]] double hardening_modulus()     const noexcept { return H_; }

    // ─── Constructors ────────────────────────────────────────────────────

    constexpr LinearIsotropicHardening(double sigma_y0, double H)
        : sigma_y0_{sigma_y0}, H_{H} {}

    constexpr LinearIsotropicHardening() = default;
};


// ─── Concept verification ────────────────────────────────────────────────────

static_assert(HardeningLaw<LinearIsotropicHardening>,
              "LinearIsotropicHardening must satisfy HardeningLaw");


#endif // FN_ISOTROPIC_HARDENING_HH
