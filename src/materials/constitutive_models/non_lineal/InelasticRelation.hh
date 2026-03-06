#ifndef INELASTIC_RELATION_HH
#define INELASTIC_RELATION_HH

#include <cmath>
#include <cstddef>
#include <iostream>

#include "../../MaterialPolicy.hh"
#include "../../MaterialState.hh"
#include "../../ConstitutiveRelation.hh"
#include "../lineal/IsotropicRelation.hh"

// =============================================================================
//  J2InternalVariables<N>  — Internal state for J2 plasticity
// =============================================================================
//
//  Stores the two internal (history) variables required by the J2 (von Mises)
//  plasticity model with isotropic hardening:
//
//    ε^p            — plastic strain tensor (Voigt notation, N components)
//    ε_bar_p        — equivalent (accumulated) plastic strain (scalar ≥ 0)
//
//  This struct is value-semantic (copyable, default-constructible) so it can
//  be stored in MemoryState for history tracking.
//
// =============================================================================

template <std::size_t N>
struct J2InternalVariables {
    static constexpr std::size_t num_components = N;

    Eigen::Vector<double, N> plastic_strain = Eigen::Vector<double, N>::Zero();
    double equivalent_plastic_strain = 0.0;

    // Named accessors
    [[nodiscard]] const auto& eps_p()     const noexcept { return plastic_strain; }
    [[nodiscard]] double      eps_bar_p() const noexcept { return equivalent_plastic_strain; }

    // Rule of five — all defaulted
    J2InternalVariables()                                         = default;
    ~J2InternalVariables()                                        = default;
    J2InternalVariables(const J2InternalVariables&)               = default;
    J2InternalVariables(J2InternalVariables&&) noexcept           = default;
    J2InternalVariables& operator=(const J2InternalVariables&)    = default;
    J2InternalVariables& operator=(J2InternalVariables&&) noexcept = default;
};


// =============================================================================
//  J2PlasticityRelation<MaterialPolicy>
// =============================================================================
//
//  Isotropic J2 (von Mises) plasticity with linear isotropic hardening.
//
//  Satisfies InelasticConstitutiveRelation:
//    - compute_response(ε) → σ         [const — trial stress from current α]
//    - tangent(ε)          → C_ep      [const — algorithmic consistent tangent]
//    - update(ε)                       [non-const — return-mapping, commits α]
//    - internal_state()    → const α&  [const — read current internal variables]
//
//  ─── Algorithm (Backward-Euler return mapping) ──────────────────────────
//
//  Given total strain ε_{n+1} and committed internal variables α_n:
//
//    1. Elastic trial:     σ_trial = C_e : (ε_{n+1} - ε^p_n)
//    2. Deviatoric part:   s_trial = dev(σ_trial)
//    3. Trial yield:       f_trial = √(3/2) ‖s_trial‖ - σ_y(ε̄^p_n)
//
//    If f_trial ≤ 0:  elastic step → σ = σ_trial, C_ep = C_e
//
//    If f_trial > 0:  plastic correction:
//      Δγ = f_trial / (3G + H)
//      n̂  = s_trial / ‖s_trial‖
//      σ  = σ_trial - 2G·Δγ·n̂
//      ε^p_{n+1}  = ε^p_n  + Δγ·n̂
//      ε̄^p_{n+1} = ε̄^p_n + √(2/3)·Δγ
//
//  ─── Consistent tangent ─────────────────────────────────────────────────
//
//    C_ep = C_e − (2G)²·Δγ/‖s_trial‖ · P_dev
//           + (2G)²·(Δγ/‖s_trial‖ − 1/(3G+H)) · (n̂ ⊗ n̂)
//
//    where P_dev = I_sym − (1/3) I⊗I  is the deviatoric projector.
//
//  ─── Memory model ──────────────────────────────────────────────────────
//
//    The relation OWNS the current internal variables.  This is intentionally
//    different from ElasticRelation (which is stateless/shared).  Each
//    MaterialInstance that wraps a J2PlasticityRelation must NOT share the
//    relation pointer when internal variables differ between points.
//
//    For shared parameters (E, ν, σ_y0, H) without shared state, construct
//    each MaterialInstance with its own relation copy:
//      MaterialInstance<J2PlasticityRelation<3D>, MemoryState>{E, nu, sy, H};
//
// =============================================================================

template <class MaterialPolicy>
class J2PlasticityRelation {

    static_assert(MaterialPolicy::StrainT::num_components == 6 ||
                  MaterialPolicy::StrainT::num_components == 3 ||
                  MaterialPolicy::StrainT::num_components == 1,
                  "J2PlasticityRelation requires N = 1, 3, or 6 components");

public:
    // --- Policy alias --------------------------------------------------------
    using MaterialPolicyT = MaterialPolicy;

    // --- Concept-required type aliases (ConstitutiveRelation Level 1) ---------
    using KinematicT = typename MaterialPolicy::StrainT;
    using ConjugateT = typename MaterialPolicy::StressT;
    using TangentT   = TangentMatrix<KinematicT, ConjugateT>;

    // --- InelasticConstitutiveRelation Level 2b ─────────────────────────────
    using InternalVariablesT = J2InternalVariables<KinematicT::num_components>;

    static constexpr std::size_t N   = KinematicT::num_components;
    static constexpr std::size_t dim = KinematicT::dim;

private:
    // Material parameters (constant after construction)
    double E_{0.0};        // Young's modulus
    double nu_{0.0};       // Poisson's ratio
    double sigma_y0_{0.0}; // Initial yield stress
    double H_{0.0};        // Isotropic hardening modulus  (σ_y = σ_y0 + H·ε̄^p)

    // Derived elastic constants (computed once)
    double G_{0.0};        // Shear modulus     = E / (2(1+ν))
    double K_{0.0};        // Bulk modulus      = E / (3(1−2ν))
    double lambda_{0.0};   // Lamé parameter    = νE / ((1+ν)(1−2ν))

    // Elastic stiffness matrix (constant)
    TangentT Ce_ = TangentT::Zero();

    // Current internal variables (owned per-instance)
    InternalVariablesT alpha_{};

    // Cached results from the last compute/update cycle
    // (avoids redundant return-mapping when tangent(k) is called right after
    //  compute_response(k) with the same k — typical in Newton-Raphson)
    mutable ConjugateT last_stress_{};
    mutable TangentT   last_tangent_{};
    mutable bool        cache_valid_{false};
    mutable Eigen::Vector<double, N> last_strain_cache_{Eigen::Vector<double, N>::Zero()};

    // ─── Helper: 4th-order identity and deviatoric projector in Voigt ────

    // Symmetric 4th-order identity  I_sym  (Voigt)
    [[nodiscard]] static TangentT I_sym() {
        TangentT I = TangentT::Zero();
        if constexpr (N == 6) {
            // Diagonal: {1,1,1, 0.5, 0.5, 0.5}
            for (int i = 0; i < 3; ++i) I(i, i) = 1.0;
            for (int i = 3; i < 6; ++i) I(i, i) = 0.5;
        } else if constexpr (N == 3) {
            I(0, 0) = 1.0; I(1, 1) = 1.0; I(2, 2) = 0.5;
        } else if constexpr (N == 1) {
            I(0, 0) = 1.0;
        }
        return I;
    }

    // I ⊗ I  in Voigt notation (outer product of identity vectors)
    [[nodiscard]] static TangentT I_otimes_I() {
        TangentT IxI = TangentT::Zero();
        if constexpr (N == 6) {
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    IxI(i, j) = 1.0;
        } else if constexpr (N == 3) {
            IxI(0, 0) = 1.0; IxI(0, 1) = 1.0;
            IxI(1, 0) = 1.0; IxI(1, 1) = 1.0;
        } else if constexpr (N == 1) {
            IxI(0, 0) = 1.0;
        }
        return IxI;
    }

    // Deviatoric projector  P_dev = I_sym - (1/3) I⊗I
    [[nodiscard]] static TangentT P_dev() {
        return I_sym() - (1.0 / 3.0) * I_otimes_I();
    }

    // ─── Helper: build isotropic elastic stiffness matrix ────────────────

    void build_elastic_tangent() {
        Ce_ = 2.0 * G_ * I_sym() + lambda_ * I_otimes_I();
    }

    // ─── Helper: deviatoric stress from Voigt stress vector ──────────────

    [[nodiscard]] static Eigen::Vector<double, N> deviatoric(
        const Eigen::Vector<double, N>& sigma)
    {
        Eigen::Vector<double, N> s = sigma;
        if constexpr (N == 6) {
            double p = (sigma[0] + sigma[1] + sigma[2]) / 3.0;
            s[0] -= p; s[1] -= p; s[2] -= p;
            // shear components (indices 3,4,5) are already deviatoric
        } else if constexpr (N == 3) {
            double p = (sigma[0] + sigma[1]) / 3.0; // plane stress/strain simplification
            s[0] -= p; s[1] -= p;
        } else if constexpr (N == 1) {
            // uniaxial: dev(σ) = (2/3)σ, but for J2 in 1D the
            // yield function simplifies to |σ| - σ_y. We keep σ as-is
            // and handle the factor in the yield function.
        }
        return s;
    }

    // ─── Helper: J2 norm  √(3/2) ‖s‖  (accounting for Voigt factors) ────
    //
    //  In Voigt notation with engineering shear:
    //    ‖s‖² = s₁² + s₂² + s₃² + 2(s₄² + s₅² + s₆²)
    //  The factor √(3/2) converts to the von Mises equivalent stress:
    //    σ_eq = √(3/2) · ‖s‖ = √(3·J₂)

    [[nodiscard]] static double von_mises_norm(const Eigen::Vector<double, N>& s) {
        if constexpr (N == 6) {
            double n2 = s[0]*s[0] + s[1]*s[1] + s[2]*s[2]
                      + 2.0 * (s[3]*s[3] + s[4]*s[4] + s[5]*s[5]);
            return std::sqrt(1.5 * n2);
        } else if constexpr (N == 3) {
            double n2 = s[0]*s[0] + s[1]*s[1] + 2.0 * s[2]*s[2];
            return std::sqrt(1.5 * n2);
        } else if constexpr (N == 1) {
            return std::abs(s[0]);  // σ_eq = |σ| for uniaxial
        }
    }

    // ─── Core: return-mapping algorithm ──────────────────────────────────

    struct ReturnMapResult {
        Eigen::Vector<double, N> stress;
        TangentT tangent;
        Eigen::Vector<double, N> eps_p_new;
        double eps_bar_p_new;
        bool plastic;
    };

    [[nodiscard]] ReturnMapResult return_mapping(
        const Eigen::Vector<double, N>& total_strain) const
    {
        ReturnMapResult res;

        // 1. Elastic trial
        Eigen::Vector<double, N> elastic_strain = total_strain - alpha_.plastic_strain;
        Eigen::Vector<double, N> sigma_trial = Ce_ * elastic_strain;

        // 2. Deviatoric trial stress
        Eigen::Vector<double, N> s_trial = deviatoric(sigma_trial);

        // 3. Von Mises equivalent stress
        double q_trial = von_mises_norm(s_trial);

        // 4. Current yield stress
        double sigma_y = sigma_y0_ + H_ * alpha_.equivalent_plastic_strain;

        // 5. Yield function
        double f_trial = q_trial - sigma_y;

        if (f_trial <= 0.0) {
            // ── Elastic step ─────────────────────────────────────────
            res.stress       = sigma_trial;
            res.tangent      = Ce_;
            res.eps_p_new    = alpha_.plastic_strain;
            res.eps_bar_p_new = alpha_.equivalent_plastic_strain;
            res.plastic      = false;
            return res;
        }

        // ── Plastic correction ───────────────────────────────────────

        // s_trial norm (Voigt)
        double s_norm;
        if constexpr (N == 6) {
            s_norm = std::sqrt(s_trial[0]*s_trial[0] + s_trial[1]*s_trial[1]
                             + s_trial[2]*s_trial[2]
                             + 2.0*(s_trial[3]*s_trial[3] + s_trial[4]*s_trial[4]
                                  + s_trial[5]*s_trial[5]));
        } else if constexpr (N == 3) {
            s_norm = std::sqrt(s_trial[0]*s_trial[0] + s_trial[1]*s_trial[1]
                             + 2.0*s_trial[2]*s_trial[2]);
        } else {
            s_norm = std::abs(s_trial[0]);
        }

        // Consistency parameter
        double denom = 3.0 * G_ + H_;
        double delta_gamma = f_trial / denom;

        // Flow direction  n̂ = s_trial / ‖s_trial‖
        Eigen::Vector<double, N> n_hat;
        if (s_norm > 1e-30) {
            n_hat = s_trial / s_norm;
        } else {
            n_hat = Eigen::Vector<double, N>::Zero();
        }

        // Corrected stress
        //   σ = σ_trial - 2G·Δγ·n̂
        res.stress = sigma_trial;
        for (std::size_t i = 0; i < N; ++i) {
            res.stress[i] -= 2.0 * G_ * delta_gamma * n_hat[i];
        }

        // Updated internal variables
        res.eps_p_new     = alpha_.plastic_strain + delta_gamma * n_hat;
        res.eps_bar_p_new = alpha_.equivalent_plastic_strain
                          + std::sqrt(2.0 / 3.0) * delta_gamma;
        res.plastic       = true;

        // ── Algorithmic consistent tangent ────────────────────────────
        //
        //  C_ep = C_e
        //       − (2G)² · (Δγ / ‖s_trial‖) · P_dev
        //       + (2G)² · (Δγ/‖s_trial‖ − 1/(3G+H)) · (n̂ ⊗ n̂)
        //
        // Equivalently, using  θ = 1 − 2G·Δγ/‖s_trial‖  and  θ̄ = 1/(1 + H/(3G)) − (1−θ):
        //
        //  C_ep = 2G·θ·P_dev + K·(I⊗I) − 2G·θ̄·(n̂⊗n̂)

        double two_G       = 2.0 * G_;
        double ratio       = (s_norm > 1e-30) ? (delta_gamma / s_norm) : 0.0;
        double factor_Pdev = two_G * two_G * ratio;
        double factor_nn   = two_G * two_G * (ratio - 1.0 / denom);

        TangentT nn = TangentT::Zero();
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < N; ++j)
                nn(i, j) = n_hat[i] * n_hat[j];

        res.tangent = Ce_ - factor_Pdev * P_dev() + factor_nn * nn;

        return res;
    }

public:
    // ─── ConstitutiveRelation interface (Level 1) — const ────────────────

    // Compute trial (or corrected) stress given total strain.
    // This is a CONST operation: it reads α_n but does NOT commit updates.
    // In a Newton-Raphson iteration, this is called to evaluate the residual.
    [[nodiscard]] ConjugateT compute_response(const KinematicT& strain) const {
        auto result = return_mapping(strain.components());

        // Cache for subsequent tangent(k) call
        last_stress_ = ConjugateT{};
        last_stress_.set_components(result.stress);
        last_tangent_       = result.tangent;
        last_strain_cache_  = strain.components();
        cache_valid_        = true;

        return last_stress_;
    }

    // Algorithmic consistent tangent ∂σ/∂ε at the given strain.
    // If compute_response(k) was just called with the same k, uses cache.
    [[nodiscard]] TangentT tangent(const KinematicT& strain) const {
        if (cache_valid_ && strain.components().isApprox(last_strain_cache_, 1e-15)) {
            return last_tangent_;
        }
        auto result = return_mapping(strain.components());
        last_tangent_      = result.tangent;
        last_strain_cache_ = strain.components();
        cache_valid_       = true;
        return last_tangent_;
    }

    // ─── InelasticConstitutiveRelation interface (Level 2b) ──────────────

    // Commit the return-mapping result: update internal variables α.
    // Called AFTER convergence of the global Newton-Raphson iteration.
    void update(const KinematicT& strain) {
        auto result = return_mapping(strain.components());
        alpha_.plastic_strain             = result.eps_p_new;
        alpha_.equivalent_plastic_strain  = result.eps_bar_p_new;
        cache_valid_ = false;  // invalidate after commit
    }

    // Read-only access to the current internal variables.
    [[nodiscard]] const InternalVariablesT& internal_state() const {
        return alpha_;
    }

    // ─── Parameter accessors ─────────────────────────────────────────────

    [[nodiscard]] double young_modulus()           const noexcept { return E_; }
    [[nodiscard]] double poisson_ratio()           const noexcept { return nu_; }
    [[nodiscard]] double initial_yield_stress()    const noexcept { return sigma_y0_; }
    [[nodiscard]] double hardening_modulus()       const noexcept { return H_; }
    [[nodiscard]] double shear_modulus()           const noexcept { return G_; }
    [[nodiscard]] double bulk_modulus()            const noexcept { return K_; }
    [[nodiscard]] const TangentT& elastic_tangent() const noexcept { return Ce_; }

    [[nodiscard]] double current_yield_stress() const noexcept {
        return sigma_y0_ + H_ * alpha_.equivalent_plastic_strain;
    }

    // ─── Constructors ────────────────────────────────────────────────────

    constexpr J2PlasticityRelation(
        double E, double nu, double sigma_y0, double H)
        : E_{E}, nu_{nu}, sigma_y0_{sigma_y0}, H_{H},
          G_{E / (2.0 * (1.0 + nu))},
          K_{E / (3.0 * (1.0 - 2.0 * nu))},
          lambda_{nu * E / ((1.0 + nu) * (1.0 - 2.0 * nu))}
    {
        build_elastic_tangent();
    }

    constexpr J2PlasticityRelation() = default;
    ~J2PlasticityRelation() = default;

    // Internal state is per-point, so copy/move are meaningful
    J2PlasticityRelation(const J2PlasticityRelation&)               = default;
    J2PlasticityRelation(J2PlasticityRelation&&) noexcept           = default;
    J2PlasticityRelation& operator=(const J2PlasticityRelation&)    = default;
    J2PlasticityRelation& operator=(J2PlasticityRelation&&) noexcept = default;

    // ─── Diagnostics ─────────────────────────────────────────────────────

    void print_constitutive_parameters() const {
        std::cout << "=== J2 Plasticity ===" << std::endl;
        std::cout << "E  = " << E_  << ",  ν = " << nu_ << std::endl;
        std::cout << "σ_y0 = " << sigma_y0_ << ",  H = " << H_ << std::endl;
        std::cout << "G  = " << G_  << ",  K = " << K_ << std::endl;
        std::cout << "ε̄^p = " << alpha_.equivalent_plastic_strain << std::endl;
        std::cout << "σ_y(current) = " << current_yield_stress() << std::endl;
    }
};


// =============================================================================
//  Static concept verification — 3D J2 Plasticity
// =============================================================================

static_assert(
    ConstitutiveRelation<J2PlasticityRelation<ThreeDimensionalMaterial>>,
    "J2PlasticityRelation<3D> must satisfy ConstitutiveRelation");

static_assert(
    InelasticConstitutiveRelation<J2PlasticityRelation<ThreeDimensionalMaterial>>,
    "J2PlasticityRelation<3D> must satisfy InelasticConstitutiveRelation");


#endif // INELASTIC_RELATION_HH