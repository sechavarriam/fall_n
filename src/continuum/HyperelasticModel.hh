#ifndef FALL_N_HYPERELASTIC_MODEL_HH
#define FALL_N_HYPERELASTIC_MODEL_HH

// =============================================================================
//  HyperelasticModel.hh — Hyperelastic stored-energy models
// =============================================================================
//
//  A hyperelastic material is defined by a stored-energy density W(E), where
//  E is the Green-Lagrange strain.  The stress and tangent follow uniquely:
//
//      S     = ∂W/∂E            2nd Piola-Kirchhoff stress
//      ℂ     = ∂²W/∂E²         Material tangent (4th order)
//
//  Each model is a lightweight value-semantic class parametrized by the
//  spatial dimension dim.  It stores only the material constants (λ, μ)
//  and provides three operations:
//
//      energy(E)                → double       (W)
//      second_piola_kirchhoff(E)→ SymTensor2   (S)
//      material_tangent(E)      → Tensor4      (ℂ, in mixed Voigt form)
//
//  All methods take the Green-Lagrange strain E as a SymmetricTensor2<dim>
//  in *tensor* Voigt form (no factor of 2 on shears).
//
//  ─── Tangent Voigt convention ───
//
//  The Tensor4 returned by material_tangent() stores the *mixed* tangent:
//
//      S_tensor = ℂ_mixed · ε_engineering
//
//  where ε_engineering is the engineering Voigt strain (γ = 2ε on shears).
//  Numerically, ℂ_mixed(I,J) = ℂ_{i(I)j(I)k(J)l(J)} — the raw 4th-order
//  tensor component at the Voigt index pair, with NO additional factors.
//
//  This convention is identical to Tensor4::isotropic_lame() and is
//  consistent with the existing ContinuumElement assembly:
//      K_mat = ∫ Bᵀ_eng · ℂ_mixed · B_eng  dV₀
//
//  ─── Available models ───
//
//    SaintVenantKirchhoff<dim>     — Linear in E, constant tangent.
//                                    W = λ/2 (tr E)² + μ (E : E)
//
//    CompressibleNeoHookean<dim>   — Logarithmic-volumetric variant.
//                                    W = μ/2(tr C − dim) − μ ln J + λ/2(ln J)²
//
// =============================================================================

#include <cstddef>
#include <cmath>
#include <concepts>

#include "SymmetricTensor2.hh"
#include "Tensor4.hh"
#include "TensorOperations.hh"

namespace continuum {


// =============================================================================
//  HyperelasticModelConcept
// =============================================================================
//
//  Minimal compile-time interface that a hyperelastic model must provide.
//  Used to constrain the HyperelasticRelation template parameter.

template <typename M>
concept HyperelasticModelConcept =
    ValidDim<M::dimension> &&
    requires(const M& m, const SymmetricTensor2<M::dimension>& E) {
        { m.energy(E)                   } -> std::convertible_to<double>;
        { m.second_piola_kirchhoff(E)   } -> std::same_as<SymmetricTensor2<M::dimension>>;
        { m.material_tangent(E)         } -> std::same_as<Tensor4<M::dimension>>;
    };


// =============================================================================
//  SaintVenantKirchhoff<dim>
// =============================================================================
//
//  The simplest hyperelastic model — a direct generalisation of Hooke's law
//  to the finite-strain setting using the Green-Lagrange strain E.
//
//      W(E) = λ/2 (tr E)² + μ (E : E)
//      S    = λ tr(E) I + 2μ E
//      ℂ    = λ (I ⊗ I) + 2μ 𝕀ˢʸᵐ    (constant, = isotropic Lamé tangent)
//
//  Properties:
//    ✓ Frame-indifferent (objective)
//    ✓ Hyperelastic (path-independent, derives from W)
//    ✓ Constant tangent in E, S variables
//    ✗ NOT polyconvex — does not resist compression (W→−∞ as J→0)
//    ✗ Should only be used for moderate strains
//
//  For small strains, SVK reduces identically to isotropic linear elasticity.
//
// -----------------------------------------------------------------------------

template <std::size_t dim>
    requires ValidDim<dim>
class SaintVenantKirchhoff {

    double lambda_;   ///< 1st Lamé parameter
    double mu_;       ///< 2nd Lamé parameter (shear modulus)

public:
    static constexpr std::size_t dimension       = dim;
    static constexpr bool        constant_tangent = true;

    // ── Constructors ─────────────────────────────────────────────────────────

    /// Construct from Lamé parameters (λ, μ).
    constexpr SaintVenantKirchhoff(double lambda, double mu) noexcept
        : lambda_{lambda}, mu_{mu} {}

    /// Named constructor from Young's modulus E and Poisson's ratio ν.
    static constexpr SaintVenantKirchhoff from_E_nu(double E, double nu) noexcept {
        double lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        double mu  = E / (2.0 * (1.0 + nu));
        return SaintVenantKirchhoff{lam, mu};
    }

    constexpr SaintVenantKirchhoff() noexcept = default;

    // ── Stored-energy density ────────────────────────────────────────────────
    //
    //  W = λ/2 (tr E)² + μ (E : E)

    [[nodiscard]] double energy(const SymmetricTensor2<dim>& E) const noexcept {
        const double trE = E.trace();
        const double E_E = E.double_contract(E);
        return 0.5 * lambda_ * trE * trE + mu_ * E_E;
    }

    // ── 2nd Piola-Kirchhoff stress ───────────────────────────────────────────
    //
    //  S = ∂W/∂E = λ tr(E) I + 2μ E

    [[nodiscard]] SymmetricTensor2<dim>
    second_piola_kirchhoff(const SymmetricTensor2<dim>& E) const noexcept {
        return SymmetricTensor2<dim>::identity() * (lambda_ * E.trace())
             + E * (2.0 * mu_);
    }

    // ── Material tangent ─────────────────────────────────────────────────────
    //
    //  ℂ = ∂S/∂E = λ (I⊗I) + 2μ 𝕀ˢʸᵐ
    //
    //  Constant (independent of E).  In mixed Voigt form this is identical
    //  to Tensor4::isotropic_lame(λ, μ).

    [[nodiscard]] Tensor4<dim>
    material_tangent([[maybe_unused]] const SymmetricTensor2<dim>& E) const noexcept {
        return Tensor4<dim>::isotropic_lame(lambda_, mu_);
    }

    /// No-argument overload (constant tangent).
    [[nodiscard]] Tensor4<dim> material_tangent() const noexcept {
        return Tensor4<dim>::isotropic_lame(lambda_, mu_);
    }

    // ── Parameter access ─────────────────────────────────────────────────────

    [[nodiscard]] constexpr double lambda()        const noexcept { return lambda_; }
    [[nodiscard]] constexpr double mu()            const noexcept { return mu_; }
    [[nodiscard]] constexpr double shear_modulus()  const noexcept { return mu_; }

    [[nodiscard]] constexpr double young_modulus() const noexcept {
        return mu_ * (3.0 * lambda_ + 2.0 * mu_) / (lambda_ + mu_);
    }

    [[nodiscard]] constexpr double poisson_ratio() const noexcept {
        return lambda_ / (2.0 * (lambda_ + mu_));
    }
};


// =============================================================================
//  CompressibleNeoHookean<dim>
// =============================================================================
//
//  A widely-used hyperelastic model for rubber-like materials and general
//  large-deformation analysis.  This is the Simo-Ciarlet logarithmic-
//  volumetric variant:
//
//      W(C) = μ/2 (tr C − dim) − μ ln J + λ/2 (ln J)²
//
//  where  C = 2E + I  (right Cauchy-Green)  and  J = √(det C).
//
//  Stresses and tangent:
//
//      S = μ(I − C⁻¹) + λ ln(J) C⁻¹
//
//      ℂ_{ijkl} = λ C⁻¹_{ij} C⁻¹_{kl}
//               + (μ − λ ln J)(C⁻¹_{ik} C⁻¹_{jl} + C⁻¹_{il} C⁻¹_{jk})
//
//  Properties:
//    ✓ Frame-indifferent (objective)
//    ✓ Hyperelastic (path-independent)
//    ✓ Polyconvex for μ > 0, λ > 0
//    ✓ W → +∞ as J → 0⁺ (resists total compression)
//    ✓ At E = 0 reduces to isotropic linear elasticity
//
// -----------------------------------------------------------------------------

template <std::size_t dim>
    requires ValidDim<dim>
class CompressibleNeoHookean {

    double lambda_;
    double mu_;

public:
    static constexpr std::size_t dimension       = dim;
    static constexpr bool        constant_tangent = false;

    // ── Constructors ─────────────────────────────────────────────────────────

    /// Construct from Lamé parameters (λ, μ).
    constexpr CompressibleNeoHookean(double lambda, double mu) noexcept
        : lambda_{lambda}, mu_{mu} {}

    /// Named constructor from Young's modulus E and Poisson's ratio ν.
    static constexpr CompressibleNeoHookean from_E_nu(double E, double nu) noexcept {
        double lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        double mu  = E / (2.0 * (1.0 + nu));
        return CompressibleNeoHookean{lam, mu};
    }

    constexpr CompressibleNeoHookean() noexcept = default;

    // ── Stored-energy density ────────────────────────────────────────────────
    //
    //  W = μ/2 (tr C − dim) − μ ln J + λ/2 (ln J)²

    [[nodiscard]] double energy(const SymmetricTensor2<dim>& E) const noexcept {
        auto C     = E * 2.0 + SymmetricTensor2<dim>::identity();
        double I1  = C.trace();
        double J   = std::sqrt(C.determinant());
        double lnJ = std::log(J);

        return 0.5 * mu_ * (I1 - static_cast<double>(dim))
             - mu_ * lnJ
             + 0.5 * lambda_ * lnJ * lnJ;
    }

    // ── 2nd Piola-Kirchhoff stress ───────────────────────────────────────────
    //
    //  S = μ(I − C⁻¹) + λ ln(J) C⁻¹

    [[nodiscard]] SymmetricTensor2<dim>
    second_piola_kirchhoff(const SymmetricTensor2<dim>& E) const noexcept {
        auto I     = SymmetricTensor2<dim>::identity();
        auto C     = E * 2.0 + I;
        auto C_inv = C.inverse();
        double J   = std::sqrt(C.determinant());
        double lnJ = std::log(J);

        // S = μ(I − C⁻¹) + λ ln(J) C⁻¹
        return (I - C_inv) * mu_ + C_inv * (lambda_ * lnJ);
    }

    // ── Material tangent ─────────────────────────────────────────────────────
    //
    //  ℂ_{ijkl} = λ C⁻¹_{ij} C⁻¹_{kl}
    //           + α (C⁻¹_{ik} C⁻¹_{jl} + C⁻¹_{il} C⁻¹_{jk})
    //
    //  where α = μ − λ ln J.
    //
    //  Returns the mixed Voigt tangent: ℂ_mixed(A,B) = ℂ_{i(A)j(A)k(B)l(B)}.
    //  No additional factors — the cancellation between the tensor-tensor
    //  factor of 2 on shear columns and the engineering-strain factor of 2
    //  produces the raw tensor component at each Voigt index pair.

    [[nodiscard]] Tensor4<dim>
    material_tangent(const SymmetricTensor2<dim>& E) const noexcept {
        auto I     = SymmetricTensor2<dim>::identity();
        auto C     = E * 2.0 + I;
        auto C_inv = C.inverse();
        double J   = std::sqrt(C.determinant());
        double lnJ = std::log(J);
        double alpha = mu_ - lambda_ * lnJ;

        constexpr std::size_t N = voigt_size<dim>();
        using VoigtMatT = Eigen::Matrix<double, static_cast<int>(N),
                                                 static_cast<int>(N)>;
        VoigtMatT CC;

        for (std::size_t A = 0; A < N; ++A) {
            auto [i, j] = SymmetricTensor2<dim>::tensor_indices(A);
            for (std::size_t B = 0; B < N; ++B) {
                auto [k, l] = SymmetricTensor2<dim>::tensor_indices(B);

                CC(static_cast<Eigen::Index>(A),
                   static_cast<Eigen::Index>(B))
                    = lambda_ * C_inv(i, j) * C_inv(k, l)
                    + alpha   * (C_inv(i, k) * C_inv(j, l)
                               + C_inv(i, l) * C_inv(j, k));
            }
        }

        return Tensor4<dim>{CC};
    }

    // ── Parameter access ─────────────────────────────────────────────────────

    [[nodiscard]] constexpr double lambda()        const noexcept { return lambda_; }
    [[nodiscard]] constexpr double mu()            const noexcept { return mu_; }
    [[nodiscard]] constexpr double shear_modulus()  const noexcept { return mu_; }

    [[nodiscard]] constexpr double young_modulus() const noexcept {
        return mu_ * (3.0 * lambda_ + 2.0 * mu_) / (lambda_ + mu_);
    }

    [[nodiscard]] constexpr double poisson_ratio() const noexcept {
        return lambda_ / (2.0 * (lambda_ + mu_));
    }
};


// =============================================================================
//  Static concept verification
// =============================================================================

static_assert(HyperelasticModelConcept<SaintVenantKirchhoff<1>>);
static_assert(HyperelasticModelConcept<SaintVenantKirchhoff<2>>);
static_assert(HyperelasticModelConcept<SaintVenantKirchhoff<3>>);

static_assert(HyperelasticModelConcept<CompressibleNeoHookean<1>>);
static_assert(HyperelasticModelConcept<CompressibleNeoHookean<2>>);
static_assert(HyperelasticModelConcept<CompressibleNeoHookean<3>>);


} // namespace continuum

#endif // FALL_N_HYPERELASTIC_MODEL_HH
