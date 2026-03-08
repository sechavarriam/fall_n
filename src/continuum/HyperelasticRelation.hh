#ifndef FALL_N_HYPERELASTIC_RELATION_HH
#define FALL_N_HYPERELASTIC_RELATION_HH

// =============================================================================
//  HyperelasticRelation.hh — ConstitutiveRelation adapter for hyperelastic
//                             models
// =============================================================================
//
//  Bridges the continuum-tensor API of HyperelasticModel (SymmetricTensor2,
//  Tensor4) to the material infrastructure (Strain<N>, Stress<N>,
//  TangentMatrix, ConstitutiveRelation concept).
//
//  ─── Usage ───
//
//    // Direct construction from Lamé parameters
//    continuum::SVKRelation<3> svk_rel{lambda, mu};
//
//    // Compute stress from engineering Voigt strain
//    Strain<6> eps;  eps.set_strain(gp.strain_voigt);
//    Stress<6> S = svk_rel.compute_response(eps);
//    auto C      = svk_rel.tangent(eps);
//
//    // Wrap in MaterialInstance for per-point state management
//    MaterialInstance<continuum::SVKRelation<3>> mat{lambda, mu};
//
//  ─── Voigt conventions ───
//
//  The adapter performs the following conversions:
//
//    Input:   Strain<N> in engineering Voigt (γ = 2ε on shears)
//             ─→ SymmetricTensor2<dim> in tensor Voigt (divide shears by 2)
//
//    Output:  SymmetricTensor2<dim> S (tensor Voigt, no factor on shears)
//             ─→ Stress<N> (directly, same convention)
//
//    Tangent: Tensor4<dim> in mixed Voigt (raw ℂ_{ijkl} at Voigt indices)
//             ─→ TangentMatrix<Strain<N>, Stress<N>>  (= Eigen NxN matrix)
//             This mixed tangent maps engineering strain to tensor stress,
//             consistent with the existing assembly:  K = ∫ Bᵀ C B dV₀
//
//  ─── Concept conformance ───
//
//    ConstitutiveRelation:       always satisfied
//    ElasticConstitutiveRelation: satisfied IFF Model::constant_tangent = true
//                                 (SVK → yes, Neo-Hookean → no)
//
// =============================================================================

#include <cstddef>
#include <concepts>
#include <memory>

#include "../materials/MaterialPolicy.hh"
#include "../materials/Strain.hh"
#include "../materials/Stress.hh"
#include "../materials/ConstitutiveRelation.hh"

#include "HyperelasticModel.hh"

namespace continuum {


// =============================================================================
//  Voigt conversion utilities
// =============================================================================

/// Convert Strain<N> (engineering Voigt) → SymmetricTensor2<dim> (tensor Voigt).
///
/// Shear components are divided by 2: γ_ij → ε_ij = γ_ij / 2.
/// Normal components are unchanged.
template <std::size_t dim>
    requires ValidDim<dim>
SymmetricTensor2<dim> strain_eng_to_tensor(
        const Strain<voigt_size<dim>()>& strain) noexcept {
    constexpr std::size_t N = voigt_size<dim>();
    Eigen::Matrix<double, static_cast<int>(N), 1> v;

    for (std::size_t k = 0; k < N; ++k) {
        v(static_cast<Eigen::Index>(k)) = strain[k];
        if (k >= dim)
            v(static_cast<Eigen::Index>(k)) *= 0.5;   // engineering → tensor
    }
    return SymmetricTensor2<dim>{v};
}

/// Convert SymmetricTensor2<dim> (tensor Voigt) → Stress<N>.
///
/// Stress uses tensor Voigt convention (no factor 2 on shears) — same
/// as SymmetricTensor2::voigt().  A direct component copy suffices.
template <std::size_t dim>
    requires ValidDim<dim>
Stress<voigt_size<dim>()> tensor_to_stress(
        const SymmetricTensor2<dim>& S) noexcept {
    Stress<voigt_size<dim>()> stress;
    for (std::size_t k = 0; k < voigt_size<dim>(); ++k)
        stress[k] = S[k];   // voigt_[k] → component
    return stress;
}


// =============================================================================
//  HyperelasticRelation<Model>
// =============================================================================

template <HyperelasticModelConcept Model>
class HyperelasticRelation {

public:
    // ── Constants ────────────────────────────────────────────────────────────

    static constexpr std::size_t dim = Model::dimension;
    static constexpr std::size_t N   = voigt_size<dim>();

    // ── Type aliases required by ConstitutiveRelation concept ─────────────────

    using MaterialPolicyT = SolidMaterial<N>;
    using MaterialPolicy  = MaterialPolicyT;    // alias for Material<> type-erasure (OwningMaterialModel)
    using KinematicT      = Strain<N>;
    using ConjugateT      = Stress<N>;
    using TangentT        = TangentMatrix<KinematicT, ConjugateT>;

    // Legacy aliases for backward compatibility
    using StrainT        = KinematicT;
    using StressT        = ConjugateT;
    using StateVariableT = KinematicT;

    // ── ConstitutiveRelation interface (Level 1) ─────────────────────────────

    /// Compute the 2nd Piola-Kirchhoff stress from engineering Voigt strain.
    ///
    /// The input strain is in engineering Voigt form (γ = 2ε on shears),
    /// as produced by KinematicPolicy::evaluate().  Internally converts
    /// to tensor Voigt, evaluates S = ∂W/∂E, and converts back.
    [[nodiscard]] ConjugateT
    compute_response(const KinematicT& strain) const {
        auto E = strain_eng_to_tensor<dim>(strain);
        auto S = model_.second_piola_kirchhoff(E);
        return tensor_to_stress<dim>(S);
    }

    /// Material tangent evaluated at the given engineering Voigt strain.
    ///
    /// Returns the mixed Voigt tangent (N × N matrix) that maps engineering
    /// strain to tensor stress:  S_tensor = ℂ_mixed · ε_engineering.
    [[nodiscard]] TangentT
    tangent(const KinematicT& strain) const {
        auto E = strain_eng_to_tensor<dim>(strain);
        return model_.material_tangent(E).voigt_matrix();
    }

    // ── ElasticConstitutiveRelation interface (Level 2a) ─────────────────────
    //
    //  Available only when the model has a constant tangent (e.g. SVK).
    //  This enables MaterialInstance<SVKRelation<3>>::C() and tangent().

    [[nodiscard]] TangentT tangent() const
        requires (Model::constant_tangent)
    {
        return model_.material_tangent().voigt_matrix();
    }

    // ── Energy evaluation ────────────────────────────────────────────────────

    /// Stored-energy density from engineering Voigt strain.
    [[nodiscard]] double energy(const KinematicT& strain) const {
        auto E = strain_eng_to_tensor<dim>(strain);
        return model_.energy(E);
    }

    // ── Model access ─────────────────────────────────────────────────────────

    [[nodiscard]] const Model& model() const noexcept { return model_; }
    [[nodiscard]]       Model& model()       noexcept { return model_; }

    // ── Constructors ─────────────────────────────────────────────────────────

    /// Perfect-forwarding constructor: forwards arguments to Model.
    ///   HyperelasticRelation<SVK<3>>{lambda, mu}
    template <typename... Args>
        requires std::constructible_from<Model, Args...>
    explicit constexpr HyperelasticRelation(Args&&... args)
        : model_{std::forward<Args>(args)...} {}

    /// Copy from a pre-built model.
    ///   auto svk = SVK<3>::from_E_nu(200.0, 0.3);
    ///   HyperelasticRelation<SVK<3>> rel{svk};
    explicit constexpr HyperelasticRelation(const Model& m) : model_{m} {}
    explicit constexpr HyperelasticRelation(Model&& m) : model_{std::move(m)} {}

    constexpr HyperelasticRelation() noexcept = default;

private:
    Model model_{};
};


// =============================================================================
//  Convenience aliases
// =============================================================================

template <std::size_t dim>
using SVKRelation = HyperelasticRelation<SaintVenantKirchhoff<dim>>;

template <std::size_t dim>
using NeoHookeanRelation = HyperelasticRelation<CompressibleNeoHookean<dim>>;


// =============================================================================
//  Static concept verification
// =============================================================================

// SVK — constant tangent → satisfies ElasticConstitutiveRelation
static_assert(ConstitutiveRelation<SVKRelation<1>>);
static_assert(ConstitutiveRelation<SVKRelation<2>>);
static_assert(ConstitutiveRelation<SVKRelation<3>>);
static_assert(ElasticConstitutiveRelation<SVKRelation<1>>);
static_assert(ElasticConstitutiveRelation<SVKRelation<2>>);
static_assert(ElasticConstitutiveRelation<SVKRelation<3>>);

// Neo-Hookean — state-dependent tangent → only ConstitutiveRelation
static_assert(ConstitutiveRelation<NeoHookeanRelation<1>>);
static_assert(ConstitutiveRelation<NeoHookeanRelation<2>>);
static_assert(ConstitutiveRelation<NeoHookeanRelation<3>>);


} // namespace continuum

#endif // FALL_N_HYPERELASTIC_RELATION_HH
