#ifndef FALL_N_CONSTITUTIVE_ORTHOTROPIC_BIMODULAR_CONCRETE_PROXY_HH
#define FALL_N_CONSTITUTIVE_ORTHOTROPIC_BIMODULAR_CONCRETE_PROXY_HH

#include "ElasticRelation.hh"

#include <array>
#include <algorithm>

// =============================================================================
//  OrthotropicBimodularConcreteProxy
// =============================================================================
//
//  Cheap path-independent proxy for RC continuum pilots.
//
//  The intent is not to replace the promoted Ko-Bathe host, but to provide a
//  low-cost host material for algorithmic audits where:
//    - compression should remain reasonably stiff,
//    - tension should be much softer,
//    - embedded steel is still allowed to carry the hysteresis.
//
//  The model is deliberately simple:
//    sigma_i = E_i(sign(eps_i)) * eps_i   for i = xx, yy, zz
//    tau_j   = G_j * gamma_j              for j = xy, yz, xz
//
//  There is no Poisson coupling in this proxy. That keeps the tangent cheap,
//  symmetric, and easy to reason about for benchmark work. The "orthotropic"
//  label reflects that the constitutive matrix is diagonal in Voigt space and
//  can admit axis-wise moduli, even though the current promoted baseline uses
//  the same axial scales on every normal direction.
//
// =============================================================================

class OrthotropicBimodularConcreteProxy {

public:
    using MaterialPolicyT = ThreeDimensionalMaterial;
    using KinematicT = typename MaterialPolicyT::StrainT;
    using ConjugateT = typename MaterialPolicyT::StressT;
    using TangentT = TangentMatrix<KinematicT, ConjugateT>;

    using StrainT = KinematicT;
    using StressT = ConjugateT;
    using ConstitutiveStateT = CommittedConstitutiveState<StrainT>;
    using MaterialStateT = ConstitutiveStateT;
    using StateVariableT = typename ConstitutiveStateT::StateVariableT;

    static constexpr std::size_t dim = KinematicT::dim;
    static constexpr std::size_t num_components = KinematicT::num_components;

private:
    std::array<double, 3> compression_moduli_mpa_{};
    std::array<double, 3> tension_moduli_mpa_{};
    std::array<double, 3> shear_moduli_mpa_{};

    [[nodiscard]] double normal_modulus_for_axis(
        std::size_t axis,
        double strain_component) const noexcept
    {
        return strain_component >= 0.0
                   ? tension_moduli_mpa_[axis]
                   : compression_moduli_mpa_[axis];
    }

public:
    OrthotropicBimodularConcreteProxy() = default;

    OrthotropicBimodularConcreteProxy(
        const std::array<double, 3>& compression_moduli_mpa,
        const std::array<double, 3>& tension_moduli_mpa,
        const std::array<double, 3>& shear_moduli_mpa)
        : compression_moduli_mpa_{compression_moduli_mpa},
          tension_moduli_mpa_{tension_moduli_mpa},
          shear_moduli_mpa_{shear_moduli_mpa}
    {
    }

    OrthotropicBimodularConcreteProxy(
        double compression_modulus_mpa,
        double tension_ratio,
        double poisson_like_ratio,
        double shear_scale = 1.0)
    {
        const double bounded_tension_ratio =
            std::clamp(tension_ratio, 0.0, 1.0);
        const double bounded_poisson =
            std::clamp(poisson_like_ratio, -0.99, 0.49);
        const double reference_shear =
            compression_modulus_mpa / (2.0 * (1.0 + bounded_poisson));
        const double tension_modulus =
            bounded_tension_ratio * compression_modulus_mpa;
        const double shear_modulus =
            std::max(reference_shear * shear_scale, 0.0);

        compression_moduli_mpa_.fill(compression_modulus_mpa);
        tension_moduli_mpa_.fill(tension_modulus);
        shear_moduli_mpa_.fill(shear_modulus);
    }

    [[nodiscard]] const auto& compression_moduli_mpa() const noexcept
    {
        return compression_moduli_mpa_;
    }

    [[nodiscard]] const auto& tension_moduli_mpa() const noexcept
    {
        return tension_moduli_mpa_;
    }

    [[nodiscard]] const auto& shear_moduli_mpa() const noexcept
    {
        return shear_moduli_mpa_;
    }

    [[nodiscard]] TangentT tangent(const KinematicT& strain) const
    {
        TangentT C = TangentT::Zero();
        C(0, 0) = normal_modulus_for_axis(0, strain[0]);
        C(1, 1) = normal_modulus_for_axis(1, strain[1]);
        C(2, 2) = normal_modulus_for_axis(2, strain[2]);
        C(3, 3) = shear_moduli_mpa_[0];
        C(4, 4) = shear_moduli_mpa_[1];
        C(5, 5) = shear_moduli_mpa_[2];
        return C;
    }

    [[nodiscard]] ConjugateT compute_response(const KinematicT& strain) const
    {
        ConjugateT stress;
        stress.set_components(tangent(strain) * strain.components());
        return stress;
    }
};

static_assert(
    ConstitutiveRelation<OrthotropicBimodularConcreteProxy>,
    "OrthotropicBimodularConcreteProxy must satisfy ConstitutiveRelation");

#endif // FALL_N_CONSTITUTIVE_ORTHOTROPIC_BIMODULAR_CONCRETE_PROXY_HH
