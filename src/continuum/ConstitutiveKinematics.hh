#ifndef FALL_N_CONTINUUM_CONSTITUTIVE_KINEMATICS_HH
#define FALL_N_CONTINUUM_CONSTITUTIVE_KINEMATICS_HH

// =============================================================================
//  ConstitutiveKinematics.hh — Continuum constitutive carrier
// =============================================================================
//
//  This header introduces a semantic bridge between:
//
//    1. element-level kinematics (B, F, detF, engineering Voigt strain), and
//    2. constitutive laws that may need richer finite-strain information than a
//       flat Strain<N> carrier can provide.
//
//  The goal is not to replace the existing Strain<N>/Stress<N> interface.  The
//  goal is to add a compile-time friendly continuum-only carrier so that
//  large-displacement / finite-strain constitutive models can consume:
//
//    • the active engineering strain vector used by the current formulation,
//    • the infinitesimal strain tensor (when meaningful),
//    • the Green-Lagrange strain tensor,
//    • the deformation gradient F and its determinant.
//
//  `Material<>` keeps the legacy `compute_response(Strain<N>)` path as the
//  fallback.  Continuum-specific models can additionally opt into
//  `compute_response(ConstitutiveKinematics<dim>)`.
//
// =============================================================================

#include <cstddef>
#include <concepts>

#include <Eigen/Dense>

#include "KinematicPolicy.hh"
#include "ContinuumSemantics.hh"
#include "../materials/ConstitutiveRelation.hh"

namespace continuum {

template <std::size_t dim>
struct ConstitutiveKinematics {
    static constexpr std::size_t dimension = dim;
    static constexpr std::size_t num_components = voigt_size<dim>();

    using VoigtVectorT = Eigen::Vector<double, static_cast<int>(num_components)>;

    StrainMeasureKind active_strain_measure{StrainMeasureKind::infinitesimal};
    StressMeasureKind conjugate_stress_measure{StressMeasureKind::cauchy};
    FormulationKind formulation_kind{FormulationKind::small_strain};
    KinematicDescriptionKind description_kind{KinematicDescriptionKind::linearized};
    ConfigurationKind assembly_configuration{ConfigurationKind::reference};
    ConfigurationKind strain_measure_configuration{ConfigurationKind::reference};
    ConfigurationKind stress_measure_configuration{ConfigurationKind::current};
    VolumeMeasureKind volume_measure{VolumeMeasureKind::reference};
    VirtualWorkCompatibilityKind virtual_work_compatibility{
        VirtualWorkCompatibilityKind::linearized_equivalent};
    FormulationMaturity formulation_maturity{FormulationMaturity::implemented};
    bool measure_pair_is_normatively_audited{true};
    FormulationAuditScope formulation_audit =
        canonical_formulation_audit_scope(FormulationKind::small_strain);

    VoigtVectorT engineering_strain = VoigtVectorT::Zero();
    SymmetricTensor2<dim> infinitesimal_strain = SymmetricTensor2<dim>::zero();
    SymmetricTensor2<dim> green_lagrange_strain = SymmetricTensor2<dim>::zero();
    SymmetricTensor2<dim> almansi_strain = SymmetricTensor2<dim>::zero();
    Tensor2<dim> F = Tensor2<dim>::identity();
    double detF{1.0};

    [[nodiscard]] bool is_finite_strain() const noexcept {
        return active_strain_measure != StrainMeasureKind::infinitesimal;
    }

    [[nodiscard]] const VoigtVectorT& active_strain_voigt() const noexcept {
        return engineering_strain;
    }

    [[nodiscard]] SymmetricTensor2<dim> active_strain_tensor() const noexcept {
        if (active_strain_measure == StrainMeasureKind::green_lagrange) {
            return green_lagrange_strain;
        }
        if (active_strain_measure == StrainMeasureKind::almansi) {
            return almansi_strain;
        }
        return infinitesimal_strain;
    }

    [[nodiscard]] ConjugateMeasureSemantics measure_semantics() const noexcept {
        return {
            active_strain_measure,
            conjugate_stress_measure,
            strain_measure_configuration,
            stress_measure_configuration
        };
    }

    [[nodiscard]] bool has_work_conjugate_measure_pair() const noexcept {
        return measure_semantics().is_work_conjugate();
    }

    [[nodiscard]] bool has_normatively_audited_work_conjugacy() const noexcept {
        return measure_pair_is_normatively_audited && has_work_conjugate_measure_pair();
    }

    [[nodiscard]] VirtualWorkSemantics virtual_work_semantics() const noexcept {
        return {
            formulation_kind,
            measure_semantics(),
            assembly_configuration,
            volume_measure,
            virtual_work_compatibility,
            measure_pair_is_normatively_audited
        };
    }

    [[nodiscard]] bool has_exact_virtual_work_semantics() const noexcept {
        return virtual_work_semantics().is_exact();
    }

    [[nodiscard]] bool has_normatively_acceptable_virtual_work_semantics() const noexcept {
        return virtual_work_semantics().is_normatively_acceptable();
    }

    [[nodiscard]] bool has_validated_continuum_3d_formulation_path() const noexcept {
        return formulation_audit.validated_for_continuum_3d;
    }

    [[nodiscard]] bool is_reference_finite_kinematics_path() const noexcept {
        return formulation_audit.is_reference_finite_kinematics_path();
    }

    [[nodiscard]] bool requires_finite_kinematics_scope_disclaimer() const noexcept {
        return formulation_audit.requires_finite_kinematics_scope_disclaimer();
    }
};

template <std::size_t dim>
    requires ValidDim<dim>
[[nodiscard]] inline SymmetricTensor2<dim> engineering_voigt_to_tensor(
    const Eigen::Vector<double, static_cast<int>(voigt_size<dim>())>& engineering_voigt) noexcept
{
    SymmetricTensor2<dim> tensor;
    tensor.set_from_engineering_voigt(engineering_voigt);
    return tensor;
}

template <typename KinematicT, std::size_t dim>
    requires (KinematicMeasure<KinematicT> &&
              ValidDim<dim> &&
              KinematicT::dim == dim)
[[nodiscard]] inline KinematicT make_kinematic_measure(
    const ConstitutiveKinematics<dim>& kin)
{
    KinematicT state{};

    if constexpr (KinematicT::num_components == voigt_size<dim>()) {
        if constexpr (KinematicT::num_components == 1 &&
                      requires(KinematicT s, double v) { s.set_components(v); }) {
            state.set_components(kin.active_strain_voigt()(0));
        }
        else if constexpr (requires(KinematicT s,
                                    const Eigen::Vector<double, static_cast<int>(KinematicT::num_components)>& v) {
            s.set_strain(v);
        }) {
            state.set_strain(kin.active_strain_voigt());
        }
        else if constexpr (requires(KinematicT s,
                                    const Eigen::Vector<double, static_cast<int>(KinematicT::num_components)>& v) {
            s.set_components(v);
        }) {
            state.set_components(kin.active_strain_voigt());
        }
        else {
            static_assert(sizeof(KinematicT) == 0,
                "KinematicT must provide set_strain(v) or set_components(v).");
        }
    }
    else if constexpr (requires(KinematicT s,
                                const Eigen::Vector<double, static_cast<int>(KinematicT::num_components)>& v) {
        s.set_components(v);
    }) {
        // Keep the erased continuum hook well-formed even for structural
        // constitutive spaces that should never consume a continuum carrier.
        // The physically meaningful path is the exact-size branch above.
        Eigen::Vector<double, static_cast<int>(KinematicT::num_components)> padded =
            Eigen::Vector<double, static_cast<int>(KinematicT::num_components)>::Zero();
        constexpr auto M = (KinematicT::num_components < voigt_size<dim>())
            ? KinematicT::num_components
            : voigt_size<dim>();
        for (std::size_t i = 0; i < M; ++i) {
            padded(static_cast<Eigen::Index>(i)) = kin.active_strain_voigt()(static_cast<Eigen::Index>(i));
        }
        state.set_components(padded);
    }
    else {
        static_assert(sizeof(KinematicT) == 0,
            "KinematicT must provide a writable component interface "
            "to be built from ContinuumConstitutiveKinematics.");
    }

    return state;
}

template <typename Policy, std::size_t dim>
    requires ValidDim<dim>
[[nodiscard]] inline ConstitutiveKinematics<dim> make_constitutive_kinematics(
    const GPKinematics<dim>& gp)
{
    ConstitutiveKinematics<dim> kin;
    using Traits = KinematicFormulationTraits<Policy>;

    kin.formulation_kind = Traits::formulation_kind;
    kin.description_kind = Traits::description_kind;
    kin.assembly_configuration = Traits::assembly_configuration;
    kin.strain_measure_configuration = Traits::conjugate_pair.strain_configuration;
    kin.stress_measure_configuration = Traits::conjugate_pair.stress_configuration;
    kin.volume_measure = Traits::volume_measure;
    kin.virtual_work_compatibility = Traits::virtual_work_compatibility;
    kin.formulation_maturity = Traits::maturity;
    kin.measure_pair_is_normatively_audited = Traits::pair_is_normatively_audited;
    kin.formulation_audit = Traits::audit_scope;
    kin.active_strain_measure = Traits::conjugate_pair.strain_measure;
    kin.conjugate_stress_measure = Traits::conjugate_pair.stress_measure;
    kin.engineering_strain = gp.strain_voigt;
    kin.F = gp.F;
    kin.detF = gp.detF;

    if constexpr (std::same_as<Policy, SmallStrain>) {
        kin.infinitesimal_strain = engineering_voigt_to_tensor<dim>(gp.strain_voigt);
        kin.green_lagrange_strain = kin.infinitesimal_strain;
        kin.almansi_strain = kin.infinitesimal_strain;
    }
    else if constexpr (std::same_as<Policy, UpdatedLagrangian>) {
        kin.green_lagrange_strain = strain::green_lagrange(gp.F);
        kin.almansi_strain = strain::almansi(gp.F);
        kin.engineering_strain = kin.almansi_strain.voigt_engineering();
        kin.infinitesimal_strain = SymmetricTensor2<dim>{
            gp.F.symmetric_part() - Tensor2<dim>::identity().symmetric_part()
        };
    }
    else if constexpr (std::same_as<Policy, TotalLagrangian>) {
        kin.green_lagrange_strain = strain::green_lagrange(gp.F);
        kin.almansi_strain = strain::almansi(gp.F);
        // For finite-strain formulations the infinitesimal strain is still
        // useful as a diagnostic / linearized companion.
        kin.infinitesimal_strain = SymmetricTensor2<dim>{
            gp.F.symmetric_part() - Tensor2<dim>::identity().symmetric_part()
        };
    }
    else {
        // Corotational continuum semantics are currently a declared slot in the
        // architecture rather than a validated formulation. Keep the carrier
        // diagnostically rich without pretending to provide a mature finite-
        // strain continuum interpretation.
        kin.infinitesimal_strain = engineering_voigt_to_tensor<dim>(gp.strain_voigt);
        kin.green_lagrange_strain = strain::green_lagrange(gp.F);
        kin.almansi_strain = strain::almansi(gp.F);
    }

    return kin;
}

template <typename Relation>
concept ContinuumKinematicsAwareConstitutiveRelation =
    ConstitutiveRelation<Relation> &&
    requires(const Relation r,
             const ConstitutiveKinematics<Relation::MaterialPolicyT::dim>& kin) {
        { r.compute_response(kin) } -> std::same_as<typename Relation::ConjugateT>;
        { r.tangent(kin) } -> std::same_as<typename Relation::TangentT>;
    };

template <typename Relation>
concept ExternallyStateDrivenContinuumRelation =
    ContinuumKinematicsAwareConstitutiveRelation<Relation> &&
    ExternallyStateDrivenConstitutiveRelation<Relation> &&
    requires(const Relation r,
             typename Relation::InternalVariablesT& alpha,
             const typename Relation::InternalVariablesT& calpha,
             const ConstitutiveKinematics<Relation::MaterialPolicyT::dim>& kin) {
        { r.compute_response(kin, calpha) } -> std::same_as<typename Relation::ConjugateT>;
        { r.tangent(kin, calpha) } -> std::same_as<typename Relation::TangentT>;
        { r.commit(alpha, kin) };
    };

} // namespace continuum

#endif // FALL_N_CONTINUUM_CONSTITUTIVE_KINEMATICS_HH
