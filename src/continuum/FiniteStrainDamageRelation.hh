#ifndef FALL_N_FINITE_STRAIN_DAMAGE_RELATION_HH
#define FALL_N_FINITE_STRAIN_DAMAGE_RELATION_HH

#include <concepts>
#include <utility>

#include "../materials/MaterialPolicy.hh"
#include "../materials/Strain.hh"
#include "../materials/Stress.hh"
#include "../materials/ConstitutiveRelation.hh"
#include "ConstitutiveKinematics.hh"
#include "FiniteStrainDamageLocalProblem.hh"
#include "HyperelasticModel.hh"
#include "HyperelasticRelation.hh"
#include "StressMeasures.hh"
#include "TensorOperations.hh"

namespace continuum {

template <
    HyperelasticModelConcept Model,
    typename DrivingForceT = damage::PositiveGreenLagrangeEquivalentStrain<Model::dimension>,
    typename DamageEvolutionT = damage::ExponentialDamageEvolution
>
    requires damage::DamageDrivingForcePolicy<DrivingForceT, Model::dimension> &&
             damage::DamageEvolutionPolicy<DamageEvolutionT>
class FiniteStrainDamageRelation {
public:
    static constexpr std::size_t dim = Model::dimension;
    static constexpr std::size_t N = voigt_size<dim>();

    using MaterialPolicyT = SolidMaterial<N>;
    using MaterialPolicy = MaterialPolicyT;
    using KinematicT = Strain<N>;
    using ConjugateT = Stress<N>;
    using TangentT = TangentMatrix<KinematicT, ConjugateT>;
    using InternalVariablesT = damage::IsotropicDamageState;
    using HyperelasticModelT = Model;
    using DrivingForcePolicyT = DrivingForceT;
    using DamageEvolutionPolicyT = DamageEvolutionT;
    using GradientVectorT = typename ConstitutiveKinematics<dim>::VoigtVectorT;

    constexpr FiniteStrainDamageRelation() noexcept = default;

    constexpr explicit FiniteStrainDamageRelation(
        Model model,
        DrivingForceT driving_force = {},
        DamageEvolutionT damage_evolution = {}) noexcept
        : model_{std::move(model)}
        , driving_force_{std::move(driving_force)}
        , damage_evolution_{std::move(damage_evolution)}
    {}

    [[nodiscard]] ConjugateT compute_response(const KinematicT& strain) const {
        return compute_response(strain, embedded_state_);
    }

    [[nodiscard]] TangentT tangent(const KinematicT& strain) const {
        return tangent(strain, embedded_state_);
    }

    void update(const KinematicT& strain) {
        commit(embedded_state_, strain);
    }

    [[nodiscard]] const InternalVariablesT& internal_state() const noexcept {
        return embedded_state_;
    }

    [[nodiscard]] ConjugateT compute_response(
        const KinematicT& strain,
        const InternalVariablesT& alpha) const
    {
        const auto E = strain_eng_to_tensor<dim>(strain);
        return tensor_to_stress<dim>(
            undamaged_second_piola(E) * degradation(alpha));
    }

    [[nodiscard]] TangentT tangent(
        const KinematicT& strain,
        const InternalVariablesT& alpha) const
    {
        const auto E = strain_eng_to_tensor<dim>(strain);
        const double Y = evaluate_driving_force(strain);
        return consistent_reference_tangent(
            E,
            alpha,
            Y,
            evaluate_driving_force_gradient_wrt_engineering(strain));
    }

    void commit(InternalVariablesT& alpha, const KinematicT& strain) const {
        const double kappa_trial =
            std::max(alpha.kappa, evaluate_driving_force(strain));
        alpha.kappa = kappa_trial;
        alpha.damage = damage_evolution_.damage(kappa_trial);
    }

    [[nodiscard]] ConjugateT compute_response(
        const ConstitutiveKinematics<dim>& kin) const
    {
        return compute_response(kin, embedded_state_);
    }

    [[nodiscard]] TangentT tangent(
        const ConstitutiveKinematics<dim>& kin) const
    {
        return tangent(kin, embedded_state_);
    }

    [[nodiscard]] ConjugateT compute_response(
        const ConstitutiveKinematics<dim>& kin,
        const InternalVariablesT& alpha) const
    {
        const auto S_eff =
            undamaged_second_piola(kin.green_lagrange_strain) * degradation(alpha);

        if (kin.conjugate_stress_measure == StressMeasureKind::second_piola_kirchhoff) {
            return tensor_to_stress<dim>(S_eff);
        }

        return tensor_to_stress<dim>(stress::cauchy_from_2pk(S_eff, kin.F));
    }

    [[nodiscard]] TangentT tangent(
        const ConstitutiveKinematics<dim>& kin,
        const InternalVariablesT& alpha) const
    {
        const auto C_ref = consistent_reference_tangent(
            kin.green_lagrange_strain,
            alpha,
            driving_force_(kin),
            evaluate_driving_force_gradient_wrt_engineering(kin));

        if (kin.conjugate_stress_measure == StressMeasureKind::second_piola_kirchhoff) {
            return C_ref;
        }

        return ops::push_forward_tangent(Tensor4<dim>{C_ref}, kin.F).voigt_matrix();
    }

    void commit(
        InternalVariablesT& alpha,
        const ConstitutiveKinematics<dim>& kin) const
    {
        const double kappa_trial =
            std::max(alpha.kappa, driving_force_(kin));
        alpha.kappa = kappa_trial;
        alpha.damage = damage_evolution_.damage(kappa_trial);
    }

    [[nodiscard]] const Model& model() const noexcept { return model_; }
    [[nodiscard]] const DrivingForceT& driving_force() const noexcept { return driving_force_; }
    [[nodiscard]] const DamageEvolutionT& damage_evolution() const noexcept {
        return damage_evolution_;
    }

private:
    // The continuum path is the primary one for this relation.  The legacy
    // Strain<N> overload remains available so existing small-strain callers can
    // still talk to the law without manufacturing a full continuum carrier.
    [[nodiscard]] double evaluate_driving_force(const KinematicT& strain) const {
        if constexpr (requires { driving_force_(strain); }) {
            return driving_force_(strain);
        } else {
            return driving_force_(make_reference_kinematics(strain_eng_to_tensor<dim>(strain)));
        }
    }

    [[nodiscard]] GradientVectorT evaluate_driving_force_gradient_wrt_engineering(
        const KinematicT& strain) const
    {
        if constexpr (requires {
            driving_force_.gradient_wrt_engineering(strain);
        }) {
            return driving_force_.gradient_wrt_engineering(strain);
        } else {
            return evaluate_driving_force_gradient_wrt_engineering(
                make_reference_kinematics(strain_eng_to_tensor<dim>(strain)));
        }
    }

    [[nodiscard]] GradientVectorT evaluate_driving_force_gradient_wrt_engineering(
        const ConstitutiveKinematics<dim>& kin) const
    {
        if constexpr (requires {
            driving_force_.gradient_wrt_engineering(kin);
        }) {
            return driving_force_.gradient_wrt_engineering(kin);
        } else if constexpr (requires {
            driving_force_.gradient(kin);
        }) {
            return driving_force_.gradient(kin).voigt();
        } else {
            return GradientVectorT::Zero();
        }
    }

    [[nodiscard]] double degradation(const InternalVariablesT& alpha) const {
        return damage_evolution_.degradation(alpha.damage);
    }

    [[nodiscard]] static ConstitutiveKinematics<dim> make_reference_kinematics(
        const SymmetricTensor2<dim>& E)
    {
        ConstitutiveKinematics<dim> kin;
        kin.active_strain_measure = StrainMeasureKind::green_lagrange;
        kin.conjugate_stress_measure = StressMeasureKind::second_piola_kirchhoff;
        kin.engineering_strain = E.voigt_engineering();
        kin.green_lagrange_strain = E;
        kin.infinitesimal_strain = E;
        kin.almansi_strain = E;
        kin.F = Tensor2<dim>::identity();
        kin.detF = 1.0;
        return kin;
    }

    [[nodiscard]] bool is_active_damage_loading(
        double driving_force,
        const InternalVariablesT& alpha) const
    {
        const double tol = 1.0e-10 * std::max(
            1.0,
            std::max(std::abs(driving_force), std::abs(alpha.kappa)));
        return alpha.kappa > damage_evolution_.threshold() + tol &&
               std::abs(alpha.kappa - driving_force) <= tol;
    }

    [[nodiscard]] TangentT consistent_reference_tangent(
        const SymmetricTensor2<dim>& E,
        const InternalVariablesT& alpha,
        double driving_force,
        const GradientVectorT& driving_force_gradient) const
    {
        TangentT tangent = degradation(alpha) *
                           undamaged_material_tangent(E).voigt_matrix();

        if constexpr (damage::DifferentiableDamageDrivingForcePolicy<DrivingForceT, dim> &&
                      damage::DifferentiableDamageEvolutionPolicy<DamageEvolutionT>) {
            if (is_active_damage_loading(driving_force, alpha)) {
                const auto S0 = undamaged_second_piola(E).voigt();
                const double dg_dk =
                    damage_evolution_.degradation_derivative(alpha.damage) *
                    damage_evolution_.damage_derivative(alpha.kappa);

                // S = g(d(kappa(E))) S0(E); during active loading the scalar
                // correction is rank-one in engineering-Voigt form.
                tangent += dg_dk * (S0 * driving_force_gradient.transpose());
            }
        }

        return tangent;
    }

    [[nodiscard]] SymmetricTensor2<dim> undamaged_second_piola(
        const SymmetricTensor2<dim>& E) const
    {
        return model_.second_piola_kirchhoff(E);
    }

    [[nodiscard]] Tensor4<dim> undamaged_material_tangent(
        const SymmetricTensor2<dim>& E) const
    {
        return model_.material_tangent(E);
    }

    Model model_{};
    DrivingForceT driving_force_{};
    DamageEvolutionT damage_evolution_{};
    InternalVariablesT embedded_state_{};
};

template <std::size_t dim>
using NeoHookeanDamageRelation = FiniteStrainDamageRelation<CompressibleNeoHookean<dim>>;

template <std::size_t dim>
using SVKDamageRelation = FiniteStrainDamageRelation<SaintVenantKirchhoff<dim>>;

static_assert(ConstitutiveRelation<NeoHookeanDamageRelation<3>>);
static_assert(InelasticConstitutiveRelation<NeoHookeanDamageRelation<3>>);
static_assert(ExternallyStateDrivenConstitutiveRelation<NeoHookeanDamageRelation<3>>);
static_assert(ContinuumKinematicsAwareConstitutiveRelation<NeoHookeanDamageRelation<3>>);
static_assert(ExternallyStateDrivenContinuumRelation<NeoHookeanDamageRelation<3>>);
static_assert(
    ContinuumLocalProblemPolicy<
        damage::RateIndependentDamageLocalProblem,
        NeoHookeanDamageRelation<3>>);

} // namespace continuum

#endif // FALL_N_FINITE_STRAIN_DAMAGE_RELATION_HH
