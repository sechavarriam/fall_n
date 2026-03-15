#ifndef FALL_N_FINITE_STRAIN_DAMAGE_LOCAL_PROBLEM_HH
#define FALL_N_FINITE_STRAIN_DAMAGE_LOCAL_PROBLEM_HH

#include <algorithm>
#include <cmath>
#include <concepts>

#include <Eigen/Dense>

#include "ConstitutiveKinematics.hh"
#include "../materials/local_problem/ContinuumLocalProblem.hh"

namespace continuum {

namespace damage {

struct IsotropicDamageState {
    double kappa{0.0};
    double damage{0.0};

    [[nodiscard]] double d() const noexcept { return damage; }
};

struct DamageTrialState {
    double driving_force{0.0};
    double target_kappa{0.0};
    double target_damage{0.0};
};

template <std::size_t dim>
struct PositiveGreenLagrangeEquivalentStrain {
    using GradientVectorT = typename ConstitutiveKinematics<dim>::VoigtVectorT;

    [[nodiscard]] double operator()(
        const ConstitutiveKinematics<dim>& kin) const
    {
        return positive_norm(kin.green_lagrange_strain);
    }

    template <typename KinematicT>
        requires (KinematicT::num_components == voigt_size<dim>())
    [[nodiscard]] double operator()(const KinematicT& k) const
    {
        SymmetricTensor2<dim> E;
        E.set_from_engineering_voigt(k.components());
        return positive_norm(E);
    }

    [[nodiscard]] SymmetricTensor2<dim> gradient(
        const ConstitutiveKinematics<dim>& kin) const
    {
        return gradient(kin.green_lagrange_strain);
    }

    template <typename KinematicT>
        requires (KinematicT::num_components == voigt_size<dim>())
    [[nodiscard]] SymmetricTensor2<dim> gradient(const KinematicT& k) const
    {
        SymmetricTensor2<dim> E;
        E.set_from_engineering_voigt(k.components());
        return gradient(E);
    }

    [[nodiscard]] GradientVectorT gradient_wrt_engineering(
        const ConstitutiveKinematics<dim>& kin) const
    {
        // The return value is the derivative with respect to the engineering
        // strain vector used by the element B-matrices. Because dgamma=2 dE12,
        // the shear entries coincide with the raw tensor-gradient components.
        return gradient(kin.green_lagrange_strain).voigt();
    }

    template <typename KinematicT>
        requires (KinematicT::num_components == voigt_size<dim>())
    [[nodiscard]] GradientVectorT gradient_wrt_engineering(
        const KinematicT& k) const
    {
        return gradient(k).voigt();
    }

private:
    [[nodiscard]] static double positive_norm(
        const SymmetricTensor2<dim>& E)
    {
        const auto spectral = E.eigendecomposition();
        double sum = 0.0;
        for (std::size_t a = 0; a < dim; ++a) {
            const double eps_pos = std::max(0.0, spectral.eigenvalues(
                static_cast<Eigen::Index>(a)));
            sum += eps_pos * eps_pos;
        }
        return std::sqrt(sum);
    }

    [[nodiscard]] static SymmetricTensor2<dim> positive_part(
        const SymmetricTensor2<dim>& E)
    {
        const auto spectral = E.eigendecomposition();
        typename SymmetricTensor2<dim>::MatrixT E_pos =
            SymmetricTensor2<dim>::MatrixT::Zero();

        for (std::size_t a = 0; a < dim; ++a) {
            const double eps_pos = std::max(
                0.0,
                spectral.eigenvalues(static_cast<Eigen::Index>(a)));
            if (eps_pos <= 0.0) {
                continue;
            }

            const auto na = spectral.eigenvectors.col(
                static_cast<Eigen::Index>(a));
            E_pos += eps_pos * (na * na.transpose()).eval();
        }

        return SymmetricTensor2<dim>{Tensor2<dim>{E_pos}};
    }

    [[nodiscard]] static SymmetricTensor2<dim> gradient(
        const SymmetricTensor2<dim>& E)
    {
        const auto E_pos = positive_part(E);
        const double norm_pos = positive_norm(E);
        if (norm_pos <= 1.0e-16) {
            return SymmetricTensor2<dim>::zero();
        }
        return E_pos / norm_pos;
    }
};

template <typename DrivingForceT, std::size_t dim>
concept DamageDrivingForcePolicy =
    requires(const DrivingForceT& driving,
             const ConstitutiveKinematics<dim>& kin)
{
    { driving(kin) } -> std::convertible_to<double>;
};

template <typename DrivingForceT, std::size_t dim>
concept DifferentiableDamageDrivingForcePolicy =
    DamageDrivingForcePolicy<DrivingForceT, dim> &&
    (
        requires(const DrivingForceT& driving,
                 const ConstitutiveKinematics<dim>& kin)
        {
            { driving.gradient_wrt_engineering(kin) }
                -> std::convertible_to<typename ConstitutiveKinematics<dim>::VoigtVectorT>;
        } ||
        requires(const DrivingForceT& driving,
                 const ConstitutiveKinematics<dim>& kin)
        {
            { driving.gradient(kin) }
                -> std::convertible_to<SymmetricTensor2<dim>>;
        }
    );

class ExponentialDamageEvolution {
public:
    constexpr ExponentialDamageEvolution(
        double kappa0 = 1.0e-4,
        double kappa_f = 5.0e-3,
        double residual_stiffness = 1.0e-8) noexcept
        : kappa0_{kappa0}
        , kappa_f_{kappa_f}
        , residual_stiffness_{residual_stiffness}
    {}

    [[nodiscard]] constexpr double threshold() const noexcept {
        return kappa0_;
    }

    [[nodiscard]] constexpr double softening_scale() const noexcept {
        return kappa_f_;
    }

    [[nodiscard]] constexpr double residual_stiffness() const noexcept {
        return residual_stiffness_;
    }

    [[nodiscard]] double damage(double kappa) const noexcept {
        if (kappa <= kappa0_) {
            return 0.0;
        }

        const double scale = std::max(kappa_f_, 1.0e-16);
        const double d = 1.0 - std::exp(-(kappa - kappa0_) / scale);
        return std::clamp(d, 0.0, 1.0);
    }

    [[nodiscard]] double degradation(double damage) const noexcept {
        const double d = std::clamp(damage, 0.0, 1.0);
        const double one_minus_d = 1.0 - d;
        return residual_stiffness_ +
               (1.0 - residual_stiffness_) * one_minus_d * one_minus_d;
    }

    [[nodiscard]] double damage_derivative(double kappa) const noexcept {
        if (kappa <= kappa0_) {
            return 0.0;
        }

        const double scale = std::max(kappa_f_, 1.0e-16);
        return std::exp(-(kappa - kappa0_) / scale) / scale;
    }

    [[nodiscard]] double degradation_derivative(double damage) const noexcept {
        const double d = std::clamp(damage, 0.0, 1.0);
        return -2.0 * (1.0 - residual_stiffness_) * (1.0 - d);
    }

private:
    double kappa0_{1.0e-4};
    double kappa_f_{5.0e-3};
    double residual_stiffness_{1.0e-8};
};

template <typename EvolutionT>
concept DamageEvolutionPolicy =
    requires(const EvolutionT& evolution,
             double kappa,
             double damage)
{
    { evolution.threshold() } -> std::convertible_to<double>;
    { evolution.damage(kappa) } -> std::convertible_to<double>;
    { evolution.degradation(damage) } -> std::convertible_to<double>;
};

template <typename EvolutionT>
concept DifferentiableDamageEvolutionPolicy =
    DamageEvolutionPolicy<EvolutionT> &&
    requires(const EvolutionT& evolution,
             double kappa,
             double damage)
{
    { evolution.damage_derivative(kappa) } -> std::convertible_to<double>;
    { evolution.degradation_derivative(damage) } -> std::convertible_to<double>;
};

template <typename Relation>
concept FiniteStrainDamageRelationLike =
    ExternallyStateDrivenContinuumRelation<Relation> &&
    requires(const Relation& relation,
             const typename Relation::InternalVariablesT& alpha,
             const ConstitutiveKinematics<Relation::MaterialPolicyT::dim>& kin)
{
    { relation.driving_force()(kin) } -> std::convertible_to<double>;
    { relation.damage_evolution().damage(alpha.kappa) } -> std::convertible_to<double>;
    { alpha.kappa } -> std::convertible_to<double>;
    { alpha.damage } -> std::convertible_to<double>;
};

// Local rate-independent damage update:
//   kappa_{n+1} = max(kappa_n, Y(kin_{n+1}))
//   d_{n+1}     = D(kappa_{n+1})
//
// written as the 2x2 local nonlinear problem
//
//   R1(kappa, d) = kappa - max(kappa_n, Y_{n+1}) = 0
//   R2(kappa, d) = d - D(kappa) = 0.
//
// Even though the present system remains triangular, promoting the local
// unknown to a fixed-size vector is architecturally important: it exercises
// the same continuum-local machinery that coupled plasticity/damage or
// finite-strain inelastic models will need later.
struct RateIndependentDamageLocalProblem {
    using UnknownT = Eigen::Vector2d;
    using ResidualT = Eigen::Vector2d;
    using JacobianT = Eigen::Matrix2d;

    template <typename Relation>
    using ContextT = continuum_local_problem::Context<Relation, DamageTrialState>;

    template <typename Relation>
    using ResultT = continuum_local_problem::UpdateResult<Relation>;

    template <typename Relation>
        requires FiniteStrainDamageRelationLike<Relation>
    [[nodiscard]] auto make_context(
        const Relation& relation,
        const ConstitutiveKinematics<Relation::MaterialPolicyT::dim>& kin,
        const typename Relation::InternalVariablesT& alpha) const
        -> ContextT<Relation>
    {
        ContextT<Relation> context{};
        context.kinematics = kin;
        context.committed_state = alpha;
        context.trial_state.driving_force = relation.driving_force()(kin);
        context.trial_state.target_kappa =
            std::max(context.committed_state.kappa, context.trial_state.driving_force);
        context.trial_state.target_damage =
            relation.damage_evolution().damage(context.trial_state.target_kappa);
        return context;
    }

    template <typename Relation>
    [[nodiscard]] UnknownT initial_guess(
        const Relation&,
        const ContextT<Relation>& context) const
    {
        UnknownT guess = UnknownT::Zero();
        guess(0) = context.trial_state.target_kappa;
        guess(1) = context.trial_state.target_damage;
        return guess;
    }

    template <typename Relation>
    [[nodiscard]] ResidualT residual(
        const Relation& relation,
        const ContextT<Relation>& context,
        const UnknownT& unknown) const
    {
        ResidualT residual = ResidualT::Zero();
        residual(0) = unknown(0) - context.trial_state.target_kappa;
        residual(1) = unknown(1) - relation.damage_evolution().damage(unknown(0));
        return residual;
    }

    template <typename Relation>
    [[nodiscard]] JacobianT jacobian(
        const Relation& relation,
        const ContextT<Relation>&,
        const UnknownT& unknown) const
    {
        JacobianT jacobian = JacobianT::Identity();
        if constexpr (DifferentiableDamageEvolutionPolicy<typename Relation::DamageEvolutionPolicyT>) {
            jacobian(1, 0) = -relation.damage_evolution().damage_derivative(unknown(0));
        } else {
            const double h = 1.0e-8 * std::max(1.0, std::abs(unknown(0)));
            const double d_plus = relation.damage_evolution().damage(unknown(0) + h);
            const double d_minus = relation.damage_evolution().damage(unknown(0) - h);
            jacobian(1, 0) = -(d_plus - d_minus) / (2.0 * h);
        }
        return jacobian;
    }

    [[nodiscard]] double residual_norm(const ResidualT& residual) const {
        return residual.norm();
    }

    void project_iterate(UnknownT& unknown) const {
        unknown(0) = std::max(0.0, unknown(0));
        unknown(1) = std::clamp(unknown(1), 0.0, 1.0);
    }

    template <typename Relation>
        requires FiniteStrainDamageRelationLike<Relation>
    [[nodiscard]] bool is_inelastic(
        [[maybe_unused]] const Relation& relation,
        const ContextT<Relation>& context) const
    {
        return context.trial_state.target_damage >
               context.committed_state.damage + 1.0e-14;
    }

    template <typename Relation>
        requires FiniteStrainDamageRelationLike<Relation>
    [[nodiscard]] ResultT<Relation> elastic_result(
        const Relation& relation,
        const ContextT<Relation>& context) const
    {
        return {
            relation.compute_response(context.kinematics, context.committed_state),
            relation.tangent(context.kinematics, context.committed_state),
            context.committed_state,
            false
        };
    }

    template <typename Relation>
        requires FiniteStrainDamageRelationLike<Relation>
    [[nodiscard]] ResultT<Relation> finalize(
        const Relation& relation,
        const ContextT<Relation>& context,
        const UnknownT& unknown) const
    {
        auto alpha_new = context.committed_state;
        alpha_new.kappa = std::max(context.committed_state.kappa, unknown(0));
        alpha_new.damage = std::max(
            context.committed_state.damage,
            std::clamp(unknown(1), 0.0, 1.0));

        return {
            relation.compute_response(context.kinematics, alpha_new),
            relation.tangent(context.kinematics, alpha_new),
            alpha_new,
            alpha_new.damage > context.committed_state.damage + 1.0e-14
        };
    }
};

} // namespace damage

} // namespace continuum

#endif // FALL_N_FINITE_STRAIN_DAMAGE_LOCAL_PROBLEM_HH
