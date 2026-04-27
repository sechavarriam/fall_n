#ifndef FALL_N_TENSILE_CRACK_BAND_DAMAGE_CONCRETE_PROXY_3D_HH
#define FALL_N_TENSILE_CRACK_BAND_DAMAGE_CONCRETE_PROXY_3D_HH

#include "../../ConstitutiveRelation.hh"
#include "../../ConstitutiveState.hh"
#include "../../MaterialPolicy.hh"
#include "../../../continuum/ConstitutiveKinematics.hh"

#include <algorithm>
#include <array>
#include <cmath>

#include <Eigen/Dense>

// =============================================================================
//  TensileCrackBandDamageConcreteProxy3D
// =============================================================================
//
//  Low-cost 3D concrete host for RC continuum validation pilots.
//
//  This model is intentionally a proxy, not a replacement for the promoted
//  Ko-Bathe fixed-crack concrete law.  It is designed for large benchmark sweeps
//  where embedded nonlinear steel must be exercised while the concrete host
//  still needs a physically meaningful tensile degradation mechanism.
//
//  Assumptions:
//    - diagonal bimodular stiffness in engineering Voigt space;
//    - compression normal stiffness is retained when a crack closes;
//    - tension normal stiffness is degraded by a scalar damage variable;
//    - shear is degraded with a retained residual fraction;
//    - damage is driven by the norm of positive principal strains;
//    - post-peak tension follows an exponential crack-band softening law.
//
//  The damage trigger follows the Mazars-style equivalent tensile strain
//
//      kappa = sqrt(sum_i <eps_i>_+^2)
//
//  while the softening branch is regularized through a characteristic length,
//  so the dissipated tensile energy scales with G_f / l_c.  The tangent is a
//  secant operator that deliberately ignores dD/deps.  That choice keeps the
//  global Newton iterations cheap and robust for validation screening; it also
//  makes this class a transparent control branch rather than an implicit local
//  nonlinear solver.
//
// =============================================================================

struct TensileCrackBandDamageConcreteProxy3DState {
    using Vec6 = Eigen::Matrix<double, 6, 1>;

    double kappa{0.0};
    double damage{0.0};

    Vec6 eps_committed = Vec6::Zero();
    Vec6 sigma_committed = Vec6::Zero();

    int num_cracks{0};
    std::array<Eigen::Vector3d, 3> crack_normals{
        Eigen::Vector3d::UnitZ(),
        Eigen::Vector3d::UnitX(),
        Eigen::Vector3d::UnitY()};
    std::array<double, 3> crack_strain{0.0, 0.0, 0.0};
    std::array<double, 3> crack_strain_max{0.0, 0.0, 0.0};
    std::array<bool, 3> crack_closed{false, false, false};

    double sigma_o_max{0.0};
    double tau_o_max{0.0};

    [[nodiscard]] double d() const noexcept { return damage; }
};

class TensileCrackBandDamageConcreteProxy3D {

public:
    using MaterialPolicyT = ThreeDimensionalMaterial;
    using KinematicT = typename MaterialPolicyT::StrainT;
    using ConjugateT = typename MaterialPolicyT::StressT;
    using TangentT = TangentMatrix<KinematicT, ConjugateT>;
    using InternalVariablesT = TensileCrackBandDamageConcreteProxy3DState;

    using StrainT = KinematicT;
    using StressT = ConjugateT;
    using ConstitutiveStateT = CommittedConstitutiveState<StrainT>;
    using MaterialStateT = ConstitutiveStateT;
    using StateVariableT = typename ConstitutiveStateT::StateVariableT;

    static constexpr std::size_t dim = KinematicT::dim;
    static constexpr std::size_t num_components = KinematicT::num_components;

private:
    using Vec6 = Eigen::Matrix<double, 6, 1>;
    using Mat3 = Eigen::Matrix3d;

    struct PrincipalStrainState {
        Eigen::Vector3d values = Eigen::Vector3d::Zero();
        Mat3 vectors = Mat3::Identity();
        double equivalent_tensile_strain{0.0};
        double max_principal_strain{0.0};
        Eigen::Vector3d max_principal_direction = Eigen::Vector3d::UnitZ();
    };

    double compression_modulus_mpa_{1.0};
    double tension_modulus_mpa_{0.1};
    double shear_modulus_mpa_{1.0};
    double tensile_strength_mpa_{0.1};
    double fracture_energy_nmm_{0.06};
    double characteristic_length_mm_{100.0};
    double residual_tension_stiffness_ratio_{1.0e-4};
    double residual_shear_stiffness_ratio_{0.20};

    [[nodiscard]] static Mat3 strain_tensor_from_voigt(
        const KinematicT& strain) noexcept
    {
        Mat3 eps = Mat3::Zero();
        eps(0, 0) = strain[0];
        eps(1, 1) = strain[1];
        eps(2, 2) = strain[2];
        eps(1, 2) = eps(2, 1) = 0.5 * strain[3];
        eps(0, 2) = eps(2, 0) = 0.5 * strain[4];
        eps(0, 1) = eps(1, 0) = 0.5 * strain[5];
        return eps;
    }

    [[nodiscard]] static Mat3 stress_tensor_from_voigt(
        const Vec6& stress) noexcept
    {
        Mat3 sig = Mat3::Zero();
        sig(0, 0) = stress(0);
        sig(1, 1) = stress(1);
        sig(2, 2) = stress(2);
        sig(1, 2) = sig(2, 1) = stress(3);
        sig(0, 2) = sig(2, 0) = stress(4);
        sig(0, 1) = sig(1, 0) = stress(5);
        return sig;
    }

    [[nodiscard]] static PrincipalStrainState principal_strain_state(
        const KinematicT& strain)
    {
        PrincipalStrainState out;
        Eigen::SelfAdjointEigenSolver<Mat3> solver{
            strain_tensor_from_voigt(strain)};

        out.values = solver.eigenvalues();
        out.vectors = solver.eigenvectors();
        out.max_principal_strain = out.values(2);
        out.max_principal_direction = out.vectors.col(2).normalized();

        double positive_norm_sq = 0.0;
        for (Eigen::Index i = 0; i < 3; ++i) {
            const double positive_part = std::max(out.values(i), 0.0);
            positive_norm_sq += positive_part * positive_part;
        }
        out.equivalent_tensile_strain = std::sqrt(positive_norm_sq);
        return out;
    }

    [[nodiscard]] double crack_onset_strain() const noexcept
    {
        return tensile_strength_mpa_ /
               std::max(tension_modulus_mpa_, 1.0e-12);
    }

    [[nodiscard]] double softening_strain_scale() const noexcept
    {
        return fracture_energy_nmm_ /
               std::max(
                   tensile_strength_mpa_ * characteristic_length_mm_,
                   1.0e-12);
    }

    [[nodiscard]] double damage_from_kappa(double kappa) const noexcept
    {
        const double eps0 = crack_onset_strain();
        if (kappa <= eps0 || eps0 <= 0.0) {
            return 0.0;
        }

        const double eps_f = std::max(softening_strain_scale(), 1.0e-12);
        const double softening_stress =
            tensile_strength_mpa_ *
            std::exp(-(kappa - eps0) / eps_f);
        const double secant_damage =
            1.0 - softening_stress /
                      std::max(tension_modulus_mpa_ * kappa, 1.0e-12);
        const double max_damage =
            1.0 - std::clamp(
                      residual_tension_stiffness_ratio_, 1.0e-8, 1.0);

        return std::clamp(secant_damage, 0.0, max_damage);
    }

    [[nodiscard]] double trial_damage(
        const KinematicT& strain,
        const InternalVariablesT& alpha) const
    {
        const auto principal = principal_strain_state(strain);
        return damage_from_kappa(
            std::max(alpha.kappa, principal.equivalent_tensile_strain));
    }

    [[nodiscard]] double normal_modulus_for_axis(
        std::size_t axis,
        double strain_component,
        double damage) const noexcept
    {
        (void)axis;
        if (strain_component >= 0.0) {
            return std::max(
                (1.0 - damage) * tension_modulus_mpa_,
                residual_tension_stiffness_ratio_ * tension_modulus_mpa_);
        }
        return compression_modulus_mpa_;
    }

    [[nodiscard]] double damaged_shear_modulus(double damage) const noexcept
    {
        const double retained_fraction =
            std::max(residual_shear_stiffness_ratio_, 1.0 - damage);
        return retained_fraction * shear_modulus_mpa_;
    }

    [[nodiscard]] TangentT secant_tangent(
        const KinematicT& strain,
        double damage) const
    {
        TangentT C = TangentT::Zero();
        C(0, 0) = normal_modulus_for_axis(0, strain[0], damage);
        C(1, 1) = normal_modulus_for_axis(1, strain[1], damage);
        C(2, 2) = normal_modulus_for_axis(2, strain[2], damage);
        const double Gd = damaged_shear_modulus(damage);
        C(3, 3) = Gd;
        C(4, 4) = Gd;
        C(5, 5) = Gd;
        return C;
    }

    [[nodiscard]] Vec6 stress_components(
        const KinematicT& strain,
        double damage) const
    {
        return secant_tangent(strain, damage) * strain.components();
    }

    [[nodiscard]] InternalVariablesT evolved_state(
        const KinematicT& strain,
        const InternalVariablesT& alpha) const
    {
        auto next = alpha;
        const auto principal = principal_strain_state(strain);

        next.kappa = std::max(
            alpha.kappa,
            principal.equivalent_tensile_strain);
        next.damage = damage_from_kappa(next.kappa);

        const Vec6 stress = stress_components(strain, next.damage);
        next.eps_committed = strain.components();
        next.sigma_committed = stress;

        if (next.damage > 1.0e-12) {
            next.num_cracks = 1;
            if (principal.max_principal_strain > 0.0) {
                next.crack_normals[0] = principal.max_principal_direction;
            }
            next.crack_strain[0] =
                std::max(principal.max_principal_strain -
                             crack_onset_strain(),
                         0.0);
            next.crack_strain_max[0] =
                std::max(alpha.crack_strain_max[0], next.crack_strain[0]);
            next.crack_closed[0] = principal.max_principal_strain <= 0.0;

            for (std::size_t i = 1; i < 3; ++i) {
                next.crack_strain[i] = 0.0;
                next.crack_strain_max[i] = alpha.crack_strain_max[i];
                next.crack_closed[i] = true;
            }
        } else {
            next.num_cracks = 0;
            next.crack_strain = {0.0, 0.0, 0.0};
            next.crack_closed = {false, false, false};
        }

        Eigen::SelfAdjointEigenSolver<Mat3> stress_solver{
            stress_tensor_from_voigt(stress)};
        const auto stress_principal = stress_solver.eigenvalues();
        const double sigma_o =
            std::max(stress_principal(2), 0.0);
        const double tau_o =
            0.5 * std::max(
                      stress_principal(2) - stress_principal(0),
                      0.0);
        next.sigma_o_max = std::max(alpha.sigma_o_max, sigma_o);
        next.tau_o_max = std::max(alpha.tau_o_max, tau_o);

        return next;
    }

public:
    TensileCrackBandDamageConcreteProxy3D() = default;

    TensileCrackBandDamageConcreteProxy3D(
        double compression_modulus_mpa,
        double tension_stiffness_ratio,
        double poisson_like_ratio,
        double tensile_strength_mpa,
        double fracture_energy_nmm,
        double characteristic_length_mm,
        double residual_tension_stiffness_ratio = 1.0e-4,
        double residual_shear_stiffness_ratio = 0.20)
        : compression_modulus_mpa_{
              std::max(compression_modulus_mpa, 1.0e-12)},
          tension_modulus_mpa_{
              std::max(compression_modulus_mpa, 1.0e-12) *
              std::clamp(tension_stiffness_ratio, 0.0, 1.0)},
          tensile_strength_mpa_{std::max(tensile_strength_mpa, 1.0e-12)},
          fracture_energy_nmm_{std::max(fracture_energy_nmm, 1.0e-12)},
          characteristic_length_mm_{std::max(characteristic_length_mm, 1.0e-9)},
          residual_tension_stiffness_ratio_{
              std::clamp(residual_tension_stiffness_ratio, 1.0e-8, 1.0)},
          residual_shear_stiffness_ratio_{
              std::clamp(residual_shear_stiffness_ratio, 1.0e-8, 1.0)}
    {
        const double bounded_poisson =
            std::clamp(poisson_like_ratio, -0.99, 0.49);
        shear_modulus_mpa_ =
            compression_modulus_mpa_ / (2.0 * (1.0 + bounded_poisson));
        tension_modulus_mpa_ =
            std::max(tension_modulus_mpa_, 1.0e-12);
    }

    [[nodiscard]] double compression_modulus_mpa() const noexcept
    {
        return compression_modulus_mpa_;
    }

    [[nodiscard]] double tension_modulus_mpa() const noexcept
    {
        return tension_modulus_mpa_;
    }

    [[nodiscard]] double tensile_strength_mpa() const noexcept
    {
        return tensile_strength_mpa_;
    }

    [[nodiscard]] double fracture_energy_nmm() const noexcept
    {
        return fracture_energy_nmm_;
    }

    [[nodiscard]] double characteristic_length_mm() const noexcept
    {
        return characteristic_length_mm_;
    }

    [[nodiscard]] ConjugateT compute_response(
        const KinematicT& strain,
        const InternalVariablesT& alpha) const
    {
        ConjugateT stress;
        stress.set_components(
            stress_components(strain, trial_damage(strain, alpha)));
        return stress;
    }

    [[nodiscard]] TangentT tangent(
        const KinematicT& strain,
        const InternalVariablesT& alpha) const
    {
        return secant_tangent(strain, trial_damage(strain, alpha));
    }

    void commit(InternalVariablesT& alpha, const KinematicT& strain) const
    {
        alpha = evolved_state(strain, alpha);
    }

    [[nodiscard]] ConjugateT compute_response(
        const continuum::ConstitutiveKinematics<3>& kin,
        const InternalVariablesT& alpha) const
    {
        return compute_response(
            continuum::make_kinematic_measure<KinematicT>(kin),
            alpha);
    }

    [[nodiscard]] TangentT tangent(
        const continuum::ConstitutiveKinematics<3>& kin,
        const InternalVariablesT& alpha) const
    {
        return tangent(
            continuum::make_kinematic_measure<KinematicT>(kin),
            alpha);
    }

    void commit(
        InternalVariablesT& alpha,
        const continuum::ConstitutiveKinematics<3>& kin) const
    {
        commit(alpha, continuum::make_kinematic_measure<KinematicT>(kin));
    }

    [[nodiscard]] ConjugateT compute_response(const KinematicT& strain) const
    {
        return compute_response(strain, state_);
    }

    [[nodiscard]] TangentT tangent(const KinematicT& strain) const
    {
        return tangent(strain, state_);
    }

    [[nodiscard]] ConjugateT compute_response(
        const continuum::ConstitutiveKinematics<3>& kin) const
    {
        return compute_response(
            continuum::make_kinematic_measure<KinematicT>(kin));
    }

    [[nodiscard]] TangentT tangent(
        const continuum::ConstitutiveKinematics<3>& kin) const
    {
        return tangent(continuum::make_kinematic_measure<KinematicT>(kin));
    }

    void update(const KinematicT& strain)
    {
        commit(state_, strain);
    }

    void update(const continuum::ConstitutiveKinematics<3>& kin)
    {
        update(continuum::make_kinematic_measure<KinematicT>(kin));
    }

    [[nodiscard]] const InternalVariablesT& internal_state() const
    {
        return state_;
    }

    void set_internal_state(const InternalVariablesT& state)
    {
        state_ = state;
    }

private:
    InternalVariablesT state_{};
};

static_assert(
    ConstitutiveRelation<TensileCrackBandDamageConcreteProxy3D>,
    "TensileCrackBandDamageConcreteProxy3D must satisfy ConstitutiveRelation");

static_assert(
    InelasticConstitutiveRelation<TensileCrackBandDamageConcreteProxy3D>,
    "TensileCrackBandDamageConcreteProxy3D must satisfy InelasticConstitutiveRelation");

static_assert(
    ExternallyStateDrivenConstitutiveRelation<
        TensileCrackBandDamageConcreteProxy3D>,
    "TensileCrackBandDamageConcreteProxy3D must support external state");

#endif // FALL_N_TENSILE_CRACK_BAND_DAMAGE_CONCRETE_PROXY_3D_HH
