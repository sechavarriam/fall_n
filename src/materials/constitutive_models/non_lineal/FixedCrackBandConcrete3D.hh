#ifndef FALL_N_FIXED_CRACK_BAND_CONCRETE_3D_HH
#define FALL_N_FIXED_CRACK_BAND_CONCRETE_3D_HH

#include "../../ConstitutiveRelation.hh"
#include "../../ConstitutiveState.hh"
#include "../../MaterialPolicy.hh"
#include "../../../continuum/ConstitutiveKinematics.hh"
#include "../../../fracture/CrackShearTransferLaw.hh"

#include <algorithm>
#include <array>
#include <cmath>

#include <Eigen/Dense>

// =============================================================================
//  FixedCrackBandConcrete3D
// =============================================================================
//
//  Low-cost multidirectional fixed-crack concrete for the reduced RC continuum
//  validation path.
//
//  CyclicCrackBandConcrete3D is intentionally cheap, but its degradation is
//  component-wise in the host axes.  This richer law keeps the same no-return-
//  mapping hot path while promoting the crack kinematics:
//
//    - crack normals are stored as fixed material directions after initiation;
//    - normal crack opening is regularized with a Bazant-Oh crack-band length;
//    - up to three orthogonal-ish crack directions may be retained;
//    - stress is computed in the fixed-crack basis and rotated back to global
//      Voigt form;
//    - shear transfer is degraded per crack pair, not by one scalar damage;
//    - unilateral crack closure recovers compression only after the crack
//      closes over its historical opening scale;
//    - the tangent is a bounded central-difference tangent for this exploratory
//      richer branch.
//
//  This is still a smeared-crack continuum material, not XFEM.  Its purpose is
//  to decide whether a richer host law can close the reduced RC local benchmark
//  before moving to an enriched-discontinuity element family.
//
// =============================================================================

struct FixedCrackBandConcrete3DState {
    using Vec6 = Eigen::Matrix<double, 6, 1>;

    std::array<double, 3> kappa{0.0, 0.0, 0.0};
    std::array<double, 3> damage{0.0, 0.0, 0.0};
    double compression_strain_min{0.0};

    Vec6 eps_committed = Vec6::Zero();
    Vec6 sigma_committed = Vec6::Zero();

    int num_cracks{0};
    std::array<Eigen::Vector3d, 3> crack_normals{
        Eigen::Vector3d::UnitX(),
        Eigen::Vector3d::UnitY(),
        Eigen::Vector3d::UnitZ()};
    std::array<double, 3> crack_strain{0.0, 0.0, 0.0};
    std::array<double, 3> crack_strain_max{0.0, 0.0, 0.0};
    std::array<bool, 3> crack_closed{false, false, false};

    double sigma_o_max{0.0};
    double tau_o_max{0.0};

    [[nodiscard]] double d() const noexcept
    {
        return std::max({damage[0], damage[1], damage[2]});
    }
};

class FixedCrackBandConcrete3D {

public:
    using MaterialPolicyT = ThreeDimensionalMaterial;
    using KinematicT = typename MaterialPolicyT::StrainT;
    using ConjugateT = typename MaterialPolicyT::StressT;
    using TangentT = TangentMatrix<KinematicT, ConjugateT>;
    using InternalVariablesT = FixedCrackBandConcrete3DState;

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
    };

    double fpc_mpa_{30.0};
    double elastic_modulus_mpa_{30000.0};
    double shear_modulus_mpa_{12500.0};
    double tensile_strength_mpa_{0.6};
    double fracture_energy_nmm_{0.06};
    double characteristic_length_mm_{100.0};
    double peak_compressive_strain_{-0.002};
    double residual_compressive_strength_ratio_{0.20};
    double residual_tension_stiffness_ratio_{1.0e-6};
    double residual_shear_stiffness_ratio_{0.20};
    double large_opening_residual_shear_stiffness_ratio_{0.20};
    double shear_retention_decay_strain_{1.0};
    fall_n::fracture::CrackShearTransferLawKind shear_transfer_law_kind_{
        fall_n::fracture::CrackShearTransferLawKind::opening_exponential};
    double crack_closure_shear_gain_{1.0};
    double open_crack_compression_transfer_ratio_{0.05};
    double crack_closure_strain_floor_multiplier_{1.0};
    double kent_park_z_slope_{0.0};
    double secondary_crack_orthogonality_limit_{0.35};
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

    [[nodiscard]] static KinematicT strain_from_components(
        const Vec6& components) noexcept
    {
        KinematicT strain;
        strain.set_components(components);
        return strain;
    }

    [[nodiscard]] static Vec6 voigt_from_tensor(
        const Mat3& tensor) noexcept
    {
        Vec6 out = Vec6::Zero();
        out(0) = tensor(0, 0);
        out(1) = tensor(1, 1);
        out(2) = tensor(2, 2);
        out(3) = tensor(1, 2);
        out(4) = tensor(0, 2);
        out(5) = tensor(0, 1);
        return out;
    }

    [[nodiscard]] static Vec6 strain_voigt_from_tensor(
        const Mat3& tensor) noexcept
    {
        Vec6 out = Vec6::Zero();
        out(0) = tensor(0, 0);
        out(1) = tensor(1, 1);
        out(2) = tensor(2, 2);
        out(3) = 2.0 * tensor(1, 2);
        out(4) = 2.0 * tensor(0, 2);
        out(5) = 2.0 * tensor(0, 1);
        return out;
    }

    [[nodiscard]] static PrincipalStrainState principal_strain_state(
        const KinematicT& strain)
    {
        PrincipalStrainState out;
        Eigen::SelfAdjointEigenSolver<Mat3> solver{
            strain_tensor_from_voigt(strain)};
        out.values = solver.eigenvalues();
        out.vectors = solver.eigenvectors();
        return out;
    }

    [[nodiscard]] double crack_onset_strain() const noexcept
    {
        return tensile_strength_mpa_ /
               std::max(elastic_modulus_mpa_, 1.0e-12);
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
                      std::max(elastic_modulus_mpa_ * kappa, 1.0e-12);
        const double max_damage =
            1.0 - std::clamp(
                      residual_tension_stiffness_ratio_, 1.0e-8, 1.0);
        return std::clamp(secant_damage, 0.0, max_damage);
    }

    [[nodiscard]] double compression_stress(double eps) const noexcept
    {
        if (eps >= 0.0) {
            return 0.0;
        }

        const double eta = eps / peak_compressive_strain_;
        if (eps >= peak_compressive_strain_) {
            return -fpc_mpa_ * (2.0 * eta - eta * eta);
        }

        const double descent =
            1.0 - kent_park_z_slope_ * (peak_compressive_strain_ - eps);
        if (descent >= residual_compressive_strength_ratio_) {
            return -fpc_mpa_ * descent;
        }
        return -residual_compressive_strength_ratio_ * fpc_mpa_;
    }

    [[nodiscard]] double compression_secant_modulus(double eps) const noexcept
    {
        if (eps >= -1.0e-14) {
            return elastic_modulus_mpa_;
        }
        return std::max(
            compression_stress(eps) / eps,
            1.0e-6 * elastic_modulus_mpa_);
    }

    [[nodiscard]] double crack_closure_strain(double kappa) const noexcept
    {
        const double historical_opening =
            std::max(kappa - crack_onset_strain(), 0.0);
        const double floor =
            crack_closure_strain_floor_multiplier_ * crack_onset_strain();
        return std::max(historical_opening, floor);
    }

    [[nodiscard]] double crack_closure_factor(
        double strain_component,
        double kappa) const noexcept
    {
        if (strain_component >= 0.0 || kappa <= crack_onset_strain()) {
            return 1.0;
        }

        const double closure_strain = crack_closure_strain(kappa);
        if (closure_strain <= 1.0e-14 ||
            strain_component <= -closure_strain) {
            return 1.0;
        }

        const double t =
            std::clamp((-strain_component) / closure_strain, 0.0, 1.0);
        const double smooth = t * t * (3.0 - 2.0 * t);
        return open_crack_compression_transfer_ratio_ +
               (1.0 - open_crack_compression_transfer_ratio_) * smooth;
    }

    [[nodiscard]] double normal_secant_modulus(
        double strain_component,
        double damage,
        double kappa) const noexcept
    {
        if (strain_component >= 0.0) {
            return std::max(
                (1.0 - damage) * elastic_modulus_mpa_,
                residual_tension_stiffness_ratio_ * elastic_modulus_mpa_);
        }
        return crack_closure_factor(strain_component, kappa) *
               compression_secant_modulus(strain_component);
    }

    [[nodiscard]] double normal_stress(
        double strain_component,
        double damage,
        double kappa) const noexcept
    {
        if (strain_component >= 0.0) {
            return normal_secant_modulus(strain_component, damage, kappa) *
                   strain_component;
        }
        return crack_closure_factor(strain_component, kappa) *
               compression_stress(strain_component);
    }

    [[nodiscard]] double residual_shear_stiffness_ratio(
        double kappa_pair) const noexcept
    {
        return fall_n::fracture::opening_exponential_shear_floor(
            shear_transfer_parameters(),
            {.opening_strain =
                 std::max(kappa_pair - crack_onset_strain(), 0.0),
             .normal_strain =
                 std::max(kappa_pair - crack_onset_strain(), 0.0),
             .damage = 1.0,
             .crack_onset_strain = crack_onset_strain()});
    }

    [[nodiscard]] fall_n::fracture::CrackShearTransferLawParameters
    shear_transfer_parameters() const noexcept
    {
        return {
            .kind = shear_transfer_law_kind_,
            .residual_ratio = residual_shear_stiffness_ratio_,
            .large_opening_ratio =
                large_opening_residual_shear_stiffness_ratio_,
            .opening_decay_strain = shear_retention_decay_strain_,
            .closure_shear_gain = crack_closure_shear_gain_,
            .max_closed_ratio = 1.0,
            .closure_reference_strain_multiplier =
                crack_closure_strain_floor_multiplier_};
    }

    [[nodiscard]] double crack_shear_floor(
        double kappa,
        double normal_strain,
        double damage) const noexcept
    {
        return fall_n::fracture::retained_crack_shear_floor(
            shear_transfer_parameters(),
            {.opening_strain = std::max(kappa - crack_onset_strain(), 0.0),
             .normal_strain = normal_strain,
             .damage = damage,
             .crack_onset_strain = crack_onset_strain()});
    }

    [[nodiscard]] double damaged_shear_modulus(
        const InternalVariablesT& alpha,
        int a,
        int b,
        const Vec6& local_strain) const noexcept
    {
        const double intact_pair = std::sqrt(
            std::max(1.0 - alpha.damage[a], 1.0e-8) *
            std::max(1.0 - alpha.damage[b], 1.0e-8));
        const double floor_pair = std::sqrt(
            crack_shear_floor(
                alpha.kappa[a],
                local_strain(a),
                alpha.damage[a]) *
            crack_shear_floor(
                alpha.kappa[b],
                local_strain(b),
                alpha.damage[b]));
        const double retained =
            std::max(floor_pair, intact_pair);
        return retained * shear_modulus_mpa_;
    }

    [[nodiscard]] static Eigen::Vector3d normalized_or(
        const Eigen::Vector3d& value,
        const Eigen::Vector3d& fallback) noexcept
    {
        const double n = value.norm();
        if (n <= 1.0e-14) {
            return fallback;
        }
        return value / n;
    }

    [[nodiscard]] bool is_new_crack_direction(
        const InternalVariablesT& alpha,
        const Eigen::Vector3d& candidate) const noexcept
    {
        for (int i = 0; i < alpha.num_cracks; ++i) {
            const double alignment =
                std::abs(alpha.crack_normals[i].normalized().dot(candidate));
            if (alignment > secondary_crack_orthogonality_limit_) {
                return false;
            }
        }
        return true;
    }

    void add_crack_if_needed(
        InternalVariablesT& next,
        const Eigen::Vector3d& direction,
        double principal_strain) const noexcept
    {
        if (principal_strain <= crack_onset_strain() ||
            next.num_cracks >= 3) {
            return;
        }
        const auto normal = normalized_or(direction, Eigen::Vector3d::UnitX());
        if (!is_new_crack_direction(next, normal)) {
            return;
        }
        next.crack_normals[static_cast<std::size_t>(next.num_cracks)] = normal;
        ++next.num_cracks;
    }

    [[nodiscard]] static Eigen::Vector3d fallback_axis(
        const Eigen::Vector3d& normal) noexcept
    {
        Eigen::Vector3d axis = Eigen::Vector3d::UnitX();
        if (std::abs(normal.dot(axis)) > 0.75) {
            axis = Eigen::Vector3d::UnitY();
        }
        return axis;
    }

    [[nodiscard]] static Mat3 crack_basis(
        const InternalVariablesT& alpha) noexcept
    {
        if (alpha.num_cracks <= 0) {
            return Mat3::Identity();
        }

        const Eigen::Vector3d e0 =
            normalized_or(alpha.crack_normals[0], Eigen::Vector3d::UnitX());
        Eigen::Vector3d e1 =
            alpha.num_cracks >= 2 ? alpha.crack_normals[1]
                                  : fallback_axis(e0);
        e1 -= e0.dot(e1) * e0;
        e1 = normalized_or(e1, fallback_axis(e0).cross(e0));
        Eigen::Vector3d e2 = normalized_or(e0.cross(e1), Eigen::Vector3d::UnitZ());

        if (alpha.num_cracks >= 3) {
            const Eigen::Vector3d candidate = alpha.crack_normals[2];
            if (std::abs(candidate.normalized().dot(e2)) >
                std::abs(candidate.normalized().dot(e1))) {
                e2 = normalized_or(candidate - e0.dot(candidate) * e0, e2);
                e1 = normalized_or(e2.cross(e0), e1);
            }
        }

        Mat3 basis = Mat3::Identity();
        basis.col(0) = e0;
        basis.col(1) = e1;
        basis.col(2) = e2;
        return basis;
    }

    [[nodiscard]] Vec6 local_strain_components(
        const KinematicT& strain,
        const InternalVariablesT& alpha) const noexcept
    {
        const Mat3 R = crack_basis(alpha);
        return strain_voigt_from_tensor(
            R.transpose() * strain_tensor_from_voigt(strain) * R);
    }

    [[nodiscard]] Vec6 local_stress_components(
        const Vec6& local_strain,
        const InternalVariablesT& alpha) const noexcept
    {
        Vec6 stress = Vec6::Zero();
        stress(0) =
            normal_stress(local_strain(0), alpha.damage[0], alpha.kappa[0]);
        stress(1) =
            normal_stress(local_strain(1), alpha.damage[1], alpha.kappa[1]);
        stress(2) =
            normal_stress(local_strain(2), alpha.damage[2], alpha.kappa[2]);
        stress(3) =
            damaged_shear_modulus(alpha, 1, 2, local_strain) * local_strain(3);
        stress(4) =
            damaged_shear_modulus(alpha, 0, 2, local_strain) * local_strain(4);
        stress(5) =
            damaged_shear_modulus(alpha, 0, 1, local_strain) * local_strain(5);
        return stress;
    }

    [[nodiscard]] Vec6 rotate_stress_to_global(
        const Vec6& local_stress,
        const InternalVariablesT& alpha) const noexcept
    {
        const Mat3 local_tensor = [&] {
            Mat3 sig = Mat3::Zero();
            sig(0, 0) = local_stress(0);
            sig(1, 1) = local_stress(1);
            sig(2, 2) = local_stress(2);
            sig(1, 2) = sig(2, 1) = local_stress(3);
            sig(0, 2) = sig(2, 0) = local_stress(4);
            sig(0, 1) = sig(1, 0) = local_stress(5);
            return sig;
        }();
        const Mat3 R = crack_basis(alpha);
        return voigt_from_tensor(R * local_tensor * R.transpose());
    }

    [[nodiscard]] Vec6 stress_components(
        const KinematicT& strain,
        const InternalVariablesT& trial) const noexcept
    {
        const Vec6 local_strain = local_strain_components(strain, trial);
        return rotate_stress_to_global(
            local_stress_components(local_strain, trial),
            trial);
    }

    [[nodiscard]] Vec6 local_secant_stiffness_diagonal(
        const Vec6& local_strain,
        const InternalVariablesT& trial) const noexcept
    {
        Vec6 diagonal = Vec6::Zero();
        diagonal(0) =
            normal_secant_modulus(local_strain(0), trial.damage[0], trial.kappa[0]);
        diagonal(1) =
            normal_secant_modulus(local_strain(1), trial.damage[1], trial.kappa[1]);
        diagonal(2) =
            normal_secant_modulus(local_strain(2), trial.damage[2], trial.kappa[2]);
        diagonal(3) = damaged_shear_modulus(trial, 1, 2, local_strain);
        diagonal(4) = damaged_shear_modulus(trial, 0, 2, local_strain);
        diagonal(5) = damaged_shear_modulus(trial, 0, 1, local_strain);
        return diagonal;
    }

    [[nodiscard]] static Mat3 unit_strain_tensor(Eigen::Index component) noexcept
    {
        Mat3 tensor = Mat3::Zero();
        switch (component) {
            case 0:
                tensor(0, 0) = 1.0;
                break;
            case 1:
                tensor(1, 1) = 1.0;
                break;
            case 2:
                tensor(2, 2) = 1.0;
                break;
            case 3:
                tensor(1, 2) = tensor(2, 1) = 0.5;
                break;
            case 4:
                tensor(0, 2) = tensor(2, 0) = 0.5;
                break;
            case 5:
                tensor(0, 1) = tensor(1, 0) = 0.5;
                break;
            default:
                break;
        }
        return tensor;
    }

    [[nodiscard]] InternalVariablesT trial_state_without_stress(
        const KinematicT& strain,
        const InternalVariablesT& alpha) const
    {
        auto next = alpha;
        const auto principal = principal_strain_state(strain);
        for (Eigen::Index i = 2; i >= 0; --i) {
            add_crack_if_needed(next, principal.vectors.col(i), principal.values(i));
        }

        const Vec6 local_strain = local_strain_components(strain, next);
        for (int i = 0; i < 3; ++i) {
            const double eps_i = local_strain(i);
            if (next.num_cracks > i || eps_i > crack_onset_strain()) {
                next.kappa[static_cast<std::size_t>(i)] =
                    std::max(alpha.kappa[static_cast<std::size_t>(i)], eps_i);
                next.damage[static_cast<std::size_t>(i)] =
                    damage_from_kappa(
                        next.kappa[static_cast<std::size_t>(i)]);
                next.crack_strain[static_cast<std::size_t>(i)] =
                    std::max(eps_i - crack_onset_strain(), 0.0);
                next.crack_strain_max[static_cast<std::size_t>(i)] =
                    std::max(
                        alpha.crack_strain_max[static_cast<std::size_t>(i)],
                        next.crack_strain[static_cast<std::size_t>(i)]);
                next.crack_closed[static_cast<std::size_t>(i)] = eps_i <= 0.0;
            }
        }
        next.compression_strain_min =
            std::min(alpha.compression_strain_min, principal.values(0));
        return next;
    }

    [[nodiscard]] InternalVariablesT evolved_state(
        const KinematicT& strain,
        const InternalVariablesT& alpha) const
    {
        auto next = trial_state_without_stress(strain, alpha);
        const Vec6 stress = stress_components(strain, next);
        next.eps_committed = strain.components();
        next.sigma_committed = stress;

        Eigen::SelfAdjointEigenSolver<Mat3> stress_solver{[&] {
            Mat3 sig = Mat3::Zero();
            sig(0, 0) = stress(0);
            sig(1, 1) = stress(1);
            sig(2, 2) = stress(2);
            sig(1, 2) = sig(2, 1) = stress(3);
            sig(0, 2) = sig(2, 0) = stress(4);
            sig(0, 1) = sig(1, 0) = stress(5);
            return sig;
        }()};
        const auto stress_principal = stress_solver.eigenvalues();
        const double sigma_o = std::max(stress_principal(2), 0.0);
        const double tau_o =
            0.5 * std::max(stress_principal(2) - stress_principal(0), 0.0);
        next.sigma_o_max = std::max(alpha.sigma_o_max, sigma_o);
        next.tau_o_max = std::max(alpha.tau_o_max, tau_o);
        return next;
    }

    [[nodiscard]] Vec6 trial_stress_components(
        const KinematicT& strain,
        const InternalVariablesT& alpha) const
    {
        const auto trial = trial_state_without_stress(strain, alpha);
        return stress_components(strain, trial);
    }

public:
    FixedCrackBandConcrete3D()
    {
        initialize_kent_park_descent();
    }

    FixedCrackBandConcrete3D(
        double fpc_mpa,
        double elastic_modulus_mpa,
        double poisson_like_ratio,
        double tensile_strength_mpa,
        double fracture_energy_nmm,
        double characteristic_length_mm,
        double residual_tension_stiffness_ratio = 1.0e-6,
        double residual_shear_stiffness_ratio = 0.20,
        double open_crack_compression_transfer_ratio = 0.05,
        double peak_compressive_strain = -0.002,
        double residual_compressive_strength_ratio = 0.20,
        double kent_park_z_slope = 0.0,
        double large_opening_residual_shear_stiffness_ratio = -1.0,
        double shear_retention_decay_strain = 1.0,
        fall_n::fracture::CrackShearTransferLawKind
            shear_transfer_law_kind =
                fall_n::fracture::CrackShearTransferLawKind::
                    opening_exponential,
        double crack_closure_shear_gain = 1.0)
        : fpc_mpa_{std::max(fpc_mpa, 1.0e-12)},
          elastic_modulus_mpa_{std::max(elastic_modulus_mpa, 1.0e-12)},
          tensile_strength_mpa_{std::max(tensile_strength_mpa, 1.0e-12)},
          fracture_energy_nmm_{std::max(fracture_energy_nmm, 1.0e-12)},
          characteristic_length_mm_{std::max(characteristic_length_mm, 1.0e-9)},
          peak_compressive_strain_{-std::max(
              std::abs(peak_compressive_strain), 1.0e-12)},
          residual_compressive_strength_ratio_{
              std::clamp(residual_compressive_strength_ratio, 1.0e-8, 1.0)},
          residual_tension_stiffness_ratio_{
              std::clamp(residual_tension_stiffness_ratio, 1.0e-8, 1.0)},
          residual_shear_stiffness_ratio_{
              std::clamp(residual_shear_stiffness_ratio, 1.0e-8, 1.0)},
          large_opening_residual_shear_stiffness_ratio_{
              std::clamp(
                  large_opening_residual_shear_stiffness_ratio >= 0.0
                      ? large_opening_residual_shear_stiffness_ratio
                      : residual_shear_stiffness_ratio,
                  1.0e-8,
                  1.0)},
          shear_retention_decay_strain_{
              std::max(shear_retention_decay_strain, 1.0e-12)},
          shear_transfer_law_kind_{shear_transfer_law_kind},
          crack_closure_shear_gain_{
              std::clamp(crack_closure_shear_gain, 0.0, 1.0)},
          open_crack_compression_transfer_ratio_{
              std::clamp(open_crack_compression_transfer_ratio, 1.0e-6, 1.0)}
    {
        const double bounded_poisson =
            std::clamp(poisson_like_ratio, -0.99, 0.49);
        shear_modulus_mpa_ =
            elastic_modulus_mpa_ / (2.0 * (1.0 + bounded_poisson));
        if (kent_park_z_slope > 0.0 && std::isfinite(kent_park_z_slope)) {
            kent_park_z_slope_ = kent_park_z_slope;
        } else {
            initialize_kent_park_descent();
        }
    }

    void initialize_kent_park_descent() noexcept
    {
        const double eps_50u = std::max(
            (3.0 + 0.29 * fpc_mpa_) / (145.0 * fpc_mpa_ - 1000.0),
            1.0e-6);
        const double denom_z =
            std::max(eps_50u + peak_compressive_strain_, 1.0e-6);
        kent_park_z_slope_ = 0.5 / denom_z;
    }

    [[nodiscard]] double tensile_cracking_strain() const noexcept
    {
        return crack_onset_strain();
    }

    [[nodiscard]] double peak_compressive_strain() const noexcept
    {
        return peak_compressive_strain_;
    }

    [[nodiscard]] double residual_shear_stiffness_ratio() const noexcept
    {
        return residual_shear_stiffness_ratio_;
    }

    [[nodiscard]] double large_opening_residual_shear_stiffness_ratio()
        const noexcept
    {
        return large_opening_residual_shear_stiffness_ratio_;
    }

    [[nodiscard]] double kent_park_z_slope() const noexcept
    {
        return kent_park_z_slope_;
    }

    [[nodiscard]] ConjugateT compute_response(
        const KinematicT& strain,
        const InternalVariablesT& alpha) const
    {
        ConjugateT stress;
        stress.set_components(trial_stress_components(strain, alpha));
        return stress;
    }

    [[nodiscard]] TangentT tangent(
        const KinematicT& strain,
        const InternalVariablesT& alpha) const
    {
        TangentT C = TangentT::Zero();
        const auto trial = trial_state_without_stress(strain, alpha);
        const Mat3 R = crack_basis(trial);
        const Vec6 local_strain = strain_voigt_from_tensor(
            R.transpose() * strain_tensor_from_voigt(strain) * R);
        const Vec6 diagonal =
            local_secant_stiffness_diagonal(local_strain, trial);

        for (Eigen::Index j = 0; j < 6; ++j) {
            const Vec6 local_column = strain_voigt_from_tensor(
                R.transpose() * unit_strain_tensor(j) * R);
            C.col(j) =
                rotate_stress_to_global(diagonal.cwiseProduct(local_column), trial);
        }
        return C;
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
    ConstitutiveRelation<FixedCrackBandConcrete3D>,
    "FixedCrackBandConcrete3D must satisfy ConstitutiveRelation");

static_assert(
    InelasticConstitutiveRelation<FixedCrackBandConcrete3D>,
    "FixedCrackBandConcrete3D must satisfy InelasticConstitutiveRelation");

static_assert(
    ExternallyStateDrivenConstitutiveRelation<FixedCrackBandConcrete3D>,
    "FixedCrackBandConcrete3D must support external state");

#endif // FALL_N_FIXED_CRACK_BAND_CONCRETE_3D_HH
