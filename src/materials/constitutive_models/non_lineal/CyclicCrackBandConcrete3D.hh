#ifndef FALL_N_CYCLIC_CRACK_BAND_CONCRETE_3D_HH
#define FALL_N_CYCLIC_CRACK_BAND_CONCRETE_3D_HH

#include "../../ConstitutiveRelation.hh"
#include "../../ConstitutiveState.hh"
#include "../../MaterialPolicy.hh"
#include "../../../continuum/ConstitutiveKinematics.hh"
#include "../../../fracture/CrackShearTransferLaw.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include <Eigen/Dense>

// =============================================================================
//  CyclicCrackBandConcrete3D
// =============================================================================
//
//  Intermediate low-cost concrete law for continuum RC validation.
//
//  The previous tensile crack-band proxy was useful as a cheap host, but its
//  reduced tensile modulus delayed crack onset with respect to the Kent-Park
//  fiber section.  This law keeps the same inexpensive no-return-mapping hot
//  path while moving the physics closer to the structural benchmark:
//
//    - initial tension and compression use the Kent-Park-compatible Ec;
//    - tension softening is regularized with a Bazant-Oh crack-band length;
//    - tensile damage is driven by a Mazars-style positive principal strain;
//    - normal compression follows Kent-Park only after the historical crack
//      opening has been closed;
//    - shear stiffness is degraded by the retained tensile damage fraction;
//    - the exported tangent is secant-positive for robust global Newton solves.
//
//  This is still a validation material, not a full 3D concrete production law:
//  it deliberately avoids local nonlinear solves, plastic flow and rotating
//  crack shear transfer.  Its role is to bridge the benchmark between the
//  overly simple proxy and the heavier Ko-Bathe fixed-crack model.
//
// =============================================================================

struct CyclicCrackBandConcrete3DState {
    using Vec6 = Eigen::Matrix<double, 6, 1>;

    double kappa{0.0};
    double damage{0.0};
    double compression_strain_min{0.0};

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

enum class CyclicCrackBandConcrete3DTangentMode {
    // Robust default for global displacement-control runs.
    SecantPositive,
    // Audit/continuation mode: captures active softening, including negative
    // tangent stiffness, through bounded perturbations of the trial strain.
    AdaptiveCentralDifference,
    // Same audit tangent, but keeps the secant column near non-smooth fronts.
    AdaptiveCentralDifferenceWithSecantFallback
};

class CyclicCrackBandConcrete3D {

public:
    using MaterialPolicyT = ThreeDimensionalMaterial;
    using KinematicT = typename MaterialPolicyT::StrainT;
    using ConjugateT = typename MaterialPolicyT::StressT;
    using TangentT = TangentMatrix<KinematicT, ConjugateT>;
    using InternalVariablesT = CyclicCrackBandConcrete3DState;

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
    CyclicCrackBandConcrete3DTangentMode tangent_mode_{
        CyclicCrackBandConcrete3DTangentMode::SecantPositive};
    double numerical_tangent_relative_step_{1.0e-4};
    double numerical_tangent_validation_tolerance_{0.50};

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

    [[nodiscard]] double trial_damage(
        const KinematicT& strain,
        const InternalVariablesT& alpha) const
    {
        const auto principal = principal_strain_state(strain);
        return damage_from_kappa(
            std::max(alpha.kappa, principal.equivalent_tensile_strain));
    }

    [[nodiscard]] double trial_kappa(
        const KinematicT& strain,
        const InternalVariablesT& alpha) const
    {
        const auto principal = principal_strain_state(strain);
        return std::max(alpha.kappa, principal.equivalent_tensile_strain);
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

    [[nodiscard]] double compression_tangent(double eps) const noexcept
    {
        if (eps >= 0.0) {
            return elastic_modulus_mpa_;
        }

        const double eta = eps / peak_compressive_strain_;
        if (eps >= peak_compressive_strain_) {
            const double tangent =
                -fpc_mpa_ * (2.0 - 2.0 * eta) /
                peak_compressive_strain_;
            return std::max(tangent, 1.0e-6 * elastic_modulus_mpa_);
        }

        const double descent =
            1.0 - kent_park_z_slope_ * (peak_compressive_strain_ - eps);
        if (descent >= residual_compressive_strength_ratio_) {
            return compression_secant_modulus(eps);
        }

        return 1.0e-6 * elastic_modulus_mpa_;
    }

    [[nodiscard]] double compression_secant_modulus(double eps) const noexcept
    {
        if (eps >= -1.0e-14) {
            return elastic_modulus_mpa_;
        }
        return std::max(compression_stress(eps) / eps, 1.0e-6 * elastic_modulus_mpa_);
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
        if (closure_strain <= 1.0e-14) {
            return 1.0;
        }
        if (strain_component <= -closure_strain) {
            return 1.0;
        }

        const double t =
            std::clamp((-strain_component) / closure_strain, 0.0, 1.0);
        const double smooth = t * t * (3.0 - 2.0 * t);
        return open_crack_compression_transfer_ratio_ +
               (1.0 - open_crack_compression_transfer_ratio_) * smooth;
    }

    [[nodiscard]] double crack_closure_factor_derivative(
        double strain_component,
        double kappa) const noexcept
    {
        if (strain_component >= 0.0 || kappa <= crack_onset_strain()) {
            return 0.0;
        }

        const double closure_strain = crack_closure_strain(kappa);
        if (closure_strain <= 1.0e-14 ||
            strain_component <= -closure_strain) {
            return 0.0;
        }

        const double t =
            std::clamp((-strain_component) / closure_strain, 0.0, 1.0);
        const double dsmooth_dt = 6.0 * t * (1.0 - t);
        return -(1.0 - open_crack_compression_transfer_ratio_) *
               dsmooth_dt / closure_strain;
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

    [[nodiscard]] double normal_algorithmic_modulus(
        double strain_component,
        double damage,
        double kappa) const noexcept
    {
        if (strain_component >= 0.0) {
            return normal_secant_modulus(strain_component, damage, kappa);
        }

        const double closure = crack_closure_factor(strain_component, kappa);
        const double dclosure =
            crack_closure_factor_derivative(strain_component, kappa);
        const double tangent =
            closure * compression_tangent(strain_component) +
            dclosure * compression_stress(strain_component);
        return std::max(tangent, 1.0e-6 * elastic_modulus_mpa_);
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
        double kappa) const noexcept
    {
        return fall_n::fracture::opening_exponential_shear_floor(
            shear_transfer_parameters(),
            {.opening_strain = std::max(kappa - crack_onset_strain(), 0.0),
             .normal_strain = std::max(kappa - crack_onset_strain(), 0.0),
             .damage = damage_from_kappa(kappa),
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

    [[nodiscard]] double crack_normal_strain(
        const KinematicT& strain,
        const InternalVariablesT& alpha) const
    {
        if (alpha.num_cracks > 0) {
            const Eigen::Vector3d n = alpha.crack_normals[0].normalized();
            return n.dot(strain_tensor_from_voigt(strain) * n);
        }
        return principal_strain_state(strain).max_principal_strain;
    }

    [[nodiscard]] double damaged_shear_modulus(
        double damage,
        double kappa,
        double normal_strain) const noexcept
    {
        const double opening = std::max(kappa - crack_onset_strain(), 0.0);
        const double retained_fraction =
            fall_n::fracture::retained_crack_shear_ratio(
                shear_transfer_parameters(),
                {.opening_strain = opening,
                 .normal_strain = normal_strain,
                 .damage = damage,
                 .crack_onset_strain = crack_onset_strain()});
        return retained_fraction * shear_modulus_mpa_;
    }

    [[nodiscard]] TangentT secant_tangent(
        const KinematicT& strain,
        const InternalVariablesT& alpha,
        double damage,
        double kappa) const
    {
        TangentT C = TangentT::Zero();
        C(0, 0) = normal_algorithmic_modulus(strain[0], damage, kappa);
        C(1, 1) = normal_algorithmic_modulus(strain[1], damage, kappa);
        C(2, 2) = normal_algorithmic_modulus(strain[2], damage, kappa);
        const double Gd =
            damaged_shear_modulus(damage, kappa, crack_normal_strain(strain, alpha));
        C(3, 3) = Gd;
        C(4, 4) = Gd;
        C(5, 5) = Gd;
        return C;
    }

    [[nodiscard]] static bool is_finite_vector(const Vec6& values) noexcept
    {
        return values.allFinite();
    }

    [[nodiscard]] static bool is_finite_matrix(const TangentT& values) noexcept
    {
        return values.allFinite();
    }

    [[nodiscard]] KinematicT perturbed_strain(
        const KinematicT& strain,
        std::size_t component,
        double perturbation) const
    {
        Vec6 values = strain.components();
        values(static_cast<Eigen::Index>(component)) += perturbation;
        KinematicT out;
        out.set_components(values);
        return out;
    }

    [[nodiscard]] Vec6 central_difference_column(
        const KinematicT& strain,
        const InternalVariablesT& alpha,
        std::size_t component,
        double step) const
    {
        const auto plus_stress = compute_response(
            perturbed_strain(strain, component, step),
            alpha);
        const auto minus_stress = compute_response(
            perturbed_strain(strain, component, -step),
            alpha);
        const Vec6 plus = plus_stress.components();
        const Vec6 minus = minus_stress.components();
        return (plus - minus) / (2.0 * step);
    }

    [[nodiscard]] TangentT adaptive_central_tangent(
        const KinematicT& strain,
        const InternalVariablesT& alpha,
        bool allow_secant_fallback) const
    {
        const double damage = trial_damage(strain, alpha);
        const double kappa = trial_kappa(strain, alpha);
        const TangentT secant = secant_tangent(strain, alpha, damage, kappa);
        TangentT tangent = secant;

        const double strain_scale_floor =
            std::max({crack_onset_strain(), 1.0e-6, alpha.kappa});
        const double rel_step =
            std::clamp(numerical_tangent_relative_step_, 1.0e-8, 1.0e-2);

        for (std::size_t j = 0; j < num_components; ++j) {
            const double strain_scale = std::max(
                std::abs(strain[static_cast<int>(j)]),
                strain_scale_floor);
            const double h = std::max(rel_step * strain_scale, 1.0e-10);
            const Vec6 coarse =
                central_difference_column(strain, alpha, j, h);
            const Vec6 fine =
                central_difference_column(strain, alpha, j, 0.5 * h);

            if (!is_finite_vector(coarse) || !is_finite_vector(fine)) {
                continue;
            }

            if (allow_secant_fallback) {
                const Vec6 secant_col =
                    secant.col(static_cast<Eigen::Index>(j));
                const double scale =
                    std::max({fine.norm(), coarse.norm(), secant_col.norm(), 1.0});
                const double relative_change =
                    (fine - coarse).norm() / scale;
                if (relative_change >
                    numerical_tangent_validation_tolerance_) {
                    continue;
                }
            }

            tangent.col(static_cast<Eigen::Index>(j)) = fine;
        }

        return is_finite_matrix(tangent) ? tangent : secant;
    }

    [[nodiscard]] Vec6 stress_components(
        const KinematicT& strain,
        const InternalVariablesT& alpha,
        double damage,
        double kappa) const
    {
        Vec6 stress = Vec6::Zero();
        stress(0) = normal_stress(strain[0], damage, kappa);
        stress(1) = normal_stress(strain[1], damage, kappa);
        stress(2) = normal_stress(strain[2], damage, kappa);
        const double Gd =
            damaged_shear_modulus(damage, kappa, crack_normal_strain(strain, alpha));
        stress(3) = Gd * strain[3];
        stress(4) = Gd * strain[4];
        stress(5) = Gd * strain[5];
        return stress;
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
        next.compression_strain_min =
            std::min(alpha.compression_strain_min, principal.values(0));

        const Vec6 stress =
            stress_components(strain, next, next.damage, next.kappa);
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
        const double sigma_o = std::max(stress_principal(2), 0.0);
        const double tau_o =
            0.5 * std::max(stress_principal(2) - stress_principal(0), 0.0);
        next.sigma_o_max = std::max(alpha.sigma_o_max, sigma_o);
        next.tau_o_max = std::max(alpha.tau_o_max, tau_o);

        return next;
    }

public:
    CyclicCrackBandConcrete3D()
    {
        initialize_kent_park_descent();
    }

    CyclicCrackBandConcrete3D(
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

    [[nodiscard]] double fpc_mpa() const noexcept { return fpc_mpa_; }

    [[nodiscard]] double elastic_modulus_mpa() const noexcept
    {
        return elastic_modulus_mpa_;
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

    [[nodiscard]] double tensile_cracking_strain() const noexcept
    {
        return crack_onset_strain();
    }

    [[nodiscard]] double peak_compressive_strain() const noexcept
    {
        return peak_compressive_strain_;
    }

    [[nodiscard]] double residual_compressive_strength_ratio() const noexcept
    {
        return residual_compressive_strength_ratio_;
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

    [[nodiscard]] double equivalent_crack_opening_mm(
        const InternalVariablesT& alpha) const noexcept
    {
        return std::max(alpha.kappa - crack_onset_strain(), 0.0) *
               characteristic_length_mm_;
    }

    [[nodiscard]] double retained_shear_stiffness_ratio(
        const InternalVariablesT& alpha) const noexcept
    {
        return std::max(
            fall_n::fracture::retained_crack_shear_floor(
                shear_transfer_parameters(),
                {.opening_strain =
                     std::max(alpha.kappa - crack_onset_strain(), 0.0),
                 .normal_strain =
                     std::max(alpha.kappa - crack_onset_strain(), 0.0),
                 .damage = alpha.damage,
                 .crack_onset_strain = crack_onset_strain()}),
            1.0 - alpha.damage);
    }

    [[nodiscard]] double shear_retention_decay_strain() const noexcept
    {
        return shear_retention_decay_strain_;
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
        stress.set_components(
            stress_components(
                strain,
                alpha,
                trial_damage(strain, alpha),
                trial_kappa(strain, alpha)));
        return stress;
    }

    [[nodiscard]] TangentT tangent(
        const KinematicT& strain,
        const InternalVariablesT& alpha) const
    {
        switch (tangent_mode_) {
        case CyclicCrackBandConcrete3DTangentMode::
            AdaptiveCentralDifference:
            return adaptive_central_tangent(strain, alpha, false);
        case CyclicCrackBandConcrete3DTangentMode::
            AdaptiveCentralDifferenceWithSecantFallback:
            return adaptive_central_tangent(strain, alpha, true);
        case CyclicCrackBandConcrete3DTangentMode::SecantPositive:
        default:
            return secant_tangent(
                strain,
                alpha,
                trial_damage(strain, alpha),
                trial_kappa(strain, alpha));
        }
    }

    void set_material_tangent_mode(
        CyclicCrackBandConcrete3DTangentMode mode) noexcept
    {
        tangent_mode_ = mode;
    }

    [[nodiscard]] CyclicCrackBandConcrete3DTangentMode
    material_tangent_mode() const noexcept
    {
        return tangent_mode_;
    }

    void set_consistent_tangent(bool enabled) noexcept
    {
        tangent_mode_ = enabled
            ? CyclicCrackBandConcrete3DTangentMode::
                  AdaptiveCentralDifferenceWithSecantFallback
            : CyclicCrackBandConcrete3DTangentMode::SecantPositive;
    }

    [[nodiscard]] bool uses_consistent_tangent() const noexcept
    {
        return tangent_mode_ !=
               CyclicCrackBandConcrete3DTangentMode::SecantPositive;
    }

    void set_numerical_tangent_relative_step(double step) noexcept
    {
        if (std::isfinite(step) && step > 0.0) {
            numerical_tangent_relative_step_ = step;
        }
    }

    void set_numerical_tangent_validation_tolerance(
        double tolerance) noexcept
    {
        if (std::isfinite(tolerance) && tolerance > 0.0) {
            numerical_tangent_validation_tolerance_ = tolerance;
        }
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
    ConstitutiveRelation<CyclicCrackBandConcrete3D>,
    "CyclicCrackBandConcrete3D must satisfy ConstitutiveRelation");

static_assert(
    InelasticConstitutiveRelation<CyclicCrackBandConcrete3D>,
    "CyclicCrackBandConcrete3D must satisfy InelasticConstitutiveRelation");

static_assert(
    ExternallyStateDrivenConstitutiveRelation<CyclicCrackBandConcrete3D>,
    "CyclicCrackBandConcrete3D must support external state");

#endif // FALL_N_CYCLIC_CRACK_BAND_CONCRETE_3D_HH
