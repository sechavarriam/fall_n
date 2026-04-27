#ifndef FALL_N_XFEM_COHESIVE_CRACK_LAW_HH
#define FALL_N_XFEM_COHESIVE_CRACK_LAW_HH

#include "CrackKinematics.hh"
#include "../fracture/CrackShearTransferLaw.hh"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fall_n::xfem {

struct BilinearCohesiveLawParameters {
    double normal_stiffness{1.0};
    double shear_stiffness{1.0};
    double tensile_strength{1.0};
    double fracture_energy{1.0};
    double mode_mixity_weight{1.0};
    double compression_stiffness{1.0};
    double residual_shear_fraction{0.0};
    fall_n::fracture::CrackShearTransferLawParameters shear_transfer_law{
        .kind = fall_n::fracture::CrackShearTransferLawKind::
            opening_exponential,
        .residual_ratio = 0.0,
        .large_opening_ratio = 0.0,
        .opening_decay_strain = 1.0,
        .closure_shear_gain = 1.0,
        .max_closed_ratio = 1.0,
        .closure_reference_strain_multiplier = 1.0};
};

struct CohesiveCrackState {
    double max_effective_separation{0.0};
};

struct CohesiveCrackResponse {
    Eigen::Vector3d traction = Eigen::Vector3d::Zero();
    Eigen::Matrix3d secant_stiffness = Eigen::Matrix3d::Zero();
    CohesiveCrackState updated_state{};
    double effective_separation{0.0};
    double damage{0.0};
};

inline void validate_bilinear_cohesive_law(
    const BilinearCohesiveLawParameters& p)
{
    if (p.normal_stiffness <= 0.0 || p.shear_stiffness <= 0.0 ||
        p.tensile_strength <= 0.0 || p.fracture_energy <= 0.0 ||
        p.compression_stiffness <= 0.0 || p.mode_mixity_weight < 0.0 ||
        p.residual_shear_fraction < 0.0 || p.residual_shear_fraction > 1.0) {
        throw std::invalid_argument(
            "BilinearCohesiveLawParameters requires positive stiffness, "
            "positive strength/energy, non-negative mode mixity, and a "
            "residual shear fraction in [0,1].");
    }

    const double delta0 = p.tensile_strength / p.normal_stiffness;
    const double deltaf = 2.0 * p.fracture_energy / p.tensile_strength;
    if (!(delta0 < deltaf)) {
        throw std::invalid_argument(
            "Bilinear cohesive law needs delta0 < deltaf. Increase fracture "
            "energy or reduce initial stiffness/strength.");
    }
}

[[nodiscard]] inline double cohesive_damage_from_history(
    const BilinearCohesiveLawParameters& p,
    double kappa)
{
    validate_bilinear_cohesive_law(p);

    const double delta0 = p.tensile_strength / p.normal_stiffness;
    const double deltaf = 2.0 * p.fracture_energy / p.tensile_strength;
    if (kappa <= delta0) {
        return 0.0;
    }
    if (kappa >= deltaf) {
        return 1.0;
    }

    // Standard scalar cohesive damage: t = (1-d) K delta follows the linear
    // softening envelope from tensile_strength at delta0 to zero at deltaf.
    return deltaf * (kappa - delta0) / (kappa * (deltaf - delta0));
}

[[nodiscard]] inline BilinearCohesiveLawParameters
make_crack_band_consistent_cohesive_law(
    double elastic_modulus,
    double shear_modulus,
    double tensile_strength,
    double fracture_energy,
    double characteristic_length,
    double penalty_scale = 1.0,
    double mode_mixity_weight = 1.0,
    double compression_stiffness_scale = 1.0,
    double residual_shear_fraction = 0.0)
{
    const double length = std::max(characteristic_length, 1.0e-12);
    return {
        .normal_stiffness =
            std::max(penalty_scale * elastic_modulus / length, 1.0e-12),
        .shear_stiffness =
            std::max(penalty_scale * shear_modulus / length, 1.0e-12),
        .tensile_strength = tensile_strength,
        .fracture_energy = fracture_energy,
        .mode_mixity_weight = mode_mixity_weight,
        .compression_stiffness = std::max(
            compression_stiffness_scale * elastic_modulus / length,
            1.0e-12),
        .residual_shear_fraction = residual_shear_fraction,
    };
}

[[nodiscard]] inline CohesiveCrackResponse evaluate_bilinear_cohesive_law(
    const BilinearCohesiveLawParameters& p,
    const CohesiveCrackState& state,
    const Eigen::Vector3d& crack_normal,
    double normal_opening,
    const Eigen::Vector3d& tangential_jump)
{
    validate_bilinear_cohesive_law(p);
    const auto split = split_crack_jump(
        crack_normal,
        normal_opening * normalized_crack_normal(crack_normal) +
            tangential_jump);
    const Eigen::Vector3d& n = split.normal;

    CohesiveCrackResponse out{};
    out.effective_separation = effective_mixed_mode_separation(
        split.normal_opening,
        split.tangential_jump,
        p.mode_mixity_weight);
    out.updated_state.max_effective_separation = std::max(
        state.max_effective_separation,
        out.effective_separation);
    out.damage = cohesive_damage_from_history(
        p,
        out.updated_state.max_effective_separation);

    const double normal_scale =
        split.normal_opening >= 0.0 ? (1.0 - out.damage) * p.normal_stiffness
                                    : p.compression_stiffness;
    auto shear_transfer = p.shear_transfer_law;
    shear_transfer.residual_ratio =
        std::max(shear_transfer.residual_ratio, p.residual_shear_fraction);
    shear_transfer.large_opening_ratio = std::max(
        shear_transfer.large_opening_ratio,
        p.residual_shear_fraction);
    const double shear_ratio =
        fall_n::fracture::retained_crack_shear_ratio(
            shear_transfer,
            {.opening_strain = std::max(split.normal_opening, 0.0),
             .normal_strain = split.normal_opening,
             .damage = out.damage,
             .crack_onset_strain =
                 p.tensile_strength / std::max(p.normal_stiffness, 1.0e-12)});
    const double shear_scale = shear_ratio * p.shear_stiffness;

    out.traction =
        normal_scale * split.normal_opening * n +
        shear_scale * split.tangential_jump;
    out.secant_stiffness =
        normal_scale * (n * n.transpose()) +
        shear_scale * (Eigen::Matrix3d::Identity() - n * n.transpose());
    return out;
}

[[nodiscard]] inline CohesiveCrackState advance_bilinear_cohesive_state(
    const CohesiveCrackResponse& response) noexcept
{
    return response.updated_state;
}

} // namespace fall_n::xfem

#endif // FALL_N_XFEM_COHESIVE_CRACK_LAW_HH
