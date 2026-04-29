#ifndef FALL_N_XFEM_COHESIVE_CRACK_LAW_HH
#define FALL_N_XFEM_COHESIVE_CRACK_LAW_HH

#include "CrackKinematics.hh"
#include "../fracture/CrackShearTransferLaw.hh"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fall_n::xfem {

enum class CohesiveCrackTangentMode {
    secant_positive,
    active_set_consistent,
    adaptive_central_difference,
    adaptive_central_difference_with_secant_fallback
};

struct BilinearCohesiveLawParameters {
    double normal_stiffness{1.0};
    double shear_stiffness{1.0};
    double tensile_strength{1.0};
    double fracture_energy{1.0};
    double mode_mixity_weight{1.0};
    double compression_stiffness{1.0};
    double residual_shear_fraction{0.0};
    double shear_traction_cap{std::numeric_limits<double>::infinity()};
    CohesiveCrackTangentMode tangent_mode{
        CohesiveCrackTangentMode::secant_positive};
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
    Eigen::Matrix3d tangent_stiffness = Eigen::Matrix3d::Zero();
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
        p.residual_shear_fraction < 0.0 ||
        p.residual_shear_fraction > 1.0 ||
        !(p.shear_traction_cap > 0.0)) {
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

[[nodiscard]] inline double cohesive_damage_history_derivative(
    const BilinearCohesiveLawParameters& p,
    double kappa)
{
    validate_bilinear_cohesive_law(p);

    const double delta0 = p.tensile_strength / p.normal_stiffness;
    const double deltaf = 2.0 * p.fracture_energy / p.tensile_strength;
    if (kappa <= delta0 || kappa >= deltaf) {
        return 0.0;
    }
    return deltaf * delta0 / ((deltaf - delta0) * kappa * kappa);
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

[[nodiscard]] inline double n_per_mm_to_mn_per_m(double value) noexcept
{
    return 1.0e-3 * value;
}

[[nodiscard]] inline BilinearCohesiveLawParameters
make_metre_jump_crack_band_consistent_cohesive_law_from_mpa_n_per_mm(
    double elastic_modulus_mpa,
    double shear_modulus_mpa,
    double tensile_strength_mpa,
    double fracture_energy_n_per_mm,
    double characteristic_length_m,
    double penalty_scale = 1.0,
    double mode_mixity_weight = 1.0,
    double compression_stiffness_scale = 1.0,
    double residual_shear_fraction = 0.0)
{
    // Global solid models in fall_n use coordinates and displacement jumps in
    // metres while retaining MPa as MN/m^2 for stresses. Therefore Gf must be
    // converted from the concrete-literature convention N/mm to MN/m. This
    // keeps delta_f = 2 Gf / f_t in metres and prevents an accidental 1000x
    // ductility/stiffness shift in XFEM interface laws.
    return make_crack_band_consistent_cohesive_law(
        elastic_modulus_mpa,
        shear_modulus_mpa,
        tensile_strength_mpa,
        n_per_mm_to_mn_per_m(fracture_energy_n_per_mm),
        characteristic_length_m,
        penalty_scale,
        mode_mixity_weight,
        compression_stiffness_scale,
        residual_shear_fraction);
}

struct RetainedShearRatioDifferential {
    double ratio{0.0};
    Eigen::Vector3d gradient = Eigen::Vector3d::Zero();
};

[[nodiscard]] inline RetainedShearRatioDifferential
retained_crack_shear_ratio_differential(
    const BilinearCohesiveLawParameters& p,
    const CrackJumpKinematics& split,
    double damage,
    const Eigen::Vector3d& damage_gradient)
{
    auto raw_shear_transfer = p.shear_transfer_law;
    raw_shear_transfer.residual_ratio =
        std::max(raw_shear_transfer.residual_ratio, p.residual_shear_fraction);
    raw_shear_transfer.large_opening_ratio = std::max(
        raw_shear_transfer.large_opening_ratio,
        p.residual_shear_fraction);
    const auto parameters =
        fall_n::fracture::bounded_crack_shear_transfer_parameters(
            raw_shear_transfer);

    const Eigen::Vector3d& n = split.normal;
    const double opening = std::max(split.normal_opening, 0.0);
    const Eigen::Vector3d opening_gradient =
        split.normal_opening > 0.0 ? n : Eigen::Vector3d::Zero();

    double open_floor = parameters.residual_ratio;
    Eigen::Vector3d open_floor_gradient = Eigen::Vector3d::Zero();
    if (parameters.kind !=
        fall_n::fracture::CrackShearTransferLawKind::constant_residual &&
        opening > 0.0 &&
        std::abs(parameters.large_opening_ratio -
                 parameters.residual_ratio) > 1.0e-14) {
        const double transition =
            std::exp(-opening / parameters.opening_decay_strain);
        open_floor =
            parameters.large_opening_ratio +
            (parameters.residual_ratio - parameters.large_opening_ratio) *
                transition;
        open_floor_gradient =
            (parameters.large_opening_ratio - parameters.residual_ratio) *
            transition / parameters.opening_decay_strain *
            opening_gradient;
    }

    double floor = open_floor;
    Eigen::Vector3d floor_gradient = open_floor_gradient;
    if (parameters.kind ==
            fall_n::fracture::CrackShearTransferLawKind::
                compression_gated_opening &&
        split.normal_opening < 0.0) {
        const double reference_floor =
            parameters.closure_reference_strain_multiplier *
            std::max(
                p.tensile_strength /
                    std::max(p.normal_stiffness, 1.0e-12),
                1.0e-14);
        const bool reference_tracks_opening = opening > reference_floor;
        const double reference =
            reference_tracks_opening ? opening : reference_floor;
        if (reference > 1.0e-14) {
            const Eigen::Vector3d reference_gradient =
                reference_tracks_opening ? opening_gradient
                                         : Eigen::Vector3d::Zero();
            const double q = (-split.normal_opening) / reference;
            const double t = std::clamp(q, 0.0, 1.0);
            const double closure = t * t * (3.0 - 2.0 * t);
            Eigen::Vector3d closure_gradient = Eigen::Vector3d::Zero();
            if (q > 0.0 && q < 1.0) {
                const Eigen::Vector3d q_gradient =
                    -n / reference +
                    split.normal_opening * reference_gradient /
                        (reference * reference);
                closure_gradient = (6.0 * t - 6.0 * t * t) * q_gradient;
            }

            const double uncapped_closed_target =
                open_floor +
                parameters.closure_shear_gain * (1.0 - open_floor);
            const bool target_capped =
                uncapped_closed_target >= parameters.max_closed_ratio;
            const double closed_target =
                target_capped ? parameters.max_closed_ratio
                              : uncapped_closed_target;
            Eigen::Vector3d closed_target_gradient =
                Eigen::Vector3d::Zero();
            if (!target_capped) {
                closed_target_gradient =
                    (1.0 - parameters.closure_shear_gain) *
                    open_floor_gradient;
            }
            floor =
                open_floor + (closed_target - open_floor) * closure;
            floor_gradient =
                open_floor_gradient +
                (closed_target_gradient - open_floor_gradient) * closure +
                (closed_target - open_floor) * closure_gradient;
        }
    }

    const double intact_ratio = std::clamp(1.0 - damage, 1.0e-8, 1.0);
    Eigen::Vector3d intact_gradient = Eigen::Vector3d::Zero();
    if (1.0 - damage > 1.0e-8 && 1.0 - damage < 1.0) {
        intact_gradient = -damage_gradient;
    }
    if (floor >= intact_ratio) {
        return {.ratio = floor, .gradient = floor_gradient};
    }
    return {.ratio = intact_ratio, .gradient = intact_gradient};
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
    const double slip = split.tangential_jump.norm();
    const double shear_cap_scale =
        std::isfinite(p.shear_traction_cap) && slip > 1.0e-14
            ? std::min(shear_scale, p.shear_traction_cap / slip)
            : shear_scale;

    out.traction =
        normal_scale * split.normal_opening * n +
        shear_cap_scale * split.tangential_jump;
    out.secant_stiffness =
        normal_scale * (n * n.transpose()) +
        shear_cap_scale *
            (Eigen::Matrix3d::Identity() - n * n.transpose());
    out.tangent_stiffness = out.secant_stiffness;

    if (p.tangent_mode == CohesiveCrackTangentMode::active_set_consistent) {
        const Eigen::Matrix3d P =
            Eigen::Matrix3d::Identity() - n * n.transpose();
        const double loading_tol = 1.0e-14 * std::max(
            {out.updated_state.max_effective_separation,
             out.effective_separation,
             1.0});
        const bool damage_loading =
            out.effective_separation + loading_tol >=
            state.max_effective_separation;
        Eigen::Vector3d damage_gradient = Eigen::Vector3d::Zero();
        if (damage_loading && out.effective_separation > 1.0e-14) {
            const double positive_opening =
                std::max(split.normal_opening, 0.0);
            const Eigen::Vector3d separation_gradient =
                (positive_opening / out.effective_separation) * n +
                (p.mode_mixity_weight * p.mode_mixity_weight /
                 out.effective_separation) *
                    split.tangential_jump;
            damage_gradient =
                cohesive_damage_history_derivative(
                    p,
                    out.updated_state.max_effective_separation) *
                separation_gradient;
        }

        if (split.normal_opening >= 0.0) {
            const Eigen::Vector3d normal_scalar_gradient =
                p.normal_stiffness *
                ((1.0 - out.damage) * n -
                 split.normal_opening * damage_gradient);
            out.tangent_stiffness =
                n * normal_scalar_gradient.transpose();
        } else {
            out.tangent_stiffness =
                p.compression_stiffness * (n * n.transpose());
        }

        const bool shear_cap_active =
            std::isfinite(p.shear_traction_cap) &&
            slip > 1.0e-14 &&
            p.shear_traction_cap / slip < shear_scale;
        if (shear_cap_active) {
            out.tangent_stiffness +=
                (p.shear_traction_cap / slip) * P -
                (p.shear_traction_cap / (slip * slip * slip)) *
                    (split.tangential_jump *
                     split.tangential_jump.transpose());
        } else {
            const auto shear_differential =
                retained_crack_shear_ratio_differential(
                    p,
                    split,
                    out.damage,
                    damage_gradient);
            out.tangent_stiffness +=
                p.shear_stiffness *
                (shear_differential.ratio * P +
                 split.tangential_jump *
                     shear_differential.gradient.transpose());
        }
    } else if (p.tangent_mode != CohesiveCrackTangentMode::secant_positive) {
        auto secant_parameters = p;
        secant_parameters.tangent_mode =
            CohesiveCrackTangentMode::secant_positive;

        const Eigen::Vector3d jump =
            split.normal_opening * n + split.tangential_jump;
        const double h = std::max(
            1.0e-9,
            1.0e-6 * std::max(
                {jump.norm(),
                 p.tensile_strength /
                     std::max(p.normal_stiffness, 1.0e-12),
                 1.0e-9}));
        Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
        bool finite = true;
        for (Eigen::Index column = 0; column < 3; ++column) {
            Eigen::Vector3d perturb = Eigen::Vector3d::Zero();
            perturb[column] = h;
            const auto plus_split =
                split_crack_jump(crack_normal, jump + perturb);
            const auto minus_split =
                split_crack_jump(crack_normal, jump - perturb);
            const auto plus = evaluate_bilinear_cohesive_law(
                secant_parameters,
                state,
                crack_normal,
                plus_split.normal_opening,
                plus_split.tangential_jump);
            const auto minus = evaluate_bilinear_cohesive_law(
                secant_parameters,
                state,
                crack_normal,
                minus_split.normal_opening,
                minus_split.tangential_jump);
            const Eigen::Vector3d derivative =
                (plus.traction - minus.traction) / (2.0 * h);
            finite = finite && derivative.allFinite();
            K.col(column) = derivative;
        }

        if (finite ||
            p.tangent_mode ==
                CohesiveCrackTangentMode::adaptive_central_difference) {
            out.tangent_stiffness = K;
        }
    }
    return out;
}

[[nodiscard]] inline CohesiveCrackState advance_bilinear_cohesive_state(
    const CohesiveCrackResponse& response) noexcept
{
    return response.updated_state;
}

} // namespace fall_n::xfem

#endif // FALL_N_XFEM_COHESIVE_CRACK_LAW_HH
