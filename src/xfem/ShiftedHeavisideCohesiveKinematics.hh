#ifndef FALL_N_XFEM_SHIFTED_HEAVISIDE_COHESIVE_KINEMATICS_HH
#define FALL_N_XFEM_SHIFTED_HEAVISIDE_COHESIVE_KINEMATICS_HH

#include "ShiftedHeavisideKinematicPolicy.hh"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <stdexcept>
#include <string_view>

namespace fall_n::xfem {

enum class ShiftedHeavisideCohesiveSurfaceMeasureKind {
    reference_area_reference_normal,
    corotated_reference_area,
    nanson_current_area
};

enum class ShiftedHeavisideCohesiveTractionMeasureKind {
    reference_nominal,
    current_spatial,
    audit_dual
};

struct ShiftedHeavisideCohesiveSurfaceKinematics {
    Eigen::Vector3d reference_normal = Eigen::Vector3d::UnitZ();
    Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
    double area_scale{1.0};
    ShiftedHeavisideCohesiveSurfaceMeasureKind measure_kind{
        ShiftedHeavisideCohesiveSurfaceMeasureKind::
            reference_area_reference_normal};
    bool geometric_tangent_is_complete{true};
    bool finite_measure_guard_required{false};
};

struct ShiftedHeavisideCohesiveSurfaceDifferential {
    Eigen::Vector3d normal_directional_derivative =
        Eigen::Vector3d::Zero();
    double area_scale_directional_derivative{0.0};
    bool includes_surface_measure_derivative{true};
};

struct ShiftedHeavisideCohesiveTractionMeasure {
    Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
    double area_scale{1.0};
    ShiftedHeavisideCohesiveTractionMeasureKind traction_kind{
        ShiftedHeavisideCohesiveTractionMeasureKind::current_spatial};
    bool uses_reference_surface{false};
    bool audits_dual_work{false};
};

struct ShiftedHeavisideCohesiveSurfaceWorkAudit {
    double reference_work_density{0.0};
    double spatial_work_density{0.0};
    double absolute_gap{0.0};
    double relative_gap{0.0};
};

[[nodiscard]] inline std::string_view to_string(
    ShiftedHeavisideCohesiveSurfaceMeasureKind kind) noexcept
{
    switch (kind) {
        case ShiftedHeavisideCohesiveSurfaceMeasureKind::
            reference_area_reference_normal:
            return "reference_area_reference_normal";
        case ShiftedHeavisideCohesiveSurfaceMeasureKind::
            corotated_reference_area:
            return "corotated_reference_area";
        case ShiftedHeavisideCohesiveSurfaceMeasureKind::nanson_current_area:
            return "nanson_current_area";
    }
    return "unknown";
}

[[nodiscard]] inline std::string_view to_string(
    ShiftedHeavisideCohesiveTractionMeasureKind kind) noexcept
{
    switch (kind) {
        case ShiftedHeavisideCohesiveTractionMeasureKind::reference_nominal:
            return "reference-nominal";
        case ShiftedHeavisideCohesiveTractionMeasureKind::current_spatial:
            return "current-spatial";
        case ShiftedHeavisideCohesiveTractionMeasureKind::audit_dual:
            return "audit-dual";
    }
    return "unknown";
}

[[nodiscard]] inline Eigen::Vector3d normalized_cohesive_normal(
    const Eigen::Vector3d& raw_normal)
{
    const double norm = raw_normal.norm();
    if (!(norm > 0.0)) {
        throw std::invalid_argument(
            "Shifted-Heaviside cohesive surface needs a non-zero normal.");
    }
    return raw_normal / norm;
}

[[nodiscard]] inline ShiftedHeavisideCohesiveTractionMeasure
select_shifted_heaviside_cohesive_traction_measure(
    ShiftedHeavisideCohesiveTractionMeasureKind kind,
    const ShiftedHeavisideCohesiveSurfaceKinematics& surface)
{
    const auto normalized_reference =
        normalized_cohesive_normal(surface.reference_normal);

    if (kind ==
        ShiftedHeavisideCohesiveTractionMeasureKind::reference_nominal) {
        return {
            .normal = normalized_reference,
            .area_scale = 1.0,
            .traction_kind = kind,
            .uses_reference_surface = true,
            .audits_dual_work = false};
    }
    if (kind == ShiftedHeavisideCohesiveTractionMeasureKind::audit_dual) {
        return {
            .normal = normalized_reference,
            .area_scale = 1.0,
            .traction_kind = kind,
            .uses_reference_surface = true,
            .audits_dual_work = true};
    }
    return {
        .normal = normalized_cohesive_normal(surface.normal),
        .area_scale = surface.area_scale,
        .traction_kind = kind,
        .uses_reference_surface = false,
        .audits_dual_work = false};
}

[[nodiscard]] inline ShiftedHeavisideCohesiveSurfaceWorkAudit
evaluate_shifted_heaviside_cohesive_surface_work_audit(
    const Eigen::Vector3d& reference_traction,
    const Eigen::Vector3d& spatial_traction,
    const Eigen::Vector3d& virtual_jump,
    double spatial_area_scale)
{
    const double reference_work = reference_traction.dot(virtual_jump);
    const double spatial_work =
        std::max(spatial_area_scale, 0.0) *
        spatial_traction.dot(virtual_jump);
    const double gap = spatial_work - reference_work;
    const double scale =
        std::max({1.0, std::abs(reference_work), std::abs(spatial_work)});
    return {
        .reference_work_density = reference_work,
        .spatial_work_density = spatial_work,
        .absolute_gap = std::abs(gap),
        .relative_gap = std::abs(gap) / scale};
}

[[nodiscard]] inline ShiftedHeavisideCohesiveSurfaceDifferential
evaluate_nanson_cohesive_surface_differential(
    const Eigen::Vector3d& reference_normal,
    const continuum::GPKinematics<3>& gp,
    const Eigen::Matrix3d& dF)
{
    const Eigen::Vector3d n0 = normalized_cohesive_normal(reference_normal);
    if (!(gp.detF > 0.0)) {
        throw std::runtime_error(
            "Shifted-Heaviside Nanson differential received det(F) <= 0.");
    }

    const Eigen::Matrix3d F_inv = gp.F.matrix().inverse();
    const Eigen::Matrix3d F_inv_T = F_inv.transpose();
    const Eigen::Vector3d area_vector = gp.detF * F_inv_T * n0;
    const double area_scale = area_vector.norm();
    if (!(area_scale > 0.0)) {
        throw std::runtime_error(
            "Shifted-Heaviside Nanson differential produced zero area.");
    }

    const Eigen::Vector3d normal = area_vector / area_scale;
    const double dJ = gp.detF * (F_inv * dF).trace();
    const Eigen::Matrix3d dF_inv_T = -F_inv_T * dF.transpose() * F_inv_T;
    const Eigen::Vector3d d_area_vector =
        dJ * F_inv_T * n0 + gp.detF * dF_inv_T * n0;
    const double d_area_scale = normal.dot(d_area_vector);
    const Eigen::Vector3d d_normal =
        (Eigen::Matrix3d::Identity() - normal * normal.transpose()) *
        d_area_vector / area_scale;

    return {
        .normal_directional_derivative = d_normal,
        .area_scale_directional_derivative = d_area_scale,
        .includes_surface_measure_derivative = true};
}

template <typename KinematicPolicy>
    requires ShiftedHeavisideKinematicPolicy<KinematicPolicy>
[[nodiscard]] inline ShiftedHeavisideCohesiveSurfaceDifferential
evaluate_shifted_heaviside_cohesive_surface_differential(
    const Eigen::Vector3d& reference_normal,
    const continuum::GPKinematics<3>& gp,
    const Eigen::Matrix3d& dF)
{
    if constexpr (std::same_as<KinematicPolicy,
                               continuum::TotalLagrangian> ||
                  std::same_as<KinematicPolicy,
                               continuum::UpdatedLagrangian>) {
        return evaluate_nanson_cohesive_surface_differential(
            reference_normal,
            gp,
            dF);
    } else {
        (void)reference_normal;
        (void)gp;
        (void)dF;
        return {
            .normal_directional_derivative = Eigen::Vector3d::Zero(),
            .area_scale_directional_derivative = 0.0,
            .includes_surface_measure_derivative = false};
    }
}

template <typename KinematicPolicy>
    requires ShiftedHeavisideKinematicPolicy<KinematicPolicy>
[[nodiscard]] inline ShiftedHeavisideCohesiveSurfaceKinematics
evaluate_shifted_heaviside_cohesive_surface_kinematics(
    const Eigen::Vector3d& reference_normal,
    const continuum::GPKinematics<3>& gp)
{
    const Eigen::Vector3d n0 = normalized_cohesive_normal(reference_normal);

    if constexpr (std::same_as<KinematicPolicy, continuum::Corotational>) {
        const Eigen::Vector3d n =
            gp.corotational_rotation.matrix() * n0;
        return {
            .reference_normal = n0,
            .normal = normalized_cohesive_normal(n),
            .area_scale = 1.0,
            .measure_kind = ShiftedHeavisideCohesiveSurfaceMeasureKind::
                corotated_reference_area,
            .geometric_tangent_is_complete = false,
            .finite_measure_guard_required = false};
    } else if constexpr (std::same_as<KinematicPolicy,
                                      continuum::TotalLagrangian> ||
                         std::same_as<KinematicPolicy,
                                      continuum::UpdatedLagrangian>) {
        if (!(gp.detF > 0.0)) {
            throw std::runtime_error(
                "Shifted-Heaviside finite cohesive surface received det(F) <= 0.");
        }
        const Eigen::Vector3d area_vector =
            gp.detF * gp.F.matrix().inverse().transpose() * n0;
        const double area_scale = area_vector.norm();
        if (!(area_scale > 0.0)) {
            throw std::runtime_error(
                "Shifted-Heaviside finite cohesive surface produced zero area.");
        }
        return {
            .reference_normal = n0,
            .normal = area_vector / area_scale,
            .area_scale = area_scale,
            .measure_kind = ShiftedHeavisideCohesiveSurfaceMeasureKind::
                nanson_current_area,
            .geometric_tangent_is_complete = false,
            .finite_measure_guard_required = true};
    } else {
        return {
            .reference_normal = n0,
            .normal = n0,
            .area_scale = 1.0,
            .measure_kind = ShiftedHeavisideCohesiveSurfaceMeasureKind::
                reference_area_reference_normal,
            .geometric_tangent_is_complete = true,
            .finite_measure_guard_required = false};
    }
}

} // namespace fall_n::xfem

#endif // FALL_N_XFEM_SHIFTED_HEAVISIDE_COHESIVE_KINEMATICS_HH
