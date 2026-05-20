#ifndef FALL_N_REDUCED_RC_MACRO_INFERRED_XFEM_SITE_POLICY_HH
#define FALL_N_REDUCED_RC_MACRO_INFERRED_XFEM_SITE_POLICY_HH

// =============================================================================
//  ReducedRCMacroInferredXfemSitePolicy.hh
// =============================================================================
//
//  Deterministic bridge from structural-scale demand indicators to the managed
//  XFEM local patch used by the FE2 validation campaign.  The macro model does
//  not create one local model per integration point; it selects a physical site
//  and provides enough information to place the local crack band and to bias the
//  independent local mesh toward the likely plastic-hinge region(s).
//
// =============================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string_view>
#include <vector>

#include "src/validation/ReducedRCManagedLocalModelReplay.hh"
#include "src/validation/ReducedRCMultiscaleReplayPlan.hh"

namespace fall_n {

struct ReducedRCMacroEndpointDemand {
    double fixed_end_score{0.0};
    double loaded_end_score{0.0};
    double macro_section_z_over_l{std::numeric_limits<double>::quiet_NaN()};
};

struct ReducedRCMacroInferredCrackPlaneDemand {
    double axial_strain{0.0};
    double curvature_y{0.0};
    double curvature_z{0.0};
    double section_width_m{0.20};
    double section_depth_m{0.20};
    double characteristic_length_m{0.10};
    double crack_z_over_l{0.40};
    std::size_t nz{4};
    int plane_id{1};
    int sequence_id{1};
    int activation_step{0};
    double activation_time{0.0};
};

struct ReducedRCMacroInferredXfemSitePolicy {
    double active_endpoint_score{1.0};
    double minimum_crack_z_over_l{0.05};
    double maximum_crack_z_over_l{0.95};
    double fallback_fixed_end_crack_z_over_l{0.10};
    double fallback_loaded_end_crack_z_over_l{0.90};
    double fallback_double_hinge_crack_z_over_l{0.50};
    double single_hinge_bias_power{1.60};
    double double_hinge_bias_power{2.20};
    bool preserve_macro_section_crack_location{true};
};

struct ReducedRCMacroInferredLocalSiteCandidate {
    double z_over_l{0.05};
    double score{0.0};
    ReducedRCLocalLongitudinalBiasLocation bias_location{
        ReducedRCLocalLongitudinalBiasLocation::fixed_end};
    std::string_view reason{"fixed_end_score"};
};

struct ReducedRCMacroInferredLocalSiteSelectionPolicy {
    double fixed_end_z_over_l{0.05};
    double center_z_over_l{0.50};
    double loaded_end_z_over_l{0.95};
    double active_endpoint_score{1.0};
    double loaded_end_relative_score{0.75};
    bool include_center_when_both_active{true};
    bool include_inactive_control_sites{false};
    bool include_center_control_site{false};
};

[[nodiscard]] inline std::vector<ReducedRCMacroInferredLocalSiteCandidate>
infer_reduced_rc_macro_local_site_candidates(
    const ReducedRCMacroEndpointDemand& demand,
    const ReducedRCMacroInferredLocalSiteSelectionPolicy& policy = {})
{
    std::vector<ReducedRCMacroInferredLocalSiteCandidate> candidates;
    candidates.reserve(3);

    const auto add_candidate =
        [&](double z_over_l,
            double score,
            ReducedRCLocalLongitudinalBiasLocation bias_location,
            std::string_view reason) {
            if (!(score > 0.0) || !std::isfinite(score)) {
                return;
            }
            const double z = std::clamp(z_over_l, 0.0, 1.0);
            const auto duplicate = std::find_if(
                candidates.begin(),
                candidates.end(),
                [z](const ReducedRCMacroInferredLocalSiteCandidate& c) {
                    return std::abs(c.z_over_l - z) < 1.0e-12;
                });
            if (duplicate != candidates.end()) {
                if (score > duplicate->score) {
                    duplicate->score = score;
                    duplicate->bias_location = bias_location;
                    duplicate->reason = reason;
                }
                return;
            }
            candidates.push_back(ReducedRCMacroInferredLocalSiteCandidate{
                .z_over_l = z,
                .score = score,
                .bias_location = bias_location,
                .reason = reason,
            });
        };

    const double fixed_score = demand.fixed_end_score;
    const double loaded_score = demand.loaded_end_score;
    const double dominant_score = std::max(fixed_score, loaded_score);

    add_candidate(policy.fixed_end_z_over_l,
                  fixed_score,
                  ReducedRCLocalLongitudinalBiasLocation::fixed_end,
                  "fixed_end_score");

    const bool loaded_active = loaded_score >= policy.active_endpoint_score;
    const bool loaded_near_dominant =
        loaded_score >=
        policy.loaded_end_relative_score * std::max(dominant_score, 1.0e-12);
    if (loaded_active || loaded_near_dominant) {
        add_candidate(policy.loaded_end_z_over_l,
                      loaded_score,
                      ReducedRCLocalLongitudinalBiasLocation::loaded_end,
                      "loaded_end_score");
    }

    const bool fixed_active = fixed_score >= policy.active_endpoint_score;
    if (policy.include_center_when_both_active && fixed_active &&
        loaded_active) {
        add_candidate(policy.center_z_over_l,
                      std::min(fixed_score, loaded_score),
                      ReducedRCLocalLongitudinalBiasLocation::both_ends,
                      "both_ends_center_probe");
    }

    if (policy.include_inactive_control_sites) {
        add_candidate(policy.loaded_end_z_over_l,
                      loaded_score,
                      ReducedRCLocalLongitudinalBiasLocation::loaded_end,
                      "loaded_end_control_from_macro_score");
        if (policy.include_center_control_site) {
            const double center_score = 0.5 * (fixed_score + loaded_score);
            add_candidate(policy.center_z_over_l,
                          center_score,
                          ReducedRCLocalLongitudinalBiasLocation::both_ends,
                          "center_control_from_macro_scores");
        }
    }

    std::sort(candidates.begin(),
              candidates.end(),
              [](const ReducedRCMacroInferredLocalSiteCandidate& a,
                 const ReducedRCMacroInferredLocalSiteCandidate& b) {
                  if (std::abs(a.score - b.score) > 1.0e-12) {
                      return a.score > b.score;
                  }
                  return a.z_over_l < b.z_over_l;
              });
    return candidates;
}

[[nodiscard]] inline std::vector<ReducedRCMacroInferredLocalSiteCandidate>
infer_reduced_rc_macro_whole_element_site_candidates(
    const ReducedRCMacroEndpointDemand& demand,
    const ReducedRCMacroInferredLocalSiteSelectionPolicy& policy = {})
{
    const double fixed_score = demand.fixed_end_score;
    const double loaded_score = demand.loaded_end_score;
    const double dominant_score = std::max(fixed_score, loaded_score);
    if (!(dominant_score > 0.0) || !std::isfinite(dominant_score)) {
        return {};
    }

    const bool fixed_active = fixed_score >= policy.active_endpoint_score;
    const bool loaded_active = loaded_score >= policy.active_endpoint_score;
    const bool paired_end_demand =
        (fixed_active && loaded_active) ||
        (std::min(fixed_score, loaded_score) >=
         policy.loaded_end_relative_score * std::max(dominant_score, 1.0e-12));

    double z_over_l = policy.center_z_over_l;
    std::string_view reason = "whole_element_both_ends";
    if (!paired_end_demand) {
        if (fixed_score >= loaded_score) {
            z_over_l = policy.fixed_end_z_over_l;
            reason = "whole_element_fixed_end_dominant";
        } else {
            z_over_l = policy.loaded_end_z_over_l;
            reason = "whole_element_loaded_end_dominant";
        }
    }

    return {ReducedRCMacroInferredLocalSiteCandidate{
        .z_over_l = std::clamp(z_over_l, 0.0, 1.0),
        .score = dominant_score,
        .bias_location = ReducedRCLocalLongitudinalBiasLocation::both_ends,
        .reason = reason,
    }};
}

[[nodiscard]] inline ReducedRCLocalLongitudinalBiasLocation
infer_reduced_rc_xfem_bias_location(
    const ReducedRCMacroEndpointDemand& demand,
    const ReducedRCMacroInferredXfemSitePolicy& policy = {}) noexcept
{
    const bool fixed_active =
        demand.fixed_end_score >= policy.active_endpoint_score;
    const bool loaded_active =
        demand.loaded_end_score >= policy.active_endpoint_score;
    if (fixed_active && loaded_active) {
        return ReducedRCLocalLongitudinalBiasLocation::both_ends;
    }
    if (loaded_active) {
        return ReducedRCLocalLongitudinalBiasLocation::loaded_end;
    }
    if (fixed_active) {
        return ReducedRCLocalLongitudinalBiasLocation::fixed_end;
    }
    return demand.loaded_end_score > demand.fixed_end_score
        ? ReducedRCLocalLongitudinalBiasLocation::loaded_end
        : ReducedRCLocalLongitudinalBiasLocation::fixed_end;
}

[[nodiscard]] inline double infer_reduced_rc_xfem_crack_z_over_l(
    const ReducedRCMacroEndpointDemand& demand,
    ReducedRCLocalLongitudinalBiasLocation bias_location,
    const ReducedRCMacroInferredXfemSitePolicy& policy = {}) noexcept
{
    const auto clamp_ratio = [&](double value) noexcept {
        return std::clamp(value,
                          policy.minimum_crack_z_over_l,
                          policy.maximum_crack_z_over_l);
    };

    if (policy.preserve_macro_section_crack_location &&
        std::isfinite(demand.macro_section_z_over_l)) {
        return clamp_ratio(demand.macro_section_z_over_l);
    }

    switch (bias_location) {
        case ReducedRCLocalLongitudinalBiasLocation::fixed_end:
            return clamp_ratio(policy.fallback_fixed_end_crack_z_over_l);
        case ReducedRCLocalLongitudinalBiasLocation::loaded_end:
            return clamp_ratio(policy.fallback_loaded_end_crack_z_over_l);
        case ReducedRCLocalLongitudinalBiasLocation::both_ends:
            return clamp_ratio(policy.fallback_double_hinge_crack_z_over_l);
    }
    return clamp_ratio(policy.fallback_fixed_end_crack_z_over_l);
}

[[nodiscard]] inline ReducedRCManagedLocalPatchSpec
make_reduced_rc_macro_inferred_xfem_patch(
    const ReducedRCMultiscaleReplaySitePlan& site,
    ReducedRCManagedLocalPatchSpec base_patch = {},
    ReducedRCMacroEndpointDemand endpoint_demand = {},
    const ReducedRCMacroInferredXfemSitePolicy& policy = {}) noexcept
{
    base_patch.site_index = site.site_index;
    base_patch.z_over_l = site.z_over_l;
    endpoint_demand.macro_section_z_over_l =
        std::isfinite(endpoint_demand.macro_section_z_over_l)
            ? endpoint_demand.macro_section_z_over_l
            : site.z_over_l;

    if (endpoint_demand.fixed_end_score <= 0.0 &&
        endpoint_demand.loaded_end_score <= 0.0) {
        const double activation = std::max(0.0, site.activation_score);
        if (site.z_over_l <= 0.5) {
            endpoint_demand.fixed_end_score = activation;
        } else {
            endpoint_demand.loaded_end_score = activation;
        }
    }

    const auto bias_location =
        infer_reduced_rc_xfem_bias_location(endpoint_demand, policy);
    base_patch.longitudinal_bias_location = bias_location;
    base_patch.double_hinge_bias_inferred_from_macro =
        bias_location == ReducedRCLocalLongitudinalBiasLocation::both_ends;
    base_patch.longitudinal_bias_power =
        base_patch.double_hinge_bias_inferred_from_macro
            ? policy.double_hinge_bias_power
            : policy.single_hinge_bias_power;
    base_patch.crack_z_over_l = infer_reduced_rc_xfem_crack_z_over_l(
        endpoint_demand,
        bias_location,
        policy);
    base_patch.crack_position_inferred_from_macro = true;
    return base_patch;
}

[[nodiscard]] inline double
adjust_reduced_rc_macro_inferred_crack_z_over_l(
    double requested_ratio,
    std::size_t nz) noexcept
{
    double ratio = std::clamp(requested_ratio, 0.0, 1.0);
    if (nz == 0) {
        return ratio;
    }
    const double scaled = ratio * static_cast<double>(nz);
    const double nearest = std::round(scaled);
    constexpr double tol = 1.0e-10;
    if (std::abs(scaled - nearest) > tol) {
        return ratio;
    }
    const double half_cell = 0.5 / static_cast<double>(nz);
    if (ratio + half_cell < 1.0 - tol) {
        ratio += half_cell;
    } else if (ratio - half_cell > tol) {
        ratio -= half_cell;
    }
    return std::clamp(ratio, 0.0, 1.0);
}

[[nodiscard]] inline ReducedRCManagedLocalCrackPlaneSpec
make_reduced_rc_macro_inferred_crack_plane(
    const ReducedRCMacroInferredCrackPlaneDemand& demand) noexcept
{
    const double half_w = 0.5 * std::max(0.0, demand.section_width_m);
    const double half_h = 0.5 * std::max(0.0, demand.section_depth_m);
    std::array<std::array<double, 2>, 4> corners{{
        {-half_w, -half_h},
        { half_w, -half_h},
        {-half_w,  half_h},
        { half_w,  half_h},
    }};

    double max_tensile_strain = -std::numeric_limits<double>::infinity();
    for (const auto& xy : corners) {
        const double eps = demand.axial_strain +
                           demand.curvature_y * xy[0] +
                           demand.curvature_z * xy[1];
        if (std::isfinite(eps)) {
            max_tensile_strain = std::max(max_tensile_strain, eps);
        }
    }
    if (!std::isfinite(max_tensile_strain)) {
        max_tensile_strain = 0.0;
    }

    Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
    const Eigen::Vector2d gradient{demand.curvature_y, demand.curvature_z};
    const double gradient_norm = gradient.norm();
    if (max_tensile_strain > 0.0 && gradient_norm > 1.0e-14) {
        const double strain_scale =
            std::max({std::abs(max_tensile_strain),
                      std::abs(demand.axial_strain),
                      1.0e-8});
        const double tilt = std::clamp(
            0.20 * gradient_norm *
                std::max(demand.section_width_m, demand.section_depth_m) /
                strain_scale,
            0.0,
            0.35);
        const Eigen::Vector2d g = gradient / gradient_norm;
        normal = Eigen::Vector3d{tilt * g.x(), tilt * g.y(), 1.0}.normalized();
    }

    const double z_ratio = adjust_reduced_rc_macro_inferred_crack_z_over_l(
        demand.crack_z_over_l, demand.nz);
    ReducedRCManagedLocalCrackPlaneSpec plane{};
    plane.point = {0.0,
                   0.0,
                   z_ratio * std::max(0.0, demand.characteristic_length_m)};
    plane.normal = {normal.x(), normal.y(), normal.z()};
    plane.plane_id = std::max(1, demand.plane_id);
    plane.sequence_id = std::max(1, demand.sequence_id);
    plane.source = ReducedRCManagedLocalCrackPlaneSource::macro_inferred;
    plane.activation_step = demand.activation_step;
    plane.activation_time = demand.activation_time;
    plane.criterion_value = max_tensile_strain;
    plane.active = true;
    return plane;
}

} // namespace fall_n

#endif // FALL_N_REDUCED_RC_MACRO_INFERRED_XFEM_SITE_POLICY_HH
