#ifndef FALL_N_REDUCED_RC_MULTISCALE_REPLAY_PLAN_HH
#define FALL_N_REDUCED_RC_MULTISCALE_REPLAY_PLAN_HH

// =============================================================================
//  ReducedRCMultiscaleReplayPlan.hh
// =============================================================================
//
//  One-way replay planner for the reduced-RC validation campaign.
//
//  The first multiscale step should not be a fully coupled FE2 jump.  The
//  validated global structural model must first replay its accepted histories
//  into a small, well documented set of local XFEM sites.  This header keeps
//  that decision deterministic and cheap:
//
//    - group structural history samples by candidate local site,
//    - score sites with work-conjugate demand indicators,
//    - estimate local XFEM cost with the mesh-scale audit,
//    - and recommend warm-start/cache/OpenMP policies before the expensive run.
//
//  It is intentionally independent from PETSc and VTK.  Drivers can serialize
//  the returned plan to JSON/CSV and later attach real local-model execution.
//
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string_view>
#include <vector>

#include "ReducedRCLocalMeshScaleAudit.hh"
#include "ReducedRCMultiscaleValidationStartCatalog.hh"

namespace fall_n {

enum class ReducedRCReplaySiteActivationKind {
    monitor_only,
    elastic_local_control,
    xfem_enriched_replay,
    guarded_two_way_candidate
};

[[nodiscard]] constexpr std::string_view to_string(
    ReducedRCReplaySiteActivationKind kind) noexcept
{
    switch (kind) {
        case ReducedRCReplaySiteActivationKind::monitor_only:
            return "monitor_only";
        case ReducedRCReplaySiteActivationKind::elastic_local_control:
            return "elastic_local_control";
        case ReducedRCReplaySiteActivationKind::xfem_enriched_replay:
            return "xfem_enriched_replay";
        case ReducedRCReplaySiteActivationKind::guarded_two_way_candidate:
            return "guarded_two_way_candidate";
    }
    return "unknown_replay_site_activation";
}

struct ReducedRCStructuralReplaySample {
    std::size_t site_index{0};
    double pseudo_time{0.0};
    double physical_time{0.0};
    double z_over_l{0.0};
    double drift_mm{0.0};
    double curvature_y{0.0};
    double moment_y_mn_m{0.0};
    double base_shear_mn{0.0};
    double steel_stress_mpa{0.0};
    double damage_indicator{0.0};
    double work_increment_mn_mm{0.0};
};

struct ReducedRCMultiscaleReplayPlanSettings {
    std::size_t max_replay_sites{3};
    double curvature_activation_threshold{0.010};
    double moment_activation_threshold_mn_m{0.015};
    double steel_activation_threshold_mpa{0.70 * 420.0};
    double damage_activation_threshold{0.20};
    double work_activation_threshold_mn_mm{2.0};
    double guarded_two_way_score_threshold{2.50};
    ReducedRCLocalMeshScaleInput local_mesh{};
    bool vtk_time_series_required{true};
    bool seed_cache_required{true};
    bool warm_start_required{true};
};

struct ReducedRCMultiscaleReplaySitePlan {
    std::size_t site_index{0};
    double z_over_l{0.0};
    std::size_t sample_count{0};
    double peak_abs_curvature_y{0.0};
    double peak_abs_moment_y_mn_m{0.0};
    double peak_abs_base_shear_mn{0.0};
    double peak_abs_steel_stress_mpa{0.0};
    double max_damage_indicator{0.0};
    double accumulated_abs_work_mn_mm{0.0};
    double activation_score{0.0};
    bool selected_for_replay{false};
    ReducedRCReplaySiteActivationKind activation_kind{
        ReducedRCReplaySiteActivationKind::monitor_only};
    std::string_view selection_reason{"below activation threshold"};
    ReducedRCLocalMeshScaleAudit local_cost{};
};

struct ReducedRCMultiscaleReplayPlan {
    std::size_t history_sample_count{0};
    std::size_t candidate_site_count{0};
    std::size_t selected_site_count{0};
    double selected_hot_state_mib{0.0};
    double selected_direct_factorization_risk_mib{0.0};
    bool vtk_contract_satisfied{false};
    bool seed_state_cache_recommended{false};
    bool newton_warm_start_recommended{false};
    bool site_level_openmp_recommended{false};
    bool avoid_direct_lu_for_batch{false};
    bool ready_for_one_way_replay{false};
    bool ready_for_two_way_fe2{false};
    std::vector<ReducedRCMultiscaleReplaySitePlan> sites{};
};

namespace detail {

[[nodiscard]] constexpr double replay_safe_abs(double value) noexcept
{
    return value < 0.0 ? -value : value;
}

[[nodiscard]] inline double replay_safe_ratio(double value,
                                             double threshold) noexcept
{
    if (!(threshold > 0.0) || !std::isfinite(threshold)) {
        return 0.0;
    }
    const double ratio = replay_safe_abs(value) / threshold;
    return std::isfinite(ratio) ? ratio
                                : std::numeric_limits<double>::infinity();
}

[[nodiscard]] inline double replay_activation_score(
    const ReducedRCMultiscaleReplaySitePlan& site,
    const ReducedRCMultiscaleReplayPlanSettings& settings) noexcept
{
    double score = 0.0;
    score = std::max(score, replay_safe_ratio(
                                site.peak_abs_curvature_y,
                                settings.curvature_activation_threshold));
    score = std::max(score, replay_safe_ratio(
                                site.peak_abs_moment_y_mn_m,
                                settings.moment_activation_threshold_mn_m));
    score = std::max(score, replay_safe_ratio(
                                site.peak_abs_steel_stress_mpa,
                                settings.steel_activation_threshold_mpa));
    score = std::max(score, replay_safe_ratio(
                                site.max_damage_indicator,
                                settings.damage_activation_threshold));
    score = std::max(score, replay_safe_ratio(
                                site.accumulated_abs_work_mn_mm,
                                settings.work_activation_threshold_mn_mm));
    return score;
}

[[nodiscard]] constexpr ReducedRCReplaySiteActivationKind
classify_replay_site_activation(double score,
                                bool selected,
                                double guarded_threshold) noexcept
{
    if (!selected) {
        return ReducedRCReplaySiteActivationKind::monitor_only;
    }
    if (score >= guarded_threshold) {
        return ReducedRCReplaySiteActivationKind::guarded_two_way_candidate;
    }
    if (score >= 1.0) {
        return ReducedRCReplaySiteActivationKind::xfem_enriched_replay;
    }
    return ReducedRCReplaySiteActivationKind::elastic_local_control;
}

} // namespace detail

[[nodiscard]] inline ReducedRCMultiscaleReplayPlan
make_reduced_rc_multiscale_replay_plan(
    const std::vector<ReducedRCStructuralReplaySample>& history,
    ReducedRCMultiscaleReplayPlanSettings settings = {})
{
    ReducedRCMultiscaleReplayPlan plan{};
    plan.history_sample_count = history.size();

    auto find_or_create_site = [&](const ReducedRCStructuralReplaySample& row)
        -> ReducedRCMultiscaleReplaySitePlan& {
        const auto it = std::ranges::find_if(
            plan.sites,
            [&](const ReducedRCMultiscaleReplaySitePlan& site) {
                return site.site_index == row.site_index;
            });
        if (it != plan.sites.end()) {
            return *it;
        }
        plan.sites.push_back(ReducedRCMultiscaleReplaySitePlan{
            .site_index = row.site_index,
            .z_over_l = row.z_over_l});
        return plan.sites.back();
    };

    for (const auto& row : history) {
        auto& site = find_or_create_site(row);
        ++site.sample_count;
        site.z_over_l = row.z_over_l;
        site.peak_abs_curvature_y =
            std::max(site.peak_abs_curvature_y,
                     detail::replay_safe_abs(row.curvature_y));
        site.peak_abs_moment_y_mn_m =
            std::max(site.peak_abs_moment_y_mn_m,
                     detail::replay_safe_abs(row.moment_y_mn_m));
        site.peak_abs_base_shear_mn =
            std::max(site.peak_abs_base_shear_mn,
                     detail::replay_safe_abs(row.base_shear_mn));
        site.peak_abs_steel_stress_mpa =
            std::max(site.peak_abs_steel_stress_mpa,
                     detail::replay_safe_abs(row.steel_stress_mpa));
        site.max_damage_indicator =
            std::max(site.max_damage_indicator,
                     std::clamp(row.damage_indicator, 0.0, 1.0));
        site.accumulated_abs_work_mn_mm +=
            detail::replay_safe_abs(row.work_increment_mn_mm);
    }

    for (auto& site : plan.sites) {
        site.activation_score =
            detail::replay_activation_score(site, settings);
        site.local_cost =
            make_reduced_rc_local_mesh_scale_audit(settings.local_mesh);
    }

    std::ranges::sort(plan.sites, [](const auto& a, const auto& b) {
        if (a.activation_score == b.activation_score) {
            return a.site_index < b.site_index;
        }
        return a.activation_score > b.activation_score;
    });

    const auto max_sites = settings.max_replay_sites == 0
                               ? plan.sites.size()
                               : std::min(settings.max_replay_sites,
                                          plan.sites.size());
    for (std::size_t i = 0; i < plan.sites.size(); ++i) {
        auto& site = plan.sites[i];
        site.selected_for_replay = i < max_sites && site.activation_score >= 1.0;
        site.activation_kind = detail::classify_replay_site_activation(
            site.activation_score,
            site.selected_for_replay,
            settings.guarded_two_way_score_threshold);
        if (site.selected_for_replay) {
            site.selection_reason =
                site.activation_kind ==
                        ReducedRCReplaySiteActivationKind::
                            guarded_two_way_candidate
                    ? "selected as one-way replay and guarded FE2 candidate"
                    : "selected for one-way XFEM replay";
            ++plan.selected_site_count;
            plan.selected_hot_state_mib += site.local_cost.estimated_hot_state_mib;
            plan.selected_direct_factorization_risk_mib +=
                site.local_cost.direct_factorization_risk_mib;
            plan.seed_state_cache_recommended =
                plan.seed_state_cache_recommended ||
                site.local_cost.seed_state_cache_recommended ||
                settings.seed_cache_required;
            plan.newton_warm_start_recommended =
                plan.newton_warm_start_recommended ||
                site.local_cost.newton_warm_start_recommended ||
                settings.warm_start_required;
            plan.site_level_openmp_recommended =
                plan.site_level_openmp_recommended ||
                site.local_cost.site_level_openmp_recommended;
            plan.avoid_direct_lu_for_batch =
                plan.avoid_direct_lu_for_batch ||
                site.local_cost.solver_advice !=
                    ReducedRCLocalSolverScalingAdviceKind::direct_lu_smoke_ok;
            continue;
        }
        site.selection_reason = site.activation_score > 0.0
                                    ? "ranked but below replay limit"
                                    : "below activation threshold";
    }

    plan.candidate_site_count = plan.sites.size();
    plan.vtk_contract_satisfied =
        !settings.vtk_time_series_required ||
        (canonical_reduced_rc_required_replay_vtk_field_count_v >= 9 &&
         vtk_field_table_has_crack_visualization(
             canonical_reduced_rc_vtk_field_table_v));
    plan.ready_for_one_way_replay =
        plan.selected_site_count > 0 && plan.vtk_contract_satisfied;
    plan.ready_for_two_way_fe2 = false;
    return plan;
}

} // namespace fall_n

#endif // FALL_N_REDUCED_RC_MULTISCALE_REPLAY_PLAN_HH
