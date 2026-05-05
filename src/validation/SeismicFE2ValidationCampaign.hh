#ifndef FALL_N_SEISMIC_FE2_VALIDATION_CAMPAIGN_HH
#define FALL_N_SEISMIC_FE2_VALIDATION_CAMPAIGN_HH

// =============================================================================
//  SeismicFE2ValidationCampaign.hh
// =============================================================================
//
//  Lightweight contracts for the 16-storey seismic FE2 validation campaign.
//  The heavy solvers remain in the executable drivers.  This header freezes the
//  scientific protocol shared by drivers, tests, scripts, wrappers and thesis
//  manifests: gravity preload, global fall_n/OpenSees baselines, one-way FE2,
//  two-way FE2, biaxial section-history transfer and synchronized VTK output.
//
//  A local model is one managed Model with its own mesh/domain per selected
//  macro site.  It is intentionally not one XFEM problem per failed fiber,
//  section sample or integration point.
//
// =============================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Dense>

#include "src/analysis/MultiscaleMaterialHistoryTransfer.hh"
#include "src/analysis/MultiscaleTypes.hh"
#include "src/validation/ReducedRCMultiscaleReplayPlan.hh"

namespace fall_n {

enum class SeismicFE2CampaignCaseKind {
    gravity_preload,
    linear_until_first_alarm,
    global_falln,
    global_opensees,
    fe2_one_way,
    fe2_two_way
};

enum class SeismicFE2ComponentSetKind {
    horizontal_2d,
    horizontal_plus_vertical_3d
};

enum class SeismicFE2TimeIntegrationProfileKind {
    newmark_average_acceleration,
    hht_alpha,
    generalized_alpha,
    petsc_ts_alpha
};

enum class SeismicFE2LocalMeshTierKind {
    xfem_3x3x6,
    xfem_5x5x10,
    xfem_7x7x25,
    kobathe_spot_check
};

enum class SeismicFE2VisualizationRole {
    global_frame,
    local_xfem_site,
    local_continuum_site,
    opensees_global_reference
};

[[nodiscard]] constexpr std::string_view to_string(
    SeismicFE2CampaignCaseKind kind) noexcept
{
    switch (kind) {
        case SeismicFE2CampaignCaseKind::gravity_preload:
            return "gravity_preload";
        case SeismicFE2CampaignCaseKind::linear_until_first_alarm:
            return "linear_until_first_alarm";
        case SeismicFE2CampaignCaseKind::global_falln:
            return "global_falln";
        case SeismicFE2CampaignCaseKind::global_opensees:
            return "global_opensees";
        case SeismicFE2CampaignCaseKind::fe2_one_way:
            return "fe2_one_way";
        case SeismicFE2CampaignCaseKind::fe2_two_way:
            return "fe2_two_way";
    }
    return "unknown_seismic_fe2_campaign_case";
}

[[nodiscard]] constexpr std::string_view to_string(
    SeismicFE2ComponentSetKind kind) noexcept
{
    switch (kind) {
        case SeismicFE2ComponentSetKind::horizontal_2d:
            return "horizontal_2d";
        case SeismicFE2ComponentSetKind::horizontal_plus_vertical_3d:
            return "horizontal_plus_vertical_3d";
    }
    return "unknown_seismic_component_set";
}

[[nodiscard]] constexpr std::string_view to_string(
    SeismicFE2TimeIntegrationProfileKind kind) noexcept
{
    switch (kind) {
        case SeismicFE2TimeIntegrationProfileKind::newmark_average_acceleration:
            return "newmark_average_acceleration";
        case SeismicFE2TimeIntegrationProfileKind::hht_alpha:
            return "hht_alpha";
        case SeismicFE2TimeIntegrationProfileKind::generalized_alpha:
            return "generalized_alpha";
        case SeismicFE2TimeIntegrationProfileKind::petsc_ts_alpha:
            return "petsc_ts_alpha";
    }
    return "unknown_time_integration_profile";
}

[[nodiscard]] constexpr std::string_view to_string(
    SeismicFE2LocalMeshTierKind kind) noexcept
{
    switch (kind) {
        case SeismicFE2LocalMeshTierKind::xfem_3x3x6:
            return "xfem_3x3x6";
        case SeismicFE2LocalMeshTierKind::xfem_5x5x10:
            return "xfem_5x5x10";
        case SeismicFE2LocalMeshTierKind::xfem_7x7x25:
            return "xfem_7x7x25";
        case SeismicFE2LocalMeshTierKind::kobathe_spot_check:
            return "kobathe_spot_check";
    }
    return "unknown_local_mesh_tier";
}

[[nodiscard]] constexpr std::string_view to_string(
    SeismicFE2VisualizationRole role) noexcept
{
    switch (role) {
        case SeismicFE2VisualizationRole::global_frame:
            return "global_frame";
        case SeismicFE2VisualizationRole::local_xfem_site:
            return "local_xfem_site";
        case SeismicFE2VisualizationRole::local_continuum_site:
            return "local_continuum_site";
        case SeismicFE2VisualizationRole::opensees_global_reference:
            return "opensees_global_reference";
    }
    return "unknown_visualization_role";
}

struct SeismicFE2AcceptanceTolerances {
    double max_elastic_roof_displacement_gap{0.15};
    double max_elastic_story_drift_gap{0.15};
    double max_two_way_force_residual_rel{0.05};
    double max_two_way_tangent_residual_rel{0.05};
    double max_local_work_gap{0.05};
    double max_gravity_damage_indicator{1.0e-6};
    double max_gravity_steel_stress_ratio{0.70};
};

struct SeismicFE2CampaignRunSpec {
    SeismicFE2CampaignCaseKind case_kind{
        SeismicFE2CampaignCaseKind::global_falln};
    SeismicFE2ComponentSetKind component_set{
        SeismicFE2ComponentSetKind::horizontal_2d};
    SeismicFE2TimeIntegrationProfileKind time_profile{
        SeismicFE2TimeIntegrationProfileKind::newmark_average_acceleration};
    SeismicFE2LocalMeshTierKind local_mesh_tier{
        SeismicFE2LocalMeshTierKind::xfem_3x3x6};

    std::string building_label{"lshaped_rc_16storey"};
    std::string earthquake_label{"MYG004_Tohoku_2011"};
    std::string output_stem{"data/output/seismic_fe2_16storey"};
    std::size_t storey_count{16};
    std::size_t max_active_local_sites{5};
    double earthquake_scale{1.0};
    double time_step_s{0.02};
    double analysis_window_s{1.5};
    double damping_ratio{0.05};

    bool run_gravity_preload{true};
    bool monitor_gravity_failure{true};
    bool run_linear_until_first_alarm{true};
    bool write_global_vtk_time_series{true};
    bool write_local_vtk_time_series{true};
    std::size_t global_vtk_stride{10};
    std::size_t local_vtk_stride{5};

    SeismicFE2AcceptanceTolerances tolerances{};
};

struct SeismicFE2LocalMeshSpec {
    int nx{3};
    int ny{3};
    int nz{6};
    bool xfem_enriched{true};
    bool kobathe_reference{false};
};

[[nodiscard]] constexpr SeismicFE2LocalMeshSpec
local_mesh_spec(SeismicFE2LocalMeshTierKind tier) noexcept
{
    switch (tier) {
        case SeismicFE2LocalMeshTierKind::xfem_3x3x6:
            return {.nx = 3, .ny = 3, .nz = 6};
        case SeismicFE2LocalMeshTierKind::xfem_5x5x10:
            return {.nx = 5, .ny = 5, .nz = 10};
        case SeismicFE2LocalMeshTierKind::xfem_7x7x25:
            return {.nx = 7, .ny = 7, .nz = 25};
        case SeismicFE2LocalMeshTierKind::kobathe_spot_check:
            return {.nx = 2, .ny = 2, .nz = 4,
                    .xfem_enriched = false,
                    .kobathe_reference = true};
    }
    return {};
}

[[nodiscard]] inline std::vector<SeismicFE2CampaignRunSpec>
make_default_lshaped_16storey_seismic_fe2_matrix()
{
    std::vector<SeismicFE2CampaignRunSpec> specs;
    auto base = SeismicFE2CampaignRunSpec{};

    base.case_kind = SeismicFE2CampaignCaseKind::gravity_preload;
    specs.push_back(base);

    base.case_kind = SeismicFE2CampaignCaseKind::linear_until_first_alarm;
    specs.push_back(base);

    base.case_kind = SeismicFE2CampaignCaseKind::global_falln;
    specs.push_back(base);

    base.case_kind = SeismicFE2CampaignCaseKind::global_opensees;
    specs.push_back(base);

    base.case_kind = SeismicFE2CampaignCaseKind::fe2_one_way;
    base.local_mesh_tier = SeismicFE2LocalMeshTierKind::xfem_3x3x6;
    specs.push_back(base);

    base.case_kind = SeismicFE2CampaignCaseKind::fe2_two_way;
    specs.push_back(base);
    return specs;
}

enum class SectionGeneralizedComponent : int {
    axial = 0,
    curvature_y = 1,
    curvature_z = 2,
    shear_y = 3,
    shear_z = 4,
    torsion = 5
};

[[nodiscard]] constexpr int section_component_index(
    SectionGeneralizedComponent component) noexcept
{
    return static_cast<int>(component);
}

struct BiaxialSectionHistorySample {
    CouplingSite site{};
    double pseudo_time{0.0};
    double physical_time{0.0};
    Eigen::Vector<double, 6> generalized_strain{
        Eigen::Vector<double, 6>::Zero()};
    Eigen::Vector<double, 6> generalized_force{
        Eigen::Vector<double, 6>::Zero()};
    double steel_stress_mpa{0.0};
    double concrete_damage_indicator{0.0};
    bool committed{true};
};

[[nodiscard]] inline BiaxialSectionHistorySample
make_biaxial_section_history_sample(
    CouplingSite site,
    double axial_strain,
    double curvature_y,
    double curvature_z,
    double axial_force_mn,
    double moment_y_mn_m,
    double moment_z_mn_m,
    double pseudo_time = 0.0,
    double physical_time = 0.0)
{
    BiaxialSectionHistorySample sample{};
    sample.site = site;
    sample.pseudo_time = pseudo_time;
    sample.physical_time = physical_time;
    sample.generalized_strain[section_component_index(
        SectionGeneralizedComponent::axial)] = axial_strain;
    sample.generalized_strain[section_component_index(
        SectionGeneralizedComponent::curvature_y)] = curvature_y;
    sample.generalized_strain[section_component_index(
        SectionGeneralizedComponent::curvature_z)] = curvature_z;
    sample.generalized_force[section_component_index(
        SectionGeneralizedComponent::axial)] = axial_force_mn;
    sample.generalized_force[section_component_index(
        SectionGeneralizedComponent::curvature_y)] = moment_y_mn_m;
    sample.generalized_force[section_component_index(
        SectionGeneralizedComponent::curvature_z)] = moment_z_mn_m;
    return sample;
}

[[nodiscard]] inline MaterialHistorySample
make_biaxial_section_material_history_sample(
    const BiaxialSectionHistorySample& row,
    MaterialHistorySiteRole role = MaterialHistorySiteRole::SectionResultant)
{
    MaterialHistorySiteKey key{};
    key.site = row.site;
    key.role = role;
    auto sample = make_section_generalized_material_history_sample(
        key,
        Eigen::VectorXd(row.generalized_strain),
        Eigen::VectorXd(row.generalized_force),
        row.pseudo_time,
        row.physical_time);
    sample.committed = row.committed;
    return sample;
}

[[nodiscard]] inline double trapezoidal_biaxial_section_work(
    const BiaxialSectionHistorySample& a,
    const BiaxialSectionHistorySample& b) noexcept
{
    const auto delta = b.generalized_strain - a.generalized_strain;
    const auto mean_force = 0.5 * (a.generalized_force + b.generalized_force);
    return mean_force.dot(delta);
}

[[nodiscard]] inline double accumulated_biaxial_section_work(
    const std::vector<BiaxialSectionHistorySample>& history) noexcept
{
    if (history.size() < 2) {
        return 0.0;
    }
    double work = 0.0;
    for (std::size_t i = 1; i < history.size(); ++i) {
        work += trapezoidal_biaxial_section_work(history[i - 1], history[i]);
    }
    return work;
}

struct BiaxialSectionHistoryAudit {
    bool has_axial{false};
    bool has_curvature_y{false};
    bool has_curvature_z{false};
    bool has_moment_y{false};
    bool has_moment_z{false};
    bool finite{true};
    double accumulated_work{0.0};

    [[nodiscard]] bool ready_for_biaxial_fe2() const noexcept
    {
        return finite && has_axial && has_curvature_y && has_curvature_z &&
               has_moment_y && has_moment_z;
    }
};

[[nodiscard]] inline BiaxialSectionHistoryAudit
audit_biaxial_section_history(
    const std::vector<BiaxialSectionHistorySample>& history,
    double activity_tol = 1.0e-14) noexcept
{
    BiaxialSectionHistoryAudit audit{};
    for (const auto& row : history) {
        audit.finite = audit.finite &&
                       row.generalized_strain.allFinite() &&
                       row.generalized_force.allFinite();
        audit.has_axial = audit.has_axial ||
            std::abs(row.generalized_strain[0]) > activity_tol ||
            std::abs(row.generalized_force[0]) > activity_tol;
        audit.has_curvature_y = audit.has_curvature_y ||
            std::abs(row.generalized_strain[1]) > activity_tol;
        audit.has_curvature_z = audit.has_curvature_z ||
            std::abs(row.generalized_strain[2]) > activity_tol;
        audit.has_moment_y = audit.has_moment_y ||
            std::abs(row.generalized_force[1]) > activity_tol;
        audit.has_moment_z = audit.has_moment_z ||
            std::abs(row.generalized_force[2]) > activity_tol;
    }
    audit.accumulated_work = accumulated_biaxial_section_work(history);
    return audit;
}

struct GravityPreloadMonitorSample {
    std::size_t macro_element_id{0};
    std::size_t storey_index{0};
    double axial_load_ratio{0.0};
    double max_steel_stress_ratio{0.0};
    double damage_indicator{0.0};
    bool converged{true};
};

struct GravityPreloadAudit {
    bool converged{true};
    bool failure_detected{false};
    std::size_t failing_sample_count{0};
    double max_axial_load_ratio{0.0};
    double max_steel_stress_ratio{0.0};
    double max_damage_indicator{0.0};
};

[[nodiscard]] inline GravityPreloadAudit audit_gravity_preload(
    const std::vector<GravityPreloadMonitorSample>& samples,
    const SeismicFE2AcceptanceTolerances& tolerances = {}) noexcept
{
    GravityPreloadAudit audit{};
    for (const auto& sample : samples) {
        audit.converged = audit.converged && sample.converged;
        audit.max_axial_load_ratio =
            std::max(audit.max_axial_load_ratio,
                     std::abs(sample.axial_load_ratio));
        audit.max_steel_stress_ratio =
            std::max(audit.max_steel_stress_ratio,
                     std::abs(sample.max_steel_stress_ratio));
        audit.max_damage_indicator =
            std::max(audit.max_damage_indicator,
                     std::clamp(sample.damage_indicator, 0.0, 1.0));
        const bool fails =
            !sample.converged ||
            std::abs(sample.max_steel_stress_ratio) >
                tolerances.max_gravity_steel_stress_ratio ||
            sample.damage_indicator > tolerances.max_gravity_damage_indicator;
        audit.failing_sample_count += fails ? 1U : 0U;
    }
    audit.failure_detected = audit.failing_sample_count > 0;
    return audit;
}

struct MultiscaleVTKTimeIndexRow {
    SeismicFE2CampaignCaseKind case_kind{
        SeismicFE2CampaignCaseKind::fe2_one_way};
    SeismicFE2VisualizationRole role{
        SeismicFE2VisualizationRole::global_frame};
    std::size_t global_step{0};
    double physical_time{0.0};
    double pseudo_time{0.0};
    std::size_t local_site_index{std::numeric_limits<std::size_t>::max()};
    std::size_t macro_element_id{std::numeric_limits<std::size_t>::max()};
    std::size_t section_gp{std::numeric_limits<std::size_t>::max()};
    std::string global_vtk_path{};
    std::string local_vtk_path{};
    std::string notes{};

    [[nodiscard]] bool is_local() const noexcept
    {
        return role == SeismicFE2VisualizationRole::local_xfem_site ||
               role == SeismicFE2VisualizationRole::local_continuum_site;
    }

    [[nodiscard]] bool has_required_paths() const noexcept
    {
        if (role == SeismicFE2VisualizationRole::global_frame ||
            role == SeismicFE2VisualizationRole::opensees_global_reference) {
            return !global_vtk_path.empty();
        }
        return !global_vtk_path.empty() && !local_vtk_path.empty() &&
               local_site_index != std::numeric_limits<std::size_t>::max();
    }
};

[[nodiscard]] inline bool write_multiscale_vtk_time_index_csv(
    const std::filesystem::path& path,
    const std::vector<MultiscaleVTKTimeIndexRow>& rows)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out) {
        return false;
    }
    out << "case_kind,role,global_step,physical_time,pseudo_time,"
           "local_site_index,macro_element_id,section_gp,global_vtk_path,"
           "local_vtk_path,notes\n";
    for (const auto& row : rows) {
        out << to_string(row.case_kind) << ','
            << to_string(row.role) << ','
            << row.global_step << ','
            << row.physical_time << ','
            << row.pseudo_time << ',';
        if (row.local_site_index == std::numeric_limits<std::size_t>::max()) {
            out << ",";
        } else {
            out << row.local_site_index << ',';
        }
        if (row.macro_element_id == std::numeric_limits<std::size_t>::max()) {
            out << ",";
        } else {
            out << row.macro_element_id << ',';
        }
        if (row.section_gp == std::numeric_limits<std::size_t>::max()) {
            out << ",";
        } else {
            out << row.section_gp << ',';
        }
        out << row.global_vtk_path << ','
            << row.local_vtk_path << ','
            << row.notes << '\n';
    }
    return true;
}

} // namespace fall_n

#endif // FALL_N_SEISMIC_FE2_VALIDATION_CAMPAIGN_HH
