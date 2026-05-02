#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "src/validation/SeismicFE2ValidationCampaign.hh"

namespace {

[[nodiscard]] bool contains(const std::string& haystack,
                            const std::string& needle)
{
    return haystack.find(needle) != std::string::npos;
}

} // namespace

int main()
{
    using namespace fall_n;

    const auto matrix = make_default_lshaped_16storey_seismic_fe2_matrix();
    assert(matrix.size() == 6);
    assert(matrix.front().case_kind ==
           SeismicFE2CampaignCaseKind::gravity_preload);
    assert(matrix.back().case_kind ==
           SeismicFE2CampaignCaseKind::fe2_two_way);
    assert(matrix.back().component_set ==
           SeismicFE2ComponentSetKind::horizontal_2d);
    assert(matrix.back().local_mesh_tier ==
           SeismicFE2LocalMeshTierKind::xfem_3x3x6);

    const std::vector<ReducedRCStructuralReplaySample> z_only_history{
        {.site_index = 9,
         .pseudo_time = 0.0,
         .physical_time = 0.0,
         .z_over_l = 0.15,
         .curvature_z = 0.025,
         .moment_z_mn_m = 0.045,
         .steel_stress_mpa = 0.0,
         .damage_indicator = 0.0}};
    ReducedRCMultiscaleReplayPlanSettings replay_settings{};
    replay_settings.max_replay_sites = 1;
    const auto replay_plan =
        make_reduced_rc_multiscale_replay_plan(z_only_history, replay_settings);
    assert(replay_plan.selected_site_count == 1);
    assert(replay_plan.sites.front().selected_for_replay);
    assert(replay_plan.sites.front().peak_abs_curvature_z > 0.024);
    assert(replay_plan.sites.front().peak_abs_moment_z_mn_m > 0.044);

    const auto mesh3 = local_mesh_spec(SeismicFE2LocalMeshTierKind::xfem_3x3x6);
    const auto mesh7 = local_mesh_spec(SeismicFE2LocalMeshTierKind::xfem_7x7x25);
    const auto kb = local_mesh_spec(SeismicFE2LocalMeshTierKind::kobathe_spot_check);
    assert(mesh3.nx == 3 && mesh3.ny == 3 && mesh3.nz == 6);
    assert(mesh7.nx == 7 && mesh7.ny == 7 && mesh7.nz == 25);
    assert(kb.kobathe_reference && !kb.xfem_enriched);

    CouplingSite site{};
    site.macro_element_id = 42;
    site.section_gp = 3;
    site.xi = -1.0;

    std::vector<BiaxialSectionHistorySample> history;
    history.push_back(make_biaxial_section_history_sample(
        site, -0.0004, 0.000, 0.000, -0.60, 0.000, 0.000, 0.0, 0.0));
    history.push_back(make_biaxial_section_history_sample(
        site, -0.0005, 0.018, -0.012, -0.62, 0.035, -0.025, 0.5, 0.02));
    history.push_back(make_biaxial_section_history_sample(
        site, -0.0006, -0.010, 0.014, -0.61, -0.020, 0.030, 1.0, 0.04));

    const auto audit = audit_biaxial_section_history(history);
    assert(audit.finite);
    assert(audit.ready_for_biaxial_fe2());
    assert(std::abs(audit.accumulated_work) > 1.0e-5);

    const auto packet_sample =
        make_biaxial_section_material_history_sample(history[1]);
    assert(packet_sample.measure_kind ==
           MaterialHistoryMeasureKind::SectionGeneralized);
    assert(packet_sample.kinematic.size() == 6);
    assert(packet_sample.conjugate.size() == 6);
    assert(std::abs(packet_sample.kinematic[1] - 0.018) < 1.0e-14);
    assert(std::abs(packet_sample.kinematic[2] + 0.012) < 1.0e-14);
    assert(std::abs(packet_sample.conjugate[1] - 0.035) < 1.0e-14);
    assert(std::abs(packet_sample.conjugate[2] + 0.025) < 1.0e-14);

    const std::vector<GravityPreloadMonitorSample> safe_gravity{
        {.macro_element_id = 1,
         .storey_index = 0,
         .axial_load_ratio = 0.18,
         .max_steel_stress_ratio = 0.20,
         .damage_indicator = 0.0,
         .converged = true}};
    const auto safe = audit_gravity_preload(safe_gravity);
    assert(safe.converged);
    assert(!safe.failure_detected);

    const std::vector<GravityPreloadMonitorSample> unsafe_gravity{
        {.macro_element_id = 2,
         .storey_index = 1,
         .axial_load_ratio = 0.30,
         .max_steel_stress_ratio = 0.82,
         .damage_indicator = 0.0,
         .converged = true}};
    const auto unsafe = audit_gravity_preload(unsafe_gravity);
    assert(unsafe.failure_detected);
    assert(unsafe.failing_sample_count == 1);

    const std::vector<MultiscaleVTKTimeIndexRow> rows{
        {.case_kind = SeismicFE2CampaignCaseKind::fe2_one_way,
         .role = SeismicFE2VisualizationRole::global_frame,
         .global_step = 10,
         .physical_time = 0.20,
         .pseudo_time = 0.20,
         .global_vtk_path = "global/frame_000010.vtm",
         .notes = "accepted global step"},
        {.case_kind = SeismicFE2CampaignCaseKind::fe2_one_way,
         .role = SeismicFE2VisualizationRole::local_xfem_site,
         .global_step = 10,
         .physical_time = 0.20,
         .pseudo_time = 0.20,
         .local_site_index = 4,
         .macro_element_id = 42,
         .section_gp = 3,
         .global_vtk_path = "global/frame_000010.vtm",
         .local_vtk_path = "local/site_004/step_000010.vtu",
         .notes = "synchronized XFEM local state"}};
    assert(rows[0].has_required_paths());
    assert(rows[1].is_local());
    assert(rows[1].has_required_paths());

    const auto out = std::filesystem::temp_directory_path() /
                     "fall_n_seismic_fe2_validation_campaign" /
                     "multiscale_time_index.csv";
    assert(write_multiscale_vtk_time_index_csv(out, rows));
    std::ifstream in(out);
    std::string text((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
    assert(contains(text, "case_kind,role,global_step"));
    assert(contains(text, "local/site_004/step_000010.vtu"));

    std::printf("[seismic_fe2_validation_campaign] matrix=%zu work=%.6e "
                "vtk_rows=%zu\n",
                matrix.size(),
                audit.accumulated_work,
                rows.size());
    return 0;
}
