#include <iostream>
#include <vector>

#include "src/validation/ReducedRCMultiscaleReplayPlan.hh"

namespace {

int passed = 0;
int failed = 0;

void report(const char* name, bool ok)
{
    if (ok) {
        ++passed;
        std::cout << "  PASS  " << name << "\n";
    } else {
        ++failed;
        std::cout << "  FAIL  " << name << "\n";
    }
}

std::vector<fall_n::ReducedRCStructuralReplaySample> make_history()
{
    using Sample = fall_n::ReducedRCStructuralReplaySample;
    return {
        Sample{
            .site_index = 0,
            .pseudo_time = 0.00,
            .z_over_l = 0.02,
            .drift_mm = 0.0,
            .curvature_y = 0.0,
            .moment_y_mn_m = 0.0},
        Sample{
            .site_index = 0,
            .pseudo_time = 0.25,
            .z_over_l = 0.02,
            .drift_mm = 50.0,
            .curvature_y = 0.018,
            .moment_y_mn_m = 0.030,
            .base_shear_mn = 0.010,
            .steel_stress_mpa = 360.0,
            .damage_indicator = 0.65,
            .work_increment_mn_mm = 1.40},
        Sample{
            .site_index = 0,
            .pseudo_time = 0.50,
            .z_over_l = 0.02,
            .drift_mm = -50.0,
            .curvature_y = -0.020,
            .moment_y_mn_m = -0.032,
            .base_shear_mn = -0.011,
            .steel_stress_mpa = -345.0,
            .damage_indicator = 0.70,
            .work_increment_mn_mm = 1.55},
        Sample{
            .site_index = 1,
            .pseudo_time = 0.25,
            .z_over_l = 0.50,
            .drift_mm = 50.0,
            .curvature_y = 0.002,
            .moment_y_mn_m = 0.003,
            .base_shear_mn = 0.010,
            .steel_stress_mpa = 55.0,
            .damage_indicator = 0.01,
            .work_increment_mn_mm = 0.04},
        Sample{
            .site_index = 2,
            .pseudo_time = 0.25,
            .z_over_l = 0.92,
            .drift_mm = 50.0,
            .curvature_y = 0.001,
            .moment_y_mn_m = 0.001,
            .base_shear_mn = 0.010,
            .steel_stress_mpa = 25.0,
            .damage_indicator = 0.0,
            .work_increment_mn_mm = 0.02}
    };
}

bool selects_base_site_and_preserves_replay_first_gate()
{
    fall_n::ReducedRCMultiscaleReplayPlanSettings settings;
    settings.max_replay_sites = 2;
    settings.local_mesh = fall_n::ReducedRCLocalMeshScaleInput{
        .nx = 5,
        .ny = 5,
        .nz = 15,
        .constitutive_cost =
            fall_n::ReducedRCLocalConstitutiveCostKind::
                cyclic_crack_band_xfem};

    const auto plan = fall_n::make_reduced_rc_multiscale_replay_plan(
        make_history(),
        settings);

    return plan.ready_for_one_way_replay &&
           !plan.ready_for_two_way_fe2 &&
           plan.candidate_site_count == 3 &&
           plan.selected_site_count == 1 &&
           !plan.sites.empty() &&
           plan.sites.front().site_index == 0 &&
           plan.sites.front().selected_for_replay &&
           plan.sites.front().activation_kind ==
               fall_n::ReducedRCReplaySiteActivationKind::
                   guarded_two_way_candidate &&
           plan.vtk_contract_satisfied &&
           plan.seed_state_cache_recommended &&
           plan.newton_warm_start_recommended;
}

bool keeps_low_demand_history_as_monitor_only()
{
    std::vector<fall_n::ReducedRCStructuralReplaySample> quiet = {
        {.site_index = 4,
         .pseudo_time = 0.1,
         .z_over_l = 0.7,
         .curvature_y = 1.0e-4,
         .moment_y_mn_m = 1.0e-4,
         .steel_stress_mpa = 5.0,
         .damage_indicator = 0.0,
         .work_increment_mn_mm = 1.0e-3}};

    const auto plan = fall_n::make_reduced_rc_multiscale_replay_plan(quiet);
    return !plan.ready_for_one_way_replay &&
           plan.selected_site_count == 0 &&
           plan.candidate_site_count == 1 &&
           plan.sites.front().activation_kind ==
               fall_n::ReducedRCReplaySiteActivationKind::monitor_only;
}

} // namespace

int main()
{
    std::cout << "=== Reduced RC Multiscale Replay Plan ===\n";
    report("selects_base_site_and_preserves_replay_first_gate",
           selects_base_site_and_preserves_replay_first_gate());
    report("keeps_low_demand_history_as_monitor_only",
           keeps_low_demand_history_as_monitor_only());

    std::cout << "\nSummary: " << passed << " passed, " << failed
              << " failed.\n";
    return failed == 0 ? 0 : 1;
}
