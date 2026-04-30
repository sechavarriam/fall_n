#include <iostream>
#include <vector>

#include "src/validation/ReducedRCMultiscaleRuntimePolicy.hh"

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

std::vector<fall_n::ReducedRCStructuralReplaySample> make_two_hot_site_history()
{
    using Sample = fall_n::ReducedRCStructuralReplaySample;
    return {
        Sample{
            .site_index = 0,
            .z_over_l = 0.02,
            .curvature_y = 0.020,
            .moment_y_mn_m = 0.034,
            .base_shear_mn = 0.012,
            .steel_stress_mpa = 365.0,
            .damage_indicator = 0.70,
            .work_increment_mn_mm = 1.60},
        Sample{
            .site_index = 1,
            .z_over_l = 0.10,
            .curvature_y = -0.016,
            .moment_y_mn_m = -0.024,
            .base_shear_mn = -0.010,
            .steel_stress_mpa = -330.0,
            .damage_indicator = 0.55,
            .work_increment_mn_mm = 1.10},
        Sample{
            .site_index = 2,
            .z_over_l = 0.75,
            .curvature_y = 0.001,
            .moment_y_mn_m = 0.001,
            .base_shear_mn = 0.006,
            .steel_stress_mpa = 35.0,
            .damage_indicator = 0.0,
            .work_increment_mn_mm = 0.01}
    };
}

bool replay_plan_promotes_bounded_runtime_and_openmp_site_loop()
{
    fall_n::ReducedRCMultiscaleReplayPlanSettings settings;
    settings.max_replay_sites = 2;
    settings.local_mesh = fall_n::ReducedRCLocalMeshScaleInput{
        .nx = 7,
        .ny = 7,
        .nz = 25,
        .constitutive_cost =
            fall_n::ReducedRCLocalConstitutiveCostKind::
                cyclic_crack_band_xfem,
        .shifted_heaviside_xfem = true,
        .longitudinal_bar_count = 8};

    const auto plan = fall_n::make_reduced_rc_multiscale_replay_plan(
        make_two_hot_site_history(),
        settings);
    const auto policy =
        fall_n::make_reduced_rc_multiscale_runtime_policy(
            plan,
            {},
            8);

    return plan.selected_site_count == 2 &&
           policy.ready_for_local_site_batch &&
           policy.cache_budget_is_bounded &&
           policy.local_runtime_settings.max_cached_seed_states == 2 &&
           policy.local_runtime_settings.seed_state_reuse_enabled &&
           policy.local_runtime_settings.restore_seed_before_solve &&
           policy.local_runtime_settings.adaptive_activation_enabled &&
           policy.local_sites_run_in_parallel &&
           policy.executor_kind ==
               fall_n::ReducedRCMultiscaleExecutorKind::
                   openmp_site_parallel &&
           policy.recommended_site_threads == 2 &&
           policy.direct_lu_kept_as_reference_only;
}

bool explicit_cache_budget_override_is_respected()
{
    fall_n::ReducedRCMultiscaleReplayPlanSettings settings;
    settings.max_replay_sites = 2;
    const auto plan = fall_n::make_reduced_rc_multiscale_replay_plan(
        make_two_hot_site_history(),
        settings);

    fall_n::LocalSubproblemRuntimeSettings runtime;
    runtime.max_cached_seed_states = 1;
    const auto policy =
        fall_n::make_reduced_rc_multiscale_runtime_policy(plan, runtime);

    return policy.local_runtime_settings.max_cached_seed_states == 1 &&
           policy.cache_budget_is_bounded;
}

bool quiet_replay_plan_stays_serial_and_not_ready()
{
    std::vector<fall_n::ReducedRCStructuralReplaySample> quiet = {
        {.site_index = 5,
         .z_over_l = 0.80,
         .curvature_y = 1.0e-4,
         .moment_y_mn_m = 1.0e-4,
         .steel_stress_mpa = 4.0,
         .damage_indicator = 0.0,
         .work_increment_mn_mm = 1.0e-3}};
    const auto plan = fall_n::make_reduced_rc_multiscale_replay_plan(quiet);
    const auto policy =
        fall_n::make_reduced_rc_multiscale_runtime_policy(plan);

    return !policy.ready_for_local_site_batch &&
           !policy.local_sites_run_in_parallel &&
           policy.executor_kind ==
               fall_n::ReducedRCMultiscaleExecutorKind::serial_site_loop;
}

} // namespace

int main()
{
    std::cout << "=== Reduced RC Multiscale Runtime Policy ===\n";
    report("replay_plan_promotes_bounded_runtime_and_openmp_site_loop",
           replay_plan_promotes_bounded_runtime_and_openmp_site_loop());
    report("explicit_cache_budget_override_is_respected",
           explicit_cache_budget_override_is_respected());
    report("quiet_replay_plan_stays_serial_and_not_ready",
           quiet_replay_plan_stays_serial_and_not_ready());

    std::cout << "\nSummary: " << passed << " passed, " << failed
              << " failed.\n";
    return failed == 0 ? 0 : 1;
}
