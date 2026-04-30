#include <cmath>
#include <iostream>
#include <vector>

#include "src/validation/ReducedRCLocalSiteReplayRunner.hh"

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

using Sample = fall_n::ReducedRCStructuralReplaySample;

std::vector<Sample> make_two_site_history()
{
    return {
        Sample{
            .site_index = 0,
            .pseudo_time = 0.0,
            .z_over_l = 0.02,
            .drift_mm = 0.0},
        Sample{
            .site_index = 0,
            .pseudo_time = 0.5,
            .z_over_l = 0.02,
            .drift_mm = 25.0,
            .curvature_y = 0.018,
            .moment_y_mn_m = 0.030,
            .steel_stress_mpa = 350.0,
            .damage_indicator = 0.60,
            .work_increment_mn_mm = 1.20},
        Sample{
            .site_index = 1,
            .pseudo_time = 0.0,
            .z_over_l = 0.10,
            .drift_mm = 0.0},
        Sample{
            .site_index = 1,
            .pseudo_time = 0.5,
            .z_over_l = 0.10,
            .drift_mm = -20.0,
            .curvature_y = -0.016,
            .moment_y_mn_m = -0.026,
            .steel_stress_mpa = -330.0,
            .damage_indicator = 0.55,
            .work_increment_mn_mm = 1.00}
    };
}

fall_n::ReducedRCLocalSiteBatchPlan make_batch_plan(
    const std::vector<Sample>& history,
    std::size_t max_sites = 2)
{
    fall_n::ReducedRCMultiscaleReplayPlanSettings settings;
    settings.max_replay_sites = max_sites;
    settings.local_mesh = fall_n::ReducedRCLocalMeshScaleInput{
        .nx = 3,
        .ny = 3,
        .nz = 8,
        .constitutive_cost =
            fall_n::ReducedRCLocalConstitutiveCostKind::
                cyclic_crack_band_xfem,
        .shifted_heaviside_xfem = true,
        .longitudinal_bar_count = 8};

    const auto replay =
        fall_n::make_reduced_rc_multiscale_replay_plan(history, settings);
    const auto runtime =
        fall_n::make_reduced_rc_multiscale_runtime_policy(replay, {}, 4);
    return fall_n::make_reduced_rc_local_site_batch_plan(replay, runtime);
}

bool runner_completes_selected_site_histories()
{
    const auto history = make_two_site_history();
    const auto batch = make_batch_plan(history);
    const auto result = fall_n::run_reduced_rc_local_site_replay_batch(
        history,
        batch,
        [](const fall_n::ReducedRCLocalSiteReplayStepContext& context) {
            return fall_n::ReducedRCLocalSiteReplayStepResult{
                .converged = true,
                .nonlinear_iterations = 3,
                .elapsed_seconds = 0.01,
                .damage_indicator = context.target_sample.damage_indicator,
                .steel_stress_mpa = context.target_sample.steel_stress_mpa,
                .local_work_increment_mn_mm =
                    context.target_sample.work_increment_mn_mm};
        });

    return result.completed &&
           result.selected_site_count == 2 &&
           result.completed_site_count == 2 &&
           result.failed_site_count == 0 &&
           result.accepted_step_count == 4 &&
           result.total_nonlinear_iterations == 12 &&
           std::abs(result.peak_abs_steel_stress_mpa - 350.0) < 1.0e-12 &&
           std::abs(result.max_damage_indicator - 0.60) < 1.0e-12;
}

bool runner_bisects_failed_large_increment()
{
    std::vector<Sample> history = {
        {.site_index = 0, .pseudo_time = 0.0, .drift_mm = 0.0},
        {.site_index = 0,
         .pseudo_time = 1.0,
         .drift_mm = 10.0,
         .curvature_y = 0.020,
         .moment_y_mn_m = 0.030,
         .steel_stress_mpa = 360.0,
         .damage_indicator = 0.60,
         .work_increment_mn_mm = 2.0}};
    const auto batch = make_batch_plan(history, 1);
    const auto result = fall_n::run_reduced_rc_local_site_replay_batch(
        history,
        batch,
        [](const fall_n::ReducedRCLocalSiteReplayStepContext& context) {
            const double du = std::abs(
                context.target_sample.drift_mm -
                context.previous_sample.drift_mm);
            const bool ok = du <= 5.0 + 1.0e-12;
            return fall_n::ReducedRCLocalSiteReplayStepResult{
                .converged = ok,
                .nonlinear_iterations = ok ? 4 : 0,
                .damage_indicator = context.target_sample.damage_indicator,
                .steel_stress_mpa = context.target_sample.steel_stress_mpa,
                .local_work_increment_mn_mm =
                    context.target_sample.work_increment_mn_mm,
                .status_label = ok ? "converged" : "needs_cutback"};
        },
        fall_n::ReducedRCLocalSiteReplaySettings{
            .adaptive_cutback_enabled = true,
            .max_cutbacks_per_increment = 2});

    return result.completed &&
           result.failed_step_count == 1 &&
           result.generated_cutback_step_count == 2 &&
           result.accepted_step_count == 3 &&
           result.sites.front().max_cutback_level == 1;
}

bool runner_reports_cutback_exhaustion()
{
    std::vector<Sample> history = {
        {.site_index = 0, .pseudo_time = 0.0, .drift_mm = 0.0},
        {.site_index = 0,
         .pseudo_time = 1.0,
         .drift_mm = 10.0,
         .curvature_y = 0.020,
         .moment_y_mn_m = 0.030,
         .steel_stress_mpa = 360.0,
         .damage_indicator = 0.60,
         .work_increment_mn_mm = 2.0}};
    const auto batch = make_batch_plan(history, 1);
    const auto result = fall_n::run_reduced_rc_local_site_replay_batch(
        history,
        batch,
        [](const fall_n::ReducedRCLocalSiteReplayStepContext&) {
            return fall_n::ReducedRCLocalSiteReplayStepResult{
                .converged = false,
                .status_label = "always_fails"};
        },
        fall_n::ReducedRCLocalSiteReplaySettings{
            .adaptive_cutback_enabled = true,
            .max_cutbacks_per_increment = 1});

    return !result.completed &&
           result.failed_site_count == 1 &&
           result.sites.front().status ==
               fall_n::ReducedRCLocalSiteReplayStatus::
                   failed_cutback_exhausted;
}

bool runner_honors_fail_fast_setting()
{
    const auto history = make_two_site_history();
    const auto batch = make_batch_plan(history);
    const auto result = fall_n::run_reduced_rc_local_site_replay_batch(
        history,
        batch,
        [](const fall_n::ReducedRCLocalSiteReplayStepContext&) {
            return fall_n::ReducedRCLocalSiteReplayStepResult{
                .converged = false,
                .hard_failure = true,
                .status_label = "hard_local_failure"};
        },
        fall_n::ReducedRCLocalSiteReplaySettings{
            .continue_after_site_failure = false});

    return !result.completed &&
           result.sites.size() == 2 &&
           result.sites[0].status ==
               fall_n::ReducedRCLocalSiteReplayStatus::local_solver_failed &&
           result.sites[1].status ==
               fall_n::ReducedRCLocalSiteReplayStatus::not_scheduled &&
           result.attempted_step_count == 1;
}

bool runner_accepts_openmp_executor_contract()
{
    const auto history = make_two_site_history();
    const auto batch = make_batch_plan(history);
    const auto result = fall_n::run_reduced_rc_local_site_replay_batch(
        history,
        batch,
        [](const fall_n::ReducedRCLocalSiteReplayStepContext& context) {
            return fall_n::ReducedRCLocalSiteReplayStepResult{
                .converged = true,
                .nonlinear_iterations = 1,
                .damage_indicator = context.target_sample.damage_indicator,
                .steel_stress_mpa = context.target_sample.steel_stress_mpa,
                .local_work_increment_mn_mm =
                    context.target_sample.work_increment_mn_mm};
        },
        fall_n::ReducedRCLocalSiteReplaySettings{},
        fall_n::ReducedRCOpenMPSiteReplayExecutor{2});

    return result.completed &&
           result.completed_site_count == 2 &&
           result.accepted_step_count == 4 &&
           result.total_nonlinear_iterations == 4;
}

} // namespace

int main()
{
    std::cout << "=== Reduced RC Local Site Replay Runner ===\n";
    report("runner_completes_selected_site_histories",
           runner_completes_selected_site_histories());
    report("runner_bisects_failed_large_increment",
           runner_bisects_failed_large_increment());
    report("runner_reports_cutback_exhaustion",
           runner_reports_cutback_exhaustion());
    report("runner_honors_fail_fast_setting",
           runner_honors_fail_fast_setting());
    report("runner_accepts_openmp_executor_contract",
           runner_accepts_openmp_executor_contract());

    std::cout << "\nSummary: " << passed << " passed, " << failed
              << " failed.\n";
    return failed == 0 ? 0 : 1;
}
