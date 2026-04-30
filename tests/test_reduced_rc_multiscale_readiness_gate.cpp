#include <iostream>
#include <vector>

#include "src/validation/ReducedRCMultiscaleReadinessGate.hh"

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

std::vector<Sample> make_history()
{
    return {
        Sample{.site_index = 0, .pseudo_time = 0.0, .drift_mm = 0.0},
        Sample{.site_index = 0,
               .pseudo_time = 1.0,
               .drift_mm = 25.0,
               .curvature_y = 0.020,
               .moment_y_mn_m = 0.030,
               .steel_stress_mpa = 360.0,
               .damage_indicator = 0.60,
               .work_increment_mn_mm = 2.5}};
}

struct Chain {
    fall_n::ReducedRCMultiscaleReplayPlan replay{};
    fall_n::ReducedRCMultiscaleRuntimePolicy runtime{};
    fall_n::ReducedRCLocalSiteBatchPlan batch{};
    fall_n::ReducedRCLocalSiteReplayRunResult result{};
};

Chain make_completed_chain()
{
    const auto history = make_history();
    fall_n::ReducedRCMultiscaleReplayPlanSettings replay_settings;
    replay_settings.local_mesh = fall_n::ReducedRCLocalMeshScaleInput{
        .nx = 3,
        .ny = 3,
        .nz = 8,
        .constitutive_cost =
            fall_n::ReducedRCLocalConstitutiveCostKind::
                cyclic_crack_band_xfem,
        .shifted_heaviside_xfem = true,
        .longitudinal_bar_count = 8};

    Chain chain;
    chain.replay =
        fall_n::make_reduced_rc_multiscale_replay_plan(history,
                                                       replay_settings);
    chain.runtime =
        fall_n::make_reduced_rc_multiscale_runtime_policy(chain.replay);
    chain.batch =
        fall_n::make_reduced_rc_local_site_batch_plan(chain.replay,
                                                      chain.runtime);
    chain.result = fall_n::run_reduced_rc_local_site_replay_batch(
        history,
        chain.batch,
        [](const fall_n::ReducedRCLocalSiteReplayStepContext& context) {
            return fall_n::ReducedRCLocalSiteReplayStepResult{
                .converged = true,
                .nonlinear_iterations = 2,
                .damage_indicator = context.target_sample.damage_indicator,
                .steel_stress_mpa = context.target_sample.steel_stress_mpa,
                .local_work_increment_mn_mm =
                    context.target_sample.work_increment_mn_mm};
        });
    return chain;
}

bool gate_allows_elastic_smoke_after_orchestration_replay()
{
    const auto chain = make_completed_chain();
    const auto gate = fall_n::make_reduced_rc_multiscale_readiness_gate(
        chain.replay,
        chain.runtime,
        chain.batch,
        chain.result,
        fall_n::ReducedRCMultiscaleReadinessGateSettings{
            .physical_local_solver_bound = false});

    return gate.ready_for_one_way_local_replay &&
           gate.ready_for_elastic_fe2_smoke &&
           !gate.ready_for_guarded_enriched_fe2_smoke &&
           gate.next_stage ==
               fall_n::ReducedRCMultiscaleReadinessStage::elastic_fe2_smoke;
}

bool gate_promotes_guarded_fe2_only_after_physical_replay()
{
    const auto chain = make_completed_chain();
    const auto gate = fall_n::make_reduced_rc_multiscale_readiness_gate(
        chain.replay,
        chain.runtime,
        chain.batch,
        chain.result,
        fall_n::ReducedRCMultiscaleReadinessGateSettings{
            .physical_local_solver_bound = true,
            .local_solver_label = "promoted_xfem_local_model"});

    return gate.ready_for_one_way_local_replay &&
           gate.ready_for_elastic_fe2_smoke &&
           gate.ready_for_guarded_enriched_fe2_smoke &&
           gate.next_stage ==
               fall_n::ReducedRCMultiscaleReadinessStage::
                   guarded_enriched_fe2_smoke;
}

bool gate_blocks_failed_replay()
{
    auto chain = make_completed_chain();
    chain.result.completed = false;
    chain.result.failed_site_count = 1;
    const auto gate = fall_n::make_reduced_rc_multiscale_readiness_gate(
        chain.replay,
        chain.runtime,
        chain.batch,
        chain.result);

    return !gate.ready_for_one_way_local_replay &&
           !gate.ready_for_elastic_fe2_smoke &&
           !gate.ready_for_guarded_enriched_fe2_smoke &&
           gate.next_stage ==
               fall_n::ReducedRCMultiscaleReadinessStage::
                   one_way_local_replay;
}

} // namespace

int main()
{
    std::cout << "=== Reduced RC Multiscale Readiness Gate ===\n";
    report("gate_allows_elastic_smoke_after_orchestration_replay",
           gate_allows_elastic_smoke_after_orchestration_replay());
    report("gate_promotes_guarded_fe2_only_after_physical_replay",
           gate_promotes_guarded_fe2_only_after_physical_replay());
    report("gate_blocks_failed_replay", gate_blocks_failed_replay());

    std::cout << "\nSummary: " << passed << " passed, " << failed
              << " failed.\n";
    return failed == 0 ? 0 : 1;
}
