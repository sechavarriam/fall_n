#include <iostream>
#include <vector>

#include "src/validation/ReducedRCLocalSiteBatchPlan.hh"

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

std::vector<fall_n::ReducedRCStructuralReplaySample> make_hot_history()
{
    using Sample = fall_n::ReducedRCStructuralReplaySample;
    return {
        Sample{
            .site_index = 0,
            .z_over_l = 0.02,
            .curvature_y = 0.022,
            .moment_y_mn_m = 0.034,
            .base_shear_mn = 0.012,
            .steel_stress_mpa = 370.0,
            .damage_indicator = 0.72,
            .work_increment_mn_mm = 1.70},
        Sample{
            .site_index = 1,
            .z_over_l = 0.10,
            .curvature_y = -0.018,
            .moment_y_mn_m = -0.026,
            .base_shear_mn = -0.011,
            .steel_stress_mpa = -345.0,
            .damage_indicator = 0.58,
            .work_increment_mn_mm = 1.25},
        Sample{
            .site_index = 2,
            .z_over_l = 0.30,
            .curvature_y = 0.012,
            .moment_y_mn_m = 0.018,
            .base_shear_mn = 0.009,
            .steel_stress_mpa = 302.0,
            .damage_indicator = 0.30,
            .work_increment_mn_mm = 0.90},
        Sample{
            .site_index = 3,
            .z_over_l = 0.80,
            .curvature_y = 0.001,
            .moment_y_mn_m = 0.001,
            .steel_stress_mpa = 20.0,
            .damage_indicator = 0.0,
            .work_increment_mn_mm = 0.01}
    };
}

fall_n::ReducedRCMultiscaleReplayPlan make_replay_plan(
    std::size_t max_sites,
    fall_n::ReducedRCLocalMeshScaleInput local_mesh)
{
    fall_n::ReducedRCMultiscaleReplayPlanSettings settings;
    settings.max_replay_sites = max_sites;
    settings.local_mesh = local_mesh;
    return fall_n::make_reduced_rc_multiscale_replay_plan(
        make_hot_history(),
        settings);
}

bool large_mesh_sites_are_batched_for_iterative_openmp_replay()
{
    const auto replay = make_replay_plan(
        3,
        fall_n::ReducedRCLocalMeshScaleInput{
            .nx = 7,
            .ny = 7,
            .nz = 25,
            .constitutive_cost =
                fall_n::ReducedRCLocalConstitutiveCostKind::
                    cyclic_crack_band_xfem,
            .shifted_heaviside_xfem = true,
            .longitudinal_bar_count = 8});
    const auto runtime =
        fall_n::make_reduced_rc_multiscale_runtime_policy(
            replay,
            {},
            8);
    const auto batch = fall_n::make_reduced_rc_local_site_batch_plan(
        replay,
        runtime,
        fall_n::ReducedRCLocalSiteBatchSettings{
            .hot_state_budget_mib = 1024.0,
            .direct_lu_factorization_budget_mib = 512.0});

    return replay.selected_site_count == 3 &&
           runtime.local_sites_run_in_parallel &&
           batch.ready_for_local_site_batch &&
           batch.ready_for_many_site_replay &&
           batch.selected_site_count == 3 &&
           batch.batch_count == 1 &&
           batch.batches.front().recommended_threads == 3 &&
           batch.batches.front().dominant_solver_kind ==
               fall_n::ReducedRCLocalSiteBatchSolverKind::
                   iterative_aij_asm_or_fieldsplit &&
           batch.rows.front().seed_restore_required &&
           batch.rows.front().warm_start_required;
}

bool tight_memory_budget_splits_batches_deterministically()
{
    const auto replay = make_replay_plan(
        3,
        fall_n::ReducedRCLocalMeshScaleInput{
            .nx = 7,
            .ny = 7,
            .nz = 25,
            .constitutive_cost =
                fall_n::ReducedRCLocalConstitutiveCostKind::
                    cyclic_crack_band_xfem,
            .shifted_heaviside_xfem = true,
            .longitudinal_bar_count = 8});
    const auto runtime =
        fall_n::make_reduced_rc_multiscale_runtime_policy(replay, {}, 8);
    const auto single_site_hot =
        replay.sites.front().local_cost.estimated_hot_state_mib;
    const auto batch = fall_n::make_reduced_rc_local_site_batch_plan(
        replay,
        runtime,
        fall_n::ReducedRCLocalSiteBatchSettings{
            .hot_state_budget_mib = single_site_hot * 1.25,
            .direct_lu_factorization_budget_mib = 512.0});

    return batch.batch_count == 3 &&
           batch.rows[0].batch_index == 0 &&
           batch.rows[1].batch_index == 1 &&
           batch.rows[2].batch_index == 2;
}

bool smoke_mesh_keeps_direct_lu_reference_path()
{
    const auto replay = make_replay_plan(
        1,
        fall_n::ReducedRCLocalMeshScaleInput{
            .nx = 1,
            .ny = 1,
            .nz = 2,
            .constitutive_cost =
                fall_n::ReducedRCLocalConstitutiveCostKind::
                    cyclic_crack_band_xfem,
            .shifted_heaviside_xfem = true,
            .longitudinal_bar_count = 8});
    const auto runtime =
        fall_n::make_reduced_rc_multiscale_runtime_policy(replay);
    const auto batch =
        fall_n::make_reduced_rc_local_site_batch_plan(replay, runtime);

    return batch.ready_for_local_site_batch &&
           !batch.ready_for_many_site_replay &&
           batch.batch_count == 1 &&
           batch.rows.front().solver_kind ==
               fall_n::ReducedRCLocalSiteBatchSolverKind::
                   direct_lu_reference &&
           batch.batches.front().direct_lu_within_budget;
}

} // namespace

int main()
{
    std::cout << "=== Reduced RC Local Site Batch Plan ===\n";
    report("large_mesh_sites_are_batched_for_iterative_openmp_replay",
           large_mesh_sites_are_batched_for_iterative_openmp_replay());
    report("tight_memory_budget_splits_batches_deterministically",
           tight_memory_budget_splits_batches_deterministically());
    report("smoke_mesh_keeps_direct_lu_reference_path",
           smoke_mesh_keeps_direct_lu_reference_path());

    std::cout << "\nSummary: " << passed << " passed, " << failed
              << " failed.\n";
    return failed == 0 ? 0 : 1;
}
