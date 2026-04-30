#ifndef FALL_N_REDUCED_RC_LOCAL_SITE_BATCH_PLAN_HH
#define FALL_N_REDUCED_RC_LOCAL_SITE_BATCH_PLAN_HH

// =============================================================================
//  ReducedRCLocalSiteBatchPlan.hh
// =============================================================================
//
//  Batch scheduler for the first reduced-RC multiscale replay runs.
//
//  The replay plan answers "which sites matter?".  The runtime policy answers
//  "what execution knobs are safe?".  This header answers the next practical
//  question before building expensive local models: how should selected local
//  sites be grouped under memory and solver constraints?
//
//  The scheduler is intentionally deterministic and conservative.  It does not
//  allocate PETSc objects, create meshes, or solve XFEM.  It produces an
//  auditable execution map for wrappers and drivers:
//
//    - selected site -> batch/slot,
//    - expected hot-state and factorization footprint,
//    - direct-LU reference versus iterative/ASM/field-split expectation,
//    - bounded seed-cache and warm-start requirements,
//    - and VTK replay responsibility.
//
// =============================================================================

#include <algorithm>
#include <cstddef>
#include <limits>
#include <string_view>
#include <vector>

#include "ReducedRCMultiscaleReplayPlan.hh"
#include "ReducedRCMultiscaleRuntimePolicy.hh"

namespace fall_n {

enum class ReducedRCLocalSiteBatchSolverKind {
    direct_lu_reference,
    iterative_aij_asm_or_fieldsplit,
    domain_decomposition_required
};

[[nodiscard]] constexpr std::string_view to_string(
    ReducedRCLocalSiteBatchSolverKind kind) noexcept
{
    switch (kind) {
        case ReducedRCLocalSiteBatchSolverKind::direct_lu_reference:
            return "direct_lu_reference";
        case ReducedRCLocalSiteBatchSolverKind::iterative_aij_asm_or_fieldsplit:
            return "iterative_aij_asm_or_fieldsplit";
        case ReducedRCLocalSiteBatchSolverKind::domain_decomposition_required:
            return "domain_decomposition_required";
    }
    return "unknown_reduced_rc_local_site_batch_solver";
}

struct ReducedRCLocalSiteBatchSettings {
    std::size_t max_concurrent_sites{0};
    double hot_state_budget_mib{1024.0};
    double direct_lu_factorization_budget_mib{512.0};
    bool allow_direct_lu_for_batch_smokes{true};
    bool vtk_time_series_required{true};
};

struct ReducedRCLocalSiteBatchRow {
    std::size_t batch_index{0};
    std::size_t slot_index{0};
    std::size_t site_index{0};
    double z_over_l{0.0};
    double activation_score{0.0};
    double estimated_hot_state_mib{0.0};
    double direct_factorization_risk_mib{0.0};
    ReducedRCLocalSiteBatchSolverKind solver_kind{
        ReducedRCLocalSiteBatchSolverKind::direct_lu_reference};
    bool seed_restore_required{false};
    bool warm_start_required{false};
    bool vtk_time_series_required{true};
    std::string_view rationale{};
};

struct ReducedRCLocalSiteBatch {
    std::size_t batch_index{0};
    std::size_t site_count{0};
    double estimated_hot_state_mib{0.0};
    double direct_factorization_risk_mib{0.0};
    std::size_t recommended_threads{1};
    ReducedRCLocalSiteBatchSolverKind dominant_solver_kind{
        ReducedRCLocalSiteBatchSolverKind::direct_lu_reference};
    bool uses_parallel_site_loop{false};
    bool within_hot_state_budget{true};
    bool direct_lu_within_budget{true};
};

struct ReducedRCLocalSiteBatchPlan {
    bool ready_for_local_site_batch{false};
    bool ready_for_many_site_replay{false};
    bool vtk_time_series_required{true};
    bool bounded_seed_cache_required{false};
    bool iterative_preconditioner_expected{false};
    std::size_t selected_site_count{0};
    std::size_t batch_count{0};
    std::size_t max_concurrent_sites{1};
    std::size_t recommended_site_threads{1};
    double total_estimated_hot_state_mib{0.0};
    double max_batch_hot_state_mib{0.0};
    double total_direct_factorization_risk_mib{0.0};
    ReducedRCMultiscaleExecutorKind executor_kind{
        ReducedRCMultiscaleExecutorKind::serial_site_loop};
    std::string_view rationale{"no selected replay sites"};
    std::vector<ReducedRCLocalSiteBatchRow> rows{};
    std::vector<ReducedRCLocalSiteBatch> batches{};
};

namespace detail {

[[nodiscard]] constexpr ReducedRCLocalSiteBatchSolverKind
promote_batch_solver_kind(ReducedRCLocalSiteBatchSolverKind a,
                          ReducedRCLocalSiteBatchSolverKind b) noexcept
{
    if (a == ReducedRCLocalSiteBatchSolverKind::domain_decomposition_required ||
        b == ReducedRCLocalSiteBatchSolverKind::domain_decomposition_required)
    {
        return ReducedRCLocalSiteBatchSolverKind::domain_decomposition_required;
    }
    if (a == ReducedRCLocalSiteBatchSolverKind::
                 iterative_aij_asm_or_fieldsplit ||
        b == ReducedRCLocalSiteBatchSolverKind::
                 iterative_aij_asm_or_fieldsplit)
    {
        return ReducedRCLocalSiteBatchSolverKind::
            iterative_aij_asm_or_fieldsplit;
    }
    return ReducedRCLocalSiteBatchSolverKind::direct_lu_reference;
}

[[nodiscard]] constexpr ReducedRCLocalSiteBatchSolverKind
classify_local_site_batch_solver(
    const ReducedRCMultiscaleReplaySitePlan& site,
    const ReducedRCMultiscaleRuntimePolicy& policy,
    const ReducedRCLocalSiteBatchSettings& settings) noexcept
{
    if (site.local_cost.solver_advice ==
        ReducedRCLocalSolverScalingAdviceKind::
            domain_decomposition_or_multiscale_batch_required)
    {
        return ReducedRCLocalSiteBatchSolverKind::
            domain_decomposition_required;
    }

    const bool smoke_direct_lu =
        site.local_cost.solver_advice ==
            ReducedRCLocalSolverScalingAdviceKind::direct_lu_smoke_ok &&
        settings.allow_direct_lu_for_batch_smokes &&
        !policy.iterative_preconditioner_expected;
    if (smoke_direct_lu) {
        return ReducedRCLocalSiteBatchSolverKind::direct_lu_reference;
    }

    return ReducedRCLocalSiteBatchSolverKind::
        iterative_aij_asm_or_fieldsplit;
}

} // namespace detail

[[nodiscard]] inline ReducedRCLocalSiteBatchPlan
make_reduced_rc_local_site_batch_plan(
    const ReducedRCMultiscaleReplayPlan& replay_plan,
    const ReducedRCMultiscaleRuntimePolicy& runtime_policy,
    ReducedRCLocalSiteBatchSettings settings = {})
{
    ReducedRCLocalSiteBatchPlan plan{};
    plan.ready_for_local_site_batch =
        runtime_policy.ready_for_local_site_batch &&
        replay_plan.ready_for_one_way_replay;
    plan.vtk_time_series_required = settings.vtk_time_series_required;
    plan.bounded_seed_cache_required = runtime_policy.cache_budget_is_bounded;
    plan.iterative_preconditioner_expected =
        runtime_policy.iterative_preconditioner_expected;
    plan.executor_kind = runtime_policy.executor_kind;
    plan.recommended_site_threads = runtime_policy.recommended_site_threads;
    plan.max_concurrent_sites =
        settings.max_concurrent_sites > 0
            ? settings.max_concurrent_sites
            : std::max<std::size_t>(1, runtime_policy.recommended_site_threads);

    if (!plan.ready_for_local_site_batch) {
        return plan;
    }

    for (const auto& site : replay_plan.sites) {
        if (!site.selected_for_replay) {
            continue;
        }
        ReducedRCLocalSiteBatchRow row{};
        row.site_index = site.site_index;
        row.z_over_l = site.z_over_l;
        row.activation_score = site.activation_score;
        row.estimated_hot_state_mib = site.local_cost.estimated_hot_state_mib;
        row.direct_factorization_risk_mib =
            site.local_cost.direct_factorization_risk_mib;
        row.solver_kind = detail::classify_local_site_batch_solver(
            site,
            runtime_policy,
            settings);
        row.seed_restore_required =
            runtime_policy.local_runtime_settings.seed_state_reuse_enabled;
        row.warm_start_required =
            runtime_policy.local_runtime_settings.restore_seed_before_solve;
        row.vtk_time_series_required = settings.vtk_time_series_required;
        row.rationale =
            row.solver_kind ==
                    ReducedRCLocalSiteBatchSolverKind::direct_lu_reference
                ? "direct LU allowed only as isolated/smoke reference"
                : "iterative or decomposed local solve expected for replay batch";
        plan.rows.push_back(row);
    }

    auto starts_new_batch = [&](const ReducedRCLocalSiteBatch& current,
                               const ReducedRCLocalSiteBatchRow& row) {
        if (current.site_count == 0) {
            return false;
        }
        if (current.site_count >= plan.max_concurrent_sites) {
            return true;
        }
        if (settings.hot_state_budget_mib > 0.0 &&
            current.estimated_hot_state_mib +
                    row.estimated_hot_state_mib >
                settings.hot_state_budget_mib)
        {
            return true;
        }
        if (row.solver_kind ==
                ReducedRCLocalSiteBatchSolverKind::direct_lu_reference &&
            settings.direct_lu_factorization_budget_mib > 0.0 &&
            current.direct_factorization_risk_mib +
                    row.direct_factorization_risk_mib >
                settings.direct_lu_factorization_budget_mib)
        {
            return true;
        }
        return false;
    };

    ReducedRCLocalSiteBatch current{};
    auto flush_batch = [&]() {
        if (current.site_count == 0) {
            return;
        }
        current.recommended_threads =
            current.uses_parallel_site_loop
                ? std::min(current.site_count, plan.recommended_site_threads)
                : std::size_t{1};
        current.within_hot_state_budget =
            settings.hot_state_budget_mib <= 0.0 ||
            current.estimated_hot_state_mib <= settings.hot_state_budget_mib;
        current.direct_lu_within_budget =
            current.dominant_solver_kind !=
                ReducedRCLocalSiteBatchSolverKind::direct_lu_reference ||
            settings.direct_lu_factorization_budget_mib <= 0.0 ||
            current.direct_factorization_risk_mib <=
                settings.direct_lu_factorization_budget_mib;
        plan.max_batch_hot_state_mib =
            std::max(plan.max_batch_hot_state_mib,
                     current.estimated_hot_state_mib);
        plan.batches.push_back(current);
        current = ReducedRCLocalSiteBatch{};
        current.batch_index = plan.batches.size();
    };

    for (auto& row : plan.rows) {
        if (starts_new_batch(current, row)) {
            flush_batch();
        }
        row.batch_index = current.batch_index;
        row.slot_index = current.site_count;
        ++current.site_count;
        current.estimated_hot_state_mib += row.estimated_hot_state_mib;
        current.direct_factorization_risk_mib +=
            row.direct_factorization_risk_mib;
        current.dominant_solver_kind =
            detail::promote_batch_solver_kind(current.dominant_solver_kind,
                                              row.solver_kind);
        current.uses_parallel_site_loop =
            runtime_policy.local_sites_run_in_parallel;
        plan.total_estimated_hot_state_mib += row.estimated_hot_state_mib;
        plan.total_direct_factorization_risk_mib +=
            row.direct_factorization_risk_mib;
    }
    flush_batch();

    plan.selected_site_count = plan.rows.size();
    plan.batch_count = plan.batches.size();
    plan.ready_for_many_site_replay =
        plan.ready_for_local_site_batch &&
        plan.selected_site_count > 1 &&
        plan.executor_kind ==
            ReducedRCMultiscaleExecutorKind::openmp_site_parallel;
    plan.rationale =
        plan.ready_for_many_site_replay
            ? "selected replay sites can run as memory-budgeted OpenMP batches"
            : "selected replay sites can run as a memory-budgeted serial batch";
    return plan;
}

} // namespace fall_n

#endif // FALL_N_REDUCED_RC_LOCAL_SITE_BATCH_PLAN_HH
