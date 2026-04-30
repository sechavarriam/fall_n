#ifndef FALL_N_REDUCED_RC_MULTISCALE_RUNTIME_POLICY_HH
#define FALL_N_REDUCED_RC_MULTISCALE_RUNTIME_POLICY_HH

// =============================================================================
//  ReducedRCMultiscaleRuntimePolicy.hh
// =============================================================================
//
//  Bridge from the reduced-RC replay catalog to the generic multiscale runtime.
//
//  The replay planner decides which structural sites deserve local XFEM
//  replay.  The analysis runtime then needs concrete, cheap execution policy:
//  how many local checkpoints may stay in memory, whether adaptive activation
//  is enabled, and whether independent sites should be dispatched through the
//  OpenMP executor.  This file keeps that translation explicit and testable
//  instead of burying it in a benchmark driver.
//
// =============================================================================

#include <algorithm>
#include <cstddef>
#include <string_view>

#include "../analysis/LocalSubproblemRuntime.hh"
#include "ReducedRCMultiscaleReplayPlan.hh"

namespace fall_n {

enum class ReducedRCMultiscaleExecutorKind {
    serial_site_loop,
    openmp_site_parallel
};

[[nodiscard]] constexpr std::string_view to_string(
    ReducedRCMultiscaleExecutorKind kind) noexcept
{
    switch (kind) {
        case ReducedRCMultiscaleExecutorKind::serial_site_loop:
            return "serial_site_loop";
        case ReducedRCMultiscaleExecutorKind::openmp_site_parallel:
            return "openmp_site_parallel";
    }
    return "unknown_reduced_rc_multiscale_executor";
}

struct ReducedRCMultiscaleRuntimePolicy {
    LocalSubproblemRuntimeSettings local_runtime_settings{};
    ReducedRCMultiscaleExecutorKind executor_kind{
        ReducedRCMultiscaleExecutorKind::serial_site_loop};
    std::size_t recommended_site_threads{1};
    bool ready_for_local_site_batch{false};
    bool cache_budget_is_bounded{false};
    bool local_sites_run_in_parallel{false};
    bool direct_lu_kept_as_reference_only{false};
    bool iterative_preconditioner_expected{false};
    std::string_view rationale{"no selected replay sites"};
};

[[nodiscard]] inline ReducedRCMultiscaleRuntimePolicy
make_reduced_rc_multiscale_runtime_policy(
    const ReducedRCMultiscaleReplayPlan& plan,
    LocalSubproblemRuntimeSettings base_settings = {},
    std::size_t hardware_threads = 0)
{
    ReducedRCMultiscaleRuntimePolicy policy{};
    policy.local_runtime_settings = base_settings;
    policy.ready_for_local_site_batch = plan.ready_for_one_way_replay &&
                                        plan.vtk_contract_satisfied &&
                                        plan.selected_site_count > 0;

    policy.local_runtime_settings.profiling_enabled = true;
    policy.local_runtime_settings.seed_state_reuse_enabled =
        plan.seed_state_cache_recommended ||
        policy.local_runtime_settings.seed_state_reuse_enabled;
    policy.local_runtime_settings.restore_seed_before_solve =
        plan.newton_warm_start_recommended ||
        policy.local_runtime_settings.restore_seed_before_solve;

    if (policy.local_runtime_settings.seed_state_reuse_enabled &&
        policy.local_runtime_settings.max_cached_seed_states == 0 &&
        plan.selected_site_count > 0)
    {
        // The default production stance is bounded by the selected hot sites.
        // Callers can override this to zero for exploratory unbounded caches.
        policy.local_runtime_settings.max_cached_seed_states =
            plan.selected_site_count;
    }

    policy.local_runtime_settings.adaptive_activation_enabled =
        policy.local_runtime_settings.adaptive_activation_enabled ||
        plan.selected_site_count < plan.candidate_site_count;
    policy.local_runtime_settings.keep_active_once_triggered = true;
    policy.local_runtime_settings.prefer_active_seed_retention = true;

    policy.cache_budget_is_bounded =
        policy.local_runtime_settings.max_cached_seed_states > 0;
    policy.direct_lu_kept_as_reference_only = plan.avoid_direct_lu_for_batch;
    policy.iterative_preconditioner_expected =
        plan.avoid_direct_lu_for_batch ||
        plan.selected_direct_factorization_risk_mib > 512.0;

    policy.local_sites_run_in_parallel =
        plan.site_level_openmp_recommended && plan.selected_site_count > 1;
    if (policy.local_sites_run_in_parallel) {
        policy.executor_kind =
            ReducedRCMultiscaleExecutorKind::openmp_site_parallel;
        const auto available =
            hardware_threads == 0 ? plan.selected_site_count
                                  : hardware_threads;
        policy.recommended_site_threads =
            std::max<std::size_t>(
                1,
                std::min(plan.selected_site_count, available));
    }

    if (policy.ready_for_local_site_batch) {
        policy.rationale =
            policy.local_sites_run_in_parallel
                ? "run selected replay sites with bounded seed cache and OpenMP site parallelism"
                : "run selected replay sites with bounded seed cache";
    }

    return policy;
}

} // namespace fall_n

#endif // FALL_N_REDUCED_RC_MULTISCALE_RUNTIME_POLICY_HH
