#ifndef FALL_N_REDUCED_RC_MULTISCALE_READINESS_GATE_HH
#define FALL_N_REDUCED_RC_MULTISCALE_READINESS_GATE_HH

// =============================================================================
//  ReducedRCMultiscaleReadinessGate.hh
// =============================================================================
//
//  Promotion gate for the reduced-RC multiscale handoff.
//
//  The validation reboot intentionally separates three facts that are easy to
//  conflate in a driver:
//
//    - a structural history is rich enough to select local sites,
//    - the local-site replay machinery ran over the selected histories, and
//    - a physical local XFEM/continuum solver has actually been bound.
//
//  This header keeps those facts explicit.  It allows an orchestration smoke to
//  unlock the next elastic FE2 wiring test, but it refuses to promote enriched
//  two-way FE2 until the replay result comes from a physical local solver.
//
// =============================================================================

#include <string_view>

#include "ReducedRCLocalSiteBatchPlan.hh"
#include "ReducedRCLocalSiteReplayRunner.hh"
#include "ReducedRCMultiscaleReplayPlan.hh"
#include "ReducedRCMultiscaleRuntimePolicy.hh"

namespace fall_n {

enum class ReducedRCMultiscaleReadinessStage {
    blocked,
    one_way_local_replay,
    elastic_fe2_smoke,
    guarded_enriched_fe2_smoke
};

[[nodiscard]] constexpr std::string_view to_string(
    ReducedRCMultiscaleReadinessStage stage) noexcept
{
    switch (stage) {
        case ReducedRCMultiscaleReadinessStage::blocked:
            return "blocked";
        case ReducedRCMultiscaleReadinessStage::one_way_local_replay:
            return "one_way_local_replay";
        case ReducedRCMultiscaleReadinessStage::elastic_fe2_smoke:
            return "elastic_fe2_smoke";
        case ReducedRCMultiscaleReadinessStage::guarded_enriched_fe2_smoke:
            return "guarded_enriched_fe2_smoke";
    }
    return "unknown_reduced_rc_multiscale_readiness_stage";
}

struct ReducedRCMultiscaleReadinessGateSettings {
    bool physical_local_solver_bound{false};
    bool allow_elastic_fe2_after_orchestration_smoke{true};
    std::string_view local_solver_label{"orchestration_smoke"};
};

struct ReducedRCMultiscaleReadinessGate {
    bool ready_for_one_way_local_replay{false};
    bool ready_for_elastic_fe2_smoke{false};
    bool ready_for_guarded_enriched_fe2_smoke{false};
    bool physical_local_solver_bound{false};
    ReducedRCMultiscaleReadinessStage next_stage{
        ReducedRCMultiscaleReadinessStage::blocked};
    std::string_view local_solver_label{"orchestration_smoke"};
    std::string_view blocking_reason{"replay plan has not been evaluated"};
};

[[nodiscard]] inline ReducedRCMultiscaleReadinessGate
make_reduced_rc_multiscale_readiness_gate(
    const ReducedRCMultiscaleReplayPlan& replay_plan,
    const ReducedRCMultiscaleRuntimePolicy& runtime_policy,
    const ReducedRCLocalSiteBatchPlan& batch_plan,
    const ReducedRCLocalSiteReplayRunResult& replay_result,
    ReducedRCMultiscaleReadinessGateSettings settings = {})
{
    ReducedRCMultiscaleReadinessGate gate{};
    gate.physical_local_solver_bound = settings.physical_local_solver_bound;
    gate.local_solver_label = settings.local_solver_label;

    if (!replay_plan.ready_for_one_way_replay) {
        gate.blocking_reason =
            "replay plan has no selected VTK-compliant local sites";
        return gate;
    }
    if (!runtime_policy.ready_for_local_site_batch) {
        gate.blocking_reason =
            "runtime policy is not ready to batch selected local sites";
        return gate;
    }
    if (!batch_plan.ready_for_local_site_batch ||
        batch_plan.selected_site_count == 0)
    {
        gate.blocking_reason =
            "local-site batch plan is not executable";
        return gate;
    }
    if (!replay_result.completed) {
        gate.next_stage =
            ReducedRCMultiscaleReadinessStage::one_way_local_replay;
        gate.blocking_reason =
            "selected local-site replay has not completed";
        return gate;
    }

    gate.ready_for_one_way_local_replay = true;
    gate.ready_for_elastic_fe2_smoke =
        settings.allow_elastic_fe2_after_orchestration_smoke;
    gate.next_stage = gate.ready_for_elastic_fe2_smoke
        ? ReducedRCMultiscaleReadinessStage::elastic_fe2_smoke
        : ReducedRCMultiscaleReadinessStage::one_way_local_replay;

    if (!settings.physical_local_solver_bound) {
        gate.blocking_reason =
            "local replay completed with an orchestration callback; bind the promoted XFEM local solver before enriched two-way FE2";
        return gate;
    }

    gate.ready_for_guarded_enriched_fe2_smoke =
        replay_result.ready_for_guarded_fe2_smoke;
    if (gate.ready_for_guarded_enriched_fe2_smoke) {
        gate.next_stage =
            ReducedRCMultiscaleReadinessStage::guarded_enriched_fe2_smoke;
        gate.blocking_reason = "all guarded enriched-FE2 prerequisites passed";
    } else {
        gate.blocking_reason =
            "physical local replay completed, but guarded enriched-FE2 metrics did not pass";
    }
    return gate;
}

} // namespace fall_n

#endif // FALL_N_REDUCED_RC_MULTISCALE_READINESS_GATE_HH
