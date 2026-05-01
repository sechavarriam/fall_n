// Plan v2 §Fase 4A regression — multiscale one-way replay PLAN smoke.
//
// Asserts that with a synthetic ramped-drift history the planner promotes a
// single site to `guarded_two_way_candidate`, marks `ready_for_one_way_replay`,
// and produces a runtime policy that is `ready_for_local_site_batch` under
// the canonical (default) settings used by the heavy XFEM driver.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "src/validation/ReducedRCMultiscaleReplayPlan.hh"
#include "src/validation/ReducedRCMultiscaleRuntimePolicy.hh"

int main() {
    using namespace fall_n;

    std::vector<ReducedRCStructuralReplaySample> hist;
    constexpr std::size_t N = 97;
    const double L_m = 0.100; // 100 mm characteristic length.
    double prev_drift = 0.0;
    double prev_vb = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(N - 1);
        // Ramp-with-cyclic synthetic loading 0..200 mm.
        const double drift_mm = 200.0 * t * std::sin(4.0 * t * 3.14159265);
        const double vb = 0.30 * std::tanh(drift_mm / 50.0); // MN
        const double curvature = (drift_mm / 1000.0) / L_m;
        const double d_drift = i == 0 ? 0.0 : drift_mm - prev_drift;
        const double d_work  = i == 0 ? 0.0 : 0.5 * (vb + prev_vb) * d_drift;
        hist.push_back(ReducedRCStructuralReplaySample{
            .site_index = 0,
            .pseudo_time = t,
            .physical_time = t,
            .z_over_l = 0.02,
            .drift_mm = drift_mm,
            .curvature_y = curvature,
            .moment_y_mn_m = 0.04 * std::tanh(curvature / 0.05),
            .base_shear_mn = vb,
            .steel_stress_mpa = 420.0 * std::tanh(drift_mm / 100.0),
            .damage_indicator = std::min(1.0, std::abs(drift_mm) / 250.0),
            .work_increment_mn_mm = d_work,
        });
        prev_drift = drift_mm; prev_vb = vb;
    }

    const auto plan = make_reduced_rc_multiscale_replay_plan(hist, {});
    assert(plan.history_sample_count == N);
    assert(plan.candidate_site_count == 1);
    assert(plan.selected_site_count == 1);
    assert(plan.ready_for_one_way_replay);
    assert(!plan.sites.empty());
    const auto& s = plan.sites.front();
    assert(s.selected_for_replay);
    // Synthetic peak drift ~200mm with damage ≈1 promotes guarded candidate.
    assert(s.activation_kind ==
           ReducedRCReplaySiteActivationKind::guarded_two_way_candidate);

    const auto policy = make_reduced_rc_multiscale_runtime_policy(plan);
    assert(policy.ready_for_local_site_batch);
    assert(policy.cache_budget_is_bounded);
    assert(policy.local_runtime_settings.seed_state_reuse_enabled);
    assert(policy.local_runtime_settings.max_cached_seed_states >= 1);

    std::printf("[fase4a-test] OK samples=%zu selected=%zu activation=%s "
                "ready_one_way=%d ready_batch=%d\n",
                plan.history_sample_count, plan.selected_site_count,
                "guarded_two_way_candidate",
                plan.ready_for_one_way_replay,
                policy.ready_for_local_site_batch);
    return 0;
}
