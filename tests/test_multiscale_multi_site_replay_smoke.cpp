// Plan v2 §Fase 4 — Multi-site replay smoke (Fase 5.3 stepping stone).
//
// Drives build_multi_site_replay_samples_from_csv with three sites along a
// synthetic cantilever (z_over_l = 0.02 / 0.10 / 0.30 with linearly decaying
// demand 1.0 / 0.7 / 0.4) and asserts that:
//   - all 3 sites are created,
//   - all 3 are selected_for_replay (selected_site_count == 3),
//   - activation scores decrease monotonically with z_over_l,
//   - the local-site batch plan packs all 3 sites,
//   - the staggered chain remains ready_for_local_site_batch.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <vector>

#include "src/validation/MultiscaleReplayDriverHelpers.hh"
#include "src/validation/ReducedRCLocalSiteBatchPlan.hh"
#include "src/validation/ReducedRCMultiscaleReplayPlan.hh"
#include "src/validation/ReducedRCMultiscaleRuntimePolicy.hh"

namespace {

[[nodiscard]] std::vector<fall_n::StructuralHistoryCsvRow>
make_synthetic_rows()
{
    std::vector<fall_n::StructuralHistoryCsvRow> rows;
    constexpr std::size_t N = 97;
    for (std::size_t i = 0; i < N; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(N - 1);
        const double drift_mm = 200.0 * t * std::sin(4.0 * t * 3.14159265);
        const double vb = 0.30 * std::tanh(drift_mm / 50.0);
        fall_n::StructuralHistoryCsvRow r{};
        r.pseudo_time = t;
        r.drift_mm = drift_mm;
        r.base_shear_mn = vb;
        r.curvature_y = (drift_mm / 1000.0) / 0.100;
        r.moment_y_mn_m = 0.04 * std::tanh(r.curvature_y / 0.05);
        r.steel_stress_mpa = 420.0 * std::tanh(drift_mm / 100.0);
        r.damage_indicator = std::min(1.0, std::abs(drift_mm) / 200.0);
        rows.push_back(r);
    }
    return rows;
}

}  // namespace

int main()
{
    using namespace fall_n;

    const auto rows = make_synthetic_rows();
    const std::vector<MultiSiteReplaySpec> sites = {
        {.site_index = 0, .z_over_l = 0.02, .demand_scale = 1.0},
        {.site_index = 1, .z_over_l = 0.10, .demand_scale = 0.7},
        {.site_index = 2, .z_over_l = 0.30, .demand_scale = 0.4},
    };
    const auto samples = build_multi_site_replay_samples_from_csv(
        rows, sites, /*characteristic_length_mm=*/100.0);
    assert(samples.size() == rows.size() * sites.size());

    ReducedRCMultiscaleReplayPlanSettings settings{};
    settings.max_replay_sites = 3;
    const auto plan = make_reduced_rc_multiscale_replay_plan(samples, settings);

    assert(plan.candidate_site_count == 3);
    assert(plan.selected_site_count == 3);
    assert(plan.ready_for_one_way_replay);

    // Plan sites are sorted by activation_score descending; the highest must
    // be the base site (z_over_l == 0.02, demand_scale 1.0). Score must be
    // monotonically non-increasing across the three sites.
    double prev_score = std::numeric_limits<double>::infinity();
    for (const auto& s : plan.sites) {
        assert(s.activation_score <= prev_score + 1.0e-12);
        prev_score = s.activation_score;
    }
    assert(plan.sites.front().z_over_l < plan.sites.back().z_over_l);

    const auto policy = make_reduced_rc_multiscale_runtime_policy(plan);
    ReducedRCLocalSiteBatchSettings bs{};
    bs.max_concurrent_sites = 0;       // auto
    bs.hot_state_budget_mib = 1024.0;
    bs.direct_lu_factorization_budget_mib = 512.0;
    const auto batch = make_reduced_rc_local_site_batch_plan(plan, policy, bs);

    assert(batch.ready_for_local_site_batch);
    assert(batch.selected_site_count == 3);
    assert(batch.batch_count >= 1);
    assert(batch.rows.size() == 3);

    std::printf(
        "[fase4-multi-site] OK sites=%zu selected=%zu batches=%zu "
        "scores=[%.3f,%.3f,%.3f]\n",
        plan.candidate_site_count, plan.selected_site_count, batch.batch_count,
        plan.sites[0].activation_score, plan.sites[1].activation_score,
        plan.sites[2].activation_score);
    return 0;
}
