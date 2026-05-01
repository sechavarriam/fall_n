// Plan v2 §Fase 5.3 multi-site chain smoke — cantilever-style 4A→4D with 3 sites.
//
// Builds the same synthetic 97-sample ramped-cyclic 200 mm history used by
// `test_multiscale_fase4_chain_smoke` and replicates it across 3 cantilever
// sites with linearly decaying demand (z_over_l = 0.02/0.10/0.30, scales
// 1.00/0.70/0.40). Then walks the canonical 4-stage chain in-memory:
//   1. ReducedRCMultiscaleReplayPlan          (Fase 4A)
//   2. ReducedRCMultiscaleRuntimePolicy       (Fase 4A)
//   3. ReducedRCLocalSiteBatchPlan            (Fase 4B)
//   4. Per-site elastic + guarded UpscalingResult  (Fase 4C/4D)
//   5. EnrichmentActivationPolicy probe per site
// and asserts every gate passes for every selected site.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include <Eigen/Dense>

#include "src/analysis/MultiscaleTypes.hh"
#include "src/reconstruction/EnrichmentActivationPolicy.hh"
#include "src/reconstruction/LocalModelKind.hh"
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
        r.damage_indicator = std::min(1.0, std::abs(drift_mm) / 250.0);
        rows.push_back(r);
    }
    return rows;
}

}  // namespace

int main()
{
    using namespace fall_n;

    const auto rows = make_synthetic_rows();
    const std::vector<MultiSiteReplaySpec> specs = {
        {.site_index = 0, .z_over_l = 0.02, .demand_scale = 1.00},
        {.site_index = 1, .z_over_l = 0.10, .demand_scale = 0.70},
        {.site_index = 2, .z_over_l = 0.30, .demand_scale = 0.40},
    };
    const auto samples = build_multi_site_replay_samples_from_csv(
        rows, specs, /*characteristic_length_mm=*/100.0);

    // ── Fase 4A ───────────────────────────────────────────────────────
    ReducedRCMultiscaleReplayPlanSettings rs{};
    rs.max_replay_sites = 3;
    const auto plan = make_reduced_rc_multiscale_replay_plan(samples, rs);
    assert(plan.candidate_site_count == 3);
    assert(plan.selected_site_count == 3);
    assert(plan.ready_for_one_way_replay);
    const auto policy = make_reduced_rc_multiscale_runtime_policy(plan);
    assert(policy.ready_for_local_site_batch);
    assert(policy.cache_budget_is_bounded);

    // ── Fase 4B ───────────────────────────────────────────────────────
    const auto bp = make_reduced_rc_local_site_batch_plan(plan, policy, {});
    assert(bp.ready_for_local_site_batch);
    assert(bp.selected_site_count == 3);
    assert(bp.batch_count >= 1);
    assert(bp.rows.size() == 3);

    // Activation scores must decrease monotonically with z_over_l.
    double prev = std::numeric_limits<double>::infinity();
    for (const auto& s : plan.sites) {
        assert(s.activation_score <= prev + 1.0e-12);
        prev = s.activation_score;
    }

    // ── Fase 4C/4D per site ──────────────────────────────────────────
    std::size_t activated_count = 0;
    std::size_t passed_gate_count = 0;
    double max_guarded_frob = 0.0;
    for (const auto& site : plan.sites) {
        if (!site.selected_for_replay) continue;
        // Elastic baseline.
        UpscalingResult elastic{};
        elastic.eps_ref = Eigen::VectorXd::Zero(2);
        elastic.eps_ref(1) = site.peak_abs_curvature_y;
        elastic.D_hom = Eigen::MatrixXd::Zero(2, 2);
        elastic.D_hom(0, 0) = 6.0e3;
        elastic.D_hom(1, 1) = 30.0;
        elastic.f_hom = elastic.D_hom * elastic.eps_ref;
        elastic.frobenius_residual = 0.0;
        elastic.snes_iters = 1;
        elastic.converged = true;
        elastic.status = ResponseStatus::Ok;
        elastic.tangent_scheme = TangentLinearizationScheme::LinearizedCondensation;
        elastic.condensed_status = CondensedTangentStatus::Success;
        assert(elastic.passes_guarded_smoke_gate(0.03, 6));

        // Activation probe.
        const EnrichmentActivationProbe probe{
            .site_kind = LocalModelKind::xfem_shifted_heaviside,
            .damage_index = site.max_damage_indicator,
            .principal_strain_magnitude =
                site.peak_abs_curvature_y * 0.5 * 0.100,
            .macro_load_step = 20,
        };
        const auto reason = classify_enrichment_activation(probe, {});

        // Guarded synthetic.
        UpscalingResult guarded = elastic;
        const double s = std::clamp(1.0 - site.max_damage_indicator, 0.05, 1.0);
        guarded.D_hom(0, 0) = s * 6.0e3;
        guarded.D_hom(1, 1) = s * 30.0;
        guarded.f_hom = guarded.D_hom * guarded.eps_ref;
        guarded.frobenius_residual =
            0.005 + 0.020 * site.max_damage_indicator;
        guarded.snes_iters = 3;
        max_guarded_frob = std::max(max_guarded_frob, guarded.frobenius_residual);

        if (reason == EnrichmentActivationReason::activated) ++activated_count;
        if (guarded.passes_guarded_smoke_gate(0.03, 6)) ++passed_gate_count;
    }
    // Top sites (highest demand) must activate. The lowest-demand site at
    // z=0.30 with scale 0.40 has damage at most 0.32 → still activates the
    // damage gate (>=0.20). Therefore all 3 must activate.
    assert(activated_count == 3);
    assert(passed_gate_count == 3);
    assert(max_guarded_frob < 0.030);

    std::printf("[fase5.3-multi-site-chain] OK sites=%zu activated=%zu "
                "passed_gate=%zu max_guarded_frob=%.4f\n",
                plan.selected_site_count, activated_count,
                passed_gate_count, max_guarded_frob);
    return 0;
}
