// Plan v2 §Fase 4 chain smoke — exercises 4A→4B→4C→4D in-memory.
//
// Builds a synthetic 97-sample ramped-cyclic 200 mm history, runs:
//   1. ReducedRCMultiscaleReplayPlan          (Fase 4A)
//   2. ReducedRCMultiscaleRuntimePolicy       (Fase 4A)
//   3. ReducedRCLocalSiteBatchPlan            (Fase 4B)
//   4. Synthetic ELASTIC UpscalingResult      (Fase 4C)
//   5. EnrichmentActivationPolicy probe +
//      synthetic GUARDED UpscalingResult      (Fase 4D)
// and asserts the stage-by-stage gates declared by the driver suite.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include <Eigen/Dense>

#include "src/analysis/MultiscaleTypes.hh"
#include "src/reconstruction/EnrichmentActivationPolicy.hh"
#include "src/reconstruction/LocalModelKind.hh"
#include "src/validation/ReducedRCLocalSiteBatchPlan.hh"
#include "src/validation/ReducedRCMultiscaleReplayPlan.hh"
#include "src/validation/ReducedRCMultiscaleRuntimePolicy.hh"

namespace {

[[nodiscard]] std::vector<fall_n::ReducedRCStructuralReplaySample>
make_synthetic_history()
{
    using namespace fall_n;
    std::vector<ReducedRCStructuralReplaySample> hist;
    constexpr std::size_t N = 97;
    const double L_m = 0.100;
    double prev_drift = 0.0, prev_vb = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(N - 1);
        const double drift_mm = 200.0 * t * std::sin(4.0 * t * 3.14159265);
        const double vb = 0.30 * std::tanh(drift_mm / 50.0);
        const double curv = (drift_mm / 1000.0) / L_m;
        const double d_drift = i == 0 ? 0.0 : drift_mm - prev_drift;
        const double d_work  = i == 0 ? 0.0 : 0.5 * (vb + prev_vb) * d_drift;
        hist.push_back(ReducedRCStructuralReplaySample{
            .site_index = 0,
            .pseudo_time = t,
            .physical_time = t,
            .z_over_l = 0.02,
            .drift_mm = drift_mm,
            .curvature_y = curv,
            .moment_y_mn_m = 0.04 * std::tanh(curv / 0.05),
            .base_shear_mn = vb,
            .steel_stress_mpa = 420.0 * std::tanh(drift_mm / 100.0),
            .damage_indicator = std::min(1.0, std::abs(drift_mm) / 250.0),
            .work_increment_mn_mm = d_work,
        });
        prev_drift = drift_mm; prev_vb = vb;
    }
    return hist;
}

}  // namespace

int main() {
    using namespace fall_n;

    // ── Fase 4A ───────────────────────────────────────────────────────
    const auto hist = make_synthetic_history();
    const auto plan = make_reduced_rc_multiscale_replay_plan(hist, {});
    assert(plan.history_sample_count == 97);
    assert(plan.selected_site_count == 1);
    assert(plan.ready_for_one_way_replay);
    const auto policy = make_reduced_rc_multiscale_runtime_policy(plan);
    assert(policy.ready_for_local_site_batch);
    assert(policy.cache_budget_is_bounded);

    // ── Fase 4B ───────────────────────────────────────────────────────
    const auto bp = make_reduced_rc_local_site_batch_plan(plan, policy, {});
    assert(bp.ready_for_local_site_batch);
    assert(bp.selected_site_count == 1);
    assert(bp.batch_count >= 1);
    assert(!bp.rows.empty());
    assert(bp.rows.front().solver_kind ==
           ReducedRCLocalSiteBatchSolverKind::direct_lu_reference);

    // ── Fase 4C synthetic elastic UpscalingResult ────────────────────
    const auto& site = plan.sites.front();
    UpscalingResult elastic{};
    elastic.eps_ref = Eigen::VectorXd::Zero(2);
    elastic.eps_ref(1) = site.peak_abs_curvature_y;
    elastic.D_hom = Eigen::MatrixXd::Zero(2, 2);
    elastic.D_hom(0, 0) = 6.0e3;   // EA MN
    elastic.D_hom(1, 1) = 30.0;    // EI MN·m²
    elastic.f_hom = elastic.D_hom * elastic.eps_ref;
    elastic.frobenius_residual = 0.0;
    elastic.snes_iters = 1;
    elastic.converged = true;
    elastic.status = ResponseStatus::Ok;
    elastic.tangent_scheme = TangentLinearizationScheme::LinearizedCondensation;
    elastic.condensed_status = CondensedTangentStatus::Success;
    assert(elastic.passes_guarded_smoke_gate(0.03, 6));

    // ── Fase 4D enrichment-activation probe + guarded UpscalingResult ──
    const EnrichmentActivationProbe probe{
        .site_kind = LocalModelKind::xfem_shifted_heaviside,
        .damage_index = site.max_damage_indicator,
        .principal_strain_magnitude =
            site.peak_abs_curvature_y * 0.5 * 0.100,
        .macro_load_step = 20,
    };
    const auto reason = classify_enrichment_activation(probe, {});
    assert(reason == EnrichmentActivationReason::activated);

    UpscalingResult guarded = elastic;
    const double s = std::clamp(1.0 - site.max_damage_indicator, 0.05, 1.0);
    guarded.D_hom(0, 0) = s * 6.0e3;
    guarded.D_hom(1, 1) = s * 30.0;
    guarded.f_hom = guarded.D_hom * guarded.eps_ref;
    guarded.frobenius_residual = 0.005 + 0.020 * site.max_damage_indicator;
    guarded.snes_iters = 3;
    // At damage=1.0 frob_residual = 0.025 < 0.030, so the canonical gate
    // (0.03, 6) must pass without loosening.
    assert(guarded.passes_guarded_smoke_gate(0.03, 6));

    std::printf("[fase4-chain] OK 4A.selected=%zu 4B.batches=%zu 4C.elastic=PASS "
                "4D.activated=1 4D.frob=%.4f\n",
                plan.selected_site_count, bp.batch_count,
                guarded.frobenius_residual);
    return 0;
}
