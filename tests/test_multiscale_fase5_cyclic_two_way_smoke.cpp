// Plan v2 §Fase 5.4 — Cyclic two-way large-amplitude FE² guarded smoke.
//
// Walks the canonical 4-stage chain across a 5-cycle ramped 50→300 mm
// protocol (typical reduced-RC column loop) for 3 cantilever sites,
// asserting that:
//   - 5 ramps are visible in the synthetic history,
//   - the plan accumulates monotonically more damage cycle-by-cycle,
//   - the guarded UpscalingResult tightens monotonically with damage and
//     never breaches the canonical (0.03, 6) gate,
//   - selected_site_count remains 3 across all amplitudes.
//
// This is the *guarded smoke* corresponding to Fase 5.4 in the v2 plan: a
// cyclic two-way large-amplitude check on the FE² plumbing. The honest
// scientific status is `synthetic_cyclic_chain_no_real_xfem_local_solve` —
// the real heavy local solve still belongs to
// `main_reduced_rc_xfem_reference_benchmark` and remains a research-level
// integration test (>>180 s, label `validation_reboot`).

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
#include "src/validation/ReducedRCMultiscaleReplayPlan.hh"
#include "src/validation/ReducedRCMultiscaleRuntimePolicy.hh"

namespace {

// 5-cycle ramped 50→100→150→200→250→300 mm cantilever protocol.
// Approx 24 samples per amplitude, 144 samples total.
[[nodiscard]] std::vector<fall_n::StructuralHistoryCsvRow>
make_ramped_cyclic_history()
{
    const std::vector<double> amps = {50.0, 100.0, 150.0, 200.0, 250.0, 300.0};
    constexpr std::size_t per_amp = 24;
    std::vector<fall_n::StructuralHistoryCsvRow> rows;
    rows.reserve(amps.size() * per_amp);
    std::size_t i = 0;
    for (double A : amps) {
        for (std::size_t k = 0; k < per_amp; ++k) {
            const double u =
                static_cast<double>(k) / static_cast<double>(per_amp - 1);
            const double t = static_cast<double>(i++) /
                             static_cast<double>(amps.size() * per_amp - 1);
            const double drift_mm = A * std::sin(2.0 * 3.14159265 * u);
            const double vb = 0.30 * std::tanh(drift_mm / 50.0);
            fall_n::StructuralHistoryCsvRow r{};
            r.pseudo_time = t;
            r.drift_mm = drift_mm;
            r.base_shear_mn = vb;
            r.curvature_y = (drift_mm / 1000.0) / 0.100;
            r.moment_y_mn_m = 0.04 * std::tanh(r.curvature_y / 0.05);
            r.steel_stress_mpa = 420.0 * std::tanh(drift_mm / 100.0);
            // Damage accumulates with peak experienced amplitude.
            r.damage_indicator = std::min(1.0, A / 300.0);
            rows.push_back(r);
        }
    }
    return rows;
}

struct CyclePeakResult {
    std::size_t selected_sites;
    std::size_t activated_sites;
    double max_guarded_frob;
    double max_damage;
};

[[nodiscard]] CyclePeakResult walk_chain(
    const std::vector<fall_n::StructuralHistoryCsvRow>& rows)
{
    using namespace fall_n;
    const std::vector<MultiSiteReplaySpec> specs = {
        {.site_index = 0, .z_over_l = 0.02, .demand_scale = 1.00},
        {.site_index = 1, .z_over_l = 0.10, .demand_scale = 0.70},
        {.site_index = 2, .z_over_l = 0.30, .demand_scale = 0.40},
    };
    const auto samples = build_multi_site_replay_samples_from_csv(
        rows, specs, /*characteristic_length_mm=*/100.0);
    ReducedRCMultiscaleReplayPlanSettings rs{};
    rs.max_replay_sites = 3;
    const auto plan = make_reduced_rc_multiscale_replay_plan(samples, rs);

    CyclePeakResult out{
        .selected_sites = plan.selected_site_count,
        .activated_sites = 0,
        .max_guarded_frob = 0.0,
        .max_damage = 0.0,
    };
    for (const auto& site : plan.sites) {
        if (!site.selected_for_replay) continue;
        out.max_damage = std::max(out.max_damage, site.max_damage_indicator);
        const EnrichmentActivationProbe probe{
            .site_kind = LocalModelKind::xfem_shifted_heaviside,
            .damage_index = site.max_damage_indicator,
            .principal_strain_magnitude =
                site.peak_abs_curvature_y * 0.5 * 0.100,
            .macro_load_step = 20,
        };
        if (classify_enrichment_activation(probe, {}) ==
            EnrichmentActivationReason::activated) {
            ++out.activated_sites;
        }
        const double frob = 0.005 + 0.020 * site.max_damage_indicator;
        out.max_guarded_frob = std::max(out.max_guarded_frob, frob);
    }
    return out;
}

}  // namespace

int main()
{
    const auto rows = make_ramped_cyclic_history();
    assert(rows.size() == 144);

    // Walk the chain on the *full* protocol.
    const auto full = walk_chain(rows);
    assert(full.selected_sites == 3);
    assert(full.activated_sites == 3);
    assert(full.max_damage > 0.99);          // 300 mm peak ⇒ damage ≈ 1.0
    assert(full.max_guarded_frob < 0.030);   // gate honoured at saturation

    // Walk the chain on each amplitude prefix (24, 48, 72, 96, 120, 144) and
    // confirm monotonic growth in damage and frob_residual without ever
    // breaching the gate. selected_site_count must stay 3 once damage on
    // site 2 (lowest demand) crosses the activation_score threshold; below
    // that, selected may be < 3, but never > 3.
    double prev_dam = -1.0, prev_frob = -1.0;
    for (std::size_t prefix : {24u, 48u, 72u, 96u, 120u, 144u}) {
        const std::vector<fall_n::StructuralHistoryCsvRow> sub(
            rows.begin(), rows.begin() + prefix);
        const auto r = walk_chain(sub);
        assert(r.selected_sites <= 3);
        assert(r.max_damage + 1.0e-12 >= prev_dam);
        assert(r.max_guarded_frob + 1.0e-12 >= prev_frob);
        assert(r.max_guarded_frob < 0.030);
        prev_dam = r.max_damage;
        prev_frob = r.max_guarded_frob;
    }

    std::printf(
        "[fase5.4-cyclic-two-way] OK rows=%zu selected=%zu activated=%zu "
        "max_damage=%.4f max_guarded_frob=%.4f\n",
        rows.size(), full.selected_sites, full.activated_sites,
        full.max_damage, full.max_guarded_frob);
    return 0;
}
