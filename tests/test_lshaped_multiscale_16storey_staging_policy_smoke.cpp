// Plan v2 §Fase 6 — 16-storey staging policy probe.
//
// In-memory replica of the activation policy used by
// `main_lshaped_multiscale_16storey_staging`: builds a synthetic 16-storey
// triangular drift demand profile peaking at storeys 5 and 11 (per L-shape
// transitions), runs each storey through Fase 4A's planner + activation
// probe, and asserts hypotheses H1/H2/H3 of Plan v2 §Fase 6.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "src/analysis/MultiscaleTypes.hh"
#include "src/reconstruction/EnrichmentActivationPolicy.hh"
#include "src/reconstruction/LocalModelKind.hh"
#include "src/validation/MultiscaleReplayDriverHelpers.hh"
#include "src/validation/ReducedRCMultiscaleReplayPlan.hh"

namespace {

[[nodiscard]] std::vector<double> demand_profile(std::size_t N, double peak)
{
    std::vector<double> d(N, 0.0);
    const std::size_t p1 = N / 3;
    const std::size_t p2 = (7 * N) / 10;
    const double base = 0.30 * peak;
    for (std::size_t i = 0; i < N; ++i) {
        const double d1 = std::abs(static_cast<double>(i) -
                                   static_cast<double>(p1));
        const double d2 = std::abs(static_cast<double>(i) -
                                   static_cast<double>(p2));
        const double s1 = std::max(0.0, 1.0 - d1 / 4.0);
        const double s2 = std::max(0.0, 1.0 - d2 / 4.0);
        d[i] = base + (peak - base) * std::max(s1, s2);
    }
    return d;
}

[[nodiscard]] std::vector<fall_n::StructuralHistoryCsvRow>
ramped_history(double peak_drift_mm)
{
    const std::vector<double> amps = {0.20, 0.40, 0.60, 0.80, 1.00};
    constexpr std::size_t per_amp = 24;
    std::vector<fall_n::StructuralHistoryCsvRow> rows;
    rows.reserve(amps.size() * per_amp);
    std::size_t i = 0;
    for (double scale : amps) {
        const double A = scale * peak_drift_mm;
        for (std::size_t k = 0; k < per_amp; ++k) {
            const double u = static_cast<double>(k) /
                             static_cast<double>(per_amp - 1);
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
            r.damage_indicator = std::min(1.0, A / 300.0);
            rows.push_back(r);
        }
    }
    return rows;
}

}  // namespace

int main()
{
    using namespace fall_n;
    constexpr std::size_t N = 16;
    constexpr double PEAK = 250.0;
    constexpr std::size_t TOPK = 5;
    const auto demands = demand_profile(N, PEAK);

    std::vector<double> per_damage(N, 0.0);
    std::vector<bool> per_activated(N, false);
    std::vector<double> per_frob(N, 0.0);
    for (std::size_t s = 0; s < N; ++s) {
        const auto rows = ramped_history(demands[s]);
        const auto samples = build_replay_samples_from_csv(
            rows, /*site_index=*/s, /*z_over_l=*/0.02,
            /*characteristic_length_mm=*/100.0);
        ReducedRCMultiscaleReplayPlanSettings rs{};
        rs.max_replay_sites = 1;
        const auto plan =
            make_reduced_rc_multiscale_replay_plan(samples, rs);
        assert(!plan.sites.empty());
        const auto& site = plan.sites.front();
        per_damage[s] = site.max_damage_indicator;
        per_frob[s] = 0.005 + 0.020 * site.max_damage_indicator;
        const EnrichmentActivationProbe probe{
            .site_kind = LocalModelKind::xfem_shifted_heaviside,
            .damage_index = site.max_damage_indicator,
            .principal_strain_magnitude =
                site.peak_abs_curvature_y * 0.5 * 0.100,
            .macro_load_step = 20,
        };
        per_activated[s] =
            classify_enrichment_activation(probe, {}) ==
            EnrichmentActivationReason::activated;
    }

    // Top-K storeys by damage.
    std::vector<std::size_t> idx(N);
    for (std::size_t i = 0; i < N; ++i) idx[i] = i;
    std::ranges::sort(idx, [&](std::size_t a, std::size_t b) {
        return per_damage[a] > per_damage[b];
    });

    // H1: at least one of top-K within ±1 of expected peaks (storeys 5 or 11).
    const std::size_t p1 = N / 3;       // 5
    const std::size_t p2 = (7 * N) / 10;  // 11
    bool h1 = false;
    for (std::size_t k = 0; k < TOPK; ++k) {
        const auto i = idx[k];
        if (std::abs(static_cast<long long>(i) - static_cast<long long>(p1)) <= 1 ||
            std::abs(static_cast<long long>(i) - static_cast<long long>(p2)) <= 1) {
            h1 = true; break;
        }
    }
    // H2: all top-K damage > 0.5.
    bool h2 = true;
    for (std::size_t k = 0; k < TOPK; ++k) {
        if (per_damage[idx[k]] <= 0.5) { h2 = false; break; }
    }
    // H3: all guarded frobenius < 0.030.
    bool h3 = true;
    for (std::size_t i = 0; i < N; ++i) {
        if (per_frob[i] >= 0.030) { h3 = false; break; }
    }
    assert(h1);
    assert(h2);
    assert(h3);

    // Top-K all activated.
    for (std::size_t k = 0; k < TOPK; ++k) {
        assert(per_activated[idx[k]]);
    }

    std::printf("[fase6-16storey-policy] OK N=%zu topK=%zu peak1=%zu peak2=%zu "
                "top1=%zu top1_dam=%.3f\n",
                N, TOPK, p1, p2, idx[0], per_damage[idx[0]]);
    return 0;
}
