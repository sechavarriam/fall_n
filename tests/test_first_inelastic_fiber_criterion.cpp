// Plan v2 — unit test for FirstInelasticFiberCriterion.

#include <cassert>
#include <cstdio>
#include <vector>

#include "src/analysis/FirstInelasticFiberCriterion.hh"
#include "src/validation/ReducedRCMultiscaleReplayPlan.hh"

int main() {
    using namespace fall_n;
    FirstInelasticFiberCriterion crit{};

    // --- Per-sample: elastic.
    ReducedRCStructuralReplaySample elastic{};
    elastic.steel_stress_mpa  = 0.5 * crit.f_y_MPa;     // half yield
    elastic.curvature_y       = 0.5 * crit.yield_strain / (crit.c_section_mm / 1000.0);
    elastic.damage_indicator  = 0.0;
    assert(crit.evaluate(elastic) ==
           FirstInelasticFiberCriterion::Reason::not_triggered);

    // --- Per-sample: rebar yield.
    ReducedRCStructuralReplaySample yielded{};
    yielded.steel_stress_mpa  = 1.05 * crit.f_y_MPa;
    assert(crit.evaluate(yielded) ==
           FirstInelasticFiberCriterion::Reason::rebar_yield);

    // --- Per-sample: extreme-fibre strain via curvature only.
    ReducedRCStructuralReplaySample curved{};
    curved.curvature_y = 1.10 * crit.yield_strain / (crit.c_section_mm / 1000.0);
    assert(crit.evaluate(curved) ==
           FirstInelasticFiberCriterion::Reason::extreme_fiber_strain);

    // --- Per-sample: damage-floor fallback.
    ReducedRCStructuralReplaySample damaged{};
    damaged.damage_indicator = 0.10;
    assert(crit.evaluate(damaged) ==
           FirstInelasticFiberCriterion::Reason::damage_floor);

    // --- Site-level: peak observables.
    ReducedRCMultiscaleReplaySitePlan site{};
    site.peak_abs_steel_stress_mpa = 1.05 * crit.f_y_MPa;
    assert(crit.evaluate(site) ==
           FirstInelasticFiberCriterion::Reason::rebar_yield);

    // --- first_trigger_index walks the history in order.
    std::vector<ReducedRCStructuralReplaySample> hist;
    for (int i = 0; i < 10; ++i) {
        ReducedRCStructuralReplaySample s{};
        s.steel_stress_mpa = (i >= 5 ? 1.10 * crit.f_y_MPa : 0.10 * crit.f_y_MPa);
        hist.push_back(s);
    }
    const auto idx = crit.first_trigger_index(hist);
    assert(idx == 5);

    // --- Empty history.
    std::vector<ReducedRCStructuralReplaySample> empty;
    assert(crit.first_trigger_index(empty) == 0);

    // --- Custom thresholds.
    FirstInelasticFiberCriterion strict{
        .yield_strain = 0.001, .f_y_MPa = 200.0,
        .c_section_mm = 50.0, .damage_floor = 0.01};
    ReducedRCStructuralReplaySample low{};
    low.steel_stress_mpa = 250.0;
    assert(strict.evaluate(low) ==
           FirstInelasticFiberCriterion::Reason::rebar_yield);

    std::printf("[first_inelastic_fiber_criterion] all assertions passed\n");
    return 0;
}
