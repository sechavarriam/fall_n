// Plan v2 — Stage A/B activation criterion.
//
// `FirstInelasticFiberCriterion` is a small, header-only, Eigen-free
// activation predicate intended to drive FE² site activation on the
// already-validated cyclic 200 mm RC column. It deliberately operates on
// the synthetic-chain replay objects (`ReducedRCStructuralReplaySample`,
// `ReducedRCMultiscaleReplaySitePlan`) so it can be reused by the Eigen-only
// standalone drivers `main_fe2_column_cyclic_one_way` and
// `main_fe2_column_cyclic_two_way` without pulling the heavy domain.
//
// The criterion fires when ANY of the following is observed at a site:
//   • |steel_stress|   ≥ f_y_MPa            (rebar yield, primary)
//   • |κ_y| · c_section_mm/1000 ≥ ε_y       (extreme fibre strain, proxy)
//   • damage_indicator ≥ damage_floor       (degradation fallback)
//
// Honest scientific status: the criterion is a domain-faithful surrogate
// of the heavy `MaxStrainDamageCriterion` / first-rebar-yield director,
// suitable for the synthetic-chain validation. The real LIBS-FULL drivers
// (e.g. `main_lshaped_multiscale.cpp`) keep using `DamageCriterion`.

#ifndef FALL_N_SRC_ANALYSIS_FIRST_INELASTIC_FIBER_CRITERION_HH
#define FALL_N_SRC_ANALYSIS_FIRST_INELASTIC_FIBER_CRITERION_HH

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string_view>

#include "../validation/ReducedRCMultiscaleReplayPlan.hh"

namespace fall_n {

struct FirstInelasticFiberCriterion {
    // Reinforcement yield: f_y / E_s. Defaults to A615 Gr. 60 (fy=420 MPa,
    // Es=200 GPa → ε_y ≈ 0.0021).
    double yield_strain{420.0 / 200000.0};
    // Rebar yield stress used as primary trigger (MPa).
    double f_y_MPa{420.0};
    // Half-section depth used to map curvature → extreme-fibre strain.
    // Default 100 mm matches the canonical 200 × 200 mm column section.
    double c_section_mm{100.0};
    // Damage-index floor (compatible with EnrichmentActivationPolicy default).
    double damage_floor{0.05};

    // Reasons used by the JSON manifests; lowercase, snake_case for stable
    // diff / regex.
    enum class Reason : int {
        not_triggered = 0,
        rebar_yield = 1,
        extreme_fiber_strain = 2,
        damage_floor = 3,
    };

    [[nodiscard]] static constexpr std::string_view reason_label(Reason r) noexcept
    {
        switch (r) {
            case Reason::not_triggered:        return "not_triggered";
            case Reason::rebar_yield:          return "rebar_yield";
            case Reason::extreme_fiber_strain: return "extreme_fiber_strain";
            case Reason::damage_floor:         return "damage_floor";
        }
        return "unknown";
    }

    // Per-sample evaluation. Returns the first reason encountered.
    [[nodiscard]] Reason evaluate(const ReducedRCStructuralReplaySample& s) const noexcept
    {
        if (std::abs(s.steel_stress_mpa) >= f_y_MPa) return Reason::rebar_yield;
        const double extreme_eps = std::abs(s.curvature_y) * (c_section_mm / 1000.0);
        if (extreme_eps >= yield_strain) return Reason::extreme_fiber_strain;
        if (s.damage_indicator >= damage_floor) return Reason::damage_floor;
        return Reason::not_triggered;
    }

    // Site-level evaluation (peak observables). Useful for the post-history
    // gate that the standalone driver applies once all samples have been
    // ingested.
    [[nodiscard]] Reason evaluate(const ReducedRCMultiscaleReplaySitePlan& site) const noexcept
    {
        if (site.peak_abs_steel_stress_mpa >= f_y_MPa) return Reason::rebar_yield;
        const double extreme_eps = site.peak_abs_curvature_y * (c_section_mm / 1000.0);
        if (extreme_eps >= yield_strain) return Reason::extreme_fiber_strain;
        if (site.max_damage_indicator >= damage_floor) return Reason::damage_floor;
        return Reason::not_triggered;
    }

    // Walks a per-site history in chronological order; returns the index of
    // the first triggering sample (or hist.size() if never triggered).
    [[nodiscard]] std::size_t first_trigger_index(
        const std::vector<ReducedRCStructuralReplaySample>& hist) const noexcept
    {
        for (std::size_t i = 0; i < hist.size(); ++i) {
            if (evaluate(hist[i]) != Reason::not_triggered) return i;
        }
        return hist.size();
    }
};

}  // namespace fall_n

#endif  // FALL_N_SRC_ANALYSIS_FIRST_INELASTIC_FIBER_CRITERION_HH
