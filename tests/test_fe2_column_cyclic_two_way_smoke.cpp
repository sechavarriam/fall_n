// Plan v2 §Stage B — in-memory smoke for FE² TWO-WAY cyclic 200 mm column.
//
// Builds a 121-sample synthetic ramped-cyclic 200 mm history, activates the
// `FirstInelasticFiberCriterion`, then runs the synthetic staggered loop
// (relax + tol identical to `main_lshaped_multiscale.cpp` defaults) on the
// selected site. Asserts:
//   • criterion fires;
//   • staggered residual converges below `staggered_tol` within
//     `max_staggered_iter` for the given relaxation factor;
//   • final D_hom diagonal is non-negative and bounded by the macro
//     reference (EA, EI).

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include <Eigen/Dense>

#include "src/analysis/FirstInelasticFiberCriterion.hh"
#include "src/validation/ReducedRCMultiscaleReplayPlan.hh"

namespace {

[[nodiscard]] std::vector<fall_n::ReducedRCStructuralReplaySample>
make_history(std::size_t N = 121, double L_m = 0.100)
{
    using namespace fall_n;
    std::vector<ReducedRCStructuralReplaySample> hist;
    hist.reserve(N);
    double prev_drift = 0.0, prev_vb = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(N - 1);
        const double drift_mm = 200.0 * t * std::sin(4.0 * t * 3.14159265);
        const double vb = 0.30 * std::tanh(drift_mm / 50.0);
        const double curv = (drift_mm / 1000.0) / L_m;
        const double d_drift = i == 0 ? 0.0 : drift_mm - prev_drift;
        const double d_work  = i == 0 ? 0.0 : 0.5 * (vb + prev_vb) * d_drift;
        ReducedRCStructuralReplaySample s{};
        s.site_index = 0;
        s.pseudo_time = t;
        s.physical_time = t;
        s.z_over_l = 0.02;
        s.drift_mm = drift_mm;
        s.curvature_y = curv;
        s.moment_y_mn_m = 0.04 * std::tanh(curv / 0.05);
        s.base_shear_mn = vb;
        s.steel_stress_mpa = 420.0 * std::tanh(drift_mm / 100.0);
        s.damage_indicator = std::min(1.0, std::abs(drift_mm) / 250.0);
        s.work_increment_mn_mm = d_work;
        hist.push_back(s);
        prev_drift = drift_mm; prev_vb = vb;
    }
    return hist;
}

}  // namespace

int main() {
    using namespace fall_n;

    const auto hist = make_history();
    const auto plan = make_reduced_rc_multiscale_replay_plan(hist, {});
    assert(plan.selected_site_count == 1);
    const auto& site = plan.sites.front();

    const FirstInelasticFiberCriterion crit{};
    const auto reason = crit.evaluate(site);
    assert(reason != FirstInelasticFiberCriterion::Reason::not_triggered);

    // --- Synthetic staggered loop (mirrors main_fe2_column_cyclic_two_way).
    constexpr double EA_MN          = 6.0e3;
    constexpr double EI_MN_m2       = 30.0;
    constexpr int    MAX_ITER       = 4;
    constexpr double STAGGERED_TOL  = 0.05;
    constexpr double STAGGERED_RELAX= 0.7;

    const double damage = std::clamp(site.max_damage_indicator, 0.0, 1.0);
    const double s_target = std::clamp(1.0 - damage, 0.05, 1.0);
    Eigen::Matrix2d D_macro = Eigen::Matrix2d::Zero();
    D_macro(0,0) = EA_MN; D_macro(1,1) = EI_MN_m2;
    Eigen::Matrix2d D_target = Eigen::Matrix2d::Zero();
    D_target(0,0) = s_target * EA_MN; D_target(1,1) = s_target * EI_MN_m2;
    Eigen::Matrix2d D_curr = D_macro;
    const double D_macro_norm = D_macro.norm();

    int    iters = 0;
    double residual = 0.0;
    bool   converged = false;
    for (int it = 1; it <= MAX_ITER; ++it) {
        const Eigen::Matrix2d D_prev = D_curr;
        D_curr = (1.0 - STAGGERED_RELAX) * D_prev + STAGGERED_RELAX * D_target;
        residual = (D_curr - D_prev).norm() / D_macro_norm;
        iters = it;
        if (residual <= STAGGERED_TOL) { converged = true; break; }
    }

    // --- Gates.
    assert(converged);
    assert(iters <= MAX_ITER);
    assert(residual <= STAGGERED_TOL);
    assert(D_curr(0,0) >= 0.0 && D_curr(0,0) <= EA_MN);
    assert(D_curr(1,1) >= 0.0 && D_curr(1,1) <= EI_MN_m2);

    std::printf("[fe2_column_cyclic_two_way_smoke] reason=%s iters=%d "
                "residual=%.6f D_hom_diag=[%.3f, %.3f]\n",
                FirstInelasticFiberCriterion::reason_label(reason).data(),
                iters, residual, D_curr(0,0), D_curr(1,1));
    return 0;
}
