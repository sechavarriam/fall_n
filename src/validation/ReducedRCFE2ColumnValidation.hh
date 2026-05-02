#ifndef FALL_N_REDUCED_RC_FE2_COLUMN_VALIDATION_HH
#define FALL_N_REDUCED_RC_FE2_COLUMN_VALIDATION_HH

// =============================================================================
//  ReducedRCFE2ColumnValidation.hh
// =============================================================================
//
//  Common contracts for the reduced-RC FE2 column validation campaign.
//
//  The intentionally narrow purpose of this header is to keep the scientific
//  status of each FE2 driver explicit:
//
//    - surrogate_smoke: cheap Eigen-only plumbing/gate test,
//    - real_xfem_replay: physical managed local-model replay candidate,
//    - iterated_two_way_fe2: macro-local feedback candidate.
//
//  The header is dependency-light and header-only so command-line drivers,
//  tests, wrappers and thesis manifest emitters can agree on the same labels,
//  thresholds and acceptance gates without pulling a concrete PETSc/XFEM local
//  solver into every translation unit.
//
//  "Local" here means one persistent Model with its own independent domain and
//  mesh per selected macro site.  It does not mean spawning one XFEM problem at
//  each failed fiber, integration point or section sample.
//
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string_view>
#include <vector>

#include <Eigen/Dense>

#include "src/analysis/FirstInelasticFiberCriterion.hh"
#include "src/analysis/MultiscaleTypes.hh"
#include "src/validation/ReducedRCMultiscaleReplayPlan.hh"

namespace fall_n {

enum class ReducedRCFE2ColumnCouplingMode {
    one_way_downscaling,
    iterated_two_way_fe2
};

enum class ReducedRCFE2ColumnLocalExecutionMode {
    surrogate_smoke,
    real_xfem_replay
};

enum class ReducedRCFE2ColumnValidationStatus {
    not_run,
    passed,
    failed_gate,
    blocked_real_local_adapter_missing,
    failed_local_solve
};

[[nodiscard]] constexpr std::string_view
to_string(ReducedRCFE2ColumnCouplingMode mode) noexcept
{
    switch (mode) {
        case ReducedRCFE2ColumnCouplingMode::one_way_downscaling:
            return "one_way_downscaling";
        case ReducedRCFE2ColumnCouplingMode::iterated_two_way_fe2:
            return "iterated_two_way_fe2";
    }
    return "unknown_reduced_rc_fe2_column_coupling_mode";
}

[[nodiscard]] constexpr std::string_view
to_string(ReducedRCFE2ColumnLocalExecutionMode mode) noexcept
{
    switch (mode) {
        case ReducedRCFE2ColumnLocalExecutionMode::surrogate_smoke:
            return "surrogate_smoke";
        case ReducedRCFE2ColumnLocalExecutionMode::real_xfem_replay:
            return "real_xfem_replay";
    }
    return "unknown_reduced_rc_fe2_column_local_execution_mode";
}

[[nodiscard]] constexpr std::string_view
to_string(ReducedRCFE2ColumnValidationStatus status) noexcept
{
    switch (status) {
        case ReducedRCFE2ColumnValidationStatus::not_run:
            return "not_run";
        case ReducedRCFE2ColumnValidationStatus::passed:
            return "passed";
        case ReducedRCFE2ColumnValidationStatus::failed_gate:
            return "failed_gate";
        case ReducedRCFE2ColumnValidationStatus::
            blocked_real_local_adapter_missing:
            return "blocked_real_local_adapter_missing";
        case ReducedRCFE2ColumnValidationStatus::failed_local_solve:
            return "failed_local_solve";
    }
    return "unknown_reduced_rc_fe2_column_validation_status";
}

struct ReducedRCFE2ColumnAcceptanceTolerances {
    double max_relative_moment_envelope_error{0.25};
    double max_frobenius_residual{0.03};
    std::size_t max_snes_iterations{6};
    double min_converged_fraction{0.95};
    double max_force_residual_rel{0.05};
    double max_tangent_residual_rel{0.05};
};

struct ReducedRCFE2ColumnRunSpec {
    ReducedRCFE2ColumnCouplingMode coupling_mode{
        ReducedRCFE2ColumnCouplingMode::one_way_downscaling};
    ReducedRCFE2ColumnLocalExecutionMode local_execution_mode{
        ReducedRCFE2ColumnLocalExecutionMode::real_xfem_replay};

    std::size_t promoted_beam_nodes{10};
    std::size_t preflight_beam_nodes{4};
    std::string_view promoted_quadrature{"gauss_lobatto"};
    std::string_view local_model_policy{
        "managed_independent_domain_per_selected_macro_site"};
    double protocol_amplitude_mm{200.0};

    double EA_MN{6.0e3};
    double EI_MN_m2{30.0};
    double damage_floor{0.05};
    double f_y_MPa{420.0};
    double yield_strain{420.0 / 200000.0};
    double c_section_mm{100.0};

    int max_staggered_iterations{4};
    double staggered_tolerance{0.05};
    double staggered_relaxation{0.70};

    ReducedRCFE2ColumnAcceptanceTolerances tolerances{};
};

class ReducedRCFE2ActivationCriterion {
public:
    explicit constexpr ReducedRCFE2ActivationCriterion(
        FirstInelasticFiberCriterion criterion = {}) noexcept
        : criterion_(criterion)
    {}

    [[nodiscard]] constexpr const FirstInelasticFiberCriterion&
    base() const noexcept
    {
        return criterion_;
    }

    [[nodiscard]] FirstInelasticFiberCriterion::Reason evaluate(
        const ReducedRCStructuralReplaySample& sample) const noexcept
    {
        return criterion_.evaluate(sample);
    }

    [[nodiscard]] FirstInelasticFiberCriterion::Reason evaluate(
        const ReducedRCMultiscaleReplaySitePlan& site) const noexcept
    {
        return criterion_.evaluate(site);
    }

    [[nodiscard]] std::size_t first_trigger_index(
        const std::vector<ReducedRCStructuralReplaySample>& history) const noexcept
    {
        return criterion_.first_trigger_index(history);
    }

private:
    FirstInelasticFiberCriterion criterion_{};
};

struct ReducedRCFE2ColumnSiteResult {
    std::size_t site_index{0};
    double z_over_l{0.0};
    FirstInelasticFiberCriterion::Reason trigger_reason{
        FirstInelasticFiberCriterion::Reason::not_triggered};
    std::size_t trigger_sample_index{0};
    double trigger_pseudo_time{-1.0};
    double trigger_drift_mm{0.0};

    double peak_abs_curvature_y{0.0};
    double peak_abs_curvature_z{0.0};
    double peak_abs_moment_y_mn_m{0.0};
    double peak_abs_moment_z_mn_m{0.0};
    double peak_abs_base_shear_mn{0.0};
    double peak_abs_steel_stress_mpa{0.0};
    double max_damage_indicator{0.0};
    double accumulated_abs_work_mn_mm{0.0};

    UpscalingResult upscaling{};
    double synthetic_moment_y_mn_m{0.0};
    double relative_moment_envelope_error{0.0};
    bool moment_envelope_available{false};
    bool activated{false};
    bool response_gate_passed{false};
    ReducedRCFE2ColumnValidationStatus status{
        ReducedRCFE2ColumnValidationStatus::not_run};
};

struct ReducedRCFE2ColumnStaggeredOutcome {
    int iterations{0};
    double final_residual{0.0};
    bool converged{false};
    Eigen::Matrix2d D_hom{Eigen::Matrix2d::Zero()};
    Eigen::Vector2d f_hom{Eigen::Vector2d::Zero()};
    Eigen::Vector2d eps_ref{Eigen::Vector2d::Zero()};
    std::vector<double> residual_history{};
};

struct ReducedRCFE2ColumnResult {
    ReducedRCFE2ColumnRunSpec spec{};
    ReducedRCFE2ColumnValidationStatus status{
        ReducedRCFE2ColumnValidationStatus::not_run};
    std::size_t history_sample_count{0};
    std::size_t selected_site_count{0};
    std::size_t activated_site_count{0};
    std::size_t passed_site_count{0};
    std::size_t failed_site_count{0};
    double converged_fraction{0.0};
    std::vector<ReducedRCFE2ColumnSiteResult> sites{};

    [[nodiscard]] bool passed() const noexcept
    {
        return status == ReducedRCFE2ColumnValidationStatus::passed;
    }
};

[[nodiscard]] inline FirstInelasticFiberCriterion
make_reduced_rc_fe2_first_inelastic_criterion(
    const ReducedRCFE2ColumnRunSpec& spec) noexcept
{
    return FirstInelasticFiberCriterion{
        .yield_strain = spec.yield_strain,
        .f_y_MPa = spec.f_y_MPa,
        .c_section_mm = spec.c_section_mm,
        .damage_floor = spec.damage_floor};
}

[[nodiscard]] inline std::vector<ReducedRCStructuralReplaySample>
history_for_fe2_site(const std::vector<ReducedRCStructuralReplaySample>& history,
                     std::size_t site_index)
{
    std::vector<ReducedRCStructuralReplaySample> out;
    for (const auto& sample : history) {
        if (sample.site_index == site_index) {
            out.push_back(sample);
        }
    }
    return out;
}

[[nodiscard]] inline UpscalingResult make_reduced_rc_fe2_surrogate_one_way_response(
    const ReducedRCMultiscaleReplaySitePlan& site,
    const ReducedRCFE2ColumnRunSpec& spec)
{
    UpscalingResult response{};
    response.eps_ref = Eigen::VectorXd::Zero(2);
    response.eps_ref(1) = site.peak_abs_curvature_y;

    const double degradation =
        std::clamp(1.0 - site.max_damage_indicator, 0.05, 1.0);
    response.D_hom = Eigen::MatrixXd::Zero(2, 2);
    response.D_hom(0, 0) = degradation * spec.EA_MN;
    response.D_hom(1, 1) = degradation * spec.EI_MN_m2;
    response.f_hom = response.D_hom * response.eps_ref;
    response.frobenius_residual =
        0.005 + 0.020 * std::clamp(site.max_damage_indicator, 0.0, 1.0);
    response.snes_iters = 3;
    response.converged = true;
    response.status = ResponseStatus::Ok;
    response.tangent_scheme =
        TangentLinearizationScheme::LinearizedCondensation;
    response.condensed_status = CondensedTangentStatus::Success;
    return response;
}

[[nodiscard]] inline ReducedRCFE2ColumnStaggeredOutcome
run_reduced_rc_fe2_surrogate_staggered_iteration(
    const ReducedRCMultiscaleReplaySitePlan& site,
    const ReducedRCFE2ColumnRunSpec& spec)
{
    ReducedRCFE2ColumnStaggeredOutcome out{};
    const double damage = std::clamp(site.max_damage_indicator, 0.0, 1.0);
    const double target_scale = std::clamp(1.0 - damage, 0.05, 1.0);

    Eigen::Matrix2d D_macro = Eigen::Matrix2d::Zero();
    D_macro(0, 0) = spec.EA_MN;
    D_macro(1, 1) = spec.EI_MN_m2;

    Eigen::Matrix2d D_target = Eigen::Matrix2d::Zero();
    D_target(0, 0) = target_scale * spec.EA_MN;
    D_target(1, 1) = target_scale * spec.EI_MN_m2;

    Eigen::Matrix2d D_current = D_macro;
    const double D_norm = std::max(D_macro.norm(), 1.0e-12);
    const int max_iter = std::max(spec.max_staggered_iterations, 1);
    const double omega = std::clamp(spec.staggered_relaxation, 0.0, 1.0);

    for (int it = 1; it <= max_iter; ++it) {
        const Eigen::Matrix2d D_previous = D_current;
        D_current = (1.0 - omega) * D_previous + omega * D_target;
        const double residual = (D_current - D_previous).norm() / D_norm;
        out.iterations = it;
        out.final_residual = residual;
        out.residual_history.push_back(residual);
        if (residual <= spec.staggered_tolerance) {
            out.converged = true;
            break;
        }
    }

    out.D_hom = D_current;
    out.eps_ref = Eigen::Vector2d::Zero();
    out.eps_ref(1) = site.peak_abs_curvature_y;
    out.f_hom = D_current * out.eps_ref;
    return out;
}

[[nodiscard]] inline ReducedRCFE2ColumnSiteResult
make_reduced_rc_fe2_site_result(
    const ReducedRCMultiscaleReplaySitePlan& site,
    const std::vector<ReducedRCStructuralReplaySample>& per_site_history,
    const ReducedRCFE2ActivationCriterion& activation,
    UpscalingResult upscaling,
    const ReducedRCFE2ColumnRunSpec& spec)
{
    ReducedRCFE2ColumnSiteResult result{};
    result.site_index = site.site_index;
    result.z_over_l = site.z_over_l;
    result.trigger_reason = activation.evaluate(site);
    result.trigger_sample_index = activation.first_trigger_index(per_site_history);
    if (result.trigger_sample_index < per_site_history.size()) {
        const auto& sample = per_site_history[result.trigger_sample_index];
        result.trigger_pseudo_time = sample.pseudo_time;
        result.trigger_drift_mm = sample.drift_mm;
    }
    result.peak_abs_curvature_y = site.peak_abs_curvature_y;
    result.peak_abs_curvature_z = site.peak_abs_curvature_z;
    result.peak_abs_moment_y_mn_m = site.peak_abs_moment_y_mn_m;
    result.peak_abs_moment_z_mn_m = site.peak_abs_moment_z_mn_m;
    result.peak_abs_base_shear_mn = site.peak_abs_base_shear_mn;
    result.peak_abs_steel_stress_mpa = site.peak_abs_steel_stress_mpa;
    result.max_damage_indicator = site.max_damage_indicator;
    result.accumulated_abs_work_mn_mm = site.accumulated_abs_work_mn_mm;
    result.upscaling = std::move(upscaling);
    result.activated =
        result.trigger_reason != FirstInelasticFiberCriterion::Reason::not_triggered;

    if (result.upscaling.is_well_formed() && result.upscaling.f_hom.size() > 1) {
        result.synthetic_moment_y_mn_m = std::abs(result.upscaling.f_hom(1));
    }
    result.moment_envelope_available = site.peak_abs_moment_y_mn_m > 1.0e-6;
    if (result.moment_envelope_available) {
        result.relative_moment_envelope_error =
            std::abs(result.synthetic_moment_y_mn_m -
                     site.peak_abs_moment_y_mn_m) /
            site.peak_abs_moment_y_mn_m;
    }

    const bool response_ok = result.upscaling.passes_guarded_smoke_gate(
        spec.tolerances.max_frobenius_residual,
        spec.tolerances.max_snes_iterations);
    const bool envelope_ok =
        !result.moment_envelope_available ||
        result.relative_moment_envelope_error <=
            spec.tolerances.max_relative_moment_envelope_error;
    result.response_gate_passed = result.activated && response_ok && envelope_ok;
    result.status = result.response_gate_passed
        ? ReducedRCFE2ColumnValidationStatus::passed
        : ReducedRCFE2ColumnValidationStatus::failed_gate;
    return result;
}

[[nodiscard]] inline ReducedRCFE2ColumnResult
summarize_reduced_rc_fe2_column_result(
    ReducedRCFE2ColumnRunSpec spec,
    std::size_t history_sample_count,
    std::vector<ReducedRCFE2ColumnSiteResult> sites)
{
    ReducedRCFE2ColumnResult result{};
    result.spec = spec;
    result.history_sample_count = history_sample_count;
    result.selected_site_count = sites.size();
    result.sites = std::move(sites);

    for (const auto& site : result.sites) {
        result.activated_site_count += site.activated ? 1U : 0U;
        result.passed_site_count += site.response_gate_passed ? 1U : 0U;
        result.failed_site_count += site.response_gate_passed ? 0U : 1U;
    }

    result.converged_fraction = result.selected_site_count == 0
        ? 0.0
        : static_cast<double>(result.passed_site_count) /
              static_cast<double>(result.selected_site_count);
    result.status =
        result.selected_site_count > 0 &&
                result.activated_site_count == result.selected_site_count &&
                result.converged_fraction >=
                    spec.tolerances.min_converged_fraction
            ? ReducedRCFE2ColumnValidationStatus::passed
            : ReducedRCFE2ColumnValidationStatus::failed_gate;
    return result;
}

[[nodiscard]] inline ReducedRCFE2ColumnResult
make_blocked_reduced_rc_fe2_column_result(
    ReducedRCFE2ColumnRunSpec spec,
    std::size_t history_sample_count,
    std::size_t selected_site_count)
{
    ReducedRCFE2ColumnResult result{};
    result.spec = spec;
    result.history_sample_count = history_sample_count;
    result.selected_site_count = selected_site_count;
    result.status =
        ReducedRCFE2ColumnValidationStatus::blocked_real_local_adapter_missing;
    return result;
}

} // namespace fall_n

#endif // FALL_N_REDUCED_RC_FE2_COLUMN_VALIDATION_HH
