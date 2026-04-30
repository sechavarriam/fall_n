#ifndef FALL_N_REDUCED_RC_LOCAL_SITE_REPLAY_RUNNER_HH
#define FALL_N_REDUCED_RC_LOCAL_SITE_REPLAY_RUNNER_HH

// =============================================================================
//  ReducedRCLocalSiteReplayRunner.hh
// =============================================================================
//
//  Generic replay executor for reduced-RC local-site batches.
//
//  This is the first executable bridge after the planning/catalog layers:
//
//    structural history -> replay plan -> runtime policy -> site batch plan
//    -> local replay runner
//
//  The runner deliberately does not know whether the local solver is XFEM,
//  standard continuum, Ko-Bathe, or a future DG variant.  It only enforces the
//  protocol semantics that matter scientifically before FE2:
//
//    - every selected site receives its own structural history,
//    - warm-start/seed intent is propagated to the local solver callback,
//    - failed increments may be bisected deterministically,
//    - batch/site timing and nonlinear-iteration metrics are accumulated, and
//    - the result is auditable before bidirectional feedback is enabled.
//
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string_view>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "ReducedRCLocalSiteBatchPlan.hh"

namespace fall_n {

class ReducedRCSerialSiteReplayExecutor {
public:
    template <typename Fn>
    void for_each(std::size_t count, Fn&& fn) const
    {
        for (std::size_t i = 0; i < count; ++i) {
            fn(i);
        }
    }
};

class ReducedRCOpenMPSiteReplayExecutor {
public:
    explicit constexpr ReducedRCOpenMPSiteReplayExecutor(
        std::size_t thread_count = 0) noexcept
        : thread_count_(thread_count)
    {}

    template <typename Fn>
    void for_each(std::size_t count, Fn&& fn) const
    {
#ifdef _OPENMP
        const auto requested =
            thread_count_ > 0 ? static_cast<int>(thread_count_) : 0;
        if (requested > 0) {
#pragma omp parallel for schedule(static) num_threads(requested)
            for (long long i = 0; i < static_cast<long long>(count); ++i) {
                fn(static_cast<std::size_t>(i));
            }
            return;
        }
#pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(count); ++i) {
            fn(static_cast<std::size_t>(i));
        }
#else
        (void)thread_count_;
        for (std::size_t i = 0; i < count; ++i) {
            fn(i);
        }
#endif
    }

private:
    std::size_t thread_count_{0};
};

enum class ReducedRCLocalSiteReplayStatus {
    not_scheduled,
    no_history,
    completed,
    failed_cutback_exhausted,
    local_solver_failed
};

[[nodiscard]] constexpr std::string_view to_string(
    ReducedRCLocalSiteReplayStatus status) noexcept
{
    switch (status) {
        case ReducedRCLocalSiteReplayStatus::not_scheduled:
            return "not_scheduled";
        case ReducedRCLocalSiteReplayStatus::no_history:
            return "no_history";
        case ReducedRCLocalSiteReplayStatus::completed:
            return "completed";
        case ReducedRCLocalSiteReplayStatus::failed_cutback_exhausted:
            return "failed_cutback_exhausted";
        case ReducedRCLocalSiteReplayStatus::local_solver_failed:
            return "local_solver_failed";
    }
    return "unknown_reduced_rc_local_site_replay_status";
}

struct ReducedRCLocalSiteReplaySettings {
    bool adaptive_cutback_enabled{true};
    std::size_t max_cutbacks_per_increment{6};
    double minimum_increment_fraction{1.0e-6};
    bool continue_after_site_failure{true};
};

struct ReducedRCLocalSiteReplayStepContext {
    ReducedRCLocalSiteBatchRow site_plan{};
    ReducedRCStructuralReplaySample previous_sample{};
    ReducedRCStructuralReplaySample target_sample{};
    std::size_t target_sample_index{0};
    std::size_t cutback_level{0};
    double increment_fraction{1.0};
    bool generated_by_cutback{false};
    bool seed_restore_requested{false};
    bool warm_start_requested{false};
};

struct ReducedRCLocalSiteReplayStepResult {
    bool converged{true};
    bool hard_failure{false};
    int nonlinear_iterations{0};
    double elapsed_seconds{0.0};
    double residual_norm{0.0};
    double damage_indicator{0.0};
    double steel_stress_mpa{0.0};
    double local_work_increment_mn_mm{
        std::numeric_limits<double>::quiet_NaN()};
    std::string_view status_label{"converged"};
};

struct ReducedRCLocalSiteReplaySiteResult {
    std::size_t batch_index{0};
    std::size_t slot_index{0};
    std::size_t site_index{0};
    ReducedRCLocalSiteReplayStatus status{
        ReducedRCLocalSiteReplayStatus::not_scheduled};
    std::size_t input_sample_count{0};
    std::size_t attempted_step_count{0};
    std::size_t accepted_step_count{0};
    std::size_t failed_step_count{0};
    std::size_t generated_cutback_step_count{0};
    std::size_t max_cutback_level{0};
    int total_nonlinear_iterations{0};
    double total_elapsed_seconds{0.0};
    double accumulated_abs_work_mn_mm{0.0};
    double peak_abs_steel_stress_mpa{0.0};
    double max_damage_indicator{0.0};
    double last_drift_mm{0.0};
    double last_pseudo_time{0.0};
    std::string_view failure_reason{};
};

struct ReducedRCLocalSiteReplayBatchResult {
    std::size_t batch_index{0};
    std::size_t site_count{0};
    std::size_t completed_site_count{0};
    std::size_t failed_site_count{0};
    std::size_t attempted_step_count{0};
    std::size_t accepted_step_count{0};
    int total_nonlinear_iterations{0};
    double total_elapsed_seconds{0.0};
    double max_site_elapsed_seconds{0.0};
};

struct ReducedRCLocalSiteReplayRunResult {
    bool completed{false};
    bool ready_for_guarded_fe2_smoke{false};
    std::size_t selected_site_count{0};
    std::size_t completed_site_count{0};
    std::size_t failed_site_count{0};
    std::size_t batch_count{0};
    std::size_t attempted_step_count{0};
    std::size_t accepted_step_count{0};
    std::size_t failed_step_count{0};
    std::size_t generated_cutback_step_count{0};
    int total_nonlinear_iterations{0};
    double total_elapsed_seconds{0.0};
    double max_site_elapsed_seconds{0.0};
    double accumulated_abs_work_mn_mm{0.0};
    double peak_abs_steel_stress_mpa{0.0};
    double max_damage_indicator{0.0};
    std::vector<ReducedRCLocalSiteReplaySiteResult> sites{};
    std::vector<ReducedRCLocalSiteReplayBatchResult> batches{};
};

namespace detail {

[[nodiscard]] inline double replay_finite_or_zero(double value) noexcept
{
    return std::isfinite(value) ? value : 0.0;
}

[[nodiscard]] inline ReducedRCStructuralReplaySample interpolate_replay_sample(
    const ReducedRCStructuralReplaySample& a,
    const ReducedRCStructuralReplaySample& b,
    double alpha) noexcept
{
    alpha = std::clamp(alpha, 0.0, 1.0);
    auto lerp = [alpha](double x, double y) {
        return x + alpha * (y - x);
    };
    ReducedRCStructuralReplaySample out = b;
    out.pseudo_time = lerp(a.pseudo_time, b.pseudo_time);
    out.physical_time = lerp(a.physical_time, b.physical_time);
    out.z_over_l = lerp(a.z_over_l, b.z_over_l);
    out.drift_mm = lerp(a.drift_mm, b.drift_mm);
    out.curvature_y = lerp(a.curvature_y, b.curvature_y);
    out.moment_y_mn_m = lerp(a.moment_y_mn_m, b.moment_y_mn_m);
    out.base_shear_mn = lerp(a.base_shear_mn, b.base_shear_mn);
    out.steel_stress_mpa = lerp(a.steel_stress_mpa, b.steel_stress_mpa);
    out.damage_indicator = lerp(a.damage_indicator, b.damage_indicator);
    out.work_increment_mn_mm = alpha * b.work_increment_mn_mm;
    return out;
}

[[nodiscard]] inline std::vector<ReducedRCStructuralReplaySample>
history_for_replay_site(
    const std::vector<ReducedRCStructuralReplaySample>& history,
    std::size_t site_index)
{
    std::vector<ReducedRCStructuralReplaySample> out;
    for (const auto& row : history) {
        if (row.site_index == site_index) {
            out.push_back(row);
        }
    }
    return out;
}

inline void accumulate_step_result(
    ReducedRCLocalSiteReplaySiteResult& site,
    const ReducedRCLocalSiteReplayStepContext& context,
    const ReducedRCLocalSiteReplayStepResult& step)
{
    ++site.accepted_step_count;
    site.total_nonlinear_iterations += step.nonlinear_iterations;
    site.total_elapsed_seconds += replay_finite_or_zero(step.elapsed_seconds);
    site.max_damage_indicator =
        std::max(site.max_damage_indicator,
                 std::clamp(step.damage_indicator, 0.0, 1.0));
    site.peak_abs_steel_stress_mpa =
        std::max(site.peak_abs_steel_stress_mpa,
                 std::abs(step.steel_stress_mpa));
    const double work =
        std::isfinite(step.local_work_increment_mn_mm)
            ? step.local_work_increment_mn_mm
            : context.target_sample.work_increment_mn_mm;
    site.accumulated_abs_work_mn_mm += std::abs(work);
    site.last_drift_mm = context.target_sample.drift_mm;
    site.last_pseudo_time = context.target_sample.pseudo_time;
    site.max_cutback_level =
        std::max(site.max_cutback_level, context.cutback_level);
    site.generated_cutback_step_count += context.generated_by_cutback ? 1U
                                                                      : 0U;
}

} // namespace detail

template <typename StepSolveFn>
bool replay_increment_with_cutback_(
    const ReducedRCLocalSiteReplaySettings& settings,
    const ReducedRCLocalSiteBatchRow& row,
    const ReducedRCStructuralReplaySample& previous,
    const ReducedRCStructuralReplaySample& target,
    std::size_t target_sample_index,
    std::size_t cutback_level,
    double increment_fraction,
    bool generated_by_cutback,
    StepSolveFn& solve_step,
    ReducedRCLocalSiteReplaySiteResult& site)
{
    ReducedRCLocalSiteReplayStepContext context{
        .site_plan = row,
        .previous_sample = previous,
        .target_sample = target,
        .target_sample_index = target_sample_index,
        .cutback_level = cutback_level,
        .increment_fraction = increment_fraction,
        .generated_by_cutback = generated_by_cutback,
        .seed_restore_requested = row.seed_restore_required,
        .warm_start_requested = row.warm_start_required};

    ++site.attempted_step_count;
    const auto step = solve_step(context);
    if (step.converged) {
        detail::accumulate_step_result(site, context, step);
        return true;
    }

    ++site.failed_step_count;
    if (step.hard_failure) {
        site.status = ReducedRCLocalSiteReplayStatus::local_solver_failed;
        site.failure_reason = step.status_label;
        return false;
    }

    const bool can_cut_back =
        settings.adaptive_cutback_enabled &&
        cutback_level < settings.max_cutbacks_per_increment &&
        increment_fraction > settings.minimum_increment_fraction;
    if (!can_cut_back) {
        site.status =
            ReducedRCLocalSiteReplayStatus::failed_cutback_exhausted;
        site.failure_reason = step.status_label;
        return false;
    }

    const auto mid =
        detail::interpolate_replay_sample(previous, target, 0.5);
    const double child_fraction = 0.5 * increment_fraction;
    if (!replay_increment_with_cutback_(
            settings,
            row,
            previous,
            mid,
            target_sample_index,
            cutback_level + 1,
            child_fraction,
            true,
            solve_step,
            site))
    {
        return false;
    }

    return replay_increment_with_cutback_(
        settings,
        row,
        mid,
        target,
        target_sample_index,
        cutback_level + 1,
        child_fraction,
        true,
        solve_step,
        site);
}

template <typename StepSolveFn>
[[nodiscard]] ReducedRCLocalSiteReplaySiteResult replay_local_site_history(
    const ReducedRCLocalSiteBatchRow& row,
    const std::vector<ReducedRCStructuralReplaySample>& site_history,
    const ReducedRCLocalSiteReplaySettings& settings,
    StepSolveFn solve_step)
{
    ReducedRCLocalSiteReplaySiteResult result{
        .batch_index = row.batch_index,
        .slot_index = row.slot_index,
        .site_index = row.site_index,
        .status = ReducedRCLocalSiteReplayStatus::no_history,
        .input_sample_count = site_history.size()};

    if (site_history.empty()) {
        result.failure_reason = "no history samples for selected site";
        return result;
    }

    ReducedRCStructuralReplaySample previous = site_history.front();
    for (std::size_t i = 0; i < site_history.size(); ++i) {
        const auto& target = site_history[i];
        const bool ok = i == 0
            ? replay_increment_with_cutback_(
                  settings,
                  row,
                  target,
                  target,
                  i,
                  0,
                  1.0,
                  false,
                  solve_step,
                  result)
            : replay_increment_with_cutback_(
                  settings,
                  row,
                  previous,
                  target,
                  i,
                  0,
                  1.0,
                  false,
                  solve_step,
                  result);
        if (!ok) {
            return result;
        }
        previous = target;
    }

    result.status = ReducedRCLocalSiteReplayStatus::completed;
    return result;
}

template <typename StepSolveFn,
          typename ExecutorT = ReducedRCSerialSiteReplayExecutor>
[[nodiscard]] ReducedRCLocalSiteReplayRunResult
run_reduced_rc_local_site_replay_batch(
    const std::vector<ReducedRCStructuralReplaySample>& history,
    const ReducedRCLocalSiteBatchPlan& batch_plan,
    StepSolveFn solve_step,
    ReducedRCLocalSiteReplaySettings settings = {},
    ExecutorT executor = {})
{
    ReducedRCLocalSiteReplayRunResult out{};
    out.selected_site_count = batch_plan.selected_site_count;
    out.batch_count = batch_plan.batch_count;
    out.sites.resize(batch_plan.rows.size());

    auto run_site = [&](std::size_t i) {
        const auto& row = batch_plan.rows[i];
        auto site_history =
            detail::history_for_replay_site(history, row.site_index);
        out.sites[i] =
            replay_local_site_history(row, site_history, settings, solve_step);
    };

    if (settings.continue_after_site_failure) {
        executor.for_each(batch_plan.rows.size(), run_site);
    } else {
        for (std::size_t i = 0; i < batch_plan.rows.size(); ++i) {
            run_site(i);
            if (out.sites[i].status !=
                ReducedRCLocalSiteReplayStatus::completed)
            {
                break;
            }
        }
    }

    out.batches.resize(batch_plan.batches.size());
    for (const auto& batch : batch_plan.batches) {
        if (batch.batch_index < out.batches.size()) {
            out.batches[batch.batch_index].batch_index = batch.batch_index;
            out.batches[batch.batch_index].site_count = batch.site_count;
        }
    }

    for (const auto& site : out.sites) {
        const bool completed =
            site.status == ReducedRCLocalSiteReplayStatus::completed;
        out.completed_site_count += completed ? 1U : 0U;
        out.failed_site_count += completed ? 0U : 1U;
        out.attempted_step_count += site.attempted_step_count;
        out.accepted_step_count += site.accepted_step_count;
        out.failed_step_count += site.failed_step_count;
        out.generated_cutback_step_count += site.generated_cutback_step_count;
        out.total_nonlinear_iterations += site.total_nonlinear_iterations;
        out.total_elapsed_seconds += site.total_elapsed_seconds;
        out.max_site_elapsed_seconds =
            std::max(out.max_site_elapsed_seconds,
                     site.total_elapsed_seconds);
        out.accumulated_abs_work_mn_mm += site.accumulated_abs_work_mn_mm;
        out.peak_abs_steel_stress_mpa =
            std::max(out.peak_abs_steel_stress_mpa,
                     site.peak_abs_steel_stress_mpa);
        out.max_damage_indicator =
            std::max(out.max_damage_indicator, site.max_damage_indicator);

        if (site.batch_index < out.batches.size()) {
            auto& batch = out.batches[site.batch_index];
            batch.completed_site_count += completed ? 1U : 0U;
            batch.failed_site_count += completed ? 0U : 1U;
            batch.attempted_step_count += site.attempted_step_count;
            batch.accepted_step_count += site.accepted_step_count;
            batch.total_nonlinear_iterations += site.total_nonlinear_iterations;
            batch.total_elapsed_seconds += site.total_elapsed_seconds;
            batch.max_site_elapsed_seconds =
                std::max(batch.max_site_elapsed_seconds,
                         site.total_elapsed_seconds);
        }
    }

    out.completed = out.failed_site_count == 0 &&
                    out.completed_site_count == out.selected_site_count &&
                    out.selected_site_count > 0;
    out.ready_for_guarded_fe2_smoke =
        out.completed &&
        batch_plan.ready_for_local_site_batch &&
        out.max_damage_indicator >= 0.0;
    return out;
}

} // namespace fall_n

#endif // FALL_N_REDUCED_RC_LOCAL_SITE_REPLAY_RUNNER_HH
