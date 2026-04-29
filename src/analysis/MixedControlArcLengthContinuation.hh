#ifndef FALL_N_SRC_ANALYSIS_MIXED_CONTROL_ARC_LENGTH_CONTINUATION_HH
#define FALL_N_SRC_ANALYSIS_MIXED_CONTROL_ARC_LENGTH_CONTINUATION_HH

// =============================================================================
//  MixedControlArcLengthContinuation
// =============================================================================
//
//  This header implements a lightweight continuation layer for displacement-
//  driven nonlinear analyses.  It is intentionally different from
//  ArcLengthSolver.hh:
//
//    * ArcLengthSolver traces a load-proportional equilibrium path in
//      (u, lambda) by solving an augmented Riks/Crisfield system.
//    * MixedControlArcLengthContinuation keeps an imposed-control protocol
//      (for example a cyclic Dirichlet drift) but adapts and, if necessary,
//      rejects converged steps using a scaled observable arc length.
//
//  The second route is the correct first closure for the XFEM reduced-column
//  benchmark because the experiment is driven by prescribed top-face motion
//  plus axial load.  It does not pretend that a Dirichlet path is a
//  load-proportional arc-length problem.  Instead, it introduces the seam that
//  a future bordered mixed-control Newton system can reuse:
//
//      ||Delta q||_mix^2 =
//          wc (Delta control / c0)^2
//        + wr (Delta reaction / r0)^2
//        + wi (Delta internal / i0)^2
//
//  where q is supplied by the caller.  The driver only needs a checkpointable
//  steppable solver, so it can wrap static, dynamic, local, or multiscale
//  engines without touching PETSc assembly kernels.
//
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

#include "SteppableSolver.hh"

namespace fall_n {

enum class MixedControlArcLengthStatus {
    completed,
    stopped,
    invalid_configuration
};

[[nodiscard]] constexpr std::string_view
to_string(MixedControlArcLengthStatus status) noexcept
{
    switch (status) {
        case MixedControlArcLengthStatus::completed:
            return "completed";
        case MixedControlArcLengthStatus::stopped:
            return "stopped";
        case MixedControlArcLengthStatus::invalid_configuration:
            return "invalid_configuration";
    }
    return "unknown_mixed_control_arc_length_status";
}

struct MixedControlArcLengthObservation {
    // Raw observable values.  Settings provide the physical scales.
    double control{0.0};
    double reaction{0.0};
    double internal{0.0};
};

struct MixedControlArcLengthScales {
    double control{1.0};
    double reaction{1.0};
    double internal{1.0};
};

struct MixedControlArcLengthWeights {
    double control{1.0};
    double reaction{1.0};
    double internal{0.0};
};

struct MixedControlArcLengthSettings {
    double target_p{1.0};
    double initial_increment{0.02};
    double min_increment{1.0e-5};
    double max_increment{0.02};
    double target_arc_length{0.10};
    double reject_arc_length_factor{1.50};
    double grow_arc_length_factor{0.45};
    double cutback_factor{0.50};
    double growth_factor{1.25};
    int max_cutbacks_per_step{10};
    int max_total_steps{10000};
    std::vector<double> guard_points{};
    MixedControlArcLengthScales scales{};
    MixedControlArcLengthWeights weights{};
};

struct MixedControlArcLengthStepRecord {
    int step{0};
    double p_start{0.0};
    double p_target{0.0};
    double p_accepted{0.0};
    double increment{0.0};
    double arc_length{0.0};
    double normalized_control_increment{0.0};
    double normalized_reaction_increment{0.0};
    double normalized_internal_increment{0.0};
    int cutbacks_before_acceptance{0};
    bool accepted{false};
    bool rejected_by_arc_length{false};
};

struct MixedControlArcLengthResult {
    MixedControlArcLengthStatus status{
        MixedControlArcLengthStatus::invalid_configuration};
    int accepted_steps{0};
    int failed_solver_attempts{0};
    int rejected_arc_attempts{0};
    int total_cutbacks{0};
    double final_p{0.0};
    double max_arc_length{0.0};
    double mean_arc_length{0.0};
    std::vector<MixedControlArcLengthStepRecord> records{};

    [[nodiscard]] bool completed() const noexcept
    {
        return status == MixedControlArcLengthStatus::completed;
    }
};

namespace detail {

[[nodiscard]] inline double safe_positive_scale(double value) noexcept
{
    return std::isfinite(value) && value > 0.0 ? value : 1.0;
}

[[nodiscard]] inline double mixed_control_normalized_increment(
    double delta,
    double scale,
    double weight) noexcept
{
    if (!(std::isfinite(weight) && weight > 0.0)) {
        return 0.0;
    }
    return std::sqrt(weight) * delta / safe_positive_scale(scale);
}

[[nodiscard]] inline MixedControlArcLengthStepRecord
make_mixed_control_arc_record(
    const MixedControlArcLengthObservation& before,
    const MixedControlArcLengthObservation& after,
    const MixedControlArcLengthSettings& settings)
{
    MixedControlArcLengthStepRecord record{};
    record.normalized_control_increment =
        mixed_control_normalized_increment(
            after.control - before.control,
            settings.scales.control,
            settings.weights.control);
    record.normalized_reaction_increment =
        mixed_control_normalized_increment(
            after.reaction - before.reaction,
            settings.scales.reaction,
            settings.weights.reaction);
    record.normalized_internal_increment =
        mixed_control_normalized_increment(
            after.internal - before.internal,
            settings.scales.internal,
            settings.weights.internal);

    record.arc_length = std::sqrt(
        record.normalized_control_increment *
            record.normalized_control_increment +
        record.normalized_reaction_increment *
            record.normalized_reaction_increment +
        record.normalized_internal_increment *
            record.normalized_internal_increment);
    return record;
}

template <typename SolverT>
void set_solver_increment_if_available(SolverT& solver, double increment)
{
    if constexpr (requires { solver.set_increment_size(increment); }) {
        solver.set_increment_size(increment);
    }
}

} // namespace detail

template <typename SolverT, typename SamplerT, typename AcceptedStepCallbackT>
requires fall_n::CheckpointableSteppableSolver<SolverT>
[[nodiscard]] MixedControlArcLengthResult
run_mixed_control_arc_length_continuation(
    SolverT& solver,
    SamplerT&& sampler,
    MixedControlArcLengthSettings settings,
    AcceptedStepCallbackT&& on_accepted_step)
{
    MixedControlArcLengthResult result{};

    const auto invalid = [&]() {
        result.status = MixedControlArcLengthStatus::invalid_configuration;
        result.final_p = solver.current_time();
        return result;
    };

    if (!(std::isfinite(settings.target_p) && settings.target_p > 0.0) ||
        !(std::isfinite(settings.target_arc_length) &&
          settings.target_arc_length > 0.0) ||
        !(std::isfinite(settings.min_increment) &&
          settings.min_increment > 0.0) ||
        !(std::isfinite(settings.max_increment) &&
          settings.max_increment >= settings.min_increment))
    {
        return invalid();
    }

    settings.initial_increment = std::clamp(
        settings.initial_increment,
        settings.min_increment,
        settings.max_increment);
    settings.cutback_factor = std::clamp(settings.cutback_factor, 0.05, 0.95);
    settings.growth_factor = std::max(settings.growth_factor, 1.0);
    settings.reject_arc_length_factor =
        std::max(settings.reject_arc_length_factor, 1.0);
    settings.grow_arc_length_factor =
        std::clamp(settings.grow_arc_length_factor, 0.0, 1.0);
    settings.max_cutbacks_per_step =
        std::max(settings.max_cutbacks_per_step, 0);
    settings.max_total_steps = std::max(settings.max_total_steps, 1);
    constexpr double p_tol = 1.0e-14;
    std::ranges::sort(settings.guard_points);
    settings.guard_points.erase(
        std::ranges::unique(settings.guard_points).begin(),
        settings.guard_points.end());

    auto clamp_to_next_guard_point = [&](double p_start, double p_trial) {
        const auto next_guard = std::ranges::upper_bound(
            settings.guard_points,
            p_start + p_tol);
        if (next_guard != settings.guard_points.end() &&
            *next_guard < p_trial - p_tol)
        {
            return *next_guard;
        }
        if (next_guard != settings.guard_points.end() &&
            std::abs(*next_guard - p_trial) <= p_tol)
        {
            return *next_guard;
        }
        return p_trial;
    };

    auto&& sample_observation = sampler;
    auto previous = sample_observation();
    double increment = settings.initial_increment;
    int step = 0;
    double arc_sum = 0.0;

    while (solver.current_time() < settings.target_p - p_tol &&
           step < settings.max_total_steps)
    {
        const double p_start = solver.current_time();
        const auto checkpoint = solver.capture_checkpoint();
        int cutbacks = 0;
        bool accepted = false;

        while (!accepted) {
            const double trial_increment = std::min(
                increment,
                settings.target_p - p_start);
            const double p_target = clamp_to_next_guard_point(
                p_start,
                std::min(
                    p_start + trial_increment,
                    settings.target_p));
            const double actual_increment = p_target - p_start;

            detail::set_solver_increment_if_available(
                solver,
                std::max(actual_increment, settings.min_increment));

            const auto verdict = solver.step_to(p_target);
            if (verdict == fall_n::StepVerdict::Continue &&
                solver.current_time() >= p_target - p_tol)
            {
                auto current = sample_observation();
                auto record = detail::make_mixed_control_arc_record(
                    previous,
                    current,
                    settings);
                record.step = step + 1;
                record.p_start = p_start;
                record.p_target = p_target;
                record.p_accepted = solver.current_time();
                record.increment = actual_increment;
                record.cutbacks_before_acceptance = cutbacks;

                const bool arc_too_large =
                    record.arc_length >
                        settings.target_arc_length *
                            settings.reject_arc_length_factor &&
                    actual_increment > settings.min_increment * (1.0 + p_tol);

                if (arc_too_large) {
                    solver.restore_checkpoint(checkpoint);
                    ++result.rejected_arc_attempts;
                    ++result.total_cutbacks;
                    ++cutbacks;
                    record.rejected_by_arc_length = true;
                    result.records.push_back(record);
                    if (cutbacks > settings.max_cutbacks_per_step) {
                        result.status = MixedControlArcLengthStatus::stopped;
                        result.final_p = solver.current_time();
                        result.mean_arc_length =
                            result.accepted_steps > 0
                                ? arc_sum /
                                      static_cast<double>(result.accepted_steps)
                                : 0.0;
                        return result;
                    }
                    increment = std::max(
                        actual_increment * settings.cutback_factor,
                        settings.min_increment);
                    continue;
                }

                record.accepted = true;
                result.records.push_back(record);
                on_accepted_step(record);
                ++result.accepted_steps;
                ++step;
                result.max_arc_length =
                    std::max(result.max_arc_length, record.arc_length);
                arc_sum += record.arc_length;
                previous = current;

                if (record.arc_length <
                    settings.target_arc_length *
                        settings.grow_arc_length_factor)
                {
                    increment = std::min(
                        std::max(actual_increment, settings.min_increment) *
                            settings.growth_factor,
                        settings.max_increment);
                } else {
                    increment = std::min(
                        std::max(actual_increment, settings.min_increment),
                        settings.max_increment);
                }
                accepted = true;
                continue;
            }

            solver.restore_checkpoint(checkpoint);
            ++result.failed_solver_attempts;
            ++result.total_cutbacks;
            ++cutbacks;
            if (cutbacks > settings.max_cutbacks_per_step ||
                increment <= settings.min_increment * (1.0 + p_tol))
            {
                result.status = MixedControlArcLengthStatus::stopped;
                result.final_p = solver.current_time();
                result.mean_arc_length =
                    result.accepted_steps > 0
                        ? arc_sum / static_cast<double>(result.accepted_steps)
                        : 0.0;
                return result;
            }

            increment = std::max(
                increment * settings.cutback_factor,
                settings.min_increment);
        }
    }

    result.final_p = solver.current_time();
    result.mean_arc_length =
        result.accepted_steps > 0
            ? arc_sum / static_cast<double>(result.accepted_steps)
            : 0.0;
    result.status =
        result.final_p >= settings.target_p - p_tol
            ? MixedControlArcLengthStatus::completed
            : MixedControlArcLengthStatus::stopped;
    return result;
}

template <typename SolverT, typename SamplerT>
requires fall_n::CheckpointableSteppableSolver<SolverT>
[[nodiscard]] MixedControlArcLengthResult
run_mixed_control_arc_length_continuation(
    SolverT& solver,
    SamplerT&& sampler,
    MixedControlArcLengthSettings settings)
{
    auto noop = [](const MixedControlArcLengthStepRecord&) {};
    return run_mixed_control_arc_length_continuation(
        solver,
        std::forward<SamplerT>(sampler),
        settings,
        noop);
}

} // namespace fall_n

#endif // FALL_N_SRC_ANALYSIS_MIXED_CONTROL_ARC_LENGTH_CONTINUATION_HH
