#ifndef FALL_N_SRC_ANALYSIS_STEP_DIRECTOR_HH
#define FALL_N_SRC_ANALYSIS_STEP_DIRECTOR_HH

// =============================================================================
//  StepDirector — Condition-based breakpoints for steppable solvers
// =============================================================================
//
//  A StepDirector is a callable that, after every converged step, inspects
//  the current state and returns a verdict:
//
//    Continue  — keep stepping normally
//    Pause     — return control to the caller (resumable)
//    Stop      — terminate the analysis (converged or user-abort)
//
//  The director is evaluated AFTER the post-step commit (material state,
//  observer notification, model state update).  This guarantees that
//  the model is in a fully consistent state when the caller inspects it.
//
//  Usage:
//
//    // Pause at specific times
//    auto dir = step_director::pause_at_times<MyModel>({1.0, 2.5, 5.0});
//
//    // Pause every N steps
//    auto dir = step_director::pause_every_n<MyModel>(100);
//
//    // Custom condition
//    auto dir = step_director::pause_on([](const StepEvent& ev, const auto& m) {
//        return ev.time > 1.0 && some_condition(m);
//    });
//
//    // Compose directors (most restrictive verdict wins)
//    auto dir = step_director::compose<MyModel>(
//        step_director::pause_at_times<MyModel>({1.0}),
//        step_director::pause_on([](auto& ev, auto&) {
//            return ev.step > 1000;
//        })
//    );
//
//    solver.set_director(dir);
//    solver.step_to(10.0);  // will pause at breakpoints
//
// =============================================================================

#include <algorithm>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <vector>

#include "AnalysisObserver.hh"

namespace fall_n {


// ─────────────────────────────────────────────────────────────────────────────
//  StepVerdict — the three possible outcomes of a director evaluation
// ─────────────────────────────────────────────────────────────────────────────

enum class StepVerdict {
    Continue,   ///< Keep stepping
    Pause,      ///< Return control to the caller (resumable)
    Stop        ///< Terminate the analysis
};

/// Most restrictive verdict wins:  Stop > Pause > Continue.
constexpr StepVerdict most_restrictive(StepVerdict a, StepVerdict b) noexcept {
    return (a > b) ? a : b;
}


// ─────────────────────────────────────────────────────────────────────────────
//  StepDirector<ModelT> — type alias for the director callable
// ─────────────────────────────────────────────────────────────────────────────

template <typename ModelT>
using StepDirector = std::function<StepVerdict(const StepEvent&, const ModelT&)>;


// ─────────────────────────────────────────────────────────────────────────────
//  Convenience factories
// ─────────────────────────────────────────────────────────────────────────────

namespace step_director {

/// Pause when the simulation time reaches any of the given breakpoints.
/// Each breakpoint fires at most once (consumed on first hit).
template <typename ModelT>
StepDirector<ModelT> pause_at_times(std::vector<double> times, double tol = 1e-12) {
    std::sort(times.begin(), times.end());
    return [times = std::move(times), tol, idx = std::size_t{0}]
           (const StepEvent& ev, [[maybe_unused]] const ModelT&) mutable
           -> StepVerdict
    {
        if (idx < times.size() && ev.time >= times[idx] - tol) {
            ++idx;
            return StepVerdict::Pause;
        }
        return StepVerdict::Continue;
    };
}

/// Pause every N converged steps.
template <typename ModelT>
StepDirector<ModelT> pause_every_n(PetscInt interval) {
    return [interval](const StepEvent& ev, [[maybe_unused]] const ModelT&)
           -> StepVerdict
    {
        return (ev.step > 0 && ev.step % interval == 0)
            ? StepVerdict::Pause
            : StepVerdict::Continue;
    };
}

/// Pause when a user-supplied predicate returns true.
///
///   auto dir = pause_on([](const StepEvent& ev, const MyModel& m) {
///       return some_condition(ev, m);
///   });
template <typename F>
auto pause_on(F&& pred) {
    return [pred = std::forward<F>(pred)]
           (const auto& ev, const auto& model) -> StepVerdict
    {
        return pred(ev, model) ? StepVerdict::Pause : StepVerdict::Continue;
    };
}

/// Stop when a user-supplied predicate returns true.
template <typename F>
auto stop_on(F&& pred) {
    return [pred = std::forward<F>(pred)]
           (const auto& ev, const auto& model) -> StepVerdict
    {
        return pred(ev, model) ? StepVerdict::Stop : StepVerdict::Continue;
    };
}

/// Compose multiple directors.  The most restrictive verdict wins.
template <typename ModelT>
StepDirector<ModelT> compose(std::vector<StepDirector<ModelT>> directors) {
    return [dirs = std::move(directors)]
           (const StepEvent& ev, const ModelT& model) -> StepVerdict
    {
        StepVerdict result = StepVerdict::Continue;
        for (auto& d : dirs) {
            result = most_restrictive(result, d(ev, model));
        }
        return result;
    };
}

/// Variadic compose overload for convenience.
template <typename ModelT, typename... Dirs>
StepDirector<ModelT> compose(Dirs&&... dirs) {
    std::vector<StepDirector<ModelT>> v;
    (v.emplace_back(std::forward<Dirs>(dirs)), ...);
    return compose<ModelT>(std::move(v));
}

} // namespace step_director

} // namespace fall_n

#endif // FALL_N_SRC_ANALYSIS_STEP_DIRECTOR_HH
