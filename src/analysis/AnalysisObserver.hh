#ifndef FALL_N_ANALYSIS_OBSERVER_HH
#define FALL_N_ANALYSIS_OBSERVER_HH

// =============================================================================
//  AnalysisObserver — Composable observation/recording for analysis solvers
// =============================================================================
//
//  Design goals:
//    1. Open/Closed principle: adding a new observer never modifies existing
//       code in DynamicAnalysis or any sibling observers.
//    2. Single Responsibility: each observer handles one concern (progress
//       printing, VTK output, node recording, fiber inspection…).
//    3. Compile-time efficiency: the CompositeObserver stores observers in a
//       std::tuple (heterogeneous, inline, no heap / no virtual dispatch)
//       when the observer set is known at compile time.  A type-erased
//       DynamicObserverList is provided for runtime composition.
//    4. Non-intrusive: observers receive a const model reference and read
//       data through existing accessors.
//
//  Two flavours:
//
//    ┌───────────────────────────────────────────────────────────┐
//    │  STATIC  (compile-time)                                  │
//    │  CompositeObserver<ModelT, Obs1, Obs2, …>                │
//    │  → zero overhead, all calls inlined                      │
//    │  → observers are value members in a std::tuple           │
//    └───────────────────────────────────────────────────────────┘
//
//    ┌───────────────────────────────────────────────────────────┐
//    │  DYNAMIC  (runtime)                                      │
//    │  DynamicObserverList<ModelT>                              │
//    │  → each observer behind unique_ptr<ObserverBase<ModelT>> │
//    │  → observers can be added/removed at runtime             │
//    └───────────────────────────────────────────────────────────┘
//
//  Integration:
//    DynamicAnalysis::set_observer(obs)  — stores a reference/pointer to
//    either flavour through a type-erased callback.  The PETSc TSMonitor
//    callback delegates to the observer on every converged step.
//
// =============================================================================

#include <cstddef>
#include <functional>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <petsc.h>


namespace fall_n {

// ─────────────────────────────────────────────────────────────────────────────
//  AnalysisEvent — data passed to every observer on each step
// ─────────────────────────────────────────────────────────────────────────────

/// Lightweight value carrying the per-step state.  Passed by value (cheap:
/// two ints + two PETSc handles) so observers never hold dangling references.
struct StepEvent {
    PetscInt step;
    double   time;
    Vec      displacement;   // PETSc global Vec — valid only during the call
    Vec      velocity;       // PETSc global Vec — valid only during the call
};


// ─────────────────────────────────────────────────────────────────────────────
//  ObserverBase<ModelT> — abstract base for DYNAMIC (runtime) composition
// ─────────────────────────────────────────────────────────────────────────────

template <typename ModelT>
class ObserverBase {
public:
    virtual ~ObserverBase() = default;

    /// Called once, right before the first time step.
    virtual void on_analysis_start([[maybe_unused]] const ModelT& model) {}

    /// Called after every converged time step.
    virtual void on_step(const StepEvent& event, const ModelT& model) = 0;

    /// Called once, after the last time step (or on divergence).
    virtual void on_analysis_end([[maybe_unused]] const ModelT& model) {}
};


// ─────────────────────────────────────────────────────────────────────────────
//  DynamicObserverList<ModelT> — runtime-polymorphic observer container
// ─────────────────────────────────────────────────────────────────────────────
//
//  Use when observers are created/configured at runtime (e.g. from input
//  files or user scripts).  Virtual dispatch cost is negligible: one
//  indirect call per observer per step, dwarfed by constitutive evaluation.

template <typename ModelT>
class DynamicObserverList {
    std::vector<std::unique_ptr<ObserverBase<ModelT>>> children_;

public:
    /// Add an observer by constructing it in-place.  Returns a reference
    /// for further configuration.
    template <typename ObsT, typename... Args>
        requires std::derived_from<ObsT, ObserverBase<ModelT>>
    ObsT& add(Args&&... args) {
        auto p = std::make_unique<ObsT>(std::forward<Args>(args)...);
        auto& ref = *p;
        children_.push_back(std::move(p));
        return ref;
    }

    /// Add a pre-constructed observer.
    template <typename ObsT>
        requires std::derived_from<ObsT, ObserverBase<ModelT>>
    ObsT& add(std::unique_ptr<ObsT> p) {
        auto& ref = *p;
        children_.push_back(std::move(p));
        return ref;
    }

    void on_analysis_start(const ModelT& m) {
        for (auto& c : children_) c->on_analysis_start(m);
    }

    void on_step(const StepEvent& ev, const ModelT& m) {
        for (auto& c : children_) c->on_step(ev, m);
    }

    void on_analysis_end(const ModelT& m) {
        for (auto& c : children_) c->on_analysis_end(m);
    }

    [[nodiscard]] std::size_t size() const noexcept { return children_.size(); }
    [[nodiscard]] bool empty()      const noexcept { return children_.empty(); }
};


// ─────────────────────────────────────────────────────────────────────────────
//  CompositeObserver<ModelT, Observers...> — compile-time observer pipeline
// ─────────────────────────────────────────────────────────────────────────────
//
//  Stores all observers in a std::tuple (no heap, no virtual).  Each call
//  is expanded at compile time via fold expressions → zero dispatch overhead.
//
//  Usage:
//    auto obs = make_composite_observer<MyModel>(
//        ConsoleProgressObserver<MyModel>{10},
//        VTKSnapshotObserver<MyModel>{dir, 100, profile, thickness},
//        NodeRecorder<MyModel>{{node_ids}, {dofs}, interval}
//    );
//    solver.set_observer(obs);

template <typename ModelT, typename... Observers>
class CompositeObserver {
    std::tuple<Observers...> observers_;

public:
    explicit CompositeObserver(Observers... obs)
        : observers_(std::move(obs)...) {}

    void on_analysis_start(const ModelT& m) {
        std::apply([&m](auto&... obs) {
            (obs.on_analysis_start(m), ...);
        }, observers_);
    }

    void on_step(const StepEvent& ev, const ModelT& m) {
        std::apply([&ev, &m](auto&... obs) {
            (obs.on_step(ev, m), ...);
        }, observers_);
    }

    void on_analysis_end(const ModelT& m) {
        std::apply([&m](auto&... obs) {
            (obs.on_analysis_end(m), ...);
        }, observers_);
    }

    /// Access the N-th observer (compile-time index).
    template <std::size_t N>
    auto& get() noexcept { return std::get<N>(observers_); }

    template <std::size_t N>
    const auto& get() const noexcept { return std::get<N>(observers_); }

    /// Access observer by type (must be unique in the pack).
    template <typename T>
    T& get() noexcept { return std::get<T>(observers_); }

    template <typename T>
    const T& get() const noexcept { return std::get<T>(observers_); }
};


/// Factory: deduces template arguments from constructor arguments.
template <typename ModelT, typename... Observers>
auto make_composite_observer(Observers&&... obs) {
    return CompositeObserver<ModelT, std::remove_cvref_t<Observers>...>(
        std::forward<Observers>(obs)...);
}


// ─────────────────────────────────────────────────────────────────────────────
//  ObserverCallback — type-erased trampoline for DynamicAnalysis integration
// ─────────────────────────────────────────────────────────────────────────────
//
//  DynamicAnalysis stores one ObserverCallback.  Both CompositeObserver and
//  DynamicObserverList (and the legacy MonitorCallback lambda) are convertible
//  to this via make_observer_callback().

template <typename ModelT>
struct ObserverCallback {
    std::function<void(const ModelT&)>                   on_start;
    std::function<void(const StepEvent&, const ModelT&)> on_step;
    std::function<void(const ModelT&)>                   on_end;
};


/// Wrap any observer-like object (duck-typed) into an ObserverCallback.
template <typename ModelT, typename Obs>
ObserverCallback<ModelT> make_observer_callback(Obs& obs) {
    return ObserverCallback<ModelT>{
        .on_start = [&obs](const ModelT& m)                   { obs.on_analysis_start(m); },
        .on_step  = [&obs](const StepEvent& ev, const ModelT& m) { obs.on_step(ev, m); },
        .on_end   = [&obs](const ModelT& m)                   { obs.on_analysis_end(m); },
    };
}


} // namespace fall_n

#endif // FALL_N_ANALYSIS_OBSERVER_HH
