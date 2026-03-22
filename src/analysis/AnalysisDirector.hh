// =============================================================================
//  AnalysisDirector.hh — Multi-phase analysis orchestrator
// =============================================================================
//
//  PURPOSE:
//    Coordinates a sequence of analysis phases that share the same model
//    but use different solver engines and/or configurations.  Typical
//    real-world scenarios:
//
//      Phase 1 (static preload):   gravity / self-weight via NonlinearAnalysis
//      Phase 2 (dynamic):          seismic excitation    via DynamicAnalysis
//      Phase 3 (static redistrib): post-event settling   via NonlinearAnalysis
//
//    Between phases the AnalysisDirector:
//      1. Captures state from the completing engine (displacement, velocity)
//      2. Injects that state into the next engine as initial conditions
//      3. Carries observers across phases (lifecycle management)
//      4. Evaluates inter-phase conditions via user callbacks
//
//  DESIGN NOTES:
//
//    The director does NOT own the solvers — callers construct them on
//    their own (potentially with different template parameters).  The
//    director stores type-erased "phase" objects that wrap the concrete
//    solver+config+target.
//
//    Phases are registered with add_phase() and executed in order by run().
//    Each phase is a callable that receives an AnalysisState (the output
//    of the prior phase) and returns an AnalysisState (its own output).
//    This gives maximum flexibility: the caller decides how to create and
//    configure each solver, apply BCs, attach observers, etc.
//
//    The director provides:
//      - Ordered phase execution with automatic state threading
//      - Phase-level event callbacks (on_begin, on_end)
//      - Abort-on-failure semantics (stop if a phase diverges)
//      - A unified log of phase outcomes
//
//    The SteppableSolver concept is NOT directly constrained here because
//    the phase callback is fully opaque (std::function).  However, the
//    typical usage pattern inside each phase callback IS to create a
//    SteppableSolver, call step_to/step_n, and return get_analysis_state().
//
//  USAGE:
//
//    AnalysisDirector director;
//
//    director.add_phase("gravity", [&](const AnalysisState& prev) {
//        NonlinearAnalysis nl{&model};
//        nl.solve_incremental(10);
//        return nl.get_analysis_state();
//    });
//
//    director.add_phase("seismic", [&](const AnalysisState& prev) {
//        DynamicAnalysis dyn{&model};
//        dyn.set_initial_displacement(prev.displacement);
//        dyn.solve(30.0, 0.01);
//        return dyn.get_analysis_state();
//    });
//
//    auto report = director.run();
//
// =============================================================================

#ifndef FALL_N_SRC_ANALYSIS_ANALYSIS_DIRECTOR_HH
#define FALL_N_SRC_ANALYSIS_ANALYSIS_DIRECTOR_HH

#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "SteppableSolver.hh"

namespace fall_n {


// =============================================================================
//  PhaseOutcome — result of a single phase execution
// =============================================================================

enum class PhaseStatus {
    Succeeded,    ///< Phase completed normally
    Failed,       ///< Phase diverged / returned failure
    Skipped       ///< Phase was skipped (precondition not met)
};

struct PhaseOutcome {
    std::string  name;
    PhaseStatus  status{PhaseStatus::Skipped};
    double       elapsed_seconds{0.0};
    AnalysisState state{};            ///< Output state of this phase
};


// =============================================================================
//  DirectorReport — summary of all phases after run()
// =============================================================================

struct DirectorReport {
    std::vector<PhaseOutcome> phases;

    bool all_succeeded() const noexcept {
        for (const auto& p : phases)
            if (p.status != PhaseStatus::Succeeded) return false;
        return !phases.empty();
    }

    std::size_t num_succeeded() const noexcept {
        std::size_t n = 0;
        for (const auto& p : phases)
            if (p.status == PhaseStatus::Succeeded) ++n;
        return n;
    }

    std::size_t num_failed() const noexcept {
        std::size_t n = 0;
        for (const auto& p : phases)
            if (p.status == PhaseStatus::Failed) ++n;
        return n;
    }

    void print_summary() const {
        std::cout << "\nAnalysisDirector — Phase Summary\n";
        std::cout << "┌────┬────────────────────────────────┬───────────┬────────────┐\n";
        std::cout << "│  # │ Phase                          │ Status    │ Time (s)   │\n";
        std::cout << "├────┼────────────────────────────────┼───────────┼────────────┤\n";
        for (std::size_t i = 0; i < phases.size(); ++i) {
            const auto& p = phases[i];
            const char* status_str = (p.status == PhaseStatus::Succeeded) ? "OK"
                                   : (p.status == PhaseStatus::Failed)    ? "FAILED"
                                   : "SKIPPED";
            char buf[128];
            std::snprintf(buf, sizeof(buf), "│ %2zu │ %-30s │ %-9s │ %10.3f │",
                          i + 1, p.name.c_str(), status_str, p.elapsed_seconds);
            std::cout << buf << "\n";
        }
        std::cout << "└────┴────────────────────────────────┴───────────┴────────────┘\n";

        if (all_succeeded()) {
            std::cout << "All " << phases.size() << " phases completed successfully.\n";
        } else {
            std::cout << num_succeeded() << "/" << phases.size()
                      << " phases succeeded, " << num_failed() << " failed.\n";
        }
    }
};


// =============================================================================
//  AnalysisDirector — the orchestrator
// =============================================================================

/// A phase callback receives the previous phase's state and returns its own.
///
/// Returning an AnalysisState with displacement == nullptr signals failure.
/// The director interprets this as PhaseStatus::Failed and aborts (unless
/// continue_on_failure is set).
using PhaseCallback = std::function<AnalysisState(const AnalysisState&)>;

/// Optional inter-phase callback: called BETWEEN phases with the
/// previous phase's outcome.  Return false to abort the sequence.
using PhaseGateCallback = std::function<bool(const PhaseOutcome&)>;


class AnalysisDirector {
public:

    // ── Phase registration ──────────────────────────────────────────

    /// Register a new phase.  Phases execute in registration order.
    AnalysisDirector& add_phase(std::string name, PhaseCallback callback) {
        phases_.push_back({std::move(name), std::move(callback), nullptr});
        return *this;
    }

    /// Register a phase with a gate: the gate is evaluated BEFORE the
    /// phase runs.  If the gate returns false, the phase is Skipped.
    AnalysisDirector& add_phase(std::string name,
                                PhaseCallback   callback,
                                PhaseGateCallback gate) {
        phases_.push_back({std::move(name), std::move(callback),
                           std::move(gate)});
        return *this;
    }

    /// If true, a failed phase does not abort the remaining phases.
    AnalysisDirector& set_continue_on_failure(bool value) noexcept {
        continue_on_failure_ = value;
        return *this;
    }


    // ── Execution ───────────────────────────────────────────────────

    /// Run all phases in sequence.  Returns a report with outcomes.
    ///
    /// The first phase receives a default-constructed AnalysisState
    /// (all nullptrs, time=0, step=0).  Each subsequent phase receives
    /// the output of its predecessor.
    DirectorReport run() {
        DirectorReport report;
        AnalysisState current_state{};   // initial: empty

        for (auto& phase : phases_) {
            PhaseOutcome outcome;
            outcome.name = phase.name;

            // Gate check
            if (phase.gate && !report.phases.empty()) {
                if (!phase.gate(report.phases.back())) {
                    outcome.status = PhaseStatus::Skipped;
                    report.phases.push_back(std::move(outcome));
                    continue;
                }
            }

            // Execute
            auto t0 = std::chrono::steady_clock::now();
            AnalysisState result = phase.callback(current_state);
            auto t1 = std::chrono::steady_clock::now();

            outcome.elapsed_seconds =
                std::chrono::duration<double>(t1 - t0).count();

            if (result.displacement != nullptr) {
                outcome.status = PhaseStatus::Succeeded;
                outcome.state  = result;
                current_state  = result;
            } else {
                outcome.status = PhaseStatus::Failed;
                outcome.state  = result;
            }

            report.phases.push_back(std::move(outcome));

            // Abort on failure unless configured otherwise
            if (report.phases.back().status == PhaseStatus::Failed
                && !continue_on_failure_) {
                break;
            }
        }

        return report;
    }


    // ── Queries ─────────────────────────────────────────────────────

    std::size_t num_phases() const noexcept {
        return phases_.size();
    }

private:

    struct Phase {
        std::string       name;
        PhaseCallback     callback;
        PhaseGateCallback gate;         ///< nullptr → always run
    };

    std::vector<Phase> phases_;
    bool continue_on_failure_{false};
};


} // namespace fall_n

#endif // FALL_N_SRC_ANALYSIS_ANALYSIS_DIRECTOR_HH
