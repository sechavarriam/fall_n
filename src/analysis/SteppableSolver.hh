#ifndef FALL_N_SRC_ANALYSIS_STEPPABLE_SOLVER_HH
#define FALL_N_SRC_ANALYSIS_STEPPABLE_SOLVER_HH

// =============================================================================
//  SteppableSolver — concept for analysis engines with single-step control
// =============================================================================
//
//  Both NonlinearAnalysis (quasi-static, SNES) and DynamicAnalysis (transient,
//  TS) expose a common single-step API that enables:
//
//    - Manual step-by-step control
//    - Pausing at runtime-defined conditions (via StepDirector)
//    - Inspecting/modifying the model between steps
//    - Switching analysis parameters mid-run
//    - Coupling between analysis engines
//
//  The "time" axis has engine-specific semantics:
//    - DynamicAnalysis:     physical time (seconds)
//    - NonlinearAnalysis:   control parameter p ∈ [0, 1]
//
//  Both engines advance monotonically along their time axis.
//
//  Minimal requirements:
//
//    bool   step()              — advance one step, return converged?
//    Verdict step_to(double)    — advance to target, return verdict
//    Verdict step_n(int)        — advance n steps, return verdict
//    double  current_time()     — position on time axis
//    PetscInt current_step()    — step counter
//
// =============================================================================

#include <concepts>
#include <petsc.h>
#include <type_traits>

#include "StepDirector.hh"

namespace fall_n {


template <typename S>
concept SteppableSolver = requires(S& solver, const S& csolver,
                                   double target, int n)
{
    { solver.step()          } -> std::convertible_to<bool>;
    { solver.step_to(target) } -> std::same_as<StepVerdict>;
    { solver.step_n(n)       } -> std::same_as<StepVerdict>;
    { csolver.current_time() } -> std::convertible_to<double>;
    { csolver.current_step() } -> std::convertible_to<PetscInt>;
};

template <typename S>
concept CheckpointableSteppableSolver =
    SteppableSolver<S> &&
    requires(
        S& solver,
        const S& csolver,
        const typename std::remove_cvref_t<S>::checkpoint_type& checkpoint)
{
    typename std::remove_cvref_t<S>::checkpoint_type;
    { csolver.capture_checkpoint() }
        -> std::same_as<typename std::remove_cvref_t<S>::checkpoint_type>;
    { solver.restore_checkpoint(checkpoint) } -> std::same_as<void>;
};

template <typename S>
concept TrialControllableSolver =
    requires(S& solver, bool enabled)
{
    { solver.set_auto_commit(enabled) } -> std::same_as<void>;
    { solver.commit_trial_state() } -> std::same_as<void>;
};


// =============================================================================
//  AnalysisState — portable snapshot for cross-engine state transfer
// =============================================================================
//
//  Captures the minimal state needed to hand off a solution between
//  analysis engines (e.g. static preload → dynamic excitation).
//
//  The Vecs are BORROWED references valid only during the lifetime of
//  the analysis object that produced them.  Copy with VecDuplicate +
//  VecCopy if you need to persist beyond the analysis scope.
//
//  Usage:
//
//    auto state = nl.get_analysis_state();
//    dynamic.set_initial_displacement(state.displacement);
//
// =============================================================================

struct AnalysisState {
    Vec      displacement{nullptr};   ///< Global displacement vector (borrowed)
    Vec      velocity{nullptr};       ///< Global velocity vector (nullptr for static)
    double   time{0.0};               ///< Physical time or control parameter
    PetscInt step{0};                 ///< Step count
};


} // namespace fall_n

#endif // FALL_N_SRC_ANALYSIS_STEPPABLE_SOLVER_HH
