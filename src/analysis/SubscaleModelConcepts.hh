#ifndef FALL_N_SRC_ANALYSIS_SUBSCALE_MODEL_CONCEPTS_HH
#define FALL_N_SRC_ANALYSIS_SUBSCALE_MODEL_CONCEPTS_HH

// =============================================================================
//  SubscaleModelConcepts -- generic compile-time contracts for local models
// =============================================================================
//
//  The original FE2 path in fall_n evolved around a section-specialized local
//  model contract: beam kinematics are imposed, a section law is recovered,
//  and that law is reinjected at the macro scale.
//
//  That path remains the production route today, but it is not yet the right
//  abstraction level for future local models that may:
//
//    - expose a different effective operator than a beam-section law,
//    - accept driving data richer than SectionKinematics,
//    - belong to a different local discretization family (XFEM, DG, DPG,
//      operator surrogates, etc.), or
//    - participate in multiscale workflows that still require rollback,
//      observation, and deterministic orchestration.
//
//  This header introduces small, orthogonal concepts that capture those
//  generic lifecycle responsibilities without imposing a specific FE2 section
//  ontology on every future subscale model.
//
//  The design goal is:
//
//    - zero runtime cost in the hot path,
//    - explicit SOLID-friendly seams at compile time,
//    - preservation of the current typed multiscale pipeline,
//    - and a migration path from "section-local model" toward "abstract local
//      subproblem" without forcing type erasure into the normative API.
//
// =============================================================================

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace fall_n {

template <typename T>
concept StepSolvableSubscaleModel = requires(T& t, double pseudo_time)
{
    { t.solve_step(pseudo_time) };
};

template <typename T>
concept CheckpointableSubscaleModel = requires(
    T& t,
    bool auto_commit,
    double pseudo_time,
    const typename std::remove_cvref_t<T>::checkpoint_type& checkpoint)
{
    { t.commit_state() };
    { t.revert_state() };
    { t.commit_trial_state() };
    { t.end_of_step(pseudo_time) };
    { t.set_auto_commit(auto_commit) };
    { t.capture_checkpoint() }
        -> std::same_as<typename std::remove_cvref_t<T>::checkpoint_type>;
    { t.restore_checkpoint(checkpoint) } -> std::same_as<void>;
};

template <typename T>
concept ObservableSubscaleModel = requires(const T& t)
{
    { t.parent_element_id() } -> std::convertible_to<std::size_t>;
};

template <typename T, typename DrivingStateT>
concept DrivenSubscaleModel = requires(T& t, const DrivingStateT& driving_state)
{
    { t.apply_driving_state(driving_state) };
};

template <typename T, typename EffectiveOperatorT>
concept EffectiveOperatorProvider = requires(T& t)
{
    { t.effective_operator() } -> std::convertible_to<EffectiveOperatorT>;
};

template <typename T, typename RequestT, typename EffectiveOperatorT>
concept RequestedEffectiveOperatorProvider = requires(
    T& t, const RequestT& request)
{
    { t.effective_operator(request) } -> std::convertible_to<EffectiveOperatorT>;
};

template <typename T,
          typename DrivingStateT,
          typename EffectiveOperatorT>
concept SubscaleModel =
    StepSolvableSubscaleModel<T> &&
    CheckpointableSubscaleModel<T> &&
    ObservableSubscaleModel<T> &&
    DrivenSubscaleModel<T, DrivingStateT> &&
    EffectiveOperatorProvider<T, EffectiveOperatorT>;

template <typename T,
          typename DrivingStateT,
          typename RequestT,
          typename EffectiveOperatorT>
concept RequestedSubscaleModel =
    StepSolvableSubscaleModel<T> &&
    CheckpointableSubscaleModel<T> &&
    ObservableSubscaleModel<T> &&
    DrivenSubscaleModel<T, DrivingStateT> &&
    RequestedEffectiveOperatorProvider<T, RequestT, EffectiveOperatorT>;

} // namespace fall_n

#endif // FALL_N_SRC_ANALYSIS_SUBSCALE_MODEL_CONCEPTS_HH
