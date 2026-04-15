#ifndef FALL_N_COMPUTATIONAL_VARIATIONAL_SLICE_AUDIT_HH
#define FALL_N_COMPUTATIONAL_VARIATIONAL_SLICE_AUDIT_HH

// =============================================================================
//  ComputationalVariationalSliceAudit.hh -- compile-time audit of the discrete
//                                           variational statement carried by a
//                                           typed Model + Solver slice
// =============================================================================
//
//  The previous audit/catalog layers answer, progressively:
//
//    1. What does a formulation mean mathematically?
//    2. For which element family is that formulation actually supported?
//    3. Which global analysis routes are available for that scope?
//    4. Which concrete element + solver pair is audited?
//    5. Which concrete Model + Solver slice is audited?
//
//  The FEM chapter, however, ultimately needs one more question answered:
//
//      "What discrete variational statement does that computational slice
//       actually assemble and linearize?"
//
//  This header combines:
//
//    - the typed slice audit (Model + Solver),
//    - the discrete carrier semantics (family + formulation),
//    - and a minimal but explicit statement of the global residual/tangent/
//      control/update topology induced by the solver route.
//
//  The goal is to make the thesis and the library talk about the same
//  computational object without pushing any additional runtime abstraction
//  into the hot numerical path.
//
// =============================================================================

#include <string_view>

#include "../continuum/DiscreteVariationalSemantics.hh"
#include "ComputationalModelSliceAudit.hh"

namespace fall_n {

enum class GlobalResidualOperatorKind {
    unavailable,
    static_equilibrium,
    incremental_static_equilibrium,
    second_order_dynamic_equilibrium,
    arc_length_augmented_equilibrium
};

enum class GlobalTangentOperatorKind {
    unavailable,
    linear_or_static_consistent_tangent,
    monolithic_incremental_consistent_tangent,
    effective_mass_damping_stiffness_tangent,
    bordered_arc_length_tangent
};

enum class GlobalControlSemanticsKind {
    unavailable,
    direct_load_or_displacement_control,
    incremental_load_or_displacement_control,
    implicit_time_control,
    arc_length_constraint_control
};

enum class IncrementalStateManagementKind {
    unavailable,
    stateless_or_direct_response,
    converged_step_commit,
    checkpointable_converged_step_commit,
    continuation_step_commit
};

[[nodiscard]] constexpr std::string_view
to_string(GlobalResidualOperatorKind kind) noexcept
{
    switch (kind) {
        case GlobalResidualOperatorKind::unavailable:
            return "unavailable";
        case GlobalResidualOperatorKind::static_equilibrium:
            return "static_equilibrium";
        case GlobalResidualOperatorKind::incremental_static_equilibrium:
            return "incremental_static_equilibrium";
        case GlobalResidualOperatorKind::second_order_dynamic_equilibrium:
            return "second_order_dynamic_equilibrium";
        case GlobalResidualOperatorKind::arc_length_augmented_equilibrium:
            return "arc_length_augmented_equilibrium";
    }
    return "unknown_global_residual_operator_kind";
}

[[nodiscard]] constexpr std::string_view
to_string(GlobalTangentOperatorKind kind) noexcept
{
    switch (kind) {
        case GlobalTangentOperatorKind::unavailable:
            return "unavailable";
        case GlobalTangentOperatorKind::linear_or_static_consistent_tangent:
            return "linear_or_static_consistent_tangent";
        case GlobalTangentOperatorKind::monolithic_incremental_consistent_tangent:
            return "monolithic_incremental_consistent_tangent";
        case GlobalTangentOperatorKind::effective_mass_damping_stiffness_tangent:
            return "effective_mass_damping_stiffness_tangent";
        case GlobalTangentOperatorKind::bordered_arc_length_tangent:
            return "bordered_arc_length_tangent";
    }
    return "unknown_global_tangent_operator_kind";
}

[[nodiscard]] constexpr std::string_view
to_string(GlobalControlSemanticsKind kind) noexcept
{
    switch (kind) {
        case GlobalControlSemanticsKind::unavailable:
            return "unavailable";
        case GlobalControlSemanticsKind::direct_load_or_displacement_control:
            return "direct_load_or_displacement_control";
        case GlobalControlSemanticsKind::incremental_load_or_displacement_control:
            return "incremental_load_or_displacement_control";
        case GlobalControlSemanticsKind::implicit_time_control:
            return "implicit_time_control";
        case GlobalControlSemanticsKind::arc_length_constraint_control:
            return "arc_length_constraint_control";
    }
    return "unknown_global_control_semantics_kind";
}

[[nodiscard]] constexpr std::string_view
to_string(IncrementalStateManagementKind kind) noexcept
{
    switch (kind) {
        case IncrementalStateManagementKind::unavailable:
            return "unavailable";
        case IncrementalStateManagementKind::stateless_or_direct_response:
            return "stateless_or_direct_response";
        case IncrementalStateManagementKind::converged_step_commit:
            return "converged_step_commit";
        case IncrementalStateManagementKind::checkpointable_converged_step_commit:
            return "checkpointable_converged_step_commit";
        case IncrementalStateManagementKind::continuation_step_commit:
            return "continuation_step_commit";
    }
    return "unknown_incremental_state_management_kind";
}

struct ComputationalVariationalSliceAuditScope {
    continuum::ElementFamilyKind element_family{
        continuum::ElementFamilyKind::continuum_solid_3d};
    continuum::FormulationKind formulation_kind{
        continuum::FormulationKind::small_strain};
    AnalysisRouteKind route_kind{AnalysisRouteKind::linear_static};

    ComputationalModelSliceAuditScope model_solver_slice{};
    continuum::DiscreteVariationalSemantics discrete_variational_semantics{};

    GlobalResidualOperatorKind global_residual_operator{
        GlobalResidualOperatorKind::unavailable};
    GlobalTangentOperatorKind global_tangent_operator{
        GlobalTangentOperatorKind::unavailable};
    GlobalControlSemanticsKind global_control_semantics{
        GlobalControlSemanticsKind::unavailable};
    IncrementalStateManagementKind incremental_state_management{
        IncrementalStateManagementKind::unavailable};

    bool assembles_monolithic_equilibrium_residual{false};
    bool linearizes_consistent_global_tangent{false};
    bool local_history_is_committed_on_converged_step{false};
    bool local_history_supports_checkpoint_restart{false};
    bool admits_effective_operator_predictor_injection{false};
    bool augments_with_inertial_terms{false};
    bool augments_with_continuation_constraint{false};

    [[nodiscard]] constexpr bool
    has_well_defined_discrete_variational_statement() const noexcept
    {
        return discrete_variational_semantics.kinematic_carrier !=
                   continuum::DiscreteKinematicCarrierKind::unavailable &&
               discrete_variational_semantics.stress_carrier !=
                   continuum::DiscreteStressCarrierKind::unavailable &&
               discrete_variational_semantics.integration_domain !=
                   continuum::DiscreteIntegrationDomainKind::unavailable &&
               discrete_variational_semantics.residual_construction !=
                   continuum::DiscreteResidualConstructionKind::unavailable &&
               discrete_variational_semantics.tangent_composition !=
                   continuum::DiscreteTangentCompositionKind::unavailable &&
               discrete_variational_semantics.algorithmic_tangent_source !=
                   continuum::AlgorithmicTangentSourceKind::unavailable &&
               global_residual_operator != GlobalResidualOperatorKind::unavailable &&
               global_tangent_operator != GlobalTangentOperatorKind::unavailable &&
               global_control_semantics != GlobalControlSemanticsKind::unavailable &&
               incremental_state_management !=
                   IncrementalStateManagementKind::unavailable;
    }

    [[nodiscard]] constexpr bool
    has_normative_variational_slice() const noexcept
    {
        return model_solver_slice.supports_normatively() &&
               discrete_variational_semantics.is_normatively_coherent() &&
               has_well_defined_discrete_variational_statement();
    }

    [[nodiscard]] constexpr bool
    requires_scope_disclaimer() const noexcept
    {
        return !has_normative_variational_slice() ||
               model_solver_slice.requires_scope_disclaimer() ||
               discrete_variational_semantics.requires_scope_disclaimer();
    }

    [[nodiscard]] constexpr bool is_structural_reduction_path() const noexcept {
        return discrete_variational_semantics.is_structural_reduction_path();
    }

    [[nodiscard]] constexpr bool integrates_on_reference_like_domain() const noexcept {
        return discrete_variational_semantics.integrates_on_reference_like_domain();
    }

    [[nodiscard]] constexpr bool integrates_on_current_like_domain() const noexcept {
        return discrete_variational_semantics.integrates_on_current_like_domain();
    }
};

template <AuditedComputationalModelType ModelT, SolverWithAuditedModelSlice SolverT>
[[nodiscard]] constexpr ComputationalVariationalSliceAuditScope
canonical_computational_variational_slice_audit_scope() noexcept
{
    using CleanModel = std::remove_cvref_t<ModelT>;
    using CleanSolver = std::remove_cvref_t<SolverT>;

    constexpr auto element_family = CleanModel::element_family_kind;
    constexpr auto formulation_kind = CleanModel::formulation_kind;
    constexpr auto route_kind = solver_analysis_route_kind_v<CleanSolver>;

    const auto model_solver_slice =
        canonical_model_solver_slice_audit_scope<ModelT, SolverT>();
    const auto discrete_variational_semantics =
        continuum::canonical_family_formulation_discrete_variational_semantics(
            element_family, formulation_kind);
    const auto& route_scope =
        model_solver_slice.element_solver_scope.family_formulation_route_scope;

    const auto make_scope =
        [&](GlobalResidualOperatorKind global_residual_operator,
            GlobalTangentOperatorKind global_tangent_operator,
            GlobalControlSemanticsKind global_control_semantics,
            IncrementalStateManagementKind incremental_state_management,
            bool assembles_monolithic_equilibrium_residual,
            bool linearizes_consistent_global_tangent,
            bool local_history_is_committed_on_converged_step,
            bool local_history_supports_checkpoint_restart,
            bool admits_effective_operator_predictor_injection,
            bool augments_with_inertial_terms,
            bool augments_with_continuation_constraint) constexpr
        {
            return ComputationalVariationalSliceAuditScope{
                .element_family = element_family,
                .formulation_kind = formulation_kind,
                .route_kind = route_kind,
                .model_solver_slice = model_solver_slice,
                .discrete_variational_semantics = discrete_variational_semantics,
                .global_residual_operator = global_residual_operator,
                .global_tangent_operator = global_tangent_operator,
                .global_control_semantics = global_control_semantics,
                .incremental_state_management = incremental_state_management,
                .assembles_monolithic_equilibrium_residual =
                    assembles_monolithic_equilibrium_residual,
                .linearizes_consistent_global_tangent =
                    linearizes_consistent_global_tangent,
                .local_history_is_committed_on_converged_step =
                    local_history_is_committed_on_converged_step,
                .local_history_supports_checkpoint_restart =
                    local_history_supports_checkpoint_restart,
                .admits_effective_operator_predictor_injection =
                    admits_effective_operator_predictor_injection,
                .augments_with_inertial_terms = augments_with_inertial_terms,
                .augments_with_continuation_constraint =
                    augments_with_continuation_constraint
            };
        };

    switch (route_kind) {
        case AnalysisRouteKind::linear_static:
            return make_scope(
                GlobalResidualOperatorKind::static_equilibrium,
                GlobalTangentOperatorKind::linear_or_static_consistent_tangent,
                GlobalControlSemanticsKind::direct_load_or_displacement_control,
                IncrementalStateManagementKind::stateless_or_direct_response,
                true,
                true,
                false,
                false,
                false,
                false,
                false);

        case AnalysisRouteKind::nonlinear_incremental_newton:
            return make_scope(
                GlobalResidualOperatorKind::incremental_static_equilibrium,
                GlobalTangentOperatorKind::monolithic_incremental_consistent_tangent,
                GlobalControlSemanticsKind::incremental_load_or_displacement_control,
                route_scope.supports_checkpoint_restart
                    ? IncrementalStateManagementKind::checkpointable_converged_step_commit
                    : IncrementalStateManagementKind::converged_step_commit,
                true,
                true,
                discrete_variational_semantics.admits_history_variables,
                discrete_variational_semantics.admits_history_variables &&
                    route_scope.supports_checkpoint_restart,
                discrete_variational_semantics.admits_effective_operator_injection,
                false,
                false);

        case AnalysisRouteKind::implicit_second_order_dynamics:
            return make_scope(
                GlobalResidualOperatorKind::second_order_dynamic_equilibrium,
                GlobalTangentOperatorKind::effective_mass_damping_stiffness_tangent,
                GlobalControlSemanticsKind::implicit_time_control,
                route_scope.supports_checkpoint_restart
                    ? IncrementalStateManagementKind::checkpointable_converged_step_commit
                    : IncrementalStateManagementKind::converged_step_commit,
                true,
                true,
                discrete_variational_semantics.admits_history_variables,
                discrete_variational_semantics.admits_history_variables &&
                    route_scope.supports_checkpoint_restart,
                false,
                true,
                false);

        case AnalysisRouteKind::arc_length_continuation:
            return make_scope(
                GlobalResidualOperatorKind::arc_length_augmented_equilibrium,
                GlobalTangentOperatorKind::bordered_arc_length_tangent,
                GlobalControlSemanticsKind::arc_length_constraint_control,
                IncrementalStateManagementKind::continuation_step_commit,
                true,
                true,
                discrete_variational_semantics.admits_history_variables,
                false,
                false,
                false,
                true);
    }

    return {};
}

template <AuditedComputationalModelType ModelT, SolverWithAuditedModelSlice SolverT>
inline constexpr ComputationalVariationalSliceAuditScope
    canonical_computational_variational_slice_audit_scope_v =
        canonical_computational_variational_slice_audit_scope<ModelT, SolverT>();

} // namespace fall_n

#endif // FALL_N_COMPUTATIONAL_VARIATIONAL_SLICE_AUDIT_HH
