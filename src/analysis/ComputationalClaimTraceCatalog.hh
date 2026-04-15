#ifndef FALL_N_COMPUTATIONAL_CLAIM_TRACE_CATALOG_HH
#define FALL_N_COMPUTATIONAL_CLAIM_TRACE_CATALOG_HH

// =============================================================================
//  ComputationalClaimTraceCatalog.hh -- canonical representative matrix tying
//                                       scientific claims to typed slices and
//                                       current evidence channels
// =============================================================================
//
//  The previous catalog layers already answer:
//
//    - which family/formulation combinations are actually supported,
//    - which solver routes are audited for that scope,
//    - which concrete Model + Solver slices exist, and
//    - which discrete variational statement each representative slice carries.
//
//  Before a physical-validation campaign, however, the thesis also needs a
//  disciplined answer to a higher-level question:
//
//      "What representative scientific claim is the code actually making,
//       through which typed computational slice, and with what kind of
//       evidence?"
//
//  This header keeps that answer as compile-time/static metadata so the thesis,
//  README, and regression surface can point to one canonical claim matrix
//  without introducing any additional runtime abstraction into the hot path.
//
// =============================================================================

#include <array>
#include <cstddef>
#include <string_view>

#include "ComputationalVariationalSliceCatalog.hh"

namespace fall_n {

enum class ComputationalClaimEvidenceBreadthKind {
    catalog_and_semantic_audit,
    formulation_regression,
    solver_slice_regression,
    structural_response_regression
};

[[nodiscard]] constexpr std::string_view
to_string(ComputationalClaimEvidenceBreadthKind kind) noexcept
{
    switch (kind) {
        case ComputationalClaimEvidenceBreadthKind::catalog_and_semantic_audit:
            return "catalog_and_semantic_audit";
        case ComputationalClaimEvidenceBreadthKind::formulation_regression:
            return "formulation_regression";
        case ComputationalClaimEvidenceBreadthKind::solver_slice_regression:
            return "solver_slice_regression";
        case ComputationalClaimEvidenceBreadthKind::structural_response_regression:
            return "structural_response_regression";
    }
    return "unknown_computational_claim_evidence_breadth_kind";
}

struct RepresentativeComputationalClaimTraceRow {
    std::string_view claim_label{};
    std::string_view theory_anchor_label{};
    std::string_view slice_label{};
    std::string_view residual_commitment_label{};
    std::string_view tangent_commitment_label{};
    std::string_view history_commitment_label{};
    std::string_view primary_evidence_label{};
    std::string_view secondary_evidence_label{};
    std::string_view next_validation_step_label{};
    ComputationalVariationalSliceAuditScope audit_scope{};
    ComputationalClaimEvidenceBreadthKind evidence_breadth{
        ComputationalClaimEvidenceBreadthKind::catalog_and_semantic_audit};
    bool is_reference_claim{false};
    bool requires_physical_validation{true};

    [[nodiscard]] constexpr ComputationalModelSliceSupportLevel
    slice_support_level() const noexcept
    {
        return audit_scope.model_solver_slice.support_level();
    }

    [[nodiscard]] constexpr bool requires_scope_disclaimer() const noexcept {
        return audit_scope.requires_scope_disclaimer();
    }
};

template <AuditedComputationalModelType ModelT, SolverWithAuditedModelSlice SolverT>
[[nodiscard]] constexpr RepresentativeComputationalClaimTraceRow
make_representative_computational_claim_trace_row(
    std::string_view claim_label,
    std::string_view theory_anchor_label,
    std::string_view slice_label,
    std::string_view residual_commitment_label,
    std::string_view tangent_commitment_label,
    std::string_view history_commitment_label,
    std::string_view primary_evidence_label,
    std::string_view secondary_evidence_label,
    std::string_view next_validation_step_label,
    ComputationalClaimEvidenceBreadthKind evidence_breadth,
    bool is_reference_claim,
    bool requires_physical_validation) noexcept
{
    return {
        .claim_label = claim_label,
        .theory_anchor_label = theory_anchor_label,
        .slice_label = slice_label,
        .residual_commitment_label = residual_commitment_label,
        .tangent_commitment_label = tangent_commitment_label,
        .history_commitment_label = history_commitment_label,
        .primary_evidence_label = primary_evidence_label,
        .secondary_evidence_label = secondary_evidence_label,
        .next_validation_step_label = next_validation_step_label,
        .audit_scope = canonical_computational_variational_slice_audit_scope<ModelT, SolverT>(),
        .evidence_breadth = evidence_breadth,
        .is_reference_claim = is_reference_claim,
        .requires_physical_validation = requires_physical_validation
    };
}

[[nodiscard]] constexpr auto
canonical_representative_computational_claim_trace_table() noexcept
{
    using namespace representative_model_solver_slices;

    return std::to_array({
        make_representative_computational_claim_trace_row<
            continuum_small_strain_model,
            continuum_linear_analysis>(
            "linearized_continuum_reference_slice",
            "chapter4_linearized_virtual_work__chapter6_displacement_fem",
            "continuum_small_strain_linear",
            "static_global_equilibrium",
            "assembled_linear_stiffness",
            "stateless_or_direct_response",
            "fall_n_continuum_interface_test",
            "fall_n_steppable_solver_test",
            "baseline_reference_frozen",
            ComputationalClaimEvidenceBreadthKind::solver_slice_regression,
            true,
            false),
        make_representative_computational_claim_trace_row<
            continuum_total_lagrangian_model,
            continuum_total_lagrangian_newton>(
            "finite_kinematics_continuum_reference_slice",
            "chapter4_material_virtual_work__chapter6_total_lagrangian_fem",
            "continuum_total_lagrangian_nonlinear",
            "incremental_global_equilibrium",
            "monolithic_consistent_tangent",
            "checkpointable_converged_step_commit",
            "fall_n_kinematic_test",
            "fall_n_steppable_solver_test",
            "structural_material_validation_campaign",
            ComputationalClaimEvidenceBreadthKind::solver_slice_regression,
            true,
            true),
        make_representative_computational_claim_trace_row<
            continuum_updated_lagrangian_model,
            continuum_updated_lagrangian_newton>(
            "spatial_finite_kinematics_continuum_slice",
            "chapter4_spatial_virtual_work__chapter6_updated_lagrangian_fem",
            "continuum_updated_lagrangian_nonlinear",
            "incremental_global_equilibrium",
            "monolithic_consistent_tangent",
            "converged_step_commit",
            "fall_n_ul_test",
            "fall_n_continuum_interface_test",
            "close_updated_lagrangian_scope_before_campaign",
            ComputationalClaimEvidenceBreadthKind::formulation_regression,
            false,
            true),
        make_representative_computational_claim_trace_row<
            continuum_total_lagrangian_model,
            continuum_total_lagrangian_dynamics>(
            "implicit_dynamic_total_lagrangian_slice",
            "chapter6_implicit_time_discretization_on_total_lagrangian_scope",
            "continuum_total_lagrangian_dynamic",
            "second_order_dynamic_equilibrium",
            "effective_mass_damping_stiffness_tangent",
            "checkpointable_converged_step_commit",
            "fall_n_steppable_dynamic_test",
            "fall_n_steppable_solver_test",
            "close_dynamic_scope_before_campaign",
            ComputationalClaimEvidenceBreadthKind::solver_slice_regression,
            false,
            true),
        make_representative_computational_claim_trace_row<
            continuum_total_lagrangian_model,
            continuum_total_lagrangian_arc_length>(
            "arc_length_total_lagrangian_slice",
            "chapter6_continuation_and_limit_point_tracking",
            "continuum_total_lagrangian_arc_length",
            "arc_length_augmented_equilibrium",
            "bordered_continuation_tangent",
            "continuation_step_commit",
            "fall_n_computational_variational_slice_catalog_test",
            "fall_n_steppable_solver_test",
            "add_runtime_arc_length_regression",
            ComputationalClaimEvidenceBreadthKind::catalog_and_semantic_audit,
            false,
            true),
        make_representative_computational_claim_trace_row<
            beam_small_rotation_model,
            beam_small_rotation_linear>(
            "beam_linear_structural_reduction_reference_slice",
            "chapter6_structural_reduction_and_section_resultants",
            "beam_small_rotation_linear",
            "sectional_static_equilibrium",
            "sectional_constitutive_stiffness",
            "stateless_or_direct_response",
            "fall_n_beam_test",
            "fall_n_seismic_infra_test",
            "baseline_reference_frozen",
            ComputationalClaimEvidenceBreadthKind::structural_response_regression,
            true,
            false),
        make_representative_computational_claim_trace_row<
            beam_corotational_model,
            beam_corotational_newton>(
            "beam_corotational_structural_slice",
            "chapter6_corotational_structural_reduction",
            "beam_corotational_nonlinear",
            "sectional_incremental_equilibrium",
            "sectional_constitutive_plus_geometric_tangent",
            "section_or_fiber_history_commit",
            "fall_n_beam_test",
            "fall_n_seismic_infra_test",
            "close_beam_corotational_solver_scope",
            ComputationalClaimEvidenceBreadthKind::structural_response_regression,
            false,
            true),
        make_representative_computational_claim_trace_row<
            shell_small_rotation_model,
            shell_small_rotation_linear>(
            "shell_linear_structural_reduction_reference_slice",
            "chapter6_shell_reduction_and_midsurface_virtual_work",
            "shell_small_rotation_linear",
            "sectional_static_equilibrium",
            "sectional_constitutive_stiffness",
            "stateless_or_direct_response",
            "fall_n_mitc_shell_test",
            "fall_n_seismic_infra_test",
            "baseline_reference_frozen",
            ComputationalClaimEvidenceBreadthKind::structural_response_regression,
            true,
            false),
        make_representative_computational_claim_trace_row<
            shell_corotational_model,
            shell_corotational_newton>(
            "shell_corotational_structural_slice",
            "chapter6_corotational_shell_reduction",
            "shell_corotational_nonlinear",
            "sectional_incremental_equilibrium",
            "sectional_constitutive_plus_geometric_tangent",
            "section_or_fiber_history_commit",
            "fall_n_mitc_shell_test",
            "fall_n_seismic_infra_test",
            "close_shell_corotational_solver_scope",
            ComputationalClaimEvidenceBreadthKind::structural_response_regression,
            false,
            true)
    });
}

inline constexpr auto canonical_representative_computational_claim_trace_table_v =
    canonical_representative_computational_claim_trace_table();

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_representative_computational_claims_by_evidence_breadth(
    const std::array<RepresentativeComputationalClaimTraceRow, N>& rows,
    ComputationalClaimEvidenceBreadthKind breadth) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.evidence_breadth == breadth) {
            ++count;
        }
    }
    return count;
}

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_representative_reference_computational_claims(
    const std::array<RepresentativeComputationalClaimTraceRow, N>& rows) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.is_reference_claim) {
            ++count;
        }
    }
    return count;
}

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_representative_computational_claims_requiring_scope_disclaimer(
    const std::array<RepresentativeComputationalClaimTraceRow, N>& rows) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.requires_scope_disclaimer()) {
            ++count;
        }
    }
    return count;
}

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_representative_computational_claims_requiring_physical_validation(
    const std::array<RepresentativeComputationalClaimTraceRow, N>& rows) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.requires_physical_validation) {
            ++count;
        }
    }
    return count;
}

template <ComputationalClaimEvidenceBreadthKind Breadth>
inline constexpr std::size_t
    canonical_representative_computational_claim_evidence_count_v =
        count_representative_computational_claims_by_evidence_breadth(
            canonical_representative_computational_claim_trace_table_v,
            Breadth);

inline constexpr std::size_t
    canonical_representative_reference_computational_claim_count_v =
        count_representative_reference_computational_claims(
            canonical_representative_computational_claim_trace_table_v);

inline constexpr std::size_t
    canonical_representative_computational_claim_scope_disclaimer_count_v =
        count_representative_computational_claims_requiring_scope_disclaimer(
            canonical_representative_computational_claim_trace_table_v);

inline constexpr std::size_t
    canonical_representative_computational_claim_physical_validation_count_v =
        count_representative_computational_claims_requiring_physical_validation(
            canonical_representative_computational_claim_trace_table_v);

} // namespace fall_n

#endif // FALL_N_COMPUTATIONAL_CLAIM_TRACE_CATALOG_HH
