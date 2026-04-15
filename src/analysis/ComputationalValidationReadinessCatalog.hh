#ifndef FALL_N_COMPUTATIONAL_VALIDATION_READINESS_CATALOG_HH
#define FALL_N_COMPUTATIONAL_VALIDATION_READINESS_CATALOG_HH

// =============================================================================
//  ComputationalValidationReadinessCatalog.hh -- canonical representative
//                                                matrix stating which audited
//                                                computational claims are
//                                                already stable enough to act
//                                                as frozen baselines, which
//                                                ones may enter a targeted
//                                                physical-validation campaign,
//                                                and which ones still require
//                                                scope closure
// =============================================================================
//
//  The previous catalog layers answer, with increasing precision:
//
//    1. What family/formulation pair is supported?
//    2. Which analysis route is audited for that scope?
//    3. Which typed Model + Solver slice exists?
//    4. Which residual/tangent/history commitments that slice carries?
//    5. Which representative scientific claim that slice is allowed to denote?
//
//  Before the validation chapters are written, one more question has to be
//  frozen in code:
//
//      "Is the current claim already fit to enter a physical-validation
//       campaign, or does it still need numerical scope closure?"
//
//  This header answers that question as static metadata.  The goal is not to
//  add a new runtime layer, but to prevent the thesis and the README from
//  silently turning any available regression into a scientific-validation
//  claim of equal weight.
//
// =============================================================================

#include <array>
#include <cstddef>
#include <string_view>

#include "ComputationalClaimTraceCatalog.hh"

namespace fall_n {

enum class ComputationalValidationReadinessKind {
    frozen_reference_baseline,
    ready_for_targeted_physical_validation,
    scope_closure_pending,
    runtime_regression_pending
};

enum class PendingPhysicalValidationExperimentKind {
    none_reference_baseline,
    structural_material_validation_campaign,
    updated_lagrangian_scope_closure_suite,
    dynamic_scope_closure_suite,
    continuation_runtime_regression_suite,
    beam_corotational_scope_closure_suite,
    shell_corotational_scope_closure_suite
};

[[nodiscard]] constexpr std::string_view
to_string(ComputationalValidationReadinessKind kind) noexcept
{
    switch (kind) {
        case ComputationalValidationReadinessKind::frozen_reference_baseline:
            return "frozen_reference_baseline";
        case ComputationalValidationReadinessKind::ready_for_targeted_physical_validation:
            return "ready_for_targeted_physical_validation";
        case ComputationalValidationReadinessKind::scope_closure_pending:
            return "scope_closure_pending";
        case ComputationalValidationReadinessKind::runtime_regression_pending:
            return "runtime_regression_pending";
    }
    return "unknown_computational_validation_readiness_kind";
}

[[nodiscard]] constexpr std::string_view
to_string(PendingPhysicalValidationExperimentKind kind) noexcept
{
    switch (kind) {
        case PendingPhysicalValidationExperimentKind::none_reference_baseline:
            return "none_reference_baseline";
        case PendingPhysicalValidationExperimentKind::structural_material_validation_campaign:
            return "structural_material_validation_campaign";
        case PendingPhysicalValidationExperimentKind::updated_lagrangian_scope_closure_suite:
            return "updated_lagrangian_scope_closure_suite";
        case PendingPhysicalValidationExperimentKind::dynamic_scope_closure_suite:
            return "dynamic_scope_closure_suite";
        case PendingPhysicalValidationExperimentKind::continuation_runtime_regression_suite:
            return "continuation_runtime_regression_suite";
        case PendingPhysicalValidationExperimentKind::beam_corotational_scope_closure_suite:
            return "beam_corotational_scope_closure_suite";
        case PendingPhysicalValidationExperimentKind::shell_corotational_scope_closure_suite:
            return "shell_corotational_scope_closure_suite";
    }
    return "unknown_pending_physical_validation_experiment_kind";
}

struct RepresentativeComputationalValidationReadinessRow {
    RepresentativeComputationalClaimTraceRow claim_trace{};
    ComputationalValidationReadinessKind readiness_kind{
        ComputationalValidationReadinessKind::scope_closure_pending};
    PendingPhysicalValidationExperimentKind pending_experiment_kind{
        PendingPhysicalValidationExperimentKind::none_reference_baseline};
    std::string_view evidence_floor_label{};
    std::string_view pending_experiment_label{};
    bool can_anchor_validation_baseline{false};
    bool can_enter_targeted_physical_validation{false};

    [[nodiscard]] constexpr bool is_frozen_baseline() const noexcept
    {
        return readiness_kind ==
               ComputationalValidationReadinessKind::frozen_reference_baseline;
    }

    [[nodiscard]] constexpr bool has_open_prevalidation_gate() const noexcept
    {
        return readiness_kind !=
               ComputationalValidationReadinessKind::frozen_reference_baseline;
    }

    [[nodiscard]] constexpr bool needs_more_numerical_scope_before_validation()
        const noexcept
    {
        return readiness_kind ==
                   ComputationalValidationReadinessKind::scope_closure_pending ||
               readiness_kind ==
                   ComputationalValidationReadinessKind::runtime_regression_pending;
    }
};

[[nodiscard]] constexpr RepresentativeComputationalValidationReadinessRow
make_representative_computational_validation_readiness_row(
    RepresentativeComputationalClaimTraceRow claim_trace,
    ComputationalValidationReadinessKind readiness_kind,
    PendingPhysicalValidationExperimentKind pending_experiment_kind,
    std::string_view evidence_floor_label,
    std::string_view pending_experiment_label,
    bool can_anchor_validation_baseline,
    bool can_enter_targeted_physical_validation) noexcept
{
    return {
        .claim_trace = claim_trace,
        .readiness_kind = readiness_kind,
        .pending_experiment_kind = pending_experiment_kind,
        .evidence_floor_label = evidence_floor_label,
        .pending_experiment_label = pending_experiment_label,
        .can_anchor_validation_baseline = can_anchor_validation_baseline,
        .can_enter_targeted_physical_validation =
            can_enter_targeted_physical_validation
    };
}

[[nodiscard]] constexpr auto
canonical_representative_computational_validation_readiness_table() noexcept
{
    constexpr auto claims = canonical_representative_computational_claim_trace_table_v;

    return std::to_array({
        make_representative_computational_validation_readiness_row(
            claims[0],
            ComputationalValidationReadinessKind::frozen_reference_baseline,
            PendingPhysicalValidationExperimentKind::none_reference_baseline,
            "solver_slice_reference_baseline",
            "none_reference_baseline",
            true,
            false),
        make_representative_computational_validation_readiness_row(
            claims[1],
            ComputationalValidationReadinessKind::ready_for_targeted_physical_validation,
            PendingPhysicalValidationExperimentKind::structural_material_validation_campaign,
            "finite_kinematics_solver_slice_ready",
            "structural_material_validation_campaign",
            false,
            true),
        make_representative_computational_validation_readiness_row(
            claims[2],
            ComputationalValidationReadinessKind::scope_closure_pending,
            PendingPhysicalValidationExperimentKind::updated_lagrangian_scope_closure_suite,
            "formulation_level_spatial_path",
            "updated_lagrangian_scope_closure_suite",
            false,
            false),
        make_representative_computational_validation_readiness_row(
            claims[3],
            ComputationalValidationReadinessKind::scope_closure_pending,
            PendingPhysicalValidationExperimentKind::dynamic_scope_closure_suite,
            "dynamic_solver_slice_ready_but_scope_narrow",
            "dynamic_scope_closure_suite",
            false,
            false),
        make_representative_computational_validation_readiness_row(
            claims[4],
            ComputationalValidationReadinessKind::runtime_regression_pending,
            PendingPhysicalValidationExperimentKind::continuation_runtime_regression_suite,
            "catalog_and_variational_audit_only",
            "continuation_runtime_regression_suite",
            false,
            false),
        make_representative_computational_validation_readiness_row(
            claims[5],
            ComputationalValidationReadinessKind::frozen_reference_baseline,
            PendingPhysicalValidationExperimentKind::none_reference_baseline,
            "structural_linear_reference_baseline",
            "none_reference_baseline",
            true,
            false),
        make_representative_computational_validation_readiness_row(
            claims[6],
            ComputationalValidationReadinessKind::scope_closure_pending,
            PendingPhysicalValidationExperimentKind::beam_corotational_scope_closure_suite,
            "structural_reduced_regression_evidence",
            "beam_corotational_scope_closure_suite",
            false,
            false),
        make_representative_computational_validation_readiness_row(
            claims[7],
            ComputationalValidationReadinessKind::frozen_reference_baseline,
            PendingPhysicalValidationExperimentKind::none_reference_baseline,
            "structural_linear_reference_baseline",
            "none_reference_baseline",
            true,
            false),
        make_representative_computational_validation_readiness_row(
            claims[8],
            ComputationalValidationReadinessKind::scope_closure_pending,
            PendingPhysicalValidationExperimentKind::shell_corotational_scope_closure_suite,
            "structural_reduced_regression_evidence",
            "shell_corotational_scope_closure_suite",
            false,
            false)
    });
}

inline constexpr auto canonical_representative_computational_validation_readiness_table_v =
    canonical_representative_computational_validation_readiness_table();

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_representative_validation_readiness_rows(
    const std::array<RepresentativeComputationalValidationReadinessRow, N>& rows,
    ComputationalValidationReadinessKind readiness_kind) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.readiness_kind == readiness_kind) {
            ++count;
        }
    }
    return count;
}

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_representative_validation_baseline_rows(
    const std::array<RepresentativeComputationalValidationReadinessRow, N>& rows) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.can_anchor_validation_baseline) {
            ++count;
        }
    }
    return count;
}

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_representative_targeted_physical_validation_rows(
    const std::array<RepresentativeComputationalValidationReadinessRow, N>& rows) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.can_enter_targeted_physical_validation) {
            ++count;
        }
    }
    return count;
}

template <ComputationalValidationReadinessKind ReadinessKind>
inline constexpr std::size_t
    canonical_representative_validation_readiness_count_v =
        count_representative_validation_readiness_rows(
            canonical_representative_computational_validation_readiness_table_v,
            ReadinessKind);

inline constexpr std::size_t
    canonical_representative_validation_baseline_count_v =
        count_representative_validation_baseline_rows(
            canonical_representative_computational_validation_readiness_table_v);

inline constexpr std::size_t
    canonical_representative_targeted_physical_validation_count_v =
        count_representative_targeted_physical_validation_rows(
            canonical_representative_computational_validation_readiness_table_v);

} // namespace fall_n

#endif // FALL_N_COMPUTATIONAL_VALIDATION_READINESS_CATALOG_HH
