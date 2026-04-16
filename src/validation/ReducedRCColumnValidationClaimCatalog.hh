#ifndef FALL_N_REDUCED_RC_COLUMN_VALIDATION_CLAIM_CATALOG_HH
#define FALL_N_REDUCED_RC_COLUMN_VALIDATION_CLAIM_CATALOG_HH

// =============================================================================
//  ReducedRCColumnValidationClaimCatalog.hh
// =============================================================================
//
//  Canonical claim -> artifact -> evidence matrix for the first reduced
//  reinforced-concrete column campaign.
//
//  The structural matrix already answers which (N, quadrature, formulation)
//  slices exist and which ones are currently runtime-ready. That is necessary,
//  but still not enough to reopen a scientific-validation campaign. We also
//  need a disciplined answer to a sharper question:
//
//      "What exactly are we allowed to claim today about the reduced RC
//       column path, through which computational artifact, and what still
//       remains open before a physical benchmark can be called closed?"
//
//  This catalog keeps that answer as compile-time metadata so the validation
//  reboot chapter, the thesis, the README, and the regression surface point to
//  one canonical matrix instead of drifting independently.
//
// =============================================================================

#include <array>
#include <cstddef>
#include <string_view>

#include "ReducedRCColumnStructuralMatrixCatalog.hh"

namespace fall_n {

enum class ReducedRCColumnClaimReadinessKind {
    enabling_contract_frozen,
    runtime_baseline_ready,
    benchmark_pending,
    instrumentation_pending
};

enum class ReducedRCColumnPendingExperimentKind {
    none_frozen_or_ready,
    reduced_column_node_refinement_suite,
    reduced_column_axial_load_suite,
    reduced_column_hysteresis_suite,
    reduced_column_moment_curvature_suite,
    reduced_column_quadrature_sensitivity_suite
};

[[nodiscard]] constexpr std::string_view
to_string(ReducedRCColumnClaimReadinessKind kind) noexcept
{
    switch (kind) {
        case ReducedRCColumnClaimReadinessKind::enabling_contract_frozen:
            return "enabling_contract_frozen";
        case ReducedRCColumnClaimReadinessKind::runtime_baseline_ready:
            return "runtime_baseline_ready";
        case ReducedRCColumnClaimReadinessKind::benchmark_pending:
            return "benchmark_pending";
        case ReducedRCColumnClaimReadinessKind::instrumentation_pending:
            return "instrumentation_pending";
    }
    return "unknown_reduced_rc_column_claim_readiness_kind";
}

[[nodiscard]] constexpr std::string_view
to_string(ReducedRCColumnPendingExperimentKind kind) noexcept
{
    switch (kind) {
        case ReducedRCColumnPendingExperimentKind::none_frozen_or_ready:
            return "none_frozen_or_ready";
        case ReducedRCColumnPendingExperimentKind::reduced_column_node_refinement_suite:
            return "reduced_column_node_refinement_suite";
        case ReducedRCColumnPendingExperimentKind::reduced_column_axial_load_suite:
            return "reduced_column_axial_load_suite";
        case ReducedRCColumnPendingExperimentKind::reduced_column_hysteresis_suite:
            return "reduced_column_hysteresis_suite";
        case ReducedRCColumnPendingExperimentKind::reduced_column_moment_curvature_suite:
            return "reduced_column_moment_curvature_suite";
        case ReducedRCColumnPendingExperimentKind::reduced_column_quadrature_sensitivity_suite:
            return "reduced_column_quadrature_sensitivity_suite";
    }
    return "unknown_reduced_rc_column_pending_experiment_kind";
}

struct ReducedRCColumnValidationClaimRow {
    std::string_view claim_label{};
    std::string_view theory_anchor_label{};
    std::string_view computational_artifact_label{};
    std::string_view current_evidence_label{};
    std::string_view current_observable_label{};
    std::string_view next_validation_step_label{};
    ReducedRCColumnClaimReadinessKind readiness_kind{
        ReducedRCColumnClaimReadinessKind::benchmark_pending};
    ReducedRCColumnPendingExperimentKind pending_experiment_kind{
        ReducedRCColumnPendingExperimentKind::none_frozen_or_ready};
    ReducedRCColumnStructuralSupportKind structural_support_kind{
        ReducedRCColumnStructuralSupportKind::unavailable_in_current_family};
    bool can_anchor_phase3_runtime_baseline{false};
    bool can_anchor_phase3_physical_benchmark{false};
    bool requires_new_runtime_instrumentation{false};
    bool depends_on_current_small_strain_runtime_family{false};

    [[nodiscard]] constexpr bool has_open_prebenchmark_gate() const noexcept
    {
        return readiness_kind !=
               ReducedRCColumnClaimReadinessKind::enabling_contract_frozen;
    }
};

[[nodiscard]] constexpr ReducedRCColumnValidationClaimRow
make_reduced_rc_column_validation_claim_row(
    std::string_view claim_label,
    std::string_view theory_anchor_label,
    std::string_view computational_artifact_label,
    std::string_view current_evidence_label,
    std::string_view current_observable_label,
    std::string_view next_validation_step_label,
    ReducedRCColumnClaimReadinessKind readiness_kind,
    ReducedRCColumnPendingExperimentKind pending_experiment_kind,
    ReducedRCColumnStructuralSupportKind structural_support_kind,
    bool can_anchor_phase3_runtime_baseline,
    bool can_anchor_phase3_physical_benchmark,
    bool requires_new_runtime_instrumentation,
    bool depends_on_current_small_strain_runtime_family) noexcept
{
    return {
        .claim_label = claim_label,
        .theory_anchor_label = theory_anchor_label,
        .computational_artifact_label = computational_artifact_label,
        .current_evidence_label = current_evidence_label,
        .current_observable_label = current_observable_label,
        .next_validation_step_label = next_validation_step_label,
        .readiness_kind = readiness_kind,
        .pending_experiment_kind = pending_experiment_kind,
        .structural_support_kind = structural_support_kind,
        .can_anchor_phase3_runtime_baseline =
            can_anchor_phase3_runtime_baseline,
        .can_anchor_phase3_physical_benchmark =
            can_anchor_phase3_physical_benchmark,
        .requires_new_runtime_instrumentation =
            requires_new_runtime_instrumentation,
        .depends_on_current_small_strain_runtime_family =
            depends_on_current_small_strain_runtime_family
    };
}

[[nodiscard]] constexpr auto
canonical_reduced_rc_column_validation_claim_table() noexcept
{
    using Pending = ReducedRCColumnPendingExperimentKind;
    using Readiness = ReducedRCColumnClaimReadinessKind;
    using Support = ReducedRCColumnStructuralSupportKind;

    return std::to_array({
        make_reduced_rc_column_validation_claim_row(
            "beam_axis_quadrature_family_is_explicit_model_axis",
            "distributed_plasticity__section_station_placement__chapter6_structural_quadrature",
            "BeamAxisQuadrature.hh + test_beam_axis_quadrature.cpp",
            "compile_time_quadrature_exactness_and_station_symmetry_regression",
            "quadrature_family_and_section_station_coordinates",
            "none_frozen_or_ready",
            Readiness::enabling_contract_frozen,
            Pending::none_frozen_or_ready,
            Support::ready_for_runtime_baseline,
            true,
            false,
            false,
            true),
        make_reduced_rc_column_validation_claim_row(
            "reduced_column_small_strain_runtime_surface_exists_for_n2_to_n10",
            "chapter6_structural_reduction__single_column_reference_slice",
            "ReducedRCColumnStructuralMatrixCatalog.hh + ReducedRCColumnStructuralBaseline.hh/.cpp + test_reduced_rc_column_structural_matrix.cpp",
            "matrix_counts_plus_runtime_smoke_across_quadrature_families",
            "base_shear_vs_drift_step_records",
            "reduced_column_node_refinement_suite",
            Readiness::runtime_baseline_ready,
            Pending::reduced_column_node_refinement_suite,
            Support::ready_for_runtime_baseline,
            true,
            false,
            false,
            true),
        make_reduced_rc_column_validation_claim_row(
            "optional_axial_compression_load_path_is_supported",
            "beam_column_interaction_under_cyclic_lateral_loading",
            "ReducedRCColumnStructuralBaseline.hh/.cpp + axial_smoke branch in test_reduced_rc_column_structural_matrix.cpp",
            "finite_runtime_smoke_under_constant_axial_preload",
            "base_shear_vs_drift_under_axial_compression",
            "reduced_column_axial_load_suite",
            Readiness::runtime_baseline_ready,
            Pending::reduced_column_axial_load_suite,
            Support::ready_for_runtime_baseline,
            true,
            false,
            false,
            true),
        make_reduced_rc_column_validation_claim_row(
            "base_shear_vs_displacement_hysteresis_output_is_reproducible",
            "global_structural_hysteresis_for_single_column_validation",
            "ReducedRCColumnStructuralBaseline.hh/.cpp + TableCyclicValidationSupport::write_csv(...)",
            "runtime_output_contract_for_hysteresis_csv",
            "hysteresis_csv_base_shear_vs_drift",
            "reduced_column_hysteresis_suite",
            Readiness::runtime_baseline_ready,
            Pending::reduced_column_hysteresis_suite,
            Support::ready_for_runtime_baseline,
            true,
            false,
            false,
            true),
        make_reduced_rc_column_validation_claim_row(
            "base_side_moment_curvature_observable_is_normatively_extracted",
            "section_resultants__curvature_history__base_side_active_station",
            "ReducedRCColumnStructuralBaseline.hh/.cpp + ReducedRCColumnSectionBaseline.hh/.cpp + ReducedRCColumnMomentCurvatureClosure.hh/.cpp",
            "runtime_output_contract_for_base_side_moment_curvature_csv_plus_preload_consistent_matrix_wide_section_closure_bundle",
            "moment_curvature_base_csv",
            "reduced_column_moment_curvature_suite",
            Readiness::runtime_baseline_ready,
            Pending::reduced_column_moment_curvature_suite,
            Support::ready_for_runtime_baseline,
            true,
            false,
            false,
            true),
        make_reduced_rc_column_validation_claim_row(
            "node_refinement_convergence_claim_is_not_yet_closed",
            "displacement_based_structural_refinement_for_single_column_response",
            "ReducedRCColumnStructuralBaseline.hh/.cpp + ReducedRCColumnStructuralMatrixCatalog.hh + ReducedRCColumnNodeRefinementStudy.hh/.cpp + ReducedRCColumnCyclicNodeRefinementStudy.hh/.cpp",
            "full_monotonic_internal_node_refinement_bundle_against_highest_n_reference_is_available_over_all_runtime_ready_n2_to_n10_slices_and_a_representative_cyclic_refinement_pilot_now_exists_over_n2_n4_n10_x_gauss_legendre_gauss_lobatto_but_full_cyclic_refinement_matrix_closure_remains_open",
            "monotonic_and_cyclic_base_side_moment_curvature_drift_vs_N",
            "reduced_column_node_refinement_suite",
            Readiness::benchmark_pending,
            Pending::reduced_column_node_refinement_suite,
            Support::ready_for_runtime_baseline,
            false,
            false,
            false,
            true),
        make_reduced_rc_column_validation_claim_row(
            "quadrature_family_sensitivity_claim_is_not_yet_closed",
            "section_history_placement_sensitivity_under_distributed_plasticity",
            "BeamAxisQuadrature.hh + ReducedRCColumnStructuralBaseline.hh/.cpp + ReducedRCColumnQuadratureSensitivityStudy.hh/.cpp",
            "full_monotonic_internal_quadrature_sensitivity_bundle_against_gauss_legendre_reference_is_available_over_all_runtime_ready_n2_to_n10_slices_but_cyclic_family_spread_closure_remains_open",
            "monotonic_base_side_moment_curvature_spread_and_controlling_station_shift_vs_quadrature_family",
            "reduced_column_quadrature_sensitivity_suite",
            Readiness::benchmark_pending,
            Pending::reduced_column_quadrature_sensitivity_suite,
            Support::ready_for_runtime_baseline,
            false,
            false,
            false,
            true)
    });
}

inline constexpr auto canonical_reduced_rc_column_validation_claim_table_v =
    canonical_reduced_rc_column_validation_claim_table();

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_reduced_rc_column_validation_claims_by_readiness(
    const std::array<ReducedRCColumnValidationClaimRow, N>& rows,
    ReducedRCColumnClaimReadinessKind readiness_kind) noexcept
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
count_reduced_rc_column_validation_claims_requiring_instrumentation(
    const std::array<ReducedRCColumnValidationClaimRow, N>& rows) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.requires_new_runtime_instrumentation) {
            ++count;
        }
    }
    return count;
}

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_reduced_rc_column_validation_claims_anchor_runtime_baseline(
    const std::array<ReducedRCColumnValidationClaimRow, N>& rows) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.can_anchor_phase3_runtime_baseline) {
            ++count;
        }
    }
    return count;
}

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_reduced_rc_column_validation_claims_anchor_physical_benchmark(
    const std::array<ReducedRCColumnValidationClaimRow, N>& rows) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.can_anchor_phase3_physical_benchmark) {
            ++count;
        }
    }
    return count;
}

template <ReducedRCColumnClaimReadinessKind ReadinessKind>
inline constexpr std::size_t canonical_reduced_rc_column_claim_readiness_count_v =
    count_reduced_rc_column_validation_claims_by_readiness(
        canonical_reduced_rc_column_validation_claim_table_v, ReadinessKind);

inline constexpr std::size_t
    canonical_reduced_rc_column_claim_instrumentation_pending_count_v =
        count_reduced_rc_column_validation_claims_requiring_instrumentation(
            canonical_reduced_rc_column_validation_claim_table_v);

inline constexpr std::size_t
    canonical_reduced_rc_column_claim_runtime_baseline_anchor_count_v =
        count_reduced_rc_column_validation_claims_anchor_runtime_baseline(
            canonical_reduced_rc_column_validation_claim_table_v);

inline constexpr std::size_t
    canonical_reduced_rc_column_claim_physical_benchmark_anchor_count_v =
        count_reduced_rc_column_validation_claims_anchor_physical_benchmark(
            canonical_reduced_rc_column_validation_claim_table_v);

} // namespace fall_n

#endif // FALL_N_REDUCED_RC_COLUMN_VALIDATION_CLAIM_CATALOG_HH
