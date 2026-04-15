#ifndef FALL_N_REDUCED_RC_COLUMN_BENCHMARK_TRACE_CATALOG_HH
#define FALL_N_REDUCED_RC_COLUMN_BENCHMARK_TRACE_CATALOG_HH

// =============================================================================
//  ReducedRCColumnBenchmarkTraceCatalog.hh
// =============================================================================
//
//  Canonical benchmark-trace matrix for the reduced reinforced-concrete column
//  reboot.
//
//  The structural matrix and the claim matrix already answer two distinct
//  questions:
//
//    1. Which structural slices exist and are runtime-ready?
//    2. Which scientific claims are currently allowed, and which ones remain
//       prebenchmark?
//
//  Before the physical-validation chapter is reopened, one more bridge is
//  needed:
//
//      "For each open claim, what concrete benchmark must be executed,
//       against which reference class, through which observable, and which
//       gate would that benchmark close if it passed?"
//
//  This header freezes that bridge as compile-time metadata so the validation
//  reboot chapters, README, and regression surface all point to one canonical
//  benchmark plan without pushing roadmap metadata into the numerical hot path.
//
// =============================================================================

#include <array>
#include <cstddef>
#include <string_view>

#include "ReducedRCColumnValidationClaimCatalog.hh"

namespace fall_n {

enum class ReducedRCColumnBenchmarkMetricKind {
    runtime_surface_completeness,
    axial_load_interaction_trend,
    hysteresis_loop_shape_and_dissipation,
    base_side_moment_curvature_envelope,
    node_refinement_convergence,
    quadrature_family_sensitivity
};

enum class ReducedRCColumnBenchmarkReferenceKind {
    internal_runtime_matrix_reference,
    analytical_or_section_baseline_reference,
    experimental_or_literature_reference
};

[[nodiscard]] constexpr std::string_view
to_string(ReducedRCColumnBenchmarkMetricKind kind) noexcept
{
    switch (kind) {
        case ReducedRCColumnBenchmarkMetricKind::runtime_surface_completeness:
            return "runtime_surface_completeness";
        case ReducedRCColumnBenchmarkMetricKind::axial_load_interaction_trend:
            return "axial_load_interaction_trend";
        case ReducedRCColumnBenchmarkMetricKind::hysteresis_loop_shape_and_dissipation:
            return "hysteresis_loop_shape_and_dissipation";
        case ReducedRCColumnBenchmarkMetricKind::base_side_moment_curvature_envelope:
            return "base_side_moment_curvature_envelope";
        case ReducedRCColumnBenchmarkMetricKind::node_refinement_convergence:
            return "node_refinement_convergence";
        case ReducedRCColumnBenchmarkMetricKind::quadrature_family_sensitivity:
            return "quadrature_family_sensitivity";
    }
    return "unknown_reduced_rc_column_benchmark_metric_kind";
}

[[nodiscard]] constexpr std::string_view
to_string(ReducedRCColumnBenchmarkReferenceKind kind) noexcept
{
    switch (kind) {
        case ReducedRCColumnBenchmarkReferenceKind::internal_runtime_matrix_reference:
            return "internal_runtime_matrix_reference";
        case ReducedRCColumnBenchmarkReferenceKind::analytical_or_section_baseline_reference:
            return "analytical_or_section_baseline_reference";
        case ReducedRCColumnBenchmarkReferenceKind::experimental_or_literature_reference:
            return "experimental_or_literature_reference";
    }
    return "unknown_reduced_rc_column_benchmark_reference_kind";
}

struct ReducedRCColumnBenchmarkTraceRow {
    ReducedRCColumnValidationClaimRow claim_row{};
    std::string_view benchmark_label{};
    std::string_view benchmark_driver_label{};
    std::string_view primary_observable_label{};
    std::string_view secondary_observable_label{};
    std::string_view reference_source_label{};
    std::string_view acceptance_gate_label{};
    ReducedRCColumnBenchmarkMetricKind metric_kind{
        ReducedRCColumnBenchmarkMetricKind::runtime_surface_completeness};
    ReducedRCColumnBenchmarkReferenceKind reference_kind{
        ReducedRCColumnBenchmarkReferenceKind::internal_runtime_matrix_reference};
    bool requires_external_reference_dataset{false};
    bool requires_independent_section_baseline{false};
    bool closes_own_prebenchmark_gate_if_passed{true};
    bool required_for_phase3_structural_physical_benchmark{false};
};

template <std::size_t N>
[[nodiscard]] constexpr ReducedRCColumnValidationClaimRow
find_reduced_rc_column_validation_claim_row(
    const std::array<ReducedRCColumnValidationClaimRow, N>& rows,
    std::string_view claim_label) noexcept
{
    for (const auto& row : rows) {
        if (row.claim_label == claim_label) {
            return row;
        }
    }

    return {};
}

[[nodiscard]] constexpr ReducedRCColumnBenchmarkTraceRow
make_reduced_rc_column_benchmark_trace_row(
    ReducedRCColumnValidationClaimRow claim_row,
    std::string_view benchmark_label,
    std::string_view benchmark_driver_label,
    std::string_view primary_observable_label,
    std::string_view secondary_observable_label,
    std::string_view reference_source_label,
    std::string_view acceptance_gate_label,
    ReducedRCColumnBenchmarkMetricKind metric_kind,
    ReducedRCColumnBenchmarkReferenceKind reference_kind,
    bool requires_external_reference_dataset,
    bool requires_independent_section_baseline,
    bool closes_own_prebenchmark_gate_if_passed,
    bool required_for_phase3_structural_physical_benchmark) noexcept
{
    return {
        .claim_row = claim_row,
        .benchmark_label = benchmark_label,
        .benchmark_driver_label = benchmark_driver_label,
        .primary_observable_label = primary_observable_label,
        .secondary_observable_label = secondary_observable_label,
        .reference_source_label = reference_source_label,
        .acceptance_gate_label = acceptance_gate_label,
        .metric_kind = metric_kind,
        .reference_kind = reference_kind,
        .requires_external_reference_dataset =
            requires_external_reference_dataset,
        .requires_independent_section_baseline =
            requires_independent_section_baseline,
        .closes_own_prebenchmark_gate_if_passed =
            closes_own_prebenchmark_gate_if_passed,
        .required_for_phase3_structural_physical_benchmark =
            required_for_phase3_structural_physical_benchmark
    };
}

[[nodiscard]] constexpr auto
canonical_reduced_rc_column_benchmark_trace_table() noexcept
{
    constexpr auto claims = canonical_reduced_rc_column_validation_claim_table_v;

    return std::to_array({
        make_reduced_rc_column_benchmark_trace_row(
            find_reduced_rc_column_validation_claim_row(
                claims,
                "reduced_column_small_strain_runtime_surface_exists_for_n2_to_n10"),
            "reduced_column_runtime_surface_matrix",
            "ReducedRCColumnStructuralBaseline + reduced_rc_column_structural_matrix harness",
            "base_shear_vs_drift_envelope_over_N_and_quadrature",
            "finite_record_count_and_output_contract_consistency",
            "internal structural matrix consistency over N=2..10 and audited quadrature families",
            "every audited small-strain baseline slice completes with finite response and auditable CSV outputs",
            ReducedRCColumnBenchmarkMetricKind::runtime_surface_completeness,
            ReducedRCColumnBenchmarkReferenceKind::internal_runtime_matrix_reference,
            false,
            false,
            true,
            false),
        make_reduced_rc_column_benchmark_trace_row(
            find_reduced_rc_column_validation_claim_row(
                claims, "optional_axial_compression_load_path_is_supported"),
            "reduced_column_axial_load_interaction_suite",
            "ReducedRCColumnStructuralBaseline under constant axial preload sweep",
            "base_shear_vs_drift_under_axial_preload",
            "peak_strength_and_secant_stiffness_vs_axial_load_ratio",
            "audited single-column cyclic datasets or literature envelopes with reported axial-load levels",
            "axial-load trend matches the audited reference class within the declared experimental/computational band",
            ReducedRCColumnBenchmarkMetricKind::axial_load_interaction_trend,
            ReducedRCColumnBenchmarkReferenceKind::experimental_or_literature_reference,
            true,
            false,
            true,
            true),
        make_reduced_rc_column_benchmark_trace_row(
            find_reduced_rc_column_validation_claim_row(
                claims,
                "base_shear_vs_displacement_hysteresis_output_is_reproducible"),
            "reduced_column_hysteresis_protocol_suite",
            "ReducedRCColumnStructuralBaseline with progressive cyclic displacement protocol",
            "base_shear_vs_drift_hysteresis",
            "cycle_energy_dissipation_and_envelope_degradation",
            "audited experimental or literature hysteresis loops for rectangular RC columns under comparable protocol and axial load",
            "loop shape, envelope degradation, and dissipated energy stay within the declared benchmark tolerance band",
            ReducedRCColumnBenchmarkMetricKind::hysteresis_loop_shape_and_dissipation,
            ReducedRCColumnBenchmarkReferenceKind::experimental_or_literature_reference,
            true,
            false,
            true,
            true),
        make_reduced_rc_column_benchmark_trace_row(
            find_reduced_rc_column_validation_claim_row(
                claims,
                "base_side_moment_curvature_observable_is_normatively_extracted"),
            "reduced_column_base_side_moment_curvature_suite",
            "ReducedRCColumnStructuralBaseline + moment_curvature_base.csv",
            "base_side_moment_curvature_envelope",
            "section_tangent_and_secant_stiffness_consistency",
            "independent section-level moment-curvature baseline built from the same audited fiber section and loading protocol",
            "base-side section observable remains consistent with the independent section baseline before column-level physical benchmarking",
            ReducedRCColumnBenchmarkMetricKind::base_side_moment_curvature_envelope,
            ReducedRCColumnBenchmarkReferenceKind::analytical_or_section_baseline_reference,
            false,
            true,
            true,
            true),
        make_reduced_rc_column_benchmark_trace_row(
            find_reduced_rc_column_validation_claim_row(
                claims,
                "node_refinement_convergence_claim_is_not_yet_closed"),
            "reduced_column_node_refinement_suite",
            "ReducedRCColumnStructuralBaseline over N=2..10",
            "envelope_strength_stiffness_and_dissipation_vs_N",
            "loop_distance_between_successive_N_families",
            "internal runtime matrix interpreted as a convergence study rather than a single smoke slice",
            "observables show stable refinement trends and no unresolved drift with increasing beam nodes/sections",
            ReducedRCColumnBenchmarkMetricKind::node_refinement_convergence,
            ReducedRCColumnBenchmarkReferenceKind::internal_runtime_matrix_reference,
            false,
            false,
            true,
            true),
        make_reduced_rc_column_benchmark_trace_row(
            find_reduced_rc_column_validation_claim_row(
                claims,
                "quadrature_family_sensitivity_claim_is_not_yet_closed"),
            "reduced_column_quadrature_sensitivity_suite",
            "ReducedRCColumnStructuralBaseline over Gauss, Lobatto, and Radau families",
            "response_spread_over_quadrature_family",
            "base_side_station_location_and_moment_curvature_spread",
            "internal runtime matrix interpreted as a controlled section-placement sensitivity study",
            "family-to-family spread is bounded and interpreted explicitly rather than hidden as numerical noise",
            ReducedRCColumnBenchmarkMetricKind::quadrature_family_sensitivity,
            ReducedRCColumnBenchmarkReferenceKind::internal_runtime_matrix_reference,
            false,
            false,
            true,
            true)
    });
}

inline constexpr auto canonical_reduced_rc_column_benchmark_trace_table_v =
    canonical_reduced_rc_column_benchmark_trace_table();

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_reduced_rc_column_benchmark_rows_by_metric(
    const std::array<ReducedRCColumnBenchmarkTraceRow, N>& rows,
    ReducedRCColumnBenchmarkMetricKind metric_kind) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.metric_kind == metric_kind) {
            ++count;
        }
    }
    return count;
}

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_reduced_rc_column_benchmark_rows_by_reference_kind(
    const std::array<ReducedRCColumnBenchmarkTraceRow, N>& rows,
    ReducedRCColumnBenchmarkReferenceKind reference_kind) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.reference_kind == reference_kind) {
            ++count;
        }
    }
    return count;
}

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_reduced_rc_column_benchmark_rows_requiring_external_reference(
    const std::array<ReducedRCColumnBenchmarkTraceRow, N>& rows) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.requires_external_reference_dataset) {
            ++count;
        }
    }
    return count;
}

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_reduced_rc_column_benchmark_rows_requiring_independent_section_baseline(
    const std::array<ReducedRCColumnBenchmarkTraceRow, N>& rows) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.requires_independent_section_baseline) {
            ++count;
        }
    }
    return count;
}

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_reduced_rc_column_phase3_structural_benchmark_gates(
    const std::array<ReducedRCColumnBenchmarkTraceRow, N>& rows) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.required_for_phase3_structural_physical_benchmark) {
            ++count;
        }
    }
    return count;
}

template <ReducedRCColumnBenchmarkMetricKind MetricKind>
inline constexpr std::size_t
    canonical_reduced_rc_column_benchmark_metric_count_v =
        count_reduced_rc_column_benchmark_rows_by_metric(
            canonical_reduced_rc_column_benchmark_trace_table_v,
            MetricKind);

template <ReducedRCColumnBenchmarkReferenceKind ReferenceKind>
inline constexpr std::size_t
    canonical_reduced_rc_column_benchmark_reference_count_v =
        count_reduced_rc_column_benchmark_rows_by_reference_kind(
            canonical_reduced_rc_column_benchmark_trace_table_v,
            ReferenceKind);

inline constexpr std::size_t
    canonical_reduced_rc_column_benchmark_external_reference_count_v =
        count_reduced_rc_column_benchmark_rows_requiring_external_reference(
            canonical_reduced_rc_column_benchmark_trace_table_v);

inline constexpr std::size_t
    canonical_reduced_rc_column_benchmark_section_baseline_count_v =
        count_reduced_rc_column_benchmark_rows_requiring_independent_section_baseline(
            canonical_reduced_rc_column_benchmark_trace_table_v);

inline constexpr std::size_t
    canonical_reduced_rc_column_phase3_structural_benchmark_gate_count_v =
        count_reduced_rc_column_phase3_structural_benchmark_gates(
            canonical_reduced_rc_column_benchmark_trace_table_v);

} // namespace fall_n

#endif // FALL_N_REDUCED_RC_COLUMN_BENCHMARK_TRACE_CATALOG_HH
