#include <cstddef>
#include <iostream>

#include "src/validation/ReducedRCColumnBenchmarkTraceCatalog.hh"

namespace {

int passed = 0;
int failed = 0;

void report(const char* name, bool ok)
{
    if (ok) {
        ++passed;
        std::cout << "  PASS  " << name << "\n";
    } else {
        ++failed;
        std::cout << "  FAIL  " << name << "\n";
    }
}

constexpr auto benchmark_table =
    fall_n::canonical_reduced_rc_column_benchmark_trace_table_v;

constexpr bool every_benchmark_row_targets_an_open_prebenchmark_claim()
{
    for (const auto& row : benchmark_table) {
        if (!row.claim_row.has_open_prebenchmark_gate() ||
            row.claim_row.requires_new_runtime_instrumentation ||
            row.claim_row.can_anchor_phase3_physical_benchmark) {
            return false;
        }
    }
    return true;
}

constexpr bool benchmark_rows_cover_only_runtime_ready_and_benchmark_pending_claims()
{
    bool found_runtime_surface = false;
    bool found_axial = false;
    bool found_hysteresis = false;
    bool found_moment_curvature = false;
    bool found_refinement = false;
    bool found_quadrature = false;

    for (const auto& row : benchmark_table) {
        switch (row.claim_row.readiness_kind) {
            case fall_n::ReducedRCColumnClaimReadinessKind::runtime_baseline_ready:
            case fall_n::ReducedRCColumnClaimReadinessKind::benchmark_pending:
                break;

            case fall_n::ReducedRCColumnClaimReadinessKind::enabling_contract_frozen:
            case fall_n::ReducedRCColumnClaimReadinessKind::instrumentation_pending:
                return false;
        }

        if (row.claim_row.claim_label ==
            "reduced_column_small_strain_runtime_surface_exists_for_n2_to_n10") {
            found_runtime_surface = true;
        } else if (row.claim_row.claim_label ==
                   "optional_axial_compression_load_path_is_supported") {
            found_axial = true;
        } else if (row.claim_row.claim_label ==
                   "base_shear_vs_displacement_hysteresis_output_is_reproducible") {
            found_hysteresis = true;
        } else if (row.claim_row.claim_label ==
                   "base_side_moment_curvature_observable_is_normatively_extracted") {
            found_moment_curvature = true;
        } else if (row.claim_row.claim_label ==
                   "node_refinement_convergence_claim_is_not_yet_closed") {
            found_refinement = true;
        } else if (row.claim_row.claim_label ==
                   "quadrature_family_sensitivity_claim_is_not_yet_closed") {
            found_quadrature = true;
        } else {
            return false;
        }
    }

    return found_runtime_surface && found_axial && found_hysteresis &&
           found_moment_curvature && found_refinement && found_quadrature;
}

constexpr bool reference_classes_are_honest()
{
    for (const auto& row : benchmark_table) {
        if (row.benchmark_label == "reduced_column_runtime_surface_matrix") {
            if (row.reference_kind !=
                    fall_n::ReducedRCColumnBenchmarkReferenceKind::
                        internal_runtime_matrix_reference ||
                row.requires_external_reference_dataset ||
                row.required_for_phase3_structural_physical_benchmark) {
                return false;
            }
        } else if (row.benchmark_label ==
                       "reduced_column_base_side_moment_curvature_suite") {
            if (row.reference_kind !=
                    fall_n::ReducedRCColumnBenchmarkReferenceKind::
                        analytical_or_section_baseline_reference ||
                !row.requires_external_reference_dataset ||
                !row.requires_independent_section_baseline) {
                return false;
            }
        } else if (row.benchmark_label ==
                       "reduced_column_axial_load_interaction_suite" ||
                   row.benchmark_label ==
                       "reduced_column_hysteresis_protocol_suite") {
            if (row.reference_kind !=
                    fall_n::ReducedRCColumnBenchmarkReferenceKind::
                        experimental_or_literature_reference ||
                !row.requires_external_reference_dataset) {
                return false;
            }
        } else if (row.benchmark_label ==
                       "reduced_column_node_refinement_suite" ||
                   row.benchmark_label ==
                       "reduced_column_quadrature_sensitivity_suite") {
            if (row.reference_kind !=
                    fall_n::ReducedRCColumnBenchmarkReferenceKind::
                        internal_runtime_matrix_reference ||
                row.requires_external_reference_dataset ||
                row.requires_independent_section_baseline) {
                return false;
            }
        } else {
            return false;
        }
    }

    return true;
}

constexpr bool phase3_structural_physical_gate_requires_all_but_runtime_surface_completeness()
{
    std::size_t gate_rows = 0;
    bool runtime_surface_is_not_gate = false;

    for (const auto& row : benchmark_table) {
        if (row.required_for_phase3_structural_physical_benchmark) {
            ++gate_rows;
        }
        if (row.benchmark_label == "reduced_column_runtime_surface_matrix" &&
            !row.required_for_phase3_structural_physical_benchmark) {
            runtime_surface_is_not_gate = true;
        }
        if (!row.closes_own_prebenchmark_gate_if_passed) {
            return false;
        }
    }

    return gate_rows == 5 && runtime_surface_is_not_gate;
}

constexpr bool benchmark_count_summary_matches_the_intended_mix()
{
    return benchmark_table.size() == 6 &&
           fall_n::canonical_reduced_rc_column_benchmark_metric_count_v<
               fall_n::ReducedRCColumnBenchmarkMetricKind::
                   runtime_surface_completeness> == 1 &&
           fall_n::canonical_reduced_rc_column_benchmark_metric_count_v<
               fall_n::ReducedRCColumnBenchmarkMetricKind::
                   axial_load_interaction_trend> == 1 &&
           fall_n::canonical_reduced_rc_column_benchmark_metric_count_v<
               fall_n::ReducedRCColumnBenchmarkMetricKind::
                   hysteresis_loop_shape_and_dissipation> == 1 &&
           fall_n::canonical_reduced_rc_column_benchmark_metric_count_v<
               fall_n::ReducedRCColumnBenchmarkMetricKind::
                   base_side_moment_curvature_envelope> == 1 &&
           fall_n::canonical_reduced_rc_column_benchmark_metric_count_v<
               fall_n::ReducedRCColumnBenchmarkMetricKind::
                   node_refinement_convergence> == 1 &&
           fall_n::canonical_reduced_rc_column_benchmark_metric_count_v<
               fall_n::ReducedRCColumnBenchmarkMetricKind::
                   quadrature_family_sensitivity> == 1 &&
           fall_n::canonical_reduced_rc_column_benchmark_reference_count_v<
               fall_n::ReducedRCColumnBenchmarkReferenceKind::
                   internal_runtime_matrix_reference> == 3 &&
           fall_n::canonical_reduced_rc_column_benchmark_reference_count_v<
               fall_n::ReducedRCColumnBenchmarkReferenceKind::
                   analytical_or_section_baseline_reference> == 1 &&
           fall_n::canonical_reduced_rc_column_benchmark_reference_count_v<
               fall_n::ReducedRCColumnBenchmarkReferenceKind::
                   experimental_or_literature_reference> == 2 &&
           fall_n::canonical_reduced_rc_column_benchmark_external_reference_count_v == 3 &&
           fall_n::canonical_reduced_rc_column_benchmark_section_baseline_count_v == 1 &&
           fall_n::canonical_reduced_rc_column_phase3_structural_benchmark_gate_count_v == 5;
}

static_assert(every_benchmark_row_targets_an_open_prebenchmark_claim());
static_assert(benchmark_rows_cover_only_runtime_ready_and_benchmark_pending_claims());
static_assert(reference_classes_are_honest());
static_assert(phase3_structural_physical_gate_requires_all_but_runtime_surface_completeness());
static_assert(benchmark_count_summary_matches_the_intended_mix());

} // namespace

int main()
{
    std::cout << "=== Reduced RC Column Benchmark Trace Catalog Tests ===\n";

    report("every_benchmark_row_targets_an_open_prebenchmark_claim",
           every_benchmark_row_targets_an_open_prebenchmark_claim());
    report("benchmark_rows_cover_only_runtime_ready_and_benchmark_pending_claims",
           benchmark_rows_cover_only_runtime_ready_and_benchmark_pending_claims());
    report("reference_classes_are_honest",
           reference_classes_are_honest());
    report("phase3_structural_physical_gate_requires_all_but_runtime_surface_completeness",
           phase3_structural_physical_gate_requires_all_but_runtime_surface_completeness());
    report("benchmark_count_summary_matches_the_intended_mix",
           benchmark_count_summary_matches_the_intended_mix());

    std::cout << "\nSummary: " << passed << " passed, " << failed << " failed.\n";
    return failed == 0 ? 0 : 1;
}
