#include <cstddef>
#include <iostream>

#include "src/validation/ReducedRCColumnEvidenceClosureCatalog.hh"

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

constexpr auto evidence_table =
    fall_n::canonical_reduced_rc_column_evidence_closure_table_v;

constexpr bool every_evidence_row_aligns_with_an_open_benchmark_trace_row()
{
    for (const auto& row : evidence_table) {
        if (row.claim_row.claim_label != row.benchmark_row.claim_row.claim_label ||
            !row.claim_row.has_open_prebenchmark_gate() ||
            !row.benchmark_row.claim_row.has_open_prebenchmark_gate()) {
            return false;
        }
    }

    return true;
}

constexpr bool evidence_artifacts_and_missing_experiments_are_nonempty()
{
    for (const auto& row : evidence_table) {
        if (row.current_artifact_label.empty() ||
            row.current_evidence_state_label.empty() ||
            row.missing_numerical_experiment_label.empty() ||
            row.required_closure_artifact_label.empty() ||
            row.acceptance_measure_label.empty() ||
            row.benchmark_dataset_scope_label.empty()) {
            return false;
        }
    }

    return true;
}

constexpr bool internal_vs_external_evidence_classes_are_honest()
{
    for (const auto& row : evidence_table) {
        if (row.benchmark_row.benchmark_label ==
            "reduced_column_runtime_surface_matrix") {
            if (row.requires_external_dataset ||
                row.requires_independent_section_baseline ||
                row.closes_phase3_structural_physical_benchmark) {
                return false;
            }
        } else if (row.benchmark_row.benchmark_label ==
                   "reduced_column_axial_load_interaction_suite" ||
                   row.benchmark_row.benchmark_label ==
                       "reduced_column_hysteresis_protocol_suite") {
            if (!row.requires_external_dataset ||
                row.requires_independent_section_baseline ||
                !row.closes_phase3_structural_physical_benchmark) {
                return false;
            }
        } else if (row.benchmark_row.benchmark_label ==
                   "reduced_column_base_side_moment_curvature_suite") {
            if (!row.requires_external_dataset ||
                !row.requires_independent_section_baseline ||
                !row.closes_phase3_structural_physical_benchmark) {
                return false;
            }
        } else if (row.benchmark_row.benchmark_label ==
                       "reduced_column_node_refinement_suite" ||
                   row.benchmark_row.benchmark_label ==
                       "reduced_column_quadrature_sensitivity_suite") {
            if (row.requires_external_dataset ||
                row.requires_independent_section_baseline ||
                !row.closes_phase3_structural_physical_benchmark) {
                return false;
            }
        } else {
            return false;
        }
    }

    return true;
}

constexpr bool matrix_sweep_obligations_are_explicit()
{
    bool runtime_surface_requires_matrix = false;
    bool node_refinement_requires_matrix = false;
    bool quadrature_requires_matrix = false;
    bool moment_curvature_requires_matrix = false;

    for (const auto& row : evidence_table) {
        if (row.benchmark_row.benchmark_label ==
            "reduced_column_runtime_surface_matrix") {
            runtime_surface_requires_matrix = row.requires_repeated_matrix_sweep;
        } else if (row.benchmark_row.benchmark_label ==
                   "reduced_column_node_refinement_suite") {
            node_refinement_requires_matrix = row.requires_repeated_matrix_sweep;
        } else if (row.benchmark_row.benchmark_label ==
                   "reduced_column_quadrature_sensitivity_suite") {
            quadrature_requires_matrix = row.requires_repeated_matrix_sweep;
        } else if (row.benchmark_row.benchmark_label ==
                   "reduced_column_base_side_moment_curvature_suite") {
            moment_curvature_requires_matrix = row.requires_repeated_matrix_sweep;
        }
    }

    return runtime_surface_requires_matrix && node_refinement_requires_matrix &&
           quadrature_requires_matrix && moment_curvature_requires_matrix;
}

constexpr bool evidence_kind_distribution_matches_the_intended_campaign()
{
    return evidence_table.size() == 6 &&
           fall_n::canonical_reduced_rc_column_missing_evidence_count_v<
               fall_n::ReducedRCColumnMissingEvidenceKind::
                   runtime_matrix_summary_bundle> == 1 &&
           fall_n::canonical_reduced_rc_column_missing_evidence_count_v<
               fall_n::ReducedRCColumnMissingEvidenceKind::
                   external_axial_interaction_dataset> == 1 &&
           fall_n::canonical_reduced_rc_column_missing_evidence_count_v<
               fall_n::ReducedRCColumnMissingEvidenceKind::
                   external_hysteresis_dataset> == 1 &&
           fall_n::canonical_reduced_rc_column_missing_evidence_count_v<
               fall_n::ReducedRCColumnMissingEvidenceKind::
                   independent_section_baseline_bundle> == 1 &&
           fall_n::canonical_reduced_rc_column_missing_evidence_count_v<
               fall_n::ReducedRCColumnMissingEvidenceKind::
                   node_refinement_convergence_bundle> == 1 &&
           fall_n::canonical_reduced_rc_column_missing_evidence_count_v<
               fall_n::ReducedRCColumnMissingEvidenceKind::
                   quadrature_sensitivity_bundle> == 1 &&
           fall_n::canonical_reduced_rc_column_evidence_external_dataset_count_v == 3 &&
           fall_n::canonical_reduced_rc_column_evidence_section_baseline_count_v == 1 &&
           fall_n::canonical_reduced_rc_column_evidence_phase3_gate_count_v == 5 &&
           fall_n::canonical_reduced_rc_column_evidence_matrix_sweep_count_v == 6;
}

static_assert(every_evidence_row_aligns_with_an_open_benchmark_trace_row());
static_assert(evidence_artifacts_and_missing_experiments_are_nonempty());
static_assert(internal_vs_external_evidence_classes_are_honest());
static_assert(matrix_sweep_obligations_are_explicit());
static_assert(evidence_kind_distribution_matches_the_intended_campaign());

} // namespace

int main()
{
    std::cout << "=== Reduced RC Column Evidence Closure Catalog Tests ===\n";

    report("every_evidence_row_aligns_with_an_open_benchmark_trace_row",
           every_evidence_row_aligns_with_an_open_benchmark_trace_row());
    report("evidence_artifacts_and_missing_experiments_are_nonempty",
           evidence_artifacts_and_missing_experiments_are_nonempty());
    report("internal_vs_external_evidence_classes_are_honest",
           internal_vs_external_evidence_classes_are_honest());
    report("matrix_sweep_obligations_are_explicit",
           matrix_sweep_obligations_are_explicit());
    report("evidence_kind_distribution_matches_the_intended_campaign",
           evidence_kind_distribution_matches_the_intended_campaign());

    std::cout << "\nSummary: " << passed << " passed, " << failed << " failed.\n";
    return failed == 0 ? 0 : 1;
}
