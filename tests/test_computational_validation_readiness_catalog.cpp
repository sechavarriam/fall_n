#include <cstddef>
#include <iostream>

#include "src/analysis/ComputationalClaimTraceCatalog.hh"
#include "src/analysis/ComputationalValidationReadinessCatalog.hh"

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

constexpr auto claim_table =
    fall_n::canonical_representative_computational_claim_trace_table_v;
constexpr auto readiness_table =
    fall_n::canonical_representative_computational_validation_readiness_table_v;

constexpr bool readiness_rows_align_with_claim_rows()
{
    if constexpr (claim_table.size() != readiness_table.size()) {
        return false;
    }

    for (std::size_t i = 0; i < claim_table.size(); ++i) {
        if (readiness_table[i].claim_trace.claim_label != claim_table[i].claim_label ||
            readiness_table[i].claim_trace.slice_label != claim_table[i].slice_label ||
            readiness_table[i].claim_trace.slice_support_level() !=
                claim_table[i].slice_support_level()) {
            return false;
        }
    }

    return true;
}

constexpr bool frozen_baselines_match_nonphysical_reference_rows()
{
    for (const auto& row : readiness_table) {
        if (!row.claim_trace.requires_physical_validation) {
            if (!row.is_frozen_baseline() || !row.can_anchor_validation_baseline ||
                row.pending_experiment_kind !=
                    fall_n::PendingPhysicalValidationExperimentKind::
                        none_reference_baseline ||
                row.pending_experiment_label != "none_reference_baseline") {
                return false;
            }
        }
    }

    return true;
}

constexpr bool only_total_lagrangian_continuum_slice_is_ready_for_targeted_validation()
{
    std::size_t ready_count = 0;

    for (const auto& row : readiness_table) {
        if (!row.can_enter_targeted_physical_validation) {
            continue;
        }

        ++ready_count;

        if (row.readiness_kind !=
                fall_n::ComputationalValidationReadinessKind::
                    ready_for_targeted_physical_validation ||
            row.claim_trace.slice_label != "continuum_total_lagrangian_nonlinear" ||
            row.pending_experiment_kind !=
                fall_n::PendingPhysicalValidationExperimentKind::
                    structural_material_validation_campaign) {
            return false;
        }
    }

    return ready_count == 1;
}

constexpr bool scope_closure_pending_rows_are_the_honest_partial_slices()
{
    bool found_ul = false;
    bool found_dynamic = false;
    bool found_beam = false;
    bool found_shell = false;

    for (const auto& row : readiness_table) {
        if (row.readiness_kind !=
            fall_n::ComputationalValidationReadinessKind::scope_closure_pending) {
            continue;
        }

        if (row.claim_trace.slice_label == "continuum_updated_lagrangian_nonlinear") {
            found_ul = true;
            if (row.pending_experiment_kind !=
                fall_n::PendingPhysicalValidationExperimentKind::
                    updated_lagrangian_scope_closure_suite) {
                return false;
            }
        } else if (row.claim_trace.slice_label == "continuum_total_lagrangian_dynamic") {
            found_dynamic = true;
            if (row.pending_experiment_kind !=
                fall_n::PendingPhysicalValidationExperimentKind::
                    dynamic_scope_closure_suite) {
                return false;
            }
        } else if (row.claim_trace.slice_label == "beam_corotational_nonlinear") {
            found_beam = true;
            if (row.pending_experiment_kind !=
                fall_n::PendingPhysicalValidationExperimentKind::
                    beam_corotational_scope_closure_suite) {
                return false;
            }
        } else if (row.claim_trace.slice_label == "shell_corotational_nonlinear") {
            found_shell = true;
            if (row.pending_experiment_kind !=
                fall_n::PendingPhysicalValidationExperimentKind::
                    shell_corotational_scope_closure_suite) {
                return false;
            }
        } else {
            return false;
        }

        if (!row.needs_more_numerical_scope_before_validation()) {
            return false;
        }
    }

    return found_ul && found_dynamic && found_beam && found_shell;
}

constexpr bool arc_length_is_runtime_regression_pending_before_validation()
{
    for (const auto& row : readiness_table) {
        if (row.claim_trace.slice_label != "continuum_total_lagrangian_arc_length") {
            continue;
        }

        return row.readiness_kind ==
                   fall_n::ComputationalValidationReadinessKind::
                       runtime_regression_pending &&
               row.pending_experiment_kind ==
                   fall_n::PendingPhysicalValidationExperimentKind::
                       continuation_runtime_regression_suite &&
               row.evidence_floor_label == "catalog_and_variational_audit_only" &&
               !row.can_enter_targeted_physical_validation;
    }

    return false;
}

static_assert(readiness_rows_align_with_claim_rows());
static_assert(frozen_baselines_match_nonphysical_reference_rows());
static_assert(only_total_lagrangian_continuum_slice_is_ready_for_targeted_validation());
static_assert(scope_closure_pending_rows_are_the_honest_partial_slices());
static_assert(arc_length_is_runtime_regression_pending_before_validation());
static_assert(
    fall_n::canonical_representative_validation_readiness_count_v<
        fall_n::ComputationalValidationReadinessKind::frozen_reference_baseline> == 3);
static_assert(
    fall_n::canonical_representative_validation_readiness_count_v<
        fall_n::ComputationalValidationReadinessKind::
            ready_for_targeted_physical_validation> == 1);
static_assert(
    fall_n::canonical_representative_validation_readiness_count_v<
        fall_n::ComputationalValidationReadinessKind::scope_closure_pending> == 4);
static_assert(
    fall_n::canonical_representative_validation_readiness_count_v<
        fall_n::ComputationalValidationReadinessKind::runtime_regression_pending> == 1);
static_assert(fall_n::canonical_representative_validation_baseline_count_v == 3);
static_assert(
    fall_n::canonical_representative_targeted_physical_validation_count_v == 1);

} // namespace

int main()
{
    std::cout << "=== Computational Validation Readiness Catalog Tests ===\n";

    report("readiness_rows_align_with_claim_rows",
           readiness_rows_align_with_claim_rows());
    report("frozen_baselines_match_nonphysical_reference_rows",
           frozen_baselines_match_nonphysical_reference_rows());
    report("only_total_lagrangian_continuum_slice_is_ready_for_targeted_validation",
           only_total_lagrangian_continuum_slice_is_ready_for_targeted_validation());
    report("scope_closure_pending_rows_are_the_honest_partial_slices",
           scope_closure_pending_rows_are_the_honest_partial_slices());
    report("arc_length_is_runtime_regression_pending_before_validation",
           arc_length_is_runtime_regression_pending_before_validation());

    std::cout << "\nSummary: " << passed << " passed, " << failed << " failed.\n";
    return failed == 0 ? 0 : 1;
}
