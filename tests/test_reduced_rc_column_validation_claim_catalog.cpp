#include <cstddef>
#include <iostream>

#include "src/validation/ReducedRCColumnStructuralMatrixCatalog.hh"
#include "src/validation/ReducedRCColumnValidationClaimCatalog.hh"

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
    fall_n::canonical_reduced_rc_column_validation_claim_table_v;
constexpr auto structural_matrix =
    fall_n::canonical_reduced_rc_column_structural_matrix_v;

constexpr bool all_current_claims_stay_on_the_runtime_ready_small_strain_family()
{
    for (const auto& row : claim_table) {
        if (!row.depends_on_current_small_strain_runtime_family) {
            continue;
        }

        if (row.structural_support_kind !=
                fall_n::ReducedRCColumnStructuralSupportKind::
                    ready_for_runtime_baseline ||
            (!row.can_anchor_phase3_runtime_baseline &&
                row.readiness_kind ==
                    fall_n::ReducedRCColumnClaimReadinessKind::
                        runtime_baseline_ready)) {
            return false;
        }
    }

    return true;
}

constexpr bool quadrature_enabler_claim_is_frozen_but_not_overclaimed()
{
    std::size_t found = 0;

    for (const auto& row : claim_table) {
        if (row.claim_label !=
            "beam_axis_quadrature_family_is_explicit_model_axis") {
            continue;
        }

        ++found;

        if (row.readiness_kind !=
                fall_n::ReducedRCColumnClaimReadinessKind::
                    enabling_contract_frozen ||
            row.pending_experiment_kind !=
                fall_n::ReducedRCColumnPendingExperimentKind::
                    none_frozen_or_ready ||
            !row.can_anchor_phase3_runtime_baseline ||
            row.can_anchor_phase3_physical_benchmark ||
            row.requires_new_runtime_instrumentation) {
            return false;
        }
    }

    return found == 1;
}

constexpr bool runtime_ready_claims_remain_prebenchmark_claims()
{
    bool found_runtime_surface = false;
    bool found_axial = false;
    bool found_hysteresis = false;
    bool found_moment_curvature = false;

    for (const auto& row : claim_table) {
        if (row.readiness_kind !=
            fall_n::ReducedRCColumnClaimReadinessKind::runtime_baseline_ready) {
            continue;
        }

        if (!row.can_anchor_phase3_runtime_baseline ||
            row.can_anchor_phase3_physical_benchmark ||
            row.requires_new_runtime_instrumentation) {
            return false;
        }

        if (row.claim_label ==
            "reduced_column_small_strain_runtime_surface_exists_for_n2_to_n10") {
            found_runtime_surface = true;
        } else if (row.claim_label ==
                   "optional_axial_compression_load_path_is_supported") {
            found_axial = true;
        } else if (row.claim_label ==
                   "base_shear_vs_displacement_hysteresis_output_is_reproducible") {
            found_hysteresis = true;
        } else if (row.claim_label ==
                   "base_side_moment_curvature_observable_is_normatively_extracted") {
            found_moment_curvature = true;
        } else {
            return false;
        }
    }

    return found_runtime_surface && found_axial && found_hysteresis &&
           found_moment_curvature;
}

constexpr bool moment_curvature_observable_is_runtime_ready_but_prebenchmark()
{
    std::size_t found = 0;

    for (const auto& row : claim_table) {
        if (row.claim_label !=
            "base_side_moment_curvature_observable_is_normatively_extracted") {
            continue;
        }

        ++found;

        if (row.readiness_kind !=
                fall_n::ReducedRCColumnClaimReadinessKind::
                    runtime_baseline_ready ||
            row.pending_experiment_kind !=
                fall_n::ReducedRCColumnPendingExperimentKind::
                    reduced_column_moment_curvature_suite ||
            row.requires_new_runtime_instrumentation ||
            !row.can_anchor_phase3_runtime_baseline ||
            row.can_anchor_phase3_physical_benchmark) {
            return false;
        }
    }

    return found == 1;
}

constexpr bool benchmark_pending_claims_are_only_refinement_and_quadrature_sensitivity()
{
    bool found_refinement = false;
    bool found_quadrature = false;

    for (const auto& row : claim_table) {
        if (row.readiness_kind !=
            fall_n::ReducedRCColumnClaimReadinessKind::benchmark_pending) {
            continue;
        }

        if (row.claim_label ==
            "node_refinement_convergence_claim_is_not_yet_closed") {
            found_refinement = true;
            if (row.pending_experiment_kind !=
                fall_n::ReducedRCColumnPendingExperimentKind::
                    reduced_column_node_refinement_suite) {
                return false;
            }
        } else if (row.claim_label ==
                   "quadrature_family_sensitivity_claim_is_not_yet_closed") {
            found_quadrature = true;
            if (row.pending_experiment_kind !=
                fall_n::ReducedRCColumnPendingExperimentKind::
                    reduced_column_quadrature_sensitivity_suite) {
                return false;
            }
        } else {
            return false;
        }

        if (row.can_anchor_phase3_runtime_baseline ||
            row.can_anchor_phase3_physical_benchmark ||
            row.requires_new_runtime_instrumentation) {
            return false;
        }
    }

    return found_refinement && found_quadrature;
}

constexpr bool structural_matrix_and_claim_matrix_have_consistent_runtime_frontier()
{
    return fall_n::canonical_reduced_rc_column_phase3_baseline_case_count_v == 36 &&
           fall_n::canonical_reduced_rc_column_claim_runtime_baseline_anchor_count_v == 5 &&
           fall_n::canonical_reduced_rc_column_claim_physical_benchmark_anchor_count_v == 0 &&
           fall_n::canonical_reduced_rc_column_claim_instrumentation_pending_count_v == 0 &&
           structural_matrix.size() == 144;
}

static_assert(all_current_claims_stay_on_the_runtime_ready_small_strain_family());
static_assert(quadrature_enabler_claim_is_frozen_but_not_overclaimed());
static_assert(runtime_ready_claims_remain_prebenchmark_claims());
static_assert(moment_curvature_observable_is_runtime_ready_but_prebenchmark());
static_assert(benchmark_pending_claims_are_only_refinement_and_quadrature_sensitivity());
static_assert(structural_matrix_and_claim_matrix_have_consistent_runtime_frontier());
static_assert(
    fall_n::canonical_reduced_rc_column_claim_readiness_count_v<
        fall_n::ReducedRCColumnClaimReadinessKind::enabling_contract_frozen> == 1);
static_assert(
    fall_n::canonical_reduced_rc_column_claim_readiness_count_v<
        fall_n::ReducedRCColumnClaimReadinessKind::runtime_baseline_ready> == 4);
static_assert(
    fall_n::canonical_reduced_rc_column_claim_readiness_count_v<
        fall_n::ReducedRCColumnClaimReadinessKind::benchmark_pending> == 2);
static_assert(
    fall_n::canonical_reduced_rc_column_claim_readiness_count_v<
        fall_n::ReducedRCColumnClaimReadinessKind::instrumentation_pending> == 0);

} // namespace

int main()
{
    std::cout << "=== Reduced RC Column Validation Claim Catalog Tests ===\n";

    report("all_current_claims_stay_on_the_runtime_ready_small_strain_family",
           all_current_claims_stay_on_the_runtime_ready_small_strain_family());
    report("quadrature_enabler_claim_is_frozen_but_not_overclaimed",
           quadrature_enabler_claim_is_frozen_but_not_overclaimed());
    report("runtime_ready_claims_remain_prebenchmark_claims",
           runtime_ready_claims_remain_prebenchmark_claims());
    report("moment_curvature_observable_is_runtime_ready_but_prebenchmark",
           moment_curvature_observable_is_runtime_ready_but_prebenchmark());
    report("benchmark_pending_claims_are_only_refinement_and_quadrature_sensitivity",
           benchmark_pending_claims_are_only_refinement_and_quadrature_sensitivity());
    report("structural_matrix_and_claim_matrix_have_consistent_runtime_frontier",
           structural_matrix_and_claim_matrix_have_consistent_runtime_frontier());

    std::cout << "\nSummary: " << passed << " passed, " << failed << " failed.\n";
    return failed == 0 ? 0 : 1;
}
