#include <cstddef>
#include <iostream>
#include <string_view>

#include "src/analysis/ComputationalClaimTraceCatalog.hh"
#include "src/analysis/ComputationalVariationalSliceCatalog.hh"

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
constexpr auto variational_slice_matrix =
    fall_n::canonical_representative_computational_variational_slice_matrix_v;

constexpr bool claim_rows_align_with_variational_slices()
{
    if constexpr (claim_table.size() != variational_slice_matrix.size()) {
        return false;
    }

    for (std::size_t i = 0; i < claim_table.size(); ++i) {
        if (claim_table[i].slice_label != variational_slice_matrix[i].slice_label ||
            claim_table[i].slice_support_level() !=
                variational_slice_matrix[i].slice_support_level() ||
            claim_table[i].requires_scope_disclaimer() !=
                variational_slice_matrix[i].requires_scope_disclaimer()) {
            return false;
        }
    }

    return true;
}

constexpr bool reference_claims_are_not_overclaimed()
{
    for (const auto& row : claim_table) {
        if (!row.is_reference_claim) {
            continue;
        }

        const auto level = row.slice_support_level();
        if (level != fall_n::ComputationalModelSliceSupportLevel::reference_linear &&
            level !=
                fall_n::ComputationalModelSliceSupportLevel::reference_geometric_nonlinearity) {
            return false;
        }
    }

    return true;
}

constexpr bool structural_evidence_stays_on_reduced_structural_slices()
{
    for (const auto& row : claim_table) {
        if (row.evidence_breadth !=
            fall_n::ComputationalClaimEvidenceBreadthKind::structural_response_regression) {
            continue;
        }

        if (!row.audit_scope.is_structural_reduction_path() ||
            row.audit_scope.element_family ==
                continuum::ElementFamilyKind::continuum_solid_3d ||
            !row.audit_scope.discrete_variational_semantics.uses_structural_resultants) {
            return false;
        }
    }

    return true;
}

constexpr bool arc_length_claim_remains_honest()
{
    for (const auto& row : claim_table) {
        if (row.slice_label != "continuum_total_lagrangian_arc_length") {
            continue;
        }

        return row.evidence_breadth ==
                   fall_n::ComputationalClaimEvidenceBreadthKind::
                       catalog_and_semantic_audit &&
               row.requires_scope_disclaimer() && row.requires_physical_validation &&
               !row.is_reference_claim;
    }

    return false;
}

constexpr bool nonphysical_baselines_are_linear_reference_slices()
{
    for (const auto& row : claim_table) {
        if (row.requires_physical_validation) {
            continue;
        }

        if (row.audit_scope.route_kind != fall_n::AnalysisRouteKind::linear_static ||
            row.slice_support_level() !=
                fall_n::ComputationalModelSliceSupportLevel::reference_linear) {
            return false;
        }
    }

    return true;
}

constexpr bool residual_tangent_and_history_commitments_match_audit_scope()
{
    for (const auto& row : claim_table) {
        switch (row.audit_scope.global_residual_operator) {
            case fall_n::GlobalResidualOperatorKind::static_equilibrium:
                if (row.residual_commitment_label != "static_global_equilibrium" &&
                    row.residual_commitment_label != "sectional_static_equilibrium") {
                    return false;
                }
                break;

            case fall_n::GlobalResidualOperatorKind::incremental_static_equilibrium:
                if (row.residual_commitment_label != "incremental_global_equilibrium" &&
                    row.residual_commitment_label !=
                        "sectional_incremental_equilibrium") {
                    return false;
                }
                break;

            case fall_n::GlobalResidualOperatorKind::second_order_dynamic_equilibrium:
                if (row.residual_commitment_label !=
                    "second_order_dynamic_equilibrium") {
                    return false;
                }
                break;

            case fall_n::GlobalResidualOperatorKind::arc_length_augmented_equilibrium:
                if (row.residual_commitment_label !=
                    "arc_length_augmented_equilibrium") {
                    return false;
                }
                break;

            case fall_n::GlobalResidualOperatorKind::unavailable:
                return false;
        }

        switch (row.audit_scope.global_tangent_operator) {
            case fall_n::GlobalTangentOperatorKind::linear_or_static_consistent_tangent:
                if (row.tangent_commitment_label != "assembled_linear_stiffness" &&
                    row.tangent_commitment_label !=
                        "sectional_constitutive_stiffness") {
                    return false;
                }
                break;

            case fall_n::GlobalTangentOperatorKind::monolithic_incremental_consistent_tangent:
                if (row.tangent_commitment_label !=
                        "monolithic_consistent_tangent" &&
                    row.tangent_commitment_label !=
                        "sectional_constitutive_plus_geometric_tangent") {
                    return false;
                }
                break;

            case fall_n::GlobalTangentOperatorKind::effective_mass_damping_stiffness_tangent:
                if (row.tangent_commitment_label !=
                    "effective_mass_damping_stiffness_tangent") {
                    return false;
                }
                break;

            case fall_n::GlobalTangentOperatorKind::bordered_arc_length_tangent:
                if (row.tangent_commitment_label !=
                    "bordered_continuation_tangent") {
                    return false;
                }
                break;

            case fall_n::GlobalTangentOperatorKind::unavailable:
                return false;
        }

        switch (row.audit_scope.incremental_state_management) {
            case fall_n::IncrementalStateManagementKind::stateless_or_direct_response:
                if (row.history_commitment_label !=
                    "stateless_or_direct_response") {
                    return false;
                }
                break;

            case fall_n::IncrementalStateManagementKind::converged_step_commit:
            case fall_n::IncrementalStateManagementKind::
                checkpointable_converged_step_commit:
                if (row.history_commitment_label !=
                        "checkpointable_converged_step_commit" &&
                    row.history_commitment_label !=
                        "section_or_fiber_history_commit" &&
                    row.history_commitment_label != "converged_step_commit") {
                    return false;
                }
                break;

            case fall_n::IncrementalStateManagementKind::continuation_step_commit:
                if (row.history_commitment_label !=
                    "continuation_step_commit") {
                    return false;
                }
                break;

            case fall_n::IncrementalStateManagementKind::unavailable:
                return false;
        }
    }

    return true;
}

constexpr bool arc_length_and_dynamic_claims_preserve_distinct_history_semantics()
{
    bool found_dynamic = false;
    bool found_arc_length = false;

    for (const auto& row : claim_table) {
        if (row.slice_label == "continuum_total_lagrangian_dynamic") {
            found_dynamic = true;
            if (row.history_commitment_label !=
                    "checkpointable_converged_step_commit" ||
                row.residual_commitment_label !=
                    "second_order_dynamic_equilibrium") {
                return false;
            }
        }

        if (row.slice_label == "continuum_total_lagrangian_arc_length") {
            found_arc_length = true;
            if (row.history_commitment_label != "continuation_step_commit" ||
                row.residual_commitment_label !=
                    "arc_length_augmented_equilibrium") {
                return false;
            }
        }
    }

    return found_dynamic && found_arc_length;
}

static_assert(claim_rows_align_with_variational_slices());
static_assert(reference_claims_are_not_overclaimed());
static_assert(structural_evidence_stays_on_reduced_structural_slices());
static_assert(arc_length_claim_remains_honest());
static_assert(nonphysical_baselines_are_linear_reference_slices());
static_assert(residual_tangent_and_history_commitments_match_audit_scope());
static_assert(arc_length_and_dynamic_claims_preserve_distinct_history_semantics());
static_assert(
    fall_n::canonical_representative_reference_computational_claim_count_v == 4);
static_assert(
    fall_n::canonical_representative_computational_claim_physical_validation_count_v == 6);
static_assert(
    fall_n::canonical_representative_computational_claim_scope_disclaimer_count_v ==
    fall_n::canonical_representative_computational_variational_slice_scope_disclaimer_count_v);
static_assert(
    fall_n::canonical_representative_computational_claim_evidence_count_v<
        fall_n::ComputationalClaimEvidenceBreadthKind::catalog_and_semantic_audit> == 1);
static_assert(
    fall_n::canonical_representative_computational_claim_evidence_count_v<
        fall_n::ComputationalClaimEvidenceBreadthKind::formulation_regression> == 1);
static_assert(
    fall_n::canonical_representative_computational_claim_evidence_count_v<
        fall_n::ComputationalClaimEvidenceBreadthKind::solver_slice_regression> == 3);
static_assert(
    fall_n::canonical_representative_computational_claim_evidence_count_v<
        fall_n::ComputationalClaimEvidenceBreadthKind::structural_response_regression> == 4);

} // namespace

int main()
{
    std::cout << "=== Computational Claim Trace Catalog Tests ===\n";

    report("claim_rows_align_with_variational_slices",
           claim_rows_align_with_variational_slices());
    report("reference_claims_are_not_overclaimed",
           reference_claims_are_not_overclaimed());
    report("structural_evidence_stays_on_reduced_structural_slices",
           structural_evidence_stays_on_reduced_structural_slices());
    report("arc_length_claim_remains_honest", arc_length_claim_remains_honest());
    report("nonphysical_baselines_are_linear_reference_slices",
           nonphysical_baselines_are_linear_reference_slices());
    report("residual_tangent_and_history_commitments_match_audit_scope",
           residual_tangent_and_history_commitments_match_audit_scope());
    report("arc_length_and_dynamic_claims_preserve_distinct_history_semantics",
           arc_length_and_dynamic_claims_preserve_distinct_history_semantics());

    std::cout << "\nSummary: " << passed << " passed, " << failed << " failed.\n";
    return failed == 0 ? 0 : 1;
}
