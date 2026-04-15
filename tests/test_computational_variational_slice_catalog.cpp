#include <cstddef>
#include <iostream>
#include <string_view>

#include "src/analysis/AnalysisRouteCatalog.hh"
#include "src/analysis/ComputationalModelSliceCatalog.hh"
#include "src/analysis/ComputationalSliceMatrixCatalog.hh"
#include "src/analysis/ComputationalVariationalSliceCatalog.hh"
#include "src/continuum/DiscreteVariationalSemantics.hh"

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

constexpr auto route_table =
    fall_n::canonical_representative_family_formulation_analysis_route_audit_table_v;
constexpr auto model_slice_table =
    fall_n::canonical_representative_model_solver_slice_audit_table_v;
constexpr auto slice_matrix =
    fall_n::canonical_representative_computational_slice_matrix_v;
constexpr auto variational_slice_matrix =
    fall_n::canonical_representative_computational_variational_slice_matrix_v;

constexpr bool same_identity_labels()
{
    if constexpr (slice_matrix.size() != variational_slice_matrix.size()) {
        return false;
    }

    for (std::size_t i = 0; i < slice_matrix.size(); ++i) {
        const auto& a = slice_matrix[i];
        const auto& b = variational_slice_matrix[i];
        if (a.family_label != b.family_label ||
            a.formulation_label != b.formulation_label ||
            a.route_label != b.route_label ||
            a.slice_label != b.slice_label ||
            a.model_label != b.model_label ||
            a.solver_label != b.solver_label) {
            return false;
        }
    }

    return true;
}

constexpr bool same_slice_support_levels()
{
    if constexpr (slice_matrix.size() != variational_slice_matrix.size()) {
        return false;
    }

    for (std::size_t i = 0; i < slice_matrix.size(); ++i) {
        if (slice_matrix[i].slice_support_level() !=
            variational_slice_matrix[i].slice_support_level()) {
            return false;
        }
    }

    return true;
}

constexpr bool same_scope_disclaimer_pattern()
{
    if constexpr (slice_matrix.size() != variational_slice_matrix.size()) {
        return false;
    }

    for (std::size_t i = 0; i < slice_matrix.size(); ++i) {
        if (slice_matrix[i].requires_scope_disclaimer() !=
            variational_slice_matrix[i].requires_scope_disclaimer()) {
            return false;
        }
    }

    return true;
}

constexpr bool discrete_semantics_match_family_formulation_rows()
{
    for (const auto& row : variational_slice_matrix) {
        const auto expected =
            continuum::canonical_family_formulation_discrete_variational_semantics(
                row.audit_scope.element_family,
                row.audit_scope.formulation_kind);
        const auto actual = row.audit_scope.discrete_variational_semantics;

        if (actual.element_family != expected.element_family ||
            actual.formulation_kind != expected.formulation_kind ||
            actual.kinematic_carrier != expected.kinematic_carrier ||
            actual.stress_carrier != expected.stress_carrier ||
            actual.integration_domain != expected.integration_domain ||
            actual.residual_construction != expected.residual_construction ||
            actual.tangent_composition != expected.tangent_composition ||
            actual.history_topology != expected.history_topology ||
            actual.algorithmic_tangent_source != expected.algorithmic_tangent_source ||
            actual.uses_structural_resultants != expected.uses_structural_resultants ||
            actual.admits_geometric_stiffness != expected.admits_geometric_stiffness ||
            actual.admits_effective_operator_injection !=
                expected.admits_effective_operator_injection ||
            actual.admits_history_variables != expected.admits_history_variables ||
            actual.requires_domain_reduction != expected.requires_domain_reduction) {
            return false;
        }
    }

    return true;
}

constexpr bool route_semantics_match_variational_slices()
{
    for (const auto& row : variational_slice_matrix) {
        switch (row.audit_scope.route_kind) {
            case fall_n::AnalysisRouteKind::linear_static:
                if (row.audit_scope.global_residual_operator !=
                        fall_n::GlobalResidualOperatorKind::static_equilibrium ||
                    row.audit_scope.global_tangent_operator !=
                        fall_n::GlobalTangentOperatorKind::linear_or_static_consistent_tangent ||
                    row.audit_scope.global_control_semantics !=
                        fall_n::GlobalControlSemanticsKind::direct_load_or_displacement_control ||
                    row.audit_scope.incremental_state_management !=
                        fall_n::IncrementalStateManagementKind::stateless_or_direct_response ||
                    row.audit_scope.augments_with_inertial_terms ||
                    row.audit_scope.augments_with_continuation_constraint) {
                    return false;
                }
                break;

            case fall_n::AnalysisRouteKind::nonlinear_incremental_newton:
                if (row.audit_scope.global_residual_operator !=
                        fall_n::GlobalResidualOperatorKind::incremental_static_equilibrium ||
                    row.audit_scope.global_tangent_operator !=
                        fall_n::GlobalTangentOperatorKind::monolithic_incremental_consistent_tangent ||
                    row.audit_scope.global_control_semantics !=
                        fall_n::GlobalControlSemanticsKind::incremental_load_or_displacement_control ||
                    row.audit_scope.incremental_state_management ==
                        fall_n::IncrementalStateManagementKind::stateless_or_direct_response ||
                    row.audit_scope.augments_with_inertial_terms ||
                    row.audit_scope.augments_with_continuation_constraint) {
                    return false;
                }
                break;

            case fall_n::AnalysisRouteKind::implicit_second_order_dynamics:
                if (row.audit_scope.global_residual_operator !=
                        fall_n::GlobalResidualOperatorKind::second_order_dynamic_equilibrium ||
                    row.audit_scope.global_tangent_operator !=
                        fall_n::GlobalTangentOperatorKind::effective_mass_damping_stiffness_tangent ||
                    row.audit_scope.global_control_semantics !=
                        fall_n::GlobalControlSemanticsKind::implicit_time_control ||
                    !row.audit_scope.augments_with_inertial_terms ||
                    row.audit_scope.augments_with_continuation_constraint) {
                    return false;
                }
                break;

            case fall_n::AnalysisRouteKind::arc_length_continuation:
                if (row.audit_scope.global_residual_operator !=
                        fall_n::GlobalResidualOperatorKind::arc_length_augmented_equilibrium ||
                    row.audit_scope.global_tangent_operator !=
                        fall_n::GlobalTangentOperatorKind::bordered_arc_length_tangent ||
                    row.audit_scope.global_control_semantics !=
                        fall_n::GlobalControlSemanticsKind::arc_length_constraint_control ||
                    row.audit_scope.incremental_state_management !=
                        fall_n::IncrementalStateManagementKind::continuation_step_commit ||
                    row.audit_scope.augments_with_inertial_terms ||
                    !row.audit_scope.augments_with_continuation_constraint) {
                    return false;
                }
                break;
        }
    }

    return true;
}

constexpr bool family_topology_matches_reduction_kind()
{
    for (const auto& row : variational_slice_matrix) {
        switch (row.audit_scope.element_family) {
            case continuum::ElementFamilyKind::continuum_solid_3d:
                if (row.audit_scope.is_structural_reduction_path() ||
                    row.audit_scope.discrete_variational_semantics
                            .uses_structural_resultants) {
                    return false;
                }
                break;

            case continuum::ElementFamilyKind::beam_1d:
            case continuum::ElementFamilyKind::shell_2d:
                if (!row.audit_scope.is_structural_reduction_path() ||
                    !row.audit_scope.discrete_variational_semantics
                         .uses_structural_resultants ||
                    !row.audit_scope.discrete_variational_semantics
                         .requires_domain_reduction) {
                    return false;
                }
                break;
        }
    }

    return true;
}

constexpr bool operator_injection_scope_is_honest()
{
    bool found_predictor_injection = false;

    for (const auto& row : variational_slice_matrix) {
        if (row.audit_scope.admits_effective_operator_predictor_injection) {
            found_predictor_injection = true;
            if (row.audit_scope.element_family != continuum::ElementFamilyKind::beam_1d ||
                row.audit_scope.route_kind !=
                    fall_n::AnalysisRouteKind::nonlinear_incremental_newton ||
                !row.audit_scope.discrete_variational_semantics
                     .admits_effective_operator_injection) {
                return false;
            }
        }
    }

    return found_predictor_injection;
}

static_assert(route_table.size() == slice_matrix.size());
static_assert(model_slice_table.size() == slice_matrix.size());
static_assert(slice_matrix.size() == variational_slice_matrix.size());
static_assert(same_identity_labels());
static_assert(same_slice_support_levels());
static_assert(same_scope_disclaimer_pattern());
static_assert(discrete_semantics_match_family_formulation_rows());
static_assert(route_semantics_match_variational_slices());
static_assert(family_topology_matches_reduction_kind());
static_assert(operator_injection_scope_is_honest());
static_assert(
    fall_n::canonical_representative_computational_variational_slice_scope_disclaimer_count_v ==
    fall_n::canonical_representative_computational_slice_matrix_scope_disclaimer_count_v);
static_assert(
    fall_n::canonical_representative_structural_reduction_variational_slice_count_v == 4);
static_assert(
    fall_n::canonical_representative_effective_operator_predictor_variational_slice_count_v == 1);

} // namespace

int main()
{
    std::cout << "=== Computational Variational Slice Catalog Tests ===\n";

    report("identity_labels_align", same_identity_labels());
    report("slice_support_levels_align", same_slice_support_levels());
    report("scope_disclaimer_pattern_align", same_scope_disclaimer_pattern());
    report("discrete_semantics_match_family_formulation_rows",
           discrete_semantics_match_family_formulation_rows());
    report("route_semantics_match_variational_slices",
           route_semantics_match_variational_slices());
    report("family_topology_matches_reduction_kind",
           family_topology_matches_reduction_kind());
    report("operator_injection_scope_is_honest",
           operator_injection_scope_is_honest());

    std::cout << "\nPassed: " << passed << "  Failed: " << failed << "\n";
    return failed == 0 ? 0 : 1;
}
