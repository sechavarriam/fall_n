#include <cstddef>
#include <iostream>

#include "src/validation/ValidationCampaignCatalog.hh"

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

constexpr auto plan = fall_n::canonical_validation_reboot_workstream_table_v;

constexpr bool governance_row_is_first_and_quarantines_legacy_only_after_replacement()
{
    const auto& row = plan.front();
    return row.row_label == "governance_reset_and_evidence_protocol" &&
           row.phase_kind ==
               fall_n::ValidationRebootPhaseKind::
                   phase0_governance_and_legacy_quarantine &&
           row.priority_kind ==
               fall_n::ValidationWorkstreamPriorityKind::mandatory_blocker &&
           row.legacy_surface_disposition ==
               fall_n::LegacyValidationSurfaceDispositionKind::
                   quarantine_to_old_when_replacement_exists &&
           row.blocks_any_stage() &&
           row.uses_legacy_surfaces_only_as_input;
}

constexpr bool reduced_order_column_precedes_continuum_column_and_both_block_escalation()
{
    std::size_t reduced_index = plan.size();
    std::size_t continuum_index = plan.size();

    for (std::size_t i = 0; i < plan.size(); ++i) {
        if (plan[i].row_label == "reduced_order_rc_column_matrix") {
            reduced_index = i;
        }
        if (plan[i].row_label == "continuum_rc_column_matrix") {
            continuum_index = i;
        }
    }

    if (reduced_index >= continuum_index || continuum_index == plan.size()) {
        return false;
    }

    return plan[reduced_index].required_for_reference_structural_column &&
           plan[reduced_index].required_for_reference_continuum_column &&
           plan[continuum_index].required_for_reference_continuum_column &&
           plan[continuum_index].required_for_full_structure_escalation;
}

constexpr bool conditional_enablers_do_not_block_the_first_column_campaign_by_default()
{
    bool found_rebar_extension = false;
    bool found_alt_concrete = false;

    for (const auto& row : plan) {
        if (row.priority_kind !=
            fall_n::ValidationWorkstreamPriorityKind::conditional_enabler) {
            continue;
        }

        if (row.row_label == "reinforcement_discretization_extension") {
            found_rebar_extension = true;
            if (row.required_for_reference_structural_column ||
                row.required_for_reference_continuum_column ||
                !row.requires_new_implementation) {
                return false;
            }
        } else if (row.row_label == "alternative_concrete_model_extension") {
            found_alt_concrete = true;
            if (row.required_for_reference_structural_column ||
                row.required_for_reference_continuum_column ||
                !row.requires_new_implementation) {
                return false;
            }
        } else {
            return false;
        }
    }

    return found_rebar_extension && found_alt_concrete;
}

constexpr bool force_based_path_is_growth_only_and_not_a_baseline_blocker()
{
    for (const auto& row : plan) {
        if (row.row_label != "force_based_structural_element_path") {
            continue;
        }

        return row.is_growth_only_path() &&
               !row.required_for_reference_structural_column &&
               !row.required_for_reference_continuum_column &&
               !row.required_for_full_structure_escalation &&
               row.requires_new_implementation;
    }

    return false;
}

constexpr bool geometrically_exact_path_is_growth_only_and_not_a_baseline_blocker()
{
    for (const auto& row : plan) {
        if (row.row_label != "geometrically_exact_beam_family_path") {
            continue;
        }

        return row.is_growth_only_path() &&
               !row.required_for_reference_structural_column &&
               !row.required_for_reference_continuum_column &&
               !row.required_for_full_structure_escalation &&
               row.requires_new_implementation;
    }

    return false;
}

constexpr bool gate_counts_match_the_reboot_strategy()
{
    return fall_n::canonical_validation_workstream_priority_count_v<
                   fall_n::ValidationWorkstreamPriorityKind::mandatory_blocker> ==
               8 &&
           fall_n::canonical_validation_workstream_priority_count_v<
                   fall_n::ValidationWorkstreamPriorityKind::conditional_enabler> ==
               2 &&
           fall_n::canonical_validation_workstream_priority_count_v<
                   fall_n::ValidationWorkstreamPriorityKind::deferred_growth_path> ==
               2 &&
           fall_n::canonical_validation_reference_structural_column_gate_count_v == 5 &&
           fall_n::canonical_validation_reference_continuum_column_gate_count_v == 7 &&
           fall_n::canonical_validation_full_structure_escalation_gate_count_v ==
               10;
}

static_assert(governance_row_is_first_and_quarantines_legacy_only_after_replacement());
static_assert(reduced_order_column_precedes_continuum_column_and_both_block_escalation());
static_assert(conditional_enablers_do_not_block_the_first_column_campaign_by_default());
static_assert(force_based_path_is_growth_only_and_not_a_baseline_blocker());
static_assert(geometrically_exact_path_is_growth_only_and_not_a_baseline_blocker());
static_assert(gate_counts_match_the_reboot_strategy());

} // namespace

int main()
{
    std::cout << "=== Validation Campaign Catalog Tests ===\n";

    report("governance_row_is_first_and_quarantines_legacy_only_after_replacement",
           governance_row_is_first_and_quarantines_legacy_only_after_replacement());
    report("reduced_order_column_precedes_continuum_column_and_both_block_escalation",
           reduced_order_column_precedes_continuum_column_and_both_block_escalation());
    report("conditional_enablers_do_not_block_the_first_column_campaign_by_default",
           conditional_enablers_do_not_block_the_first_column_campaign_by_default());
    report("force_based_path_is_growth_only_and_not_a_baseline_blocker",
           force_based_path_is_growth_only_and_not_a_baseline_blocker());
    report("geometrically_exact_path_is_growth_only_and_not_a_baseline_blocker",
           geometrically_exact_path_is_growth_only_and_not_a_baseline_blocker());
    report("gate_counts_match_the_reboot_strategy",
           gate_counts_match_the_reboot_strategy());

    std::cout << "\nSummary: " << passed << " passed, " << failed << " failed.\n";
    return failed == 0 ? 0 : 1;
}
