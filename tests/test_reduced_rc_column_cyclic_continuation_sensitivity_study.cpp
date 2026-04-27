#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <petsc.h>

#include "src/validation/ReducedRCColumnCyclicContinuationSensitivityStudy.hh"

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

fall_n::table_cyclic_validation::CyclicValidationRunConfig make_protocol()
{
    return {
        .protocol_name = "reduced_node_refinement_cyclic",
        .execution_profile_name = "default",
        .amplitudes_m = {1.25e-3, 2.50e-3},
        .steps_per_segment = 2,
        .max_steps = 0,
        .max_bisections = 4,
    };
}

fall_n::validation_reboot::ReducedRCColumnCyclicNodeRefinementRunSpec
make_full_baseline_spec()
{
    using fall_n::validation_reboot::ReducedRCColumnCyclicNodeRefinementRunSpec;

    return ReducedRCColumnCyclicNodeRefinementRunSpec{
        .structural_spec = {
            .axial_compression_force_mn = 0.02,
            .write_hysteresis_csv = false,
            .write_section_response_csv = true,
            .print_progress = false,
        },
        .structural_protocol = make_protocol(),
        .include_only_phase3_runtime_baseline = true,
        .write_case_outputs = true,
        .write_csv = true,
        .print_progress = false,
    };
}

fall_n::validation_reboot::ReducedRCColumnCyclicNodeRefinementRunSpec
make_full_guarded_spec()
{
    auto spec = make_full_baseline_spec();
    spec.structural_spec.continuation_kind =
        fall_n::validation_reboot::ReducedRCColumnContinuationKind::
            reversal_guarded_incremental_displacement_control;
    spec.structural_spec.continuation_segment_substep_factor = 2;
    return spec;
}

fall_n::validation_reboot::ReducedRCColumnCyclicContinuationSensitivityRunSpec
make_full_continuation_sensitivity_spec()
{
    using fall_n::validation_reboot::
        ReducedRCColumnCyclicContinuationSensitivityRunSpec;

    return ReducedRCColumnCyclicContinuationSensitivityRunSpec{
        .baseline_spec = make_full_baseline_spec(),
        .candidate_spec = make_full_guarded_spec(),
        .write_csv = true,
        .print_progress = false,
    };
}

bool reduced_rc_column_cyclic_continuation_sensitivity_study_compares_full_runtime_matrix()
{
    using fall_n::canonical_reduced_rc_column_phase3_baseline_case_count_v;
    using fall_n::validation_reboot::ReducedRCColumnContinuationKind;
    using fall_n::validation_reboot::
        run_reduced_rc_column_cyclic_continuation_sensitivity_study;

    const auto result = run_reduced_rc_column_cyclic_continuation_sensitivity_study(
        make_full_continuation_sensitivity_spec(),
        "data/output/cyclic_validation/reboot_cyclic_continuation_sensitivity_study_full");

    if (result.empty() ||
        result.case_rows.size() !=
            canonical_reduced_rc_column_phase3_baseline_case_count_v ||
        result.summary_rows.size() != 9u ||
        result.summary.total_case_count !=
            canonical_reduced_rc_column_phase3_baseline_case_count_v ||
        result.summary.compared_case_count !=
            canonical_reduced_rc_column_phase3_baseline_case_count_v ||
        result.summary.baseline_completed_case_count !=
            canonical_reduced_rc_column_phase3_baseline_case_count_v ||
        result.summary.candidate_completed_case_count !=
            canonical_reduced_rc_column_phase3_baseline_case_count_v ||
        result.summary.baseline_representative_pass_count != 8u ||
        result.summary.candidate_representative_pass_count != 8u ||
        result.summary.candidate_additional_representative_pass_count != 0u ||
        result.summary.candidate_lost_representative_pass_count != 0u ||
        result.summary.candidate_improves_terminal_return_count != 16u ||
        result.summary.candidate_improves_moment_history_count != 19u ||
        result.summary.candidate_improves_tangent_history_count != 3u ||
        result.summary.candidate_improves_secant_history_count != 15u ||
        result.summary.candidate_improves_turning_point_count != 19u ||
        result.summary.candidate_improves_axial_force_count != 32u ||
        result.summary.candidate_improves_station_shift_count != 0u ||
        !result.summary.baseline_and_candidate_all_cases_completed() ||
        result.baseline_result.case_rows.size() != result.case_rows.size() ||
        result.candidate_result.case_rows.size() != result.case_rows.size()) {
        return false;
    }

    for (const auto& row : result.case_rows) {
        if (row.case_id.empty() ||
            !row.baseline_execution_ok ||
            !row.candidate_execution_ok ||
            row.baseline_continuation_kind !=
                ReducedRCColumnContinuationKind::
                    monolithic_incremental_displacement_control ||
            row.baseline_continuation_segment_substep_factor != 1 ||
            row.candidate_continuation_kind !=
                ReducedRCColumnContinuationKind::
                    reversal_guarded_incremental_displacement_control ||
            row.candidate_continuation_segment_substep_factor != 2 ||
            row.baseline_history_point_count != 13u ||
            row.baseline_turning_point_count != 6u ||
            row.candidate_history_point_count != 25u ||
            row.candidate_turning_point_count != 6u ||
            !std::isfinite(row.delta_rel_terminal_return_moment_drift) ||
            !std::isfinite(row.delta_max_rel_moment_history_drift) ||
            !std::isfinite(row.delta_max_rel_tangent_history_drift) ||
            !std::isfinite(row.delta_max_rel_secant_history_drift) ||
            !std::isfinite(row.delta_max_rel_turning_point_moment_drift) ||
            !std::isfinite(row.delta_max_rel_axial_force_history_drift) ||
            !std::isfinite(row.delta_abs_station_xi_shift)) {
            return false;
        }
    }

    return std::isfinite(result.summary.max_abs_delta_rel_terminal_return_moment_drift) &&
           std::isfinite(result.summary.max_abs_delta_max_rel_moment_history_drift) &&
           std::isfinite(result.summary.max_abs_delta_max_rel_tangent_history_drift) &&
           std::isfinite(result.summary.max_abs_delta_max_rel_secant_history_drift) &&
           std::isfinite(result.summary.max_abs_delta_max_rel_turning_point_moment_drift) &&
           std::isfinite(result.summary.max_abs_delta_max_rel_axial_force_history_drift) &&
           std::isfinite(result.summary.max_abs_delta_abs_station_xi_shift) &&
           result.summary.max_abs_delta_abs_station_xi_shift == 0.0;
}

bool reduced_rc_column_cyclic_continuation_sensitivity_study_writes_csv_contract()
{
    using fall_n::validation_reboot::
        run_reduced_rc_column_cyclic_continuation_sensitivity_study;

    const auto out_dir = std::filesystem::path{
        "data/output/cyclic_validation/reboot_cyclic_continuation_sensitivity_study_csv"};
    std::filesystem::remove_all(out_dir);

    const auto result = run_reduced_rc_column_cyclic_continuation_sensitivity_study(
        make_full_continuation_sensitivity_spec(),
        out_dir.string());

    if (result.case_rows.size() != 36u || result.summary_rows.size() != 9u) {
        return false;
    }

    const auto cases_csv =
        out_dir / "cyclic_continuation_sensitivity_case_comparisons.csv";
    const auto summary_csv =
        out_dir / "cyclic_continuation_sensitivity_summary.csv";
    const auto overall_csv =
        out_dir / "cyclic_continuation_sensitivity_overall_summary.csv";

    if (!std::filesystem::exists(cases_csv) ||
        !std::filesystem::exists(summary_csv) ||
        !std::filesystem::exists(overall_csv)) {
        return false;
    }

    std::ifstream cases_stream(cases_csv);
    std::ifstream summary_stream(summary_csv);
    std::ifstream overall_stream(overall_csv);

    std::string cases_header;
    std::string summary_header;
    std::string overall_header;
    std::getline(cases_stream, cases_header);
    std::getline(summary_stream, summary_header);
    std::getline(overall_stream, overall_header);

    return
        cases_header ==
            "case_id,beam_nodes,beam_axis_quadrature_family,formulation_kind,baseline_continuation_kind,baseline_continuation_segment_substep_factor,baseline_execution_ok,baseline_representative_internal_cyclic_refinement_passes,baseline_history_point_count,baseline_turning_point_count,baseline_rel_terminal_return_moment_drift,baseline_max_rel_moment_history_drift,baseline_max_rel_tangent_history_drift,baseline_max_rel_secant_history_drift,baseline_max_rel_turning_point_moment_drift,baseline_max_rel_axial_force_history_drift,baseline_abs_station_xi_shift,candidate_continuation_kind,candidate_continuation_segment_substep_factor,candidate_execution_ok,candidate_representative_internal_cyclic_refinement_passes,candidate_history_point_count,candidate_turning_point_count,candidate_rel_terminal_return_moment_drift,candidate_max_rel_moment_history_drift,candidate_max_rel_tangent_history_drift,candidate_max_rel_secant_history_drift,candidate_max_rel_turning_point_moment_drift,candidate_max_rel_axial_force_history_drift,candidate_abs_station_xi_shift,delta_rel_terminal_return_moment_drift,delta_max_rel_moment_history_drift,delta_max_rel_tangent_history_drift,delta_max_rel_secant_history_drift,delta_max_rel_turning_point_moment_drift,delta_max_rel_axial_force_history_drift,delta_abs_station_xi_shift,candidate_improves_terminal_return_drift,candidate_improves_moment_history_drift,candidate_improves_tangent_history_drift,candidate_improves_secant_history_drift,candidate_improves_turning_point_drift,candidate_improves_axial_force_drift,candidate_improves_station_shift,scope_label,rationale_label" &&
        summary_header ==
            "beam_nodes,case_count,baseline_completed_case_count,candidate_completed_case_count,baseline_representative_pass_count,candidate_representative_pass_count,candidate_additional_representative_pass_count,candidate_lost_representative_pass_count,candidate_improves_terminal_return_count,candidate_improves_moment_history_count,candidate_improves_tangent_history_count,candidate_improves_secant_history_count,candidate_improves_turning_point_count,candidate_improves_axial_force_count,candidate_improves_station_shift_count,max_abs_delta_rel_terminal_return_moment_drift,max_abs_delta_max_rel_moment_history_drift,max_abs_delta_max_rel_tangent_history_drift,max_abs_delta_max_rel_secant_history_drift,max_abs_delta_max_rel_turning_point_moment_drift,max_abs_delta_max_rel_axial_force_history_drift,max_abs_delta_abs_station_xi_shift" &&
        overall_header ==
            "total_case_count,compared_case_count,baseline_completed_case_count,candidate_completed_case_count,baseline_representative_pass_count,candidate_representative_pass_count,candidate_additional_representative_pass_count,candidate_lost_representative_pass_count,candidate_improves_terminal_return_count,candidate_improves_moment_history_count,candidate_improves_tangent_history_count,candidate_improves_secant_history_count,candidate_improves_turning_point_count,candidate_improves_axial_force_count,candidate_improves_station_shift_count,max_abs_delta_rel_terminal_return_moment_drift,max_abs_delta_terminal_return_case_id,max_abs_delta_max_rel_moment_history_drift,max_abs_delta_moment_history_case_id,max_abs_delta_max_rel_tangent_history_drift,max_abs_delta_tangent_history_case_id,max_abs_delta_max_rel_secant_history_drift,max_abs_delta_secant_history_case_id,max_abs_delta_max_rel_turning_point_moment_drift,max_abs_delta_turning_point_case_id,max_abs_delta_max_rel_axial_force_history_drift,max_abs_delta_axial_force_case_id,max_abs_delta_abs_station_xi_shift,max_abs_delta_station_shift_case_id,baseline_and_candidate_all_cases_completed";
}

bool reduced_rc_column_cyclic_continuation_sensitivity_study_detects_real_policy_differences()
{
    using fall_n::validation_reboot::
        run_reduced_rc_column_cyclic_continuation_sensitivity_study;

    const auto result = run_reduced_rc_column_cyclic_continuation_sensitivity_study(
        make_full_continuation_sensitivity_spec(),
        "data/output/cyclic_validation/reboot_cyclic_continuation_sensitivity_study_deltas");

    if (result.empty()) {
        return false;
    }

    bool saw_history_refinement = false;
    for (const auto& row : result.case_rows) {
        if (row.candidate_history_point_count > row.baseline_history_point_count) {
            saw_history_refinement = true;
            break;
        }
    }

    return saw_history_refinement &&
           result.summary.max_abs_delta_rel_terminal_return_moment_drift > 0.0 &&
           result.summary.max_abs_delta_max_rel_moment_history_drift > 0.0 &&
           result.summary.max_abs_delta_max_rel_axial_force_history_drift > 0.0;
}

} // namespace

int main(int argc, char** argv)
{
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    std::cout
        << "=== Reduced RC Column Cyclic Continuation Sensitivity Study Tests ===\n";

    report(
        "reduced_rc_column_cyclic_continuation_sensitivity_study_compares_full_runtime_matrix",
        reduced_rc_column_cyclic_continuation_sensitivity_study_compares_full_runtime_matrix());
    report(
        "reduced_rc_column_cyclic_continuation_sensitivity_study_writes_csv_contract",
        reduced_rc_column_cyclic_continuation_sensitivity_study_writes_csv_contract());
    report(
        "reduced_rc_column_cyclic_continuation_sensitivity_study_detects_real_policy_differences",
        reduced_rc_column_cyclic_continuation_sensitivity_study_detects_real_policy_differences());

    std::cout << "\nSummary: " << passed << " passed, " << failed << " failed.\n";

    PetscFinalize();
    return failed == 0 ? 0 : 1;
}
