#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <petsc.h>

#include "src/validation/ReducedRCColumnCyclicNodeRefinementStudy.hh"

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

fall_n::table_cyclic_validation::CyclicValidationRunConfig
make_cyclic_node_refinement_protocol()
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
make_full_cyclic_node_refinement_spec()
{
    using fall_n::validation_reboot::ReducedRCColumnCyclicNodeRefinementRunSpec;

    return ReducedRCColumnCyclicNodeRefinementRunSpec{
        .structural_spec = {
            .axial_compression_force_mn = 0.02,
            .write_hysteresis_csv = false,
            .write_section_response_csv = true,
            .print_progress = false,
        },
        .structural_protocol = make_cyclic_node_refinement_protocol(),
        .include_only_phase3_runtime_baseline = true,
        .write_case_outputs = true,
        .write_csv = true,
        .print_progress = false,
    };
}

fall_n::validation_reboot::ReducedRCColumnCyclicNodeRefinementRunSpec
make_pilot_cyclic_node_refinement_spec()
{
    auto spec = make_full_cyclic_node_refinement_spec();
    spec.beam_nodes_filter = {2, 4, 10};
    spec.quadrature_filter = {
        BeamAxisQuadratureFamily::GaussLegendre,
        BeamAxisQuadratureFamily::GaussLobatto,
    };
    return spec;
}

fall_n::validation_reboot::ReducedRCColumnCyclicNodeRefinementRunSpec
make_guarded_pilot_cyclic_node_refinement_spec()
{
    auto spec = make_pilot_cyclic_node_refinement_spec();
    spec.structural_spec.continuation_kind =
        fall_n::validation_reboot::ReducedRCColumnContinuationKind::
            reversal_guarded_incremental_displacement_control;
    spec.structural_spec.continuation_segment_substep_factor = 2;
    return spec;
}

bool reduced_rc_column_cyclic_node_refinement_study_completes_full_runtime_matrix()
{
    using fall_n::canonical_reduced_rc_column_phase3_baseline_case_count_v;
    using fall_n::validation_reboot::run_reduced_rc_column_cyclic_node_refinement_study;

    const auto result = run_reduced_rc_column_cyclic_node_refinement_study(
        make_full_cyclic_node_refinement_spec(),
        "data/output/cyclic_validation/reboot_cyclic_node_refinement_study_full");

    if (result.empty() ||
        result.case_rows.size() !=
            canonical_reduced_rc_column_phase3_baseline_case_count_v ||
        result.summary.total_case_count !=
            canonical_reduced_rc_column_phase3_baseline_case_count_v ||
        result.summary.completed_case_count !=
            canonical_reduced_rc_column_phase3_baseline_case_count_v ||
        result.summary.failed_case_count != 0u ||
        result.summary_rows.size() != 9u ||
        result.reference_rows.size() != 4u) {
        return false;
    }

    for (const auto& row : result.case_rows) {
        if (!row.execution_ok ||
            row.continuation_kind !=
                fall_n::validation_reboot::ReducedRCColumnContinuationKind::
                    monolithic_incremental_displacement_control ||
            row.continuation_segment_substep_factor != 1 ||
            row.case_id.empty() ||
            row.reference_case_id.empty() ||
            row.case_out_dir.empty() ||
            row.history_point_count != 13u ||
            row.turning_point_count != 6u ||
            !std::isfinite(row.controlling_station_xi) ||
            !std::isfinite(row.reference_controlling_station_xi) ||
            !std::isfinite(row.abs_station_xi_shift) ||
            !std::isfinite(row.rel_terminal_return_moment_drift) ||
            !std::isfinite(row.max_rel_moment_history_drift) ||
            !std::isfinite(row.rms_rel_moment_history_drift) ||
            !std::isfinite(row.max_rel_tangent_history_drift) ||
            !std::isfinite(row.max_rel_secant_history_drift) ||
            !std::isfinite(row.max_rel_turning_point_moment_drift) ||
            !std::isfinite(row.max_rel_axial_force_history_drift)) {
            return false;
        }
    }

    for (const auto& row : result.summary_rows) {
        if (row.case_count != 4u || row.completed_case_count != 4u) {
            return false;
        }
    }

    for (const auto& row : result.reference_rows) {
        if (row.reference_beam_nodes != 10u ||
            row.reference_case_id.find("n10_") != 0 ||
            row.compared_case_count != 9u ||
            row.history_point_count != 13u ||
            row.turning_point_count != 6u ||
            !std::isfinite(row.reference_controlling_station_xi) ||
            !std::isfinite(row.reference_terminal_return_moment_y)) {
            return false;
        }
    }

    return std::isfinite(result.summary.worst_rel_terminal_return_moment_drift) &&
           std::isfinite(result.summary.worst_max_rel_moment_history_drift) &&
           std::isfinite(result.summary.worst_max_rel_tangent_history_drift) &&
           std::isfinite(result.summary.worst_max_rel_secant_history_drift) &&
           std::isfinite(result.summary.worst_max_rel_turning_point_moment_drift) &&
           std::isfinite(result.summary.worst_max_rel_axial_force_history_drift) &&
           std::isfinite(result.summary.worst_abs_station_xi_shift);
}

bool reduced_rc_column_cyclic_node_refinement_study_completes_pilot_frontier()
{
    using fall_n::validation_reboot::run_reduced_rc_column_cyclic_node_refinement_study;

    const auto result = run_reduced_rc_column_cyclic_node_refinement_study(
        make_pilot_cyclic_node_refinement_spec(),
        "data/output/cyclic_validation/reboot_cyclic_node_refinement_study_pilot");

    if (result.empty() ||
        result.case_rows.size() != 6u ||
        result.summary.total_case_count != 6u ||
        result.summary.completed_case_count != 6u ||
        result.summary.failed_case_count != 0u ||
        result.summary_rows.size() != 9u ||
        result.reference_rows.size() != 2u) {
        return false;
    }

    for (const auto& row : result.case_rows) {
        if (!row.execution_ok ||
            row.case_id.empty() ||
            row.reference_case_id.empty() ||
            row.case_out_dir.empty() ||
            row.continuation_kind !=
                fall_n::validation_reboot::ReducedRCColumnContinuationKind::
                    monolithic_incremental_displacement_control ||
            row.continuation_segment_substep_factor != 1 ||
            row.history_point_count != 13u ||
            row.turning_point_count != 6u ||
            !std::isfinite(row.controlling_station_xi) ||
            !std::isfinite(row.reference_controlling_station_xi) ||
            !std::isfinite(row.abs_station_xi_shift) ||
            !std::isfinite(row.rel_terminal_return_moment_drift) ||
            !std::isfinite(row.max_rel_moment_history_drift) ||
            !std::isfinite(row.rms_rel_moment_history_drift) ||
            !std::isfinite(row.max_rel_tangent_history_drift) ||
            !std::isfinite(row.max_rel_secant_history_drift) ||
            !std::isfinite(row.max_rel_turning_point_moment_drift) ||
            !std::isfinite(row.max_rel_axial_force_history_drift)) {
            return false;
        }
    }

    for (const auto& row : result.summary_rows) {
        if (row.case_count > 0u && row.case_count != 2u) {
            return false;
        }
    }

    for (const auto& row : result.reference_rows) {
        if (row.reference_beam_nodes != 10u ||
            row.reference_case_id.find("n10_") != 0 ||
            row.compared_case_count != 3u ||
            row.history_point_count != 13u ||
            row.turning_point_count != 6u ||
            !std::isfinite(row.reference_controlling_station_xi) ||
            !std::isfinite(row.reference_terminal_return_moment_y)) {
            return false;
        }
    }

    return std::isfinite(result.summary.worst_rel_terminal_return_moment_drift) &&
           std::isfinite(result.summary.worst_max_rel_moment_history_drift) &&
           std::isfinite(result.summary.worst_max_rel_tangent_history_drift) &&
           std::isfinite(result.summary.worst_max_rel_secant_history_drift) &&
           std::isfinite(result.summary.worst_max_rel_turning_point_moment_drift) &&
           std::isfinite(result.summary.worst_max_rel_axial_force_history_drift) &&
           std::isfinite(result.summary.worst_abs_station_xi_shift);
}

bool reduced_rc_column_cyclic_node_refinement_study_writes_csv_contract()
{
    using fall_n::validation_reboot::run_reduced_rc_column_cyclic_node_refinement_study;

    const auto out_dir = std::filesystem::path{
        "data/output/cyclic_validation/reboot_cyclic_node_refinement_study_csv"};
    std::filesystem::remove_all(out_dir);

    const auto result = run_reduced_rc_column_cyclic_node_refinement_study(
        make_pilot_cyclic_node_refinement_spec(),
        out_dir.string());

    if (result.case_rows.size() != 6u || result.reference_rows.size() != 2u) {
        return false;
    }

    const auto cases_csv = out_dir / "cyclic_node_refinement_case_comparisons.csv";
    const auto summary_csv = out_dir / "cyclic_node_refinement_summary.csv";
    const auto refs_csv = out_dir / "cyclic_node_refinement_reference_cases.csv";
    const auto overall_csv = out_dir / "cyclic_node_refinement_overall_summary.csv";

    if (!std::filesystem::exists(cases_csv) ||
        !std::filesystem::exists(summary_csv) ||
        !std::filesystem::exists(refs_csv) ||
        !std::filesystem::exists(overall_csv)) {
        return false;
    }

    std::ifstream cases_stream(cases_csv);
    std::ifstream summary_stream(summary_csv);
    std::ifstream refs_stream(refs_csv);
    std::ifstream overall_stream(overall_csv);

    std::string cases_header;
    std::string summary_header;
    std::string refs_header;
    std::string overall_header;
    std::getline(cases_stream, cases_header);
    std::getline(summary_stream, summary_header);
    std::getline(refs_stream, refs_header);
    std::getline(overall_stream, overall_header);

    return
        cases_header ==
            "case_id,reference_case_id,beam_nodes,beam_axis_quadrature_family,formulation_kind,continuation_kind,continuation_segment_substep_factor,execution_ok,history_point_count,turning_point_count,controlling_station_xi,reference_controlling_station_xi,abs_station_xi_shift,terminal_return_moment_y,reference_terminal_return_moment_y,rel_terminal_return_moment_drift,max_rel_moment_history_drift,rms_rel_moment_history_drift,max_rel_tangent_history_drift,max_rel_secant_history_drift,max_rel_turning_point_moment_drift,max_rel_axial_force_history_drift,terminal_return_within_representative_tolerance,moment_history_within_representative_tolerance,tangent_history_within_representative_tolerance,secant_history_within_representative_tolerance,turning_point_within_representative_tolerance,axial_force_history_within_representative_tolerance,representative_internal_cyclic_refinement_passes,scope_label,error_message" &&
        summary_header ==
            "beam_nodes,case_count,completed_case_count,representative_pass_count,min_rel_terminal_return_moment_drift,max_rel_terminal_return_moment_drift,avg_rel_terminal_return_moment_drift,min_max_rel_moment_history_drift,max_max_rel_moment_history_drift,avg_max_rel_moment_history_drift,min_max_rel_tangent_history_drift,max_max_rel_tangent_history_drift,avg_max_rel_tangent_history_drift,min_max_rel_secant_history_drift,max_max_rel_secant_history_drift,avg_max_rel_secant_history_drift,min_max_rel_turning_point_moment_drift,max_max_rel_turning_point_moment_drift,avg_max_rel_turning_point_moment_drift,min_max_rel_axial_force_history_drift,max_max_rel_axial_force_history_drift,avg_max_rel_axial_force_history_drift,max_abs_station_xi_shift" &&
        refs_header ==
            "beam_axis_quadrature_family,reference_beam_nodes,reference_case_id,compared_case_count,history_point_count,turning_point_count,reference_controlling_station_xi,reference_terminal_return_moment_y" &&
        overall_header ==
            "total_case_count,completed_case_count,failed_case_count,representative_pass_count,worst_rel_terminal_return_moment_drift,worst_terminal_return_case_id,worst_max_rel_moment_history_drift,worst_moment_history_case_id,worst_max_rel_tangent_history_drift,worst_tangent_history_case_id,worst_max_rel_secant_history_drift,worst_secant_history_case_id,worst_max_rel_turning_point_moment_drift,worst_turning_point_case_id,worst_max_rel_axial_force_history_drift,worst_axial_force_history_case_id,worst_abs_station_xi_shift,worst_station_shift_case_id,all_cases_completed,all_completed_cases_pass_representative_internal_cyclic_refinement";
}

bool reduced_rc_column_cyclic_node_refinement_study_uses_highest_n_reference_per_family()
{
    using fall_n::validation_reboot::run_reduced_rc_column_cyclic_node_refinement_study;

    const auto result = run_reduced_rc_column_cyclic_node_refinement_study(
        make_pilot_cyclic_node_refinement_spec(),
        "data/output/cyclic_validation/reboot_cyclic_node_refinement_study_reference");

    if (result.reference_rows.size() != 2u) {
        return false;
    }

    std::size_t zero_drift_reference_cases = 0;
    for (const auto& row : result.case_rows) {
        if (row.beam_nodes != 10u) {
            continue;
        }

        if (!row.execution_ok ||
            row.reference_case_id != row.case_id ||
            row.abs_station_xi_shift > 1.0e-12 ||
            row.rel_terminal_return_moment_drift > 1.0e-12 ||
            row.max_rel_moment_history_drift > 1.0e-12 ||
            row.max_rel_tangent_history_drift > 1.0e-12 ||
            row.max_rel_secant_history_drift > 1.0e-12 ||
            row.max_rel_turning_point_moment_drift > 1.0e-12 ||
            row.max_rel_axial_force_history_drift > 1.0e-12) {
            return false;
        }

        ++zero_drift_reference_cases;
    }

    return zero_drift_reference_cases == 2u;
}

bool reduced_rc_column_cyclic_node_refinement_study_produces_finite_history_metrics()
{
    using fall_n::validation_reboot::run_reduced_rc_column_cyclic_node_refinement_study;

    const auto result = run_reduced_rc_column_cyclic_node_refinement_study(
        make_pilot_cyclic_node_refinement_spec(),
        "data/output/cyclic_validation/reboot_cyclic_node_refinement_study_finite");

    if (result.empty() ||
        result.summary.total_case_count != 6u ||
        result.summary.completed_case_count != 6u ||
        result.summary.failed_case_count != 0u) {
        return false;
    }

    bool saw_nonreference_case = false;
    for (const auto& row : result.case_rows) {
        if (!row.execution_ok ||
            row.history_point_count < 2u ||
            row.turning_point_count == 0u ||
            !std::isfinite(row.terminal_return_moment_y) ||
            !std::isfinite(row.reference_terminal_return_moment_y) ||
            !std::isfinite(row.rel_terminal_return_moment_drift) ||
            !std::isfinite(row.max_rel_moment_history_drift) ||
            !std::isfinite(row.rms_rel_moment_history_drift) ||
            !std::isfinite(row.max_rel_tangent_history_drift) ||
            !std::isfinite(row.max_rel_secant_history_drift) ||
            !std::isfinite(row.max_rel_turning_point_moment_drift) ||
            !std::isfinite(row.max_rel_axial_force_history_drift)) {
            return false;
        }

        if (row.beam_nodes != 10u) {
            saw_nonreference_case = true;
        }
    }

    if (!saw_nonreference_case ||
        !std::isfinite(result.summary.worst_rel_terminal_return_moment_drift) ||
        !std::isfinite(result.summary.worst_max_rel_moment_history_drift) ||
        !std::isfinite(result.summary.worst_max_rel_tangent_history_drift) ||
        !std::isfinite(result.summary.worst_max_rel_secant_history_drift) ||
        !std::isfinite(result.summary.worst_max_rel_turning_point_moment_drift) ||
        !std::isfinite(result.summary.worst_max_rel_axial_force_history_drift) ||
        !std::isfinite(result.summary.worst_abs_station_xi_shift)) {
        return false;
    }

    const auto active_summary_rows = std::count_if(
        result.summary_rows.begin(),
        result.summary_rows.end(),
        [](const auto& row) { return row.case_count > 0u; });

    return active_summary_rows == 3u;
}

bool reduced_rc_column_cyclic_node_refinement_study_supports_reversal_guarded_continuation()
{
    using fall_n::validation_reboot::ReducedRCColumnContinuationKind;
    using fall_n::validation_reboot::run_reduced_rc_column_cyclic_node_refinement_study;

    const auto result = run_reduced_rc_column_cyclic_node_refinement_study(
        make_guarded_pilot_cyclic_node_refinement_spec(),
        "data/output/cyclic_validation/reboot_cyclic_node_refinement_study_guarded");

    if (result.case_rows.size() != 6u ||
        result.summary.total_case_count != 6u ||
        result.summary.completed_case_count != 6u ||
        result.summary.failed_case_count != 0u) {
        return false;
    }

    for (const auto& row : result.case_rows) {
        if (!row.execution_ok ||
            row.continuation_kind !=
                ReducedRCColumnContinuationKind::
                    reversal_guarded_incremental_displacement_control ||
            row.continuation_segment_substep_factor != 2 ||
            row.history_point_count != 25u ||
            row.turning_point_count != 6u ||
            !std::isfinite(row.rel_terminal_return_moment_drift) ||
            !std::isfinite(row.max_rel_moment_history_drift) ||
            !std::isfinite(row.max_rel_tangent_history_drift) ||
            !std::isfinite(row.max_rel_turning_point_moment_drift)) {
            return false;
        }
    }

    return true;
}

} // namespace

int main(int argc, char** argv)
{
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    std::cout << "=== Reduced RC Column Cyclic Node-Refinement Study Tests ===\n";

    report(
        "reduced_rc_column_cyclic_node_refinement_study_completes_full_runtime_matrix",
        reduced_rc_column_cyclic_node_refinement_study_completes_full_runtime_matrix());
    report(
        "reduced_rc_column_cyclic_node_refinement_study_completes_pilot_frontier",
        reduced_rc_column_cyclic_node_refinement_study_completes_pilot_frontier());
    report(
        "reduced_rc_column_cyclic_node_refinement_study_writes_csv_contract",
        reduced_rc_column_cyclic_node_refinement_study_writes_csv_contract());
    report(
        "reduced_rc_column_cyclic_node_refinement_study_uses_highest_n_reference_per_family",
        reduced_rc_column_cyclic_node_refinement_study_uses_highest_n_reference_per_family());
    report(
        "reduced_rc_column_cyclic_node_refinement_study_produces_finite_history_metrics",
        reduced_rc_column_cyclic_node_refinement_study_produces_finite_history_metrics());
    report(
        "reduced_rc_column_cyclic_node_refinement_study_supports_reversal_guarded_continuation",
        reduced_rc_column_cyclic_node_refinement_study_supports_reversal_guarded_continuation());

    std::cout << "\nSummary: " << passed << " passed, " << failed << " failed.\n";

    PetscFinalize();
    return failed == 0 ? 0 : 1;
}
