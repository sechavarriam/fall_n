#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <petsc.h>

#include "src/validation/ReducedRCColumnCyclicQuadratureSensitivityStudy.hh"

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
        .protocol_name = "reduced_cyclic_quadrature_sensitivity",
        .execution_profile_name = "default",
        .amplitudes_m = {1.25e-3, 2.50e-3},
        .steps_per_segment = 2,
        .max_steps = 0,
        .max_bisections = 4,
    };
}

fall_n::validation_reboot::ReducedRCColumnCyclicQuadratureSensitivityRunSpec
make_full_spec()
{
    using fall_n::validation_reboot::
        ReducedRCColumnCyclicQuadratureSensitivityRunSpec;
    using fall_n::validation_reboot::ReducedRCColumnContinuationKind;

    return ReducedRCColumnCyclicQuadratureSensitivityRunSpec{
        .structural_spec = {
            .axial_compression_force_mn = 0.02,
            .use_equilibrated_axial_preload_stage = true,
            .axial_preload_steps = 4,
            .continuation_kind =
                ReducedRCColumnContinuationKind::
                    reversal_guarded_incremental_displacement_control,
            .continuation_segment_substep_factor = 2,
            .write_hysteresis_csv = false,
            .write_section_response_csv = true,
            .print_progress = false,
        },
        .structural_protocol = make_protocol(),
        .reference_family = BeamAxisQuadratureFamily::GaussLegendre,
        .include_only_phase3_runtime_baseline = true,
        .write_case_outputs = true,
        .write_csv = true,
        .print_progress = false,
    };
}

fall_n::validation_reboot::ReducedRCColumnCyclicQuadratureSensitivityRunSpec
make_pilot_spec()
{
    auto spec = make_full_spec();
    spec.beam_nodes_filter = {2, 4, 10};
    spec.quadrature_filter = {
        BeamAxisQuadratureFamily::GaussLegendre,
        BeamAxisQuadratureFamily::GaussLobatto,
        BeamAxisQuadratureFamily::GaussRadauLeft,
        BeamAxisQuadratureFamily::GaussRadauRight,
    };
    return spec;
}

bool reduced_rc_column_cyclic_quadrature_sensitivity_study_completes_full_runtime_matrix()
{
    using fall_n::canonical_reduced_rc_column_phase3_baseline_case_count_v;
    using fall_n::validation_reboot::ReducedRCColumnContinuationKind;
    using fall_n::validation_reboot::
        run_reduced_rc_column_cyclic_quadrature_sensitivity_study;

    const auto result = run_reduced_rc_column_cyclic_quadrature_sensitivity_study(
        make_full_spec(),
        "data/output/cyclic_validation/reboot_cyclic_quadrature_sensitivity_study_full");

    if (result.empty() ||
        result.case_rows.size() !=
            canonical_reduced_rc_column_phase3_baseline_case_count_v ||
        result.summary.total_case_count !=
            canonical_reduced_rc_column_phase3_baseline_case_count_v ||
        result.summary.completed_case_count !=
            canonical_reduced_rc_column_phase3_baseline_case_count_v ||
        result.summary.failed_case_count != 0u ||
        result.node_rows.size() != 9u ||
        result.family_rows.size() != 4u ||
        result.reference_rows.size() != 9u) {
        return false;
    }

    for (const auto& row : result.case_rows) {
        if (!row.execution_ok ||
            row.continuation_kind !=
                ReducedRCColumnContinuationKind::
                    reversal_guarded_incremental_displacement_control ||
            row.continuation_segment_substep_factor != 2 ||
            row.case_id.empty() ||
            row.reference_case_id.empty() ||
            row.case_out_dir.empty() ||
            row.history_point_count < 2u ||
            row.turning_point_count < 2u ||
            !std::isfinite(row.abs_station_xi_shift) ||
            !std::isfinite(row.rel_terminal_return_moment_spread) ||
            !std::isfinite(row.max_rel_moment_history_spread) ||
            !std::isfinite(row.max_rel_tangent_history_spread) ||
            !std::isfinite(row.max_rel_secant_history_spread) ||
            !std::isfinite(row.max_rel_turning_point_moment_spread) ||
            !std::isfinite(row.max_rel_axial_force_history_spread)) {
            return false;
        }
    }

    for (const auto& row : result.node_rows) {
        if (row.case_count != 4u || row.completed_case_count != 4u) {
            return false;
        }
    }

    for (const auto& row : result.family_rows) {
        if (row.case_count != 9u || row.completed_case_count != 9u) {
            return false;
        }
    }

    for (const auto& row : result.reference_rows) {
        if (row.reference_family != BeamAxisQuadratureFamily::GaussLegendre ||
            row.continuation_kind !=
                ReducedRCColumnContinuationKind::
                    reversal_guarded_incremental_displacement_control ||
            row.continuation_segment_substep_factor != 2 ||
            row.reference_case_id.find("gauss_legendre") == std::string::npos ||
            row.compared_case_count != 4u ||
            !std::isfinite(row.reference_controlling_station_xi) ||
            !std::isfinite(row.reference_terminal_return_moment_y)) {
            return false;
        }
    }

    return std::isfinite(result.summary.worst_rel_terminal_return_moment_spread) &&
           std::isfinite(result.summary.worst_max_rel_moment_history_spread) &&
           std::isfinite(result.summary.worst_max_rel_tangent_history_spread) &&
           std::isfinite(result.summary.worst_max_rel_secant_history_spread) &&
           std::isfinite(result.summary.worst_max_rel_turning_point_moment_spread) &&
           std::isfinite(result.summary.worst_max_rel_axial_force_history_spread) &&
           std::isfinite(result.summary.worst_abs_station_xi_shift);
}

bool reduced_rc_column_cyclic_quadrature_sensitivity_study_writes_csv_contract()
{
    using fall_n::validation_reboot::
        run_reduced_rc_column_cyclic_quadrature_sensitivity_study;

    const auto out_dir = std::filesystem::path{
        "data/output/cyclic_validation/reboot_cyclic_quadrature_sensitivity_study_csv"};
    std::filesystem::remove_all(out_dir);

    const auto result = run_reduced_rc_column_cyclic_quadrature_sensitivity_study(
        make_pilot_spec(),
        out_dir.string());

    if (result.case_rows.size() != 12u ||
        result.reference_rows.size() != 3u ||
        result.node_rows.size() != 9u ||
        result.family_rows.size() != 4u) {
        return false;
    }

    const auto cases_csv =
        out_dir / "cyclic_quadrature_sensitivity_case_comparisons.csv";
    const auto node_csv =
        out_dir / "cyclic_quadrature_sensitivity_node_summary.csv";
    const auto family_csv =
        out_dir / "cyclic_quadrature_sensitivity_family_summary.csv";
    const auto refs_csv =
        out_dir / "cyclic_quadrature_sensitivity_reference_cases.csv";
    const auto overall_csv =
        out_dir / "cyclic_quadrature_sensitivity_overall_summary.csv";

    if (!std::filesystem::exists(cases_csv) ||
        !std::filesystem::exists(node_csv) ||
        !std::filesystem::exists(family_csv) ||
        !std::filesystem::exists(refs_csv) ||
        !std::filesystem::exists(overall_csv)) {
        return false;
    }

    std::ifstream cases_stream(cases_csv);
    std::ifstream node_stream(node_csv);
    std::ifstream family_stream(family_csv);
    std::ifstream refs_stream(refs_csv);
    std::ifstream overall_stream(overall_csv);

    std::string cases_header;
    std::string node_header;
    std::string family_header;
    std::string refs_header;
    std::string overall_header;
    std::getline(cases_stream, cases_header);
    std::getline(node_stream, node_header);
    std::getline(family_stream, family_header);
    std::getline(refs_stream, refs_header);
    std::getline(overall_stream, overall_header);

    return
        cases_header ==
            "case_id,reference_case_id,beam_nodes,beam_axis_quadrature_family,formulation_kind,continuation_kind,continuation_segment_substep_factor,execution_ok,history_point_count,turning_point_count,controlling_station_xi,reference_controlling_station_xi,abs_station_xi_shift,terminal_return_moment_y,reference_terminal_return_moment_y,rel_terminal_return_moment_spread,max_rel_moment_history_spread,rms_rel_moment_history_spread,max_rel_tangent_history_spread,max_rel_secant_history_spread,max_rel_turning_point_moment_spread,max_rel_axial_force_history_spread,terminal_return_within_representative_tolerance,moment_history_within_representative_tolerance,tangent_history_within_representative_tolerance,secant_history_within_representative_tolerance,turning_point_within_representative_tolerance,axial_force_history_within_representative_tolerance,representative_internal_cyclic_family_spread_passes,scope_label,error_message" &&
        node_header ==
            "beam_nodes,case_count,completed_case_count,representative_pass_count,min_rel_terminal_return_moment_spread,max_rel_terminal_return_moment_spread,avg_rel_terminal_return_moment_spread,min_max_rel_moment_history_spread,max_max_rel_moment_history_spread,avg_max_rel_moment_history_spread,min_max_rel_tangent_history_spread,max_max_rel_tangent_history_spread,avg_max_rel_tangent_history_spread,min_max_rel_secant_history_spread,max_max_rel_secant_history_spread,avg_max_rel_secant_history_spread,min_max_rel_turning_point_moment_spread,max_max_rel_turning_point_moment_spread,avg_max_rel_turning_point_moment_spread,min_max_rel_axial_force_history_spread,max_max_rel_axial_force_history_spread,avg_max_rel_axial_force_history_spread,max_abs_station_xi_shift" &&
        family_header ==
            "beam_axis_quadrature_family,case_count,completed_case_count,representative_pass_count,min_rel_terminal_return_moment_spread,max_rel_terminal_return_moment_spread,avg_rel_terminal_return_moment_spread,min_max_rel_moment_history_spread,max_max_rel_moment_history_spread,avg_max_rel_moment_history_spread,min_max_rel_tangent_history_spread,max_max_rel_tangent_history_spread,avg_max_rel_tangent_history_spread,min_max_rel_secant_history_spread,max_max_rel_secant_history_spread,avg_max_rel_secant_history_spread,min_max_rel_turning_point_moment_spread,max_max_rel_turning_point_moment_spread,avg_max_rel_turning_point_moment_spread,min_max_rel_axial_force_history_spread,max_max_rel_axial_force_history_spread,avg_max_rel_axial_force_history_spread,max_abs_station_xi_shift" &&
        refs_header ==
            "beam_nodes,reference_family,continuation_kind,continuation_segment_substep_factor,reference_case_id,compared_case_count,history_point_count,turning_point_count,reference_controlling_station_xi,reference_terminal_return_moment_y" &&
        overall_header ==
            "total_case_count,completed_case_count,failed_case_count,representative_pass_count,worst_rel_terminal_return_moment_spread,worst_terminal_return_case_id,worst_max_rel_moment_history_spread,worst_moment_history_case_id,worst_max_rel_tangent_history_spread,worst_tangent_history_case_id,worst_max_rel_secant_history_spread,worst_secant_history_case_id,worst_max_rel_turning_point_moment_spread,worst_turning_point_case_id,worst_max_rel_axial_force_history_spread,worst_axial_force_history_case_id,worst_abs_station_xi_shift,worst_station_shift_case_id,all_cases_completed,all_completed_cases_pass_representative_internal_cyclic_family_spread";
}

bool reduced_rc_column_cyclic_quadrature_sensitivity_study_uses_declared_reference_family()
{
    using fall_n::validation_reboot::
        run_reduced_rc_column_cyclic_quadrature_sensitivity_study;

    const auto result = run_reduced_rc_column_cyclic_quadrature_sensitivity_study(
        make_pilot_spec(),
        "data/output/cyclic_validation/reboot_cyclic_quadrature_sensitivity_study_reference");

    if (result.reference_rows.size() != 3u) {
        return false;
    }

    for (const auto& row : result.reference_rows) {
        if (row.reference_family != BeamAxisQuadratureFamily::GaussLegendre ||
            row.reference_case_id.find("gauss_legendre") == std::string::npos ||
            row.compared_case_count != 4u ||
            !std::isfinite(row.reference_controlling_station_xi) ||
            !std::isfinite(row.reference_terminal_return_moment_y)) {
            return false;
        }
    }

    std::size_t zero_spread_reference_cases = 0;
    for (const auto& row : result.case_rows) {
        if (row.beam_axis_quadrature_family !=
            BeamAxisQuadratureFamily::GaussLegendre) {
            continue;
        }

        if (!row.execution_ok ||
            row.reference_case_id != row.case_id ||
            row.abs_station_xi_shift > 1.0e-12 ||
            row.rel_terminal_return_moment_spread > 1.0e-12 ||
            row.max_rel_moment_history_spread > 1.0e-12 ||
            row.max_rel_tangent_history_spread > 1.0e-12 ||
            row.max_rel_secant_history_spread > 1.0e-12 ||
            row.max_rel_turning_point_moment_spread > 1.0e-12 ||
            row.max_rel_axial_force_history_spread > 1.0e-12) {
            return false;
        }

        ++zero_spread_reference_cases;
    }

    return zero_spread_reference_cases == 3u;
}

bool reduced_rc_column_cyclic_quadrature_sensitivity_study_produces_finite_family_spread_metrics()
{
    using fall_n::validation_reboot::
        run_reduced_rc_column_cyclic_quadrature_sensitivity_study;

    const auto result = run_reduced_rc_column_cyclic_quadrature_sensitivity_study(
        make_pilot_spec(),
        "data/output/cyclic_validation/reboot_cyclic_quadrature_sensitivity_study_finite");

    if (result.empty() ||
        result.summary.total_case_count != 12u ||
        result.summary.completed_case_count != 12u ||
        result.summary.failed_case_count != 0u) {
        return false;
    }

    bool saw_non_reference_family_spread = false;
    for (const auto& row : result.case_rows) {
        if (row.beam_axis_quadrature_family ==
            BeamAxisQuadratureFamily::GaussLegendre) {
            continue;
        }

        if (row.rel_terminal_return_moment_spread > 0.0 ||
            row.max_rel_moment_history_spread > 0.0 ||
            row.max_rel_tangent_history_spread > 0.0 ||
            row.max_rel_secant_history_spread > 0.0 ||
            row.max_rel_turning_point_moment_spread > 0.0 ||
            row.max_rel_axial_force_history_spread > 0.0 ||
            row.abs_station_xi_shift > 0.0) {
            saw_non_reference_family_spread = true;
            break;
        }
    }

    return saw_non_reference_family_spread &&
           std::isfinite(result.summary.worst_rel_terminal_return_moment_spread) &&
           std::isfinite(result.summary.worst_max_rel_moment_history_spread) &&
           std::isfinite(result.summary.worst_max_rel_tangent_history_spread) &&
           std::isfinite(result.summary.worst_max_rel_secant_history_spread) &&
           std::isfinite(result.summary.worst_max_rel_turning_point_moment_spread) &&
           std::isfinite(result.summary.worst_max_rel_axial_force_history_spread) &&
           std::isfinite(result.summary.worst_abs_station_xi_shift);
}

} // namespace

int main(int argc, char** argv)
{
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    std::cout
        << "=== Reduced RC Column Cyclic Quadrature Sensitivity Study Tests ===\n";

    report(
        "reduced_rc_column_cyclic_quadrature_sensitivity_study_completes_full_runtime_matrix",
        reduced_rc_column_cyclic_quadrature_sensitivity_study_completes_full_runtime_matrix());
    report(
        "reduced_rc_column_cyclic_quadrature_sensitivity_study_writes_csv_contract",
        reduced_rc_column_cyclic_quadrature_sensitivity_study_writes_csv_contract());
    report(
        "reduced_rc_column_cyclic_quadrature_sensitivity_study_uses_declared_reference_family",
        reduced_rc_column_cyclic_quadrature_sensitivity_study_uses_declared_reference_family());
    report(
        "reduced_rc_column_cyclic_quadrature_sensitivity_study_produces_finite_family_spread_metrics",
        reduced_rc_column_cyclic_quadrature_sensitivity_study_produces_finite_family_spread_metrics());

    std::cout << "\nSummary: " << passed << " passed, " << failed << " failed.\n";

    PetscFinalize();
    return failed == 0 ? 0 : 1;
}
