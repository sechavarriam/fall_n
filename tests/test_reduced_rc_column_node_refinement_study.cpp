#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <petsc.h>

#include "src/validation/ReducedRCColumnNodeRefinementStudy.hh"

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
make_monotonic_node_refinement_protocol()
{
    return {
        .protocol_name = "reduced_node_refinement",
        .execution_profile_name = "default",
        .amplitudes_m = {2.5e-3},
        .steps_per_segment = 4,
        .max_steps = 0,
        .max_bisections = 4,
    };
}

fall_n::validation_reboot::ReducedRCColumnNodeRefinementRunSpec
make_full_node_refinement_spec()
{
    using fall_n::validation_reboot::ReducedRCColumnNodeRefinementRunSpec;

    return ReducedRCColumnNodeRefinementRunSpec{
        .closure_spec = {
            .structural_spec = {
                .axial_compression_force_mn = 0.02,
                .write_hysteresis_csv = false,
                .write_section_response_csv = true,
                .print_progress = false,
            },
            .section_spec = {
                .steps = 12,
                .write_csv = true,
                .print_progress = false,
            },
            .structural_protocol = make_monotonic_node_refinement_protocol(),
            .write_closure_csv = true,
            .print_progress = false,
        },
        .include_only_phase3_runtime_baseline = true,
        .write_case_outputs = true,
        .write_csv = true,
        .print_progress = false,
    };
}

fall_n::validation_reboot::ReducedRCColumnNodeRefinementRunSpec
make_pilot_node_refinement_spec()
{
    using fall_n::validation_reboot::ReducedRCColumnNodeRefinementRunSpec;

    auto spec = make_full_node_refinement_spec();
    spec.beam_nodes_filter = {2, 4, 10};
    spec.quadrature_filter = {
        BeamAxisQuadratureFamily::GaussLegendre,
        BeamAxisQuadratureFamily::GaussLobatto,
    };
    return spec;
}

bool reduced_rc_column_node_refinement_study_completes_runtime_frontier()
{
    using fall_n::canonical_reduced_rc_column_phase3_baseline_case_count_v;
    using fall_n::validation_reboot::run_reduced_rc_column_node_refinement_study;

    const auto result = run_reduced_rc_column_node_refinement_study(
        make_full_node_refinement_spec(),
        "data/output/cyclic_validation/reboot_node_refinement_study_full");

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
            row.case_id.empty() ||
            row.reference_case_id.empty() ||
            row.case_out_dir.empty() ||
            row.positive_branch_point_count < 2u ||
            row.overlap_point_count < 2u ||
            !std::isfinite(row.rel_terminal_moment_drift) ||
            !std::isfinite(row.max_rel_moment_drift) ||
            !std::isfinite(row.max_rel_tangent_drift) ||
            !std::isfinite(row.max_rel_secant_drift)) {
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
            !std::isfinite(row.reference_max_curvature_y) ||
            !std::isfinite(row.reference_terminal_structural_moment_y)) {
            return false;
        }
    }

    return std::isfinite(result.summary.worst_rel_terminal_moment_drift) &&
           std::isfinite(result.summary.worst_max_rel_moment_drift) &&
           std::isfinite(result.summary.worst_max_rel_tangent_drift) &&
           std::isfinite(result.summary.worst_max_rel_secant_drift);
}

bool reduced_rc_column_node_refinement_study_writes_csv_contract()
{
    using fall_n::validation_reboot::run_reduced_rc_column_node_refinement_study;

    const auto out_dir = std::filesystem::path{
        "data/output/cyclic_validation/reboot_node_refinement_study_csv"};
    std::filesystem::remove_all(out_dir);

    const auto result =
        run_reduced_rc_column_node_refinement_study(
            make_pilot_node_refinement_spec(),
            out_dir.string());

    if (result.case_rows.size() != 6u || result.reference_rows.size() != 2u) {
        return false;
    }

    const auto cases_csv = out_dir / "node_refinement_case_comparisons.csv";
    const auto summary_csv = out_dir / "node_refinement_summary.csv";
    const auto refs_csv = out_dir / "node_refinement_reference_cases.csv";
    const auto overall_csv = out_dir / "node_refinement_overall_summary.csv";

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
            "case_id,reference_case_id,beam_nodes,beam_axis_quadrature_family,formulation_kind,execution_ok,positive_branch_point_count,overlap_point_count,compared_max_curvature_y,reference_max_curvature_y,terminal_structural_moment_y,reference_terminal_structural_moment_y,rel_terminal_moment_drift,max_rel_moment_drift,rms_rel_moment_drift,max_rel_tangent_drift,max_rel_secant_drift,terminal_moment_within_representative_tolerance,moment_drift_within_representative_tolerance,tangent_drift_within_representative_tolerance,secant_drift_within_representative_tolerance,representative_internal_refinement_passes,scope_label,error_message" &&
        summary_header ==
            "beam_nodes,case_count,completed_case_count,representative_pass_count,min_rel_terminal_moment_drift,max_rel_terminal_moment_drift,avg_rel_terminal_moment_drift,min_max_rel_moment_drift,max_max_rel_moment_drift,avg_max_rel_moment_drift,min_max_rel_tangent_drift,max_max_rel_tangent_drift,avg_max_rel_tangent_drift,min_max_rel_secant_drift,max_max_rel_secant_drift,avg_max_rel_secant_drift" &&
        refs_header ==
            "beam_axis_quadrature_family,reference_beam_nodes,reference_case_id,compared_case_count,reference_max_curvature_y,reference_terminal_structural_moment_y" &&
        overall_header ==
            "total_case_count,completed_case_count,failed_case_count,representative_pass_count,worst_rel_terminal_moment_drift,worst_terminal_moment_case_id,worst_max_rel_moment_drift,worst_moment_case_id,worst_max_rel_tangent_drift,worst_tangent_case_id,worst_max_rel_secant_drift,worst_secant_case_id,all_cases_completed,all_completed_cases_pass_representative_internal_refinement";
}

bool reduced_rc_column_node_refinement_study_uses_highest_n_reference_per_family()
{
    using fall_n::validation_reboot::run_reduced_rc_column_node_refinement_study;

    const auto result =
        run_reduced_rc_column_node_refinement_study(
            make_pilot_node_refinement_spec(),
            "data/output/cyclic_validation/reboot_node_refinement_study_reference");

    if (result.reference_rows.size() != 2u) {
        return false;
    }

    for (const auto& row : result.reference_rows) {
        if (row.reference_beam_nodes != 10u ||
            row.reference_case_id.find("n10_") != 0 ||
            row.compared_case_count != 3u ||
            !std::isfinite(row.reference_max_curvature_y) ||
            !std::isfinite(row.reference_terminal_structural_moment_y)) {
            return false;
        }
    }

    std::size_t zero_drift_reference_cases = 0;
    for (const auto& row : result.case_rows) {
        if (row.beam_nodes != 10u) {
            continue;
        }
        if (!row.execution_ok ||
            row.reference_case_id != row.case_id ||
            row.overlap_point_count < 2u ||
            row.rel_terminal_moment_drift > 1.0e-12 ||
            row.max_rel_moment_drift > 1.0e-12 ||
            row.max_rel_tangent_drift > 1.0e-12 ||
            row.max_rel_secant_drift > 1.0e-12) {
            return false;
        }
        ++zero_drift_reference_cases;
    }

    return zero_drift_reference_cases == 2u;
}

bool reduced_rc_column_node_refinement_study_produces_finite_internal_drift_metrics()
{
    using fall_n::validation_reboot::run_reduced_rc_column_node_refinement_study;

    const auto result =
        run_reduced_rc_column_node_refinement_study(
            make_pilot_node_refinement_spec(),
            "data/output/cyclic_validation/reboot_node_refinement_study_finite");

    if (result.empty() ||
        result.summary.total_case_count != 6u ||
        result.summary.completed_case_count != 6u ||
        result.summary.failed_case_count != 0u) {
        return false;
    }

    bool saw_non_reference_case = false;
    for (const auto& row : result.case_rows) {
        if (!row.execution_ok ||
            row.positive_branch_point_count < 2u ||
            row.overlap_point_count < 2u ||
            !std::isfinite(row.compared_max_curvature_y) ||
            !std::isfinite(row.reference_max_curvature_y) ||
            !std::isfinite(row.rel_terminal_moment_drift) ||
            !std::isfinite(row.max_rel_moment_drift) ||
            !std::isfinite(row.rms_rel_moment_drift) ||
            !std::isfinite(row.max_rel_tangent_drift) ||
            !std::isfinite(row.max_rel_secant_drift)) {
            return false;
        }

        if (row.beam_nodes != 10u) {
            saw_non_reference_case = true;
        }
    }

    if (!saw_non_reference_case ||
        !std::isfinite(result.summary.worst_rel_terminal_moment_drift) ||
        !std::isfinite(result.summary.worst_max_rel_moment_drift) ||
        !std::isfinite(result.summary.worst_max_rel_tangent_drift) ||
        !std::isfinite(result.summary.worst_max_rel_secant_drift)) {
        return false;
    }

    const auto active_summary_rows = std::count_if(
        result.summary_rows.begin(),
        result.summary_rows.end(),
        [](const auto& row) { return row.case_count > 0u; });

    return active_summary_rows == 3;
}

} // namespace

int main(int argc, char** argv)
{
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    std::cout << "=== Reduced RC Column Node-Refinement Study Tests ===\n";

    report(
        "reduced_rc_column_node_refinement_study_completes_runtime_frontier",
        reduced_rc_column_node_refinement_study_completes_runtime_frontier());
    report(
        "reduced_rc_column_node_refinement_study_writes_csv_contract",
        reduced_rc_column_node_refinement_study_writes_csv_contract());
    report(
        "reduced_rc_column_node_refinement_study_uses_highest_n_reference_per_family",
        reduced_rc_column_node_refinement_study_uses_highest_n_reference_per_family());
    report(
        "reduced_rc_column_node_refinement_study_produces_finite_internal_drift_metrics",
        reduced_rc_column_node_refinement_study_produces_finite_internal_drift_metrics());

    std::cout << "\nSummary: " << passed << " passed, " << failed << " failed.\n";

    PetscFinalize();
    return failed == 0 ? 0 : 1;
}
