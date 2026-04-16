#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <petsc.h>

#include "src/validation/ReducedRCColumnMomentCurvatureClosureMatrix.hh"

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
make_monotonic_matrix_protocol()
{
    return {
        .protocol_name = "reduced_moment_curvature_matrix",
        .execution_profile_name = "default",
        .amplitudes_m = {2.5e-3},
        .steps_per_segment = 4,
        .max_steps = 0,
        .max_bisections = 4,
    };
}

fall_n::validation_reboot::ReducedRCColumnMomentCurvatureClosureMatrixRunSpec
make_matrix_spec()
{
    using fall_n::validation_reboot::ReducedRCColumnMomentCurvatureClosureMatrixRunSpec;

    return ReducedRCColumnMomentCurvatureClosureMatrixRunSpec{
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
            .structural_protocol = make_monotonic_matrix_protocol(),
            .write_closure_csv = true,
            .print_progress = false,
        },
        .include_only_phase3_runtime_baseline = true,
        .write_case_outputs = true,
        .write_matrix_csv = true,
        .print_progress = false,
    };
}

bool reduced_rc_column_moment_curvature_closure_matrix_completes_runtime_frontier()
{
    using fall_n::canonical_reduced_rc_column_phase3_baseline_case_count_v;
    using fall_n::validation_reboot::run_reduced_rc_column_moment_curvature_closure_matrix;

    const auto result = run_reduced_rc_column_moment_curvature_closure_matrix(
        make_matrix_spec(),
        "data/output/cyclic_validation/reboot_moment_curvature_closure_matrix_full");

    if (result.empty() ||
        result.case_rows.size() !=
            canonical_reduced_rc_column_phase3_baseline_case_count_v ||
        result.summary.total_case_count !=
            canonical_reduced_rc_column_phase3_baseline_case_count_v ||
        result.summary.completed_case_count !=
            canonical_reduced_rc_column_phase3_baseline_case_count_v ||
        result.summary.failed_case_count != 0 ||
        result.node_spread_rows.size() != 9u ||
        result.quadrature_spread_rows.size() != 4u) {
        return false;
    }

    for (const auto& row : result.case_rows) {
        if (!row.execution_ok ||
            row.case_id.empty() ||
            row.case_out_dir.empty() ||
            !std::isfinite(row.max_rel_moment_error) ||
            !std::isfinite(row.max_rel_tangent_error) ||
            !std::isfinite(row.max_rel_secant_error) ||
            row.positive_branch_point_count < 2u) {
            return false;
        }
    }

    if (result.summary.worst_max_rel_axial_force_error >= 1.0e-4) {
        return false;
    }

    for (const auto& row : result.node_spread_rows) {
        if (row.case_count != 4u || row.completed_case_count != 4u) {
            return false;
        }
    }

    for (const auto& row : result.quadrature_spread_rows) {
        if (row.case_count != 9u || row.completed_case_count != 9u) {
            return false;
        }
    }

    return std::isfinite(result.summary.worst_max_rel_moment_error) &&
           std::isfinite(result.summary.worst_max_rel_axial_force_error) &&
           std::isfinite(result.summary.worst_max_rel_tangent_error) &&
           std::isfinite(result.summary.worst_max_rel_secant_error);
}

bool reduced_rc_column_moment_curvature_closure_matrix_writes_csv_contract()
{
    using fall_n::validation_reboot::run_reduced_rc_column_moment_curvature_closure_matrix;

    auto spec = make_matrix_spec();
    spec.beam_nodes_filter = {2, 4};
    spec.quadrature_filter = {
        BeamAxisQuadratureFamily::GaussLegendre,
        BeamAxisQuadratureFamily::GaussLobatto};

    const auto out_dir = std::filesystem::path{
        "data/output/cyclic_validation/reboot_moment_curvature_closure_matrix_csv"};
    std::filesystem::remove_all(out_dir);

    const auto result =
        run_reduced_rc_column_moment_curvature_closure_matrix(spec, out_dir.string());

    if (result.case_rows.size() != 4u) {
        return false;
    }

    const auto cases_csv = out_dir / "moment_curvature_closure_matrix_cases.csv";
    const auto summary_csv = out_dir / "moment_curvature_closure_matrix_summary.csv";
    const auto node_csv = out_dir / "moment_curvature_closure_node_spread.csv";
    const auto quad_csv = out_dir / "moment_curvature_closure_quadrature_spread.csv";

    if (!std::filesystem::exists(cases_csv) ||
        !std::filesystem::exists(summary_csv) ||
        !std::filesystem::exists(node_csv) ||
        !std::filesystem::exists(quad_csv)) {
        return false;
    }

    std::ifstream cases_stream(cases_csv);
    std::ifstream summary_stream(summary_csv);
    std::ifstream node_stream(node_csv);
    std::ifstream quad_stream(quad_csv);

    std::string cases_header;
    std::string summary_header;
    std::string node_header;
    std::string quad_header;
    std::getline(cases_stream, cases_header);
    std::getline(summary_stream, summary_header);
    std::getline(node_stream, node_header);
    std::getline(quad_stream, quad_header);

    return cases_header ==
               "case_id,beam_nodes,beam_axis_quadrature_family,formulation_kind,execution_ok,positive_branch_point_count,structural_max_curvature_y,section_baseline_max_curvature_y,max_rel_axial_force_error,max_rel_moment_error,rms_rel_moment_error,max_rel_tangent_error,max_rel_secant_error,moment_within_representative_tolerance,tangent_within_representative_tolerance,secant_within_representative_tolerance,axial_force_within_representative_tolerance,representative_closure_passes,scope_label,error_message" &&
           summary_header ==
               "total_case_count,completed_case_count,failed_case_count,representative_pass_count,worst_max_rel_axial_force_error,worst_axial_force_case_id,worst_max_rel_moment_error,worst_moment_case_id,worst_max_rel_tangent_error,worst_tangent_case_id,worst_max_rel_secant_error,worst_secant_case_id,all_cases_completed,all_completed_cases_pass_representative_closure" &&
           node_header ==
               "beam_nodes,case_count,completed_case_count,representative_pass_count,min_max_rel_moment_error,max_max_rel_moment_error,avg_max_rel_moment_error,min_max_rel_tangent_error,max_max_rel_tangent_error,avg_max_rel_tangent_error,min_max_rel_secant_error,max_max_rel_secant_error,avg_max_rel_secant_error" &&
           quad_header ==
               "beam_axis_quadrature_family,case_count,completed_case_count,representative_pass_count,min_max_rel_moment_error,max_max_rel_moment_error,avg_max_rel_moment_error,min_max_rel_tangent_error,max_max_rel_tangent_error,avg_max_rel_tangent_error,min_max_rel_secant_error,max_max_rel_secant_error,avg_max_rel_secant_error";
}

bool reduced_rc_column_moment_curvature_closure_matrix_case_ids_are_unique()
{
    using fall_n::validation_reboot::run_reduced_rc_column_moment_curvature_closure_matrix;

    auto spec = make_matrix_spec();
    spec.beam_nodes_filter = {3, 5, 7};

    const auto result = run_reduced_rc_column_moment_curvature_closure_matrix(
        spec,
        "data/output/cyclic_validation/reboot_moment_curvature_closure_matrix_ids");

    std::vector<std::string> case_ids;
    case_ids.reserve(result.case_rows.size());
    for (const auto& row : result.case_rows) {
        case_ids.push_back(row.case_id);
    }

    std::ranges::sort(case_ids);
    const auto it = std::ranges::adjacent_find(case_ids);
    if (it != case_ids.end()) {
        return false;
    }

    return result.summary.total_case_count == 12u &&
           result.summary.completed_case_count == 12u &&
           result.node_spread_rows.size() == 9u &&
           result.quadrature_spread_rows.size() == 4u;
}

} // namespace

int main(int argc, char** argv)
{
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    std::cout << "=== Reduced RC Column Moment-Curvature Closure Matrix Tests ===\n";

    report(
        "reduced_rc_column_moment_curvature_closure_matrix_completes_runtime_frontier",
        reduced_rc_column_moment_curvature_closure_matrix_completes_runtime_frontier());
    report(
        "reduced_rc_column_moment_curvature_closure_matrix_writes_csv_contract",
        reduced_rc_column_moment_curvature_closure_matrix_writes_csv_contract());
    report(
        "reduced_rc_column_moment_curvature_closure_matrix_case_ids_are_unique",
        reduced_rc_column_moment_curvature_closure_matrix_case_ids_are_unique());

    std::cout << "\nSummary: " << passed << " passed, " << failed << " failed.\n";

    PetscFinalize();
    return failed == 0 ? 0 : 1;
}
