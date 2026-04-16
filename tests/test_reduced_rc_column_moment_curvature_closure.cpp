#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <petsc.h>

#include "src/validation/ReducedRCColumnMomentCurvatureClosure.hh"

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
make_monotonic_reduced_column_protocol()
{
    return {
        .protocol_name = "reduced_moment_curvature_closure",
        .execution_profile_name = "default",
        .amplitudes_m = {2.5e-3},
        .steps_per_segment = 8,
        .max_steps = 0,
        .max_bisections = 4,
    };
}

bool reduced_rc_column_moment_curvature_closure_produces_finite_bundle()
{
    using fall_n::validation_reboot::ReducedRCColumnMomentCurvatureClosureRunSpec;
    using fall_n::validation_reboot::run_reduced_rc_column_moment_curvature_closure;

    const auto result = run_reduced_rc_column_moment_curvature_closure(
        ReducedRCColumnMomentCurvatureClosureRunSpec{
            .structural_spec = {
                .beam_nodes = 4,
                .beam_axis_quadrature_family = BeamAxisQuadratureFamily::GaussLobatto,
                .axial_compression_force_mn = 0.02,
                .write_hysteresis_csv = false,
                .write_section_response_csv = true,
                .print_progress = false,
            },
            .section_spec = {
                .steps = 24,
                .write_csv = true,
                .print_progress = false,
            },
            .structural_protocol = make_monotonic_reduced_column_protocol(),
            .write_closure_csv = false,
            .print_progress = false,
        },
        "data/output/cyclic_validation/reboot_moment_curvature_closure_finite");

    if (result.empty() ||
        result.summary.positive_branch_point_count != result.closure_records.size() ||
        result.summary.positive_branch_point_count < 3u) {
        return false;
    }

    for (const auto& row : result.closure_records) {
        if (!std::isfinite(row.curvature_y) ||
            !std::isfinite(row.structural_moment_y) ||
            !std::isfinite(row.section_moment_y) ||
            !std::isfinite(row.rel_moment_error) ||
            !std::isfinite(row.rel_tangent_error) ||
            !std::isfinite(row.rel_secant_error) ||
            !std::isfinite(row.rel_axial_force_error)) {
            return false;
        }
    }

    return std::isfinite(result.summary.max_rel_moment_error) &&
           std::isfinite(result.summary.max_rel_tangent_error) &&
           std::isfinite(result.summary.max_rel_secant_error) &&
           std::isfinite(result.summary.max_rel_axial_force_error) &&
           result.summary.structural_max_curvature_y > 0.0 &&
           result.summary.section_baseline_max_curvature_y >=
               result.summary.structural_max_curvature_y;
}

bool reduced_rc_column_moment_curvature_closure_writes_csv_contract()
{
    using fall_n::validation_reboot::ReducedRCColumnMomentCurvatureClosureRunSpec;
    using fall_n::validation_reboot::run_reduced_rc_column_moment_curvature_closure;

    const auto out_dir = std::filesystem::path{
        "data/output/cyclic_validation/reboot_moment_curvature_closure_csv"};
    std::filesystem::remove_all(out_dir);

    const auto result = run_reduced_rc_column_moment_curvature_closure(
        ReducedRCColumnMomentCurvatureClosureRunSpec{
            .structural_spec = {
                .beam_nodes = 4,
                .beam_axis_quadrature_family = BeamAxisQuadratureFamily::GaussLobatto,
                .axial_compression_force_mn = 0.02,
                .write_hysteresis_csv = false,
                .write_section_response_csv = true,
                .print_progress = false,
            },
            .section_spec = {
                .steps = 16,
                .write_csv = true,
                .print_progress = false,
            },
            .structural_protocol = make_monotonic_reduced_column_protocol(),
            .write_closure_csv = true,
            .print_progress = false,
        },
        out_dir.string());

    const auto closure_csv = out_dir / "moment_curvature_closure.csv";
    const auto summary_csv = out_dir / "moment_curvature_closure_summary.csv";
    const auto structural_csv =
        out_dir / "structural_baseline" / "moment_curvature_base.csv";
    const auto section_csv =
        out_dir / "section_baseline" / "section_moment_curvature_baseline.csv";

    if (!std::filesystem::exists(closure_csv) ||
        !std::filesystem::exists(summary_csv) ||
        !std::filesystem::exists(structural_csv) ||
        !std::filesystem::exists(section_csv) ||
        result.empty()) {
        return false;
    }

    std::ifstream closure_stream(closure_csv);
    std::ifstream summary_stream(summary_csv);
    std::string closure_header;
    std::string summary_header;
    std::getline(closure_stream, closure_header);
    std::getline(summary_stream, summary_header);

    return closure_header ==
               "step,p,drift_m,curvature_y,structural_axial_force_MN,section_axial_force_MN,structural_moment_y_MNm,section_moment_y_MNm,structural_tangent_eiy,section_tangent_eiy,structural_secant_eiy,section_secant_eiy,abs_axial_force_error_MN,rel_axial_force_error,abs_moment_error_MNm,rel_moment_error,abs_tangent_error,rel_tangent_error,abs_secant_error,rel_secant_error" &&
           summary_header ==
               "positive_branch_point_count,structural_max_curvature_y,section_baseline_max_curvature_y,max_abs_axial_force_error_MN,max_rel_axial_force_error,max_abs_moment_error_MNm,max_rel_moment_error,rms_rel_moment_error,max_abs_tangent_error,max_rel_tangent_error,max_abs_secant_error,max_rel_secant_error,moment_within_representative_tolerance,tangent_within_representative_tolerance,secant_within_representative_tolerance,axial_force_within_representative_tolerance,representative_closure_passes";
}

bool reduced_rc_column_moment_curvature_closure_keeps_representative_errors_bounded()
{
    using fall_n::validation_reboot::ReducedRCColumnMomentCurvatureClosureRunSpec;
    using fall_n::validation_reboot::run_reduced_rc_column_moment_curvature_closure;

    const auto result = run_reduced_rc_column_moment_curvature_closure(
        ReducedRCColumnMomentCurvatureClosureRunSpec{
            .structural_spec = {
                .beam_nodes = 4,
                .beam_axis_quadrature_family = BeamAxisQuadratureFamily::GaussLobatto,
                .axial_compression_force_mn = 0.02,
                .write_hysteresis_csv = false,
                .write_section_response_csv = false,
                .print_progress = false,
            },
            .section_spec = {
                .steps = 24,
                .write_csv = false,
                .print_progress = false,
            },
            .structural_protocol = make_monotonic_reduced_column_protocol(),
            .write_closure_csv = false,
            .print_progress = false,
        },
        "data/output/cyclic_validation/reboot_moment_curvature_closure_bounds");

    return result.summary.representative_closure_passes() &&
           std::isfinite(result.summary.max_rel_axial_force_error) &&
           result.summary.max_rel_axial_force_error < 1.0e-4 &&
           result.summary.max_rel_moment_error < 1.0e-2 &&
           result.summary.max_rel_tangent_error < 1.0e-2 &&
           result.summary.max_rel_secant_error < 1.0e-2;
}

} // namespace

int main()
{
    PetscInitialize(nullptr, nullptr, nullptr, nullptr);

    std::cout << "=== Reduced RC Column Moment-Curvature Closure Tests ===\n";

    report(
        "reduced_rc_column_moment_curvature_closure_produces_finite_bundle",
        reduced_rc_column_moment_curvature_closure_produces_finite_bundle());
    report(
        "reduced_rc_column_moment_curvature_closure_writes_csv_contract",
        reduced_rc_column_moment_curvature_closure_writes_csv_contract());
    report(
        "reduced_rc_column_moment_curvature_closure_keeps_representative_errors_bounded",
        reduced_rc_column_moment_curvature_closure_keeps_representative_errors_bounded());

    std::cout << "\nSummary: " << passed << " passed, " << failed << " failed.\n";
    PetscFinalize();
    return failed == 0 ? 0 : 1;
}
