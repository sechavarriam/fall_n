#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "src/validation/ReducedRCColumnSectionBaseline.hh"

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

bool reduced_rc_column_section_baseline_produces_finite_records()
{
    const auto result =
        fall_n::validation_reboot::run_reduced_rc_column_section_moment_curvature_baseline(
            {
                .target_axial_compression_force_mn = 0.0,
                .max_curvature_y = 0.02,
                .steps = 24,
                .write_csv = false,
                .print_progress = false,
            },
            "data/output/cyclic_validation/reboot_section_baseline_finite");

    if (result.records.size() != 25u) {
        return false;
    }

    bool found_nonzero_moment = false;
    for (std::size_t i = 1; i < result.records.size(); ++i) {
        const auto& row = result.records[i];
        if (!std::isfinite(row.solved_axial_strain) ||
            !std::isfinite(row.curvature_y) ||
            !std::isfinite(row.axial_force) ||
            !std::isfinite(row.moment_y) ||
            !std::isfinite(row.tangent_ea) ||
            !std::isfinite(row.tangent_eiy) ||
            !std::isfinite(row.final_axial_force_residual)) {
            return false;
        }

        if (std::abs(row.moment_y) > 1.0e-8) {
            found_nonzero_moment = true;
        }
    }

    return found_nonzero_moment;
}

bool reduced_rc_column_section_baseline_closes_target_axial_force()
{
    constexpr double target_axial_compression_force_mn = 0.02;
    constexpr double newton_tolerance_mn = 1.0e-9;

    const auto result =
        fall_n::validation_reboot::run_reduced_rc_column_section_moment_curvature_baseline(
            {
                .target_axial_compression_force_mn =
                    target_axial_compression_force_mn,
                .max_curvature_y = 0.015,
                .steps = 18,
                .axial_force_newton_tolerance_mn = newton_tolerance_mn,
                .write_csv = false,
                .print_progress = false,
            },
            "data/output/cyclic_validation/reboot_section_baseline_axial");

    if (result.records.size() != 19u) {
        return false;
    }

    const double target_internal_force_mn = -target_axial_compression_force_mn;
    for (std::size_t i = 1; i < result.records.size(); ++i) {
        const auto& row = result.records[i];
        if (std::abs(row.axial_force - target_internal_force_mn) >
                5.0 * newton_tolerance_mn ||
            std::abs(row.final_axial_force_residual) >
                5.0 * newton_tolerance_mn ||
            row.newton_iterations <= 0) {
            return false;
        }
    }

    return true;
}

bool reduced_rc_column_section_baseline_writes_csv_contract()
{
    const auto out_dir = std::filesystem::path{
        "data/output/cyclic_validation/reboot_section_baseline_csv"};
    std::filesystem::remove_all(out_dir);

    const auto result =
        fall_n::validation_reboot::run_reduced_rc_column_section_moment_curvature_baseline(
            {
                .target_axial_compression_force_mn = 0.01,
                .max_curvature_y = 0.01,
                .steps = 8,
                .write_csv = true,
                .print_progress = false,
            },
            out_dir.string());

    const auto csv_path = out_dir / "section_moment_curvature_baseline.csv";
    const auto diag_csv_path = out_dir / "section_tangent_diagnostics.csv";
    const auto fiber_csv_path = out_dir / "section_fiber_state_history.csv";
    const auto control_csv_path = out_dir / "section_control_trace.csv";
    if (!std::filesystem::exists(csv_path) || result.records.empty()) {
        return false;
    }

    std::ifstream ifs(csv_path);
    std::string header;
    std::getline(ifs, header);

    std::ifstream diag_ifs(diag_csv_path);
    std::string diag_header;
    std::getline(diag_ifs, diag_header);

    std::ifstream fiber_ifs(fiber_csv_path);
    std::string fiber_header;
    std::getline(fiber_ifs, fiber_header);

    std::ifstream control_ifs(control_csv_path);
    std::string control_header;
    std::getline(control_ifs, control_header);

    return
        header ==
            "step,load_factor,target_axial_force_MN,solved_axial_strain,curvature_y,curvature_z,axial_force_MN,moment_y_MNm,moment_z_MNm,tangent_ea,tangent_eiy,tangent_eiz,tangent_eiy_direct_raw,tangent_eiz_direct_raw,newton_iterations,final_axial_force_residual_MN" &&
        std::filesystem::exists(diag_csv_path) &&
        std::filesystem::exists(fiber_csv_path) &&
        std::filesystem::exists(control_csv_path) &&
        diag_header ==
            "step,load_factor,curvature_y,moment_y_MNm,zero_curvature_anchor,tangent_eiy_condensed,tangent_eiy_direct_raw,tangent_eiy_numerical,tangent_eiy_left,tangent_eiy_right,tangent_consistency_rel_error,raw_k00,raw_k0y,raw_ky0,raw_kyy" &&
        fiber_header ==
            "step,load_factor,solved_axial_strain,curvature_y,zero_curvature_anchor,fiber_index,y,z,area,zone,material_role,strain_xx,stress_xx_MPa,tangent_xx_MPa,axial_force_contribution_MN,moment_y_contribution_MNm,raw_k00_contribution,raw_k0y_contribution,raw_kyy_contribution" &&
        control_header ==
            "step,load_factor,stage,target_curvature_y,actual_curvature_y,delta_target_curvature_y,delta_actual_curvature_y,pseudo_time_before,pseudo_time_after,pseudo_time_increment,domain_time_before,domain_time_after,domain_time_increment,control_dof_before,control_dof_after,target_increment_direction,actual_increment_direction,protocol_branch_id,reversal_index,branch_step_index,accepted_substep_count,max_bisection_level,newton_iterations,newton_iterations_per_substep,target_axial_force_MN,actual_axial_force_MN,axial_force_residual_MN";
}

bool reduced_rc_column_section_baseline_reports_positive_timing_surface()
{
    const auto result =
        fall_n::validation_reboot::run_reduced_rc_column_section_moment_curvature_baseline(
            {
                .target_axial_compression_force_mn = 0.01,
                .max_curvature_y = 0.01,
                .steps = 6,
                .write_csv = false,
                .print_progress = false,
            },
            "data/output/cyclic_validation/reboot_section_baseline_timing");

    return result.completed_successfully &&
           result.timing.total_wall_seconds > 0.0 &&
           result.timing.solve_wall_seconds > 0.0 &&
           result.timing.output_write_wall_seconds >= 0.0 &&
           result.timing.total_wall_seconds >= result.timing.solve_wall_seconds;
}

bool reduced_rc_column_section_elasticized_mode_produces_finite_monotonic_branch()
{
    const auto result =
        fall_n::validation_reboot::run_reduced_rc_column_section_moment_curvature_baseline(
            {
                .material_mode =
                    fall_n::validation_reboot::ReducedRCColumnSectionMaterialMode::elasticized,
                .target_axial_compression_force_mn = 0.02,
                .max_curvature_y = 0.01,
                .steps = 10,
                .write_csv = false,
                .print_progress = false,
            },
            "data/output/cyclic_validation/reboot_section_baseline_elasticized");

    if (!result.completed_successfully || result.records.size() != 11u) {
        return false;
    }

    double previous_moment = -1.0;
    for (std::size_t i = 1; i < result.records.size(); ++i) {
        const auto& row = result.records[i];
        if (!std::isfinite(row.moment_y) ||
            !std::isfinite(row.tangent_eiy) ||
            row.moment_y <= previous_moment) {
            return false;
        }
        if (std::abs(row.curvature_y) > 1.0e-12) {
            const double secant_eiy = row.moment_y / row.curvature_y;
            if (std::abs(secant_eiy - row.tangent_eiy) >
                1.0e-8 * std::max({1.0, std::abs(secant_eiy), std::abs(row.tangent_eiy)})) {
                return false;
            }
        }
        previous_moment = row.moment_y;
    }
    return true;
}

bool reduced_rc_column_section_baseline_supports_cyclic_curvature_history()
{
    using fall_n::validation_reboot::ReducedRCColumnSectionProtocolKind;

    const auto result =
        fall_n::validation_reboot::run_reduced_rc_column_section_moment_curvature_baseline(
            {
                .protocol_kind = ReducedRCColumnSectionProtocolKind::cyclic,
                .target_axial_compression_force_mn = 0.02,
                .max_curvature_y = 0.006,
                .cyclic_curvature_levels_y = {0.003, 0.006},
                .steps_per_segment = 2,
                .write_csv = false,
                .print_progress = false,
            },
            "data/output/cyclic_validation/reboot_section_baseline_cyclic");

    if (!result.completed_successfully || result.records.size() != 17u) {
        return false;
    }

    bool saw_positive = false;
    bool saw_negative = false;
    bool saw_return_near_zero = false;
    for (std::size_t i = 1; i < result.records.size(); ++i) {
        const auto& row = result.records[i];
        if (!std::isfinite(row.curvature_y) ||
            !std::isfinite(row.moment_y) ||
            !std::isfinite(row.axial_force) ||
            !std::isfinite(row.tangent_eiy)) {
            return false;
        }
        saw_positive = saw_positive || row.curvature_y > 1.0e-12;
        saw_negative = saw_negative || row.curvature_y < -1.0e-12;
        saw_return_near_zero =
            saw_return_near_zero || (i > 1 && std::abs(row.curvature_y) < 1.0e-12);
    }

    return saw_positive && saw_negative && saw_return_near_zero;
}

} // namespace

int main()
{
    std::cout << "=== Reduced RC Column Section Baseline Tests ===\n";

    report(
        "reduced_rc_column_section_baseline_produces_finite_records",
        reduced_rc_column_section_baseline_produces_finite_records());
    report(
        "reduced_rc_column_section_baseline_closes_target_axial_force",
        reduced_rc_column_section_baseline_closes_target_axial_force());
    report(
        "reduced_rc_column_section_baseline_writes_csv_contract",
        reduced_rc_column_section_baseline_writes_csv_contract());
    report(
        "reduced_rc_column_section_baseline_reports_positive_timing_surface",
        reduced_rc_column_section_baseline_reports_positive_timing_surface());
    report(
        "reduced_rc_column_section_elasticized_mode_produces_finite_monotonic_branch",
        reduced_rc_column_section_elasticized_mode_produces_finite_monotonic_branch());
    report(
        "reduced_rc_column_section_baseline_supports_cyclic_curvature_history",
        reduced_rc_column_section_baseline_supports_cyclic_curvature_history());

    std::cout << "\nSummary: " << passed << " passed, " << failed << " failed.\n";
    return failed == 0 ? 0 : 1;
}
