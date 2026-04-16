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
    if (!std::filesystem::exists(csv_path) || result.records.empty()) {
        return false;
    }

    std::ifstream ifs(csv_path);
    std::string header;
    std::getline(ifs, header);

    return
        header ==
        "step,load_factor,target_axial_force_MN,solved_axial_strain,curvature_y,curvature_z,axial_force_MN,moment_y_MNm,moment_z_MNm,tangent_ea,tangent_eiy,tangent_eiz,newton_iterations,final_axial_force_residual_MN";
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

    std::cout << "\nSummary: " << passed << " passed, " << failed << " failed.\n";
    return failed == 0 ? 0 : 1;
}
