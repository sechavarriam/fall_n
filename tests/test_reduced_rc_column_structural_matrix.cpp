#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <petsc.h>

#include "src/validation/ReducedRCColumnStructuralBaseline.hh"
#include "src/validation/ReducedRCColumnStructuralMatrixCatalog.hh"

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

constexpr auto matrix =
    fall_n::canonical_reduced_rc_column_structural_matrix_v;

constexpr bool reduced_rc_column_matrix_counts_are_honest()
{
    using fall_n::ReducedRCColumnStructuralSupportKind;
    using continuum::FormulationKind;

    return matrix.size() == 144 &&
           fall_n::canonical_reduced_rc_column_phase3_baseline_case_count_v == 36 &&
           fall_n::canonical_reduced_rc_column_structural_case_count_v<
               ReducedRCColumnStructuralSupportKind::ready_for_runtime_baseline> == 36 &&
           fall_n::canonical_reduced_rc_column_structural_case_count_v<
               ReducedRCColumnStructuralSupportKind::planned_family_extension> == 36 &&
           fall_n::canonical_reduced_rc_column_structural_case_count_v<
               ReducedRCColumnStructuralSupportKind::unavailable_in_current_family> == 72 &&
           fall_n::canonical_reduced_rc_column_structural_formulation_count_v<
               FormulationKind::small_strain> == 36 &&
           fall_n::canonical_reduced_rc_column_structural_formulation_count_v<
               FormulationKind::corotational> == 36 &&
           fall_n::canonical_reduced_rc_column_structural_formulation_count_v<
               FormulationKind::total_lagrangian> == 36 &&
           fall_n::canonical_reduced_rc_column_structural_formulation_count_v<
               FormulationKind::updated_lagrangian> == 36;
}

constexpr bool only_small_strain_cases_can_anchor_the_current_phase3_baseline()
{
    using continuum::FormulationKind;

    for (const auto& row : matrix) {
        if (row.is_current_baseline_case()) {
            if (row.formulation_kind != FormulationKind::small_strain ||
                !row.has_runtime_path ||
                !row.keeps_compile_time_hot_path_static) {
                return false;
            }
        } else {
            if (row.formulation_kind == FormulationKind::small_strain &&
                row.has_runtime_path &&
                !row.can_anchor_phase3_structural_baseline) {
                return false;
            }
        }
    }

    return true;
}

constexpr bool blocked_rows_distinguish_corotational_extension_from_tl_ul_absence()
{
    using continuum::FormulationKind;

    bool found_corotational = false;
    bool found_tl = false;
    bool found_ul = false;

    for (const auto& row : matrix) {
        if (row.formulation_kind == FormulationKind::corotational) {
            found_corotational = true;
            if (!row.requires_new_kinematic_extension ||
                row.requires_new_beam_family_or_formulation ||
                row.has_runtime_path) {
                return false;
            }
        } else if (row.formulation_kind == FormulationKind::total_lagrangian) {
            found_tl = true;
            if (row.requires_new_kinematic_extension ||
                !row.requires_new_beam_family_or_formulation ||
                row.has_runtime_path) {
                return false;
            }
        } else if (row.formulation_kind == FormulationKind::updated_lagrangian) {
            found_ul = true;
            if (row.requires_new_kinematic_extension ||
                !row.requires_new_beam_family_or_formulation ||
                row.has_runtime_path) {
                return false;
            }
        }
    }

    return found_corotational && found_tl && found_ul;
}

bool run_small_strain_runtime_family_smoke()
{
    using fall_n::table_cyclic_validation::CyclicValidationRunConfig;
    using fall_n::validation_reboot::ReducedRCColumnStructuralRunSpec;
    using fall_n::validation_reboot::run_reduced_rc_column_small_strain_beam_case;

    const CyclicValidationRunConfig cfg{
        .protocol_name = "reboot_smoke",
        .execution_profile_name = "default",
        .amplitudes_m = {2.5e-4},
        .steps_per_segment = 1,
        .max_steps = 0,
        .max_bisections = 3,
    };

    const std::array<BeamAxisQuadratureFamily, 3> families{
        BeamAxisQuadratureFamily::GaussLegendre,
        BeamAxisQuadratureFamily::GaussLobatto,
        BeamAxisQuadratureFamily::GaussRadauLeft
    };

    bool ok = true;
    for (const auto family : families) {
        const auto records = run_reduced_rc_column_small_strain_beam_case(
            ReducedRCColumnStructuralRunSpec{
                .beam_nodes = 3,
                .beam_axis_quadrature_family = family,
                .axial_compression_force_mn = 0.0,
                .write_hysteresis_csv = false,
                .print_progress = false,
            },
            "data/output/cyclic_validation/reboot_structural_matrix_smoke",
            cfg);

        if (records.size() != static_cast<std::size_t>(cfg.total_steps()) + 1) {
            return false;
        }

        double peak_shear = 0.0;
        for (const auto& row : records) {
            if (!std::isfinite(row.base_shear) || !std::isfinite(row.drift)) {
                return false;
            }
            peak_shear = std::max(peak_shear, std::abs(row.base_shear));
        }

        ok = ok && peak_shear > 1.0e-8;
    }

    return ok;
}

bool axial_compression_option_preserves_finite_response()
{
    using fall_n::table_cyclic_validation::CyclicValidationRunConfig;
    using fall_n::validation_reboot::ReducedRCColumnStructuralRunSpec;
    using fall_n::validation_reboot::run_reduced_rc_column_small_strain_beam_case;

    const CyclicValidationRunConfig cfg{
        .protocol_name = "reboot_axial_smoke",
        .execution_profile_name = "default",
        .amplitudes_m = {2.5e-4},
        .steps_per_segment = 1,
        .max_steps = 0,
        .max_bisections = 3,
    };

    const auto records = run_reduced_rc_column_small_strain_beam_case(
        ReducedRCColumnStructuralRunSpec{
            .beam_nodes = 2,
            .beam_axis_quadrature_family = BeamAxisQuadratureFamily::GaussLegendre,
            .axial_compression_force_mn = 0.02,
            .write_hysteresis_csv = false,
            .print_progress = false,
        },
        "data/output/cyclic_validation/reboot_structural_matrix_axial",
        cfg);

    if (records.empty()) {
        return false;
    }

    return std::all_of(
        records.begin(),
        records.end(),
        [](const auto& row) {
            return std::isfinite(row.base_shear) && std::isfinite(row.drift);
        });
}

bool base_side_moment_curvature_observable_is_finite_and_ordered()
{
    using fall_n::table_cyclic_validation::CyclicValidationRunConfig;
    using fall_n::validation_reboot::ReducedRCColumnStructuralRunSpec;
    using fall_n::validation_reboot::run_reduced_rc_column_small_strain_beam_case_result;

    const CyclicValidationRunConfig cfg{
        .protocol_name = "reboot_section_observable",
        .execution_profile_name = "default",
        .amplitudes_m = {2.5e-4},
        .steps_per_segment = 1,
        .max_steps = 0,
        .max_bisections = 3,
    };

    const auto result = run_reduced_rc_column_small_strain_beam_case_result(
        ReducedRCColumnStructuralRunSpec{
            .beam_nodes = 4,
            .beam_axis_quadrature_family = BeamAxisQuadratureFamily::GaussLobatto,
            .axial_compression_force_mn = 0.0,
            .write_hysteresis_csv = false,
            .write_section_response_csv = false,
            .print_progress = false,
        },
        "data/output/cyclic_validation/reboot_structural_matrix_section",
        cfg);

    if (result.section_response_records.empty()) {
        return false;
    }

    const auto expected_records =
        static_cast<std::size_t>(cfg.total_steps() + 1) * (4u - 1u);
    if (result.section_response_records.size() != expected_records) {
        return false;
    }

    double min_xi = result.section_response_records.front().xi;
    std::size_t controlling_gp = result.section_response_records.front().section_gp;
    for (const auto& row : result.section_response_records) {
        if (!std::isfinite(row.curvature_y) ||
            !std::isfinite(row.moment_y) ||
            !std::isfinite(row.tangent_eiy)) {
            return false;
        }
        if (row.xi < min_xi ||
            (row.xi == min_xi && row.section_gp < controlling_gp)) {
            min_xi = row.xi;
            controlling_gp = row.section_gp;
        }
    }

    const auto controlling_count = static_cast<std::size_t>(std::count_if(
        result.section_response_records.begin(),
        result.section_response_records.end(),
        [controlling_gp](const auto& row) {
            return row.section_gp == controlling_gp;
        }));

    return controlling_count ==
           static_cast<std::size_t>(cfg.total_steps() + 1);
}

bool section_observable_csv_contract_is_written()
{
    using fall_n::table_cyclic_validation::CyclicValidationRunConfig;
    using fall_n::validation_reboot::ReducedRCColumnStructuralRunSpec;
    using fall_n::validation_reboot::run_reduced_rc_column_small_strain_beam_case_result;

    const auto out_dir =
        std::filesystem::path{
            "data/output/cyclic_validation/reboot_structural_matrix_section_csv"};
    std::filesystem::remove_all(out_dir);

    const CyclicValidationRunConfig cfg{
        .protocol_name = "reboot_section_csv",
        .execution_profile_name = "default",
        .amplitudes_m = {2.5e-4},
        .steps_per_segment = 1,
        .max_steps = 0,
        .max_bisections = 3,
    };

    const auto result = run_reduced_rc_column_small_strain_beam_case_result(
        ReducedRCColumnStructuralRunSpec{
            .beam_nodes = 3,
            .beam_axis_quadrature_family = BeamAxisQuadratureFamily::GaussLegendre,
            .axial_compression_force_mn = 0.02,
            .write_hysteresis_csv = false,
            .write_section_response_csv = true,
            .print_progress = false,
        },
        out_dir.string(),
        cfg);

    const auto section_csv = out_dir / "section_response.csv";
    const auto mk_csv = out_dir / "moment_curvature_base.csv";
    if (!std::filesystem::exists(section_csv) ||
        !std::filesystem::exists(mk_csv) ||
        !result.has_section_response_observable()) {
        return false;
    }

    std::ifstream section_stream(section_csv);
    std::ifstream mk_stream(mk_csv);
    std::string section_header;
    std::string mk_header;
    std::getline(section_stream, section_header);
    std::getline(mk_stream, mk_header);

    return section_header ==
               "step,p,drift_m,section_gp,xi,axial_strain,curvature_y,curvature_z,"
               "axial_force_MN,moment_y_MNm,moment_z_MNm,tangent_ea,tangent_eiy,tangent_eiz" &&
           mk_header ==
               "step,p,drift_m,section_gp,xi,curvature_y,moment_y_MNm,axial_force_MN,tangent_eiy";
}

static_assert(reduced_rc_column_matrix_counts_are_honest());
static_assert(only_small_strain_cases_can_anchor_the_current_phase3_baseline());
static_assert(blocked_rows_distinguish_corotational_extension_from_tl_ul_absence());

} // namespace

int main(int argc, char** argv)
{
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    std::cout << "=== Reduced RC Column Structural Matrix Tests ===\n";

    report("reduced_rc_column_matrix_counts_are_honest",
           reduced_rc_column_matrix_counts_are_honest());
    report("only_small_strain_cases_can_anchor_the_current_phase3_baseline",
           only_small_strain_cases_can_anchor_the_current_phase3_baseline());
    report("blocked_rows_distinguish_corotational_extension_from_tl_ul_absence",
           blocked_rows_distinguish_corotational_extension_from_tl_ul_absence());
    report("run_small_strain_runtime_family_smoke",
           run_small_strain_runtime_family_smoke());
    report("axial_compression_option_preserves_finite_response",
           axial_compression_option_preserves_finite_response());
    report("base_side_moment_curvature_observable_is_finite_and_ordered",
           base_side_moment_curvature_observable_is_finite_and_ordered());
    report("section_observable_csv_contract_is_written",
           section_observable_csv_contract_is_written());

    std::cout << "\nSummary: " << passed << " passed, " << failed << " failed.\n";

    PetscFinalize();
    return failed == 0 ? 0 : 1;
}
