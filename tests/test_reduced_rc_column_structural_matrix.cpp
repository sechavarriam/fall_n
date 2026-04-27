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

bool equilibrated_axial_preload_stage_seeds_step_zero_section_force()
{
    using fall_n::table_cyclic_validation::CyclicValidationRunConfig;
    using fall_n::validation_reboot::ReducedRCColumnStructuralRunSpec;
    using fall_n::validation_reboot::run_reduced_rc_column_small_strain_beam_case_result;

    const CyclicValidationRunConfig cfg{
        .protocol_name = "reboot_axial_preload_seed",
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
            .axial_compression_force_mn = 0.02,
            .use_equilibrated_axial_preload_stage = true,
            .axial_preload_steps = 3,
            .write_hysteresis_csv = false,
            .write_section_response_csv = false,
            .print_progress = false,
        },
        "data/output/cyclic_validation/reboot_structural_matrix_preload_seed",
        cfg);

    if (result.hysteresis_records.size() !=
            static_cast<std::size_t>(cfg.total_steps()) + 1u ||
        result.section_response_records.empty()) {
        return false;
    }

    const auto first_step_count = static_cast<std::size_t>(std::count_if(
        result.section_response_records.begin(),
        result.section_response_records.end(),
        [](const auto& row) { return row.step == 0; }));
    if (first_step_count != 3u) {
        return false;
    }

    constexpr double target_internal_force_mn = -0.02;
    constexpr double rel_tol = 1.0e-4;
    for (const auto& row : result.section_response_records) {
        if (row.step != 0) {
            continue;
        }
        const double rel_error =
            std::abs(row.axial_force - target_internal_force_mn) /
            std::max(std::abs(target_internal_force_mn), 1.0e-12);
        if (!std::isfinite(rel_error) || rel_error > rel_tol) {
            return false;
        }
    }

    return std::abs(result.hysteresis_records.front().drift) <= 1.0e-14;
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
               "axial_force_MN,moment_y_MNm,moment_z_MNm,tangent_ea,tangent_eiy,"
               "tangent_eiz,tangent_eiy_direct_raw,tangent_eiz_direct_raw,"
               "raw_k00,raw_k0y,raw_ky0,raw_kyy" &&
           mk_header ==
               "step,p,drift_m,section_gp,xi,curvature_y,moment_y_MNm,axial_force_MN,tangent_eiy";
}

bool element_tangent_audit_csv_contract_is_written()
{
    using fall_n::table_cyclic_validation::CyclicValidationRunConfig;
    using fall_n::validation_reboot::ReducedRCColumnStructuralRunSpec;
    using fall_n::validation_reboot::run_reduced_rc_column_small_strain_beam_case_result;

    const auto out_dir =
        std::filesystem::path{
            "data/output/cyclic_validation/reboot_structural_matrix_tangent_csv"};
    std::filesystem::remove_all(out_dir);

    const CyclicValidationRunConfig cfg{
        .protocol_name = "reboot_tangent_csv",
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
            .write_section_response_csv = false,
            .write_section_fiber_history_csv = false,
            .write_element_tangent_audit_csv = true,
            .print_progress = false,
        },
        out_dir.string(),
        cfg);

    const auto tangent_csv = out_dir / "element_tangent_audit.csv";
    const auto section_tangent_csv = out_dir / "section_tangent_audit.csv";
    if (!std::filesystem::exists(tangent_csv) ||
        !std::filesystem::exists(section_tangent_csv) ||
        result.element_tangent_audit_records.empty() ||
        result.section_tangent_audit_records.empty()) {
        return false;
    }

    std::ifstream tangent_stream(tangent_csv);
    std::ifstream section_tangent_stream(section_tangent_csv);
    std::string tangent_header;
    std::string section_tangent_header;
    std::getline(tangent_stream, tangent_header);
    std::getline(section_tangent_stream, section_tangent_header);

    return tangent_header ==
           "runtime_step,step,p,runtime_p,drift_m,failed_attempt,fd_step_reference,"
           "displacement_inf_norm,internal_force_norm,tangent_frobenius_norm,"
           "fd_tangent_frobenius_norm,tangent_fd_rel_error,max_column_rel_error,"
           "worst_column_index,worst_column_node,worst_column_local_dof,"
           "worst_column_tangent_norm,worst_column_fd_norm,"
           "top_control_column_rel_error,top_bending_rotation_column_rel_error" &&
           section_tangent_header ==
               "runtime_step,step,p,runtime_p,drift_m,failed_attempt,section_gp,xi,"
               "axial_strain,curvature_y,curvature_z,fd_step_reference,"
               "tangent_frobenius_norm,fd_tangent_frobenius_norm,tangent_fd_rel_error,"
               "max_column_rel_error,worst_column_index,axial_column_rel_error,"
               "curvature_y_column_rel_error,shear_z_column_rel_error,"
               "raw_tangent_k00,raw_tangent_k0y,raw_tangent_ky0,raw_tangent_kyy,"
               "fd_tangent_k00,fd_tangent_k0y,fd_tangent_ky0,fd_tangent_kyy,"
               "rel_error_k00,rel_error_k0y,rel_error_ky0,rel_error_kyy";
}

bool elasticized_tangent_audits_remain_frozen_and_consistent()
{
    using fall_n::table_cyclic_validation::CyclicValidationRunConfig;
    using fall_n::validation_reboot::ReducedRCColumnStructuralMaterialMode;
    using fall_n::validation_reboot::ReducedRCColumnStructuralRunSpec;
    using fall_n::validation_reboot::run_reduced_rc_column_small_strain_beam_case_result;

    const auto out_dir =
        std::filesystem::path{
            "data/output/cyclic_validation/reboot_structural_matrix_tangent_frozen"};
    std::filesystem::remove_all(out_dir);

    const CyclicValidationRunConfig cfg{
        .protocol_name = "reboot_tangent_frozen",
        .execution_profile_name = "default",
        .amplitudes_m = {2.5e-4},
        .steps_per_segment = 1,
        .max_steps = 0,
        .max_bisections = 3,
    };

    const auto result = run_reduced_rc_column_small_strain_beam_case_result(
        ReducedRCColumnStructuralRunSpec{
            .material_mode = ReducedRCColumnStructuralMaterialMode::elasticized,
            .beam_nodes = 3,
            .beam_axis_quadrature_family = BeamAxisQuadratureFamily::GaussLegendre,
            .axial_compression_force_mn = 0.02,
            .write_hysteresis_csv = false,
            .write_section_response_csv = false,
            .write_section_fiber_history_csv = false,
            .write_element_tangent_audit_csv = true,
            .print_progress = false,
        },
        out_dir.string(),
        cfg);

    if (result.element_tangent_audit_records.empty() ||
        result.section_tangent_audit_records.empty()) {
        return false;
    }

    const auto element_ok = std::all_of(
        result.element_tangent_audit_records.begin(),
        result.element_tangent_audit_records.end(),
        [](const auto& row) {
            return std::isfinite(row.tangent_fd_rel_error) &&
                   std::isfinite(row.max_column_rel_error) &&
                   row.tangent_fd_rel_error < 1.0e-8 &&
                   row.max_column_rel_error < 1.0e-8;
        });
    const auto section_ok = std::all_of(
        result.section_tangent_audit_records.begin(),
        result.section_tangent_audit_records.end(),
        [](const auto& row) {
            return std::isfinite(row.tangent_fd_rel_error) &&
                   std::isfinite(row.max_column_rel_error) &&
                   std::isfinite(row.rel_error_k00) &&
                   std::isfinite(row.rel_error_k0y) &&
                   std::isfinite(row.rel_error_ky0) &&
                   std::isfinite(row.rel_error_kyy) &&
                   row.tangent_fd_rel_error < 1.0e-8 &&
                   row.max_column_rel_error < 1.0e-8 &&
                   row.rel_error_k00 < 1.0e-8 &&
                   row.rel_error_k0y < 1.0e-8 &&
                   row.rel_error_ky0 < 1.0e-8 &&
                   row.rel_error_kyy < 1.0e-8;
        });

    return element_ok && section_ok;
}

bool segmented_continuation_preserves_the_baseline_history_contract()
{
    using fall_n::table_cyclic_validation::CyclicValidationRunConfig;
    using fall_n::validation_reboot::ReducedRCColumnContinuationKind;
    using fall_n::validation_reboot::ReducedRCColumnStructuralRunSpec;
    using fall_n::validation_reboot::run_reduced_rc_column_small_strain_beam_case_result;

    const CyclicValidationRunConfig cfg{
        .protocol_name = "reboot_segmented_continuation",
        .execution_profile_name = "default",
        .amplitudes_m = {2.5e-4, 5.0e-4},
        .steps_per_segment = 2,
        .max_steps = 0,
        .max_bisections = 3,
    };

    const auto result = run_reduced_rc_column_small_strain_beam_case_result(
        ReducedRCColumnStructuralRunSpec{
            .beam_nodes = 3,
            .beam_axis_quadrature_family = BeamAxisQuadratureFamily::GaussLegendre,
            .continuation_kind =
                ReducedRCColumnContinuationKind::
                    segmented_incremental_displacement_control,
            .write_hysteresis_csv = false,
            .write_section_response_csv = false,
            .print_progress = false,
        },
        "data/output/cyclic_validation/reboot_structural_matrix_segmented",
        cfg);

    return result.hysteresis_records.size() ==
               static_cast<std::size_t>(cfg.total_steps()) + 1u &&
           std::all_of(
               result.hysteresis_records.begin(),
               result.hysteresis_records.end(),
               [](const auto& row) {
                   return std::isfinite(row.base_shear) &&
                          std::isfinite(row.drift);
               });
}

bool reversal_guarded_continuation_refines_the_cyclic_history()
{
    using fall_n::table_cyclic_validation::CyclicValidationRunConfig;
    using fall_n::validation_reboot::ReducedRCColumnContinuationKind;
    using fall_n::validation_reboot::ReducedRCColumnStructuralRunSpec;
    using fall_n::validation_reboot::run_reduced_rc_column_small_strain_beam_case_result;

    const CyclicValidationRunConfig cfg{
        .protocol_name = "reboot_guarded_continuation",
        .execution_profile_name = "default",
        .amplitudes_m = {2.5e-4, 5.0e-4},
        .steps_per_segment = 2,
        .max_steps = 0,
        .max_bisections = 3,
    };

    const auto result = run_reduced_rc_column_small_strain_beam_case_result(
        ReducedRCColumnStructuralRunSpec{
            .beam_nodes = 3,
            .beam_axis_quadrature_family = BeamAxisQuadratureFamily::GaussLegendre,
            .continuation_kind =
                ReducedRCColumnContinuationKind::
                    reversal_guarded_incremental_displacement_control,
            .continuation_segment_substep_factor = 2,
            .write_hysteresis_csv = false,
            .write_section_response_csv = false,
            .print_progress = false,
        },
        "data/output/cyclic_validation/reboot_structural_matrix_guarded",
        cfg);

    const auto expected_history_points =
        static_cast<std::size_t>(
            fall_n::cyclic_segment_count(cfg.amplitudes_m.size()) *
                cfg.steps_per_segment * 2 +
            1);

    return result.hysteresis_records.size() == expected_history_points &&
           std::all_of(
               result.hysteresis_records.begin(),
               result.hysteresis_records.end(),
               [](const auto& row) {
                   return std::isfinite(row.base_shear) &&
                          std::isfinite(row.drift);
               });
}

bool structural_control_state_exports_increment_diagnostics()
{
    using fall_n::table_cyclic_validation::CyclicValidationRunConfig;
    using fall_n::validation_reboot::ReducedRCColumnContinuationKind;
    using fall_n::validation_reboot::ReducedRCColumnStructuralRunSpec;
    using fall_n::validation_reboot::run_reduced_rc_column_small_strain_beam_case_result;

    const CyclicValidationRunConfig cfg{
        .protocol_name = "reboot_structural_control_trace",
        .execution_profile_name = "default",
        .amplitudes_m = {2.5e-4, 5.0e-4},
        .steps_per_segment = 2,
        .max_steps = 0,
        .max_bisections = 3,
    };

    const auto result = run_reduced_rc_column_small_strain_beam_case_result(
        ReducedRCColumnStructuralRunSpec{
            .beam_nodes = 4,
            .beam_axis_quadrature_family = BeamAxisQuadratureFamily::GaussLegendre,
            .axial_compression_force_mn = 0.02,
            .use_equilibrated_axial_preload_stage = true,
            .axial_preload_steps = 3,
            .continuation_kind =
                ReducedRCColumnContinuationKind::
                    reversal_guarded_incremental_displacement_control,
            .continuation_segment_substep_factor = 2,
            .write_hysteresis_csv = false,
            .write_section_response_csv = false,
            .print_progress = false,
        },
        "data/output/cyclic_validation/reboot_structural_control_trace",
        cfg);

    if (!result.completed_successfully ||
        result.control_state_records.size() != result.hysteresis_records.size() ||
        result.control_state_records.empty()) {
        return false;
    }

    bool saw_reversal = false;
    for (const auto& row : result.control_state_records) {
        if (row.runtime_step < 0 ||
            row.runtime_p < -1.0e-14 ||
            row.runtime_p > 1.0 + 1.0e-14 ||
            row.accepted_substep_count < 0 ||
            row.max_bisection_level < 0 ||
            row.newton_iterations < 0.0 ||
            row.last_snes_reason < 0 ||
            !std::isfinite(row.last_function_norm) ||
            !row.converged) {
            return false;
        }

        if (row.step == 0) {
            continue;
        }

        if (row.protocol_branch_id < 1 ||
            row.branch_step_index < 1 ||
            row.target_increment_direction == 0 ||
            row.accepted_substep_count < 1 ||
            row.newton_iterations < static_cast<double>(row.accepted_substep_count)) {
            return false;
        }

        saw_reversal = saw_reversal || row.reversal_index > 0;
    }

    return saw_reversal;
}

bool structural_monotonic_protocol_uses_declared_history()
{
    using fall_n::table_cyclic_validation::CyclicValidationRunConfig;
    using fall_n::validation_reboot::ReducedRCColumnStructuralRunSpec;
    using fall_n::validation_reboot::run_reduced_rc_column_small_strain_beam_case_result;

    const CyclicValidationRunConfig cfg{
        .protocol_name = "monotonic_smoke",
        .execution_profile_name = "default",
        .amplitudes_m = {5.0e-4},
        .steps_per_segment = 4,
        .max_steps = 0,
        .max_bisections = 3,
    };

    const auto result = run_reduced_rc_column_small_strain_beam_case_result(
        ReducedRCColumnStructuralRunSpec{
            .beam_nodes = 4,
            .beam_axis_quadrature_family = BeamAxisQuadratureFamily::GaussLobatto,
            .axial_compression_force_mn = 0.02,
            .use_equilibrated_axial_preload_stage = true,
            .axial_preload_steps = 3,
            .write_hysteresis_csv = false,
            .write_section_response_csv = false,
            .print_progress = false,
        },
        "data/output/cyclic_validation/reboot_structural_monotonic_smoke",
        cfg);

    if (!result.completed_successfully ||
        result.hysteresis_records.size() !=
            static_cast<std::size_t>(cfg.steps_per_segment) + 1u ||
        result.control_state_records.size() != result.hysteresis_records.size()) {
        return false;
    }

    for (std::size_t i = 1; i < result.hysteresis_records.size(); ++i) {
        if (result.hysteresis_records[i].drift + 1.0e-14 <
            result.hysteresis_records[i - 1].drift) {
            return false;
        }

        const auto& row = result.control_state_records[i];
        if (row.reversal_index != 0 ||
            row.protocol_branch_id > 1 ||
            row.target_increment_direction < 0 ||
            row.actual_increment_direction < 0) {
            return false;
        }
    }

    const double final_target = cfg.max_amplitude_m();
    const double final_drift = result.hysteresis_records.back().drift;
    return std::isfinite(final_drift) &&
           std::abs(final_drift - final_target) <=
               std::max(1.0e-12, 1.0e-8 * std::abs(final_target));
}

bool structural_baseline_reports_positive_timing_surface()
{
    using fall_n::table_cyclic_validation::CyclicValidationRunConfig;
    using fall_n::validation_reboot::ReducedRCColumnStructuralRunSpec;
    using fall_n::validation_reboot::run_reduced_rc_column_small_strain_beam_case_result;

    const CyclicValidationRunConfig cfg{
        .protocol_name = "reboot_timing_surface",
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
            .write_hysteresis_csv = false,
            .write_section_response_csv = false,
            .print_progress = false,
        },
        "data/output/cyclic_validation/reboot_structural_matrix_timing",
        cfg);

    return result.completed_successfully &&
           result.timing.total_wall_seconds > 0.0 &&
           result.timing.solve_wall_seconds > 0.0 &&
           result.timing.output_write_wall_seconds >= 0.0 &&
           result.timing.total_wall_seconds >= result.timing.solve_wall_seconds;
}

bool structural_elasticized_mode_preserves_finite_response()
{
    using fall_n::table_cyclic_validation::CyclicValidationRunConfig;
    using fall_n::validation_reboot::ReducedRCColumnStructuralMaterialMode;
    using fall_n::validation_reboot::ReducedRCColumnStructuralRunSpec;
    using fall_n::validation_reboot::run_reduced_rc_column_small_strain_beam_case_result;

    const CyclicValidationRunConfig cfg{
        .protocol_name = "reboot_structural_elasticized",
        .execution_profile_name = "default",
        .amplitudes_m = {2.5e-4},
        .steps_per_segment = 1,
        .max_steps = 0,
        .max_bisections = 3,
    };

    const auto result = run_reduced_rc_column_small_strain_beam_case_result(
        ReducedRCColumnStructuralRunSpec{
            .material_mode = ReducedRCColumnStructuralMaterialMode::elasticized,
            .beam_nodes = 4,
            .beam_axis_quadrature_family = BeamAxisQuadratureFamily::GaussLegendre,
            .axial_compression_force_mn = 0.02,
            .write_hysteresis_csv = false,
            .write_section_response_csv = false,
            .print_progress = false,
        },
        "data/output/cyclic_validation/reboot_structural_matrix_elasticized",
        cfg);

    return result.completed_successfully &&
           std::all_of(
               result.hysteresis_records.begin(),
               result.hysteresis_records.end(),
               [](const auto& row) {
                   return std::isfinite(row.base_shear) &&
                          std::isfinite(row.drift);
               });
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
    report("equilibrated_axial_preload_stage_seeds_step_zero_section_force",
           equilibrated_axial_preload_stage_seeds_step_zero_section_force());
    report("base_side_moment_curvature_observable_is_finite_and_ordered",
           base_side_moment_curvature_observable_is_finite_and_ordered());
    report("section_observable_csv_contract_is_written",
           section_observable_csv_contract_is_written());
    report("element_tangent_audit_csv_contract_is_written",
           element_tangent_audit_csv_contract_is_written());
    report("elasticized_tangent_audits_remain_frozen_and_consistent",
           elasticized_tangent_audits_remain_frozen_and_consistent());
    report("segmented_continuation_preserves_the_baseline_history_contract",
           segmented_continuation_preserves_the_baseline_history_contract());
    report("reversal_guarded_continuation_refines_the_cyclic_history",
           reversal_guarded_continuation_refines_the_cyclic_history());
    report("structural_control_state_exports_increment_diagnostics",
           structural_control_state_exports_increment_diagnostics());
    report("structural_monotonic_protocol_uses_declared_history",
           structural_monotonic_protocol_uses_declared_history());
    report("structural_baseline_reports_positive_timing_surface",
           structural_baseline_reports_positive_timing_surface());
    report("structural_elasticized_mode_preserves_finite_response",
           structural_elasticized_mode_preserves_finite_response());

    std::cout << "\nSummary: " << passed << " passed, " << failed << " failed.\n";

    PetscFinalize();
    return failed == 0 ? 0 : 1;
}
