// =============================================================================
//  main_table_cyclic_validation.cpp
//
//  Thin CLI entrypoint for the progressive cyclic validation suite.
//  The heavy numerical implementations live in
//  src/validation/TableCyclicValidationStructural.cpp and
//  src/validation/TableCyclicValidationFE2.cpp so edits to one validation path
//  do not force recompilation of the entire driver.
// =============================================================================

#include "src/validation/TableCyclicValidationAPI.hh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <limits>
#include <print>
#include <stdexcept>
#include <string>
#include <vector>

#include <petsc.h>

using namespace fall_n::table_cyclic_validation;

namespace {

static void sep(char c = '=', int n = 72)
{
    for (int i = 0; i < n; ++i) {
        std::print("{}", c);
    }
    std::print("\n");
}

} // namespace

int main(int argc, char* argv[])
{
    setvbuf(stdout, nullptr, _IONBF, 0);

    std::string case_id = "all";
    ValidationProtocolPreset protocol_preset =
        ValidationProtocolPreset::Extended50;
    ValidationExecutionProfile execution_profile =
        ValidationExecutionProfile::Default;
    int steps_per_segment_override = -1;
    int max_steps_override = -1;
    int max_bisections_override = -1;
    int max_staggered_iterations_override = -1;
    double staggered_tol_override = -1.0;
    double staggered_relaxation_override = -1.0;
    int predictor_backtracking_attempts_override = -1;
    double predictor_backtracking_factor_override = -1.0;
    double predictor_min_symmetric_eigenvalue_override =
        std::numeric_limits<double>::quiet_NaN();
    int macro_step_cutback_attempts_override = -1;
    double macro_step_cutback_factor_override = -1.0;
    int macro_backtracking_attempts_override = -1;
    double macro_backtracking_factor_override = -1.0;
    int submodel_increment_steps_override = -1;
    int submodel_max_bisections_override = -1;
    bool submodel_enable_arc_length_from_start = false;
    int submodel_arc_length_threshold_override = -1;
    int submodel_adaptive_max_substeps_override = -1;
    int submodel_adaptive_max_bisections_override = -1;
    int submodel_output_interval_override = -1;
    int global_output_interval_override = -2;
    int submodel_snes_max_it_override = -1;
    double submodel_snes_atol_override = -1.0;
    double submodel_snes_rtol_override = -1.0;
    double min_crack_opening_override = -1.0;
    int submodel_use_consistent_material_tangent_override = -1;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--case" && i + 1 < argc) {
            case_id = argv[++i];
        } else if (arg == "--protocol" && i + 1 < argc) {
            const std::string value = argv[++i];
            if (value == "legacy20" || value == "legacy") {
                protocol_preset = ValidationProtocolPreset::Legacy20;
            } else if (value == "extended50" || value == "v3") {
                protocol_preset = ValidationProtocolPreset::Extended50;
            } else {
                throw std::invalid_argument(
                    "Unknown protocol. Use legacy20 or extended50.");
            }
        } else if (arg == "--fe2-profile" && i + 1 < argc) {
            const std::string value = argv[++i];
            if (value == "default") {
                execution_profile = ValidationExecutionProfile::Default;
            } else if (value == "crack50" || value == "fe2_crack50") {
                execution_profile =
                    ValidationExecutionProfile::FE2Crack50Exploratory;
            } else if (value == "frontier" || value == "fe2_frontier_audit") {
                execution_profile =
                    ValidationExecutionProfile::FE2FrontierAudit;
            } else {
                throw std::invalid_argument(
                    "Unknown FE2 execution profile. Use default, crack50, or frontier.");
            }
        } else if (arg == "--steps-per-segment" && i + 1 < argc) {
            steps_per_segment_override = std::stoi(argv[++i]);
        } else if (arg == "--max-steps" && i + 1 < argc) {
            max_steps_override = std::stoi(argv[++i]);
        } else if (arg == "--max-bisections" && i + 1 < argc) {
            max_bisections_override = std::stoi(argv[++i]);
        } else if (arg == "--staggered-iters" && i + 1 < argc) {
            max_staggered_iterations_override = std::stoi(argv[++i]);
        } else if (arg == "--staggered-tol" && i + 1 < argc) {
            staggered_tol_override = std::stod(argv[++i]);
        } else if (arg == "--staggered-relaxation" && i + 1 < argc) {
            staggered_relaxation_override = std::stod(argv[++i]);
        } else if (arg == "--predictor-backtracking-attempts" && i + 1 < argc) {
            predictor_backtracking_attempts_override = std::stoi(argv[++i]);
        } else if (arg == "--predictor-backtracking-factor" && i + 1 < argc) {
            predictor_backtracking_factor_override = std::stod(argv[++i]);
        } else if (arg == "--predictor-min-sym-eig" && i + 1 < argc) {
            predictor_min_symmetric_eigenvalue_override = std::stod(argv[++i]);
        } else if (arg == "--macro-step-cutback-attempts" && i + 1 < argc) {
            macro_step_cutback_attempts_override = std::stoi(argv[++i]);
        } else if (arg == "--macro-step-cutback-factor" && i + 1 < argc) {
            macro_step_cutback_factor_override = std::stod(argv[++i]);
        } else if (arg == "--macro-backtracking-attempts" && i + 1 < argc) {
            macro_backtracking_attempts_override = std::stoi(argv[++i]);
        } else if (arg == "--macro-backtracking-factor" && i + 1 < argc) {
            macro_backtracking_factor_override = std::stod(argv[++i]);
        } else if (arg == "--submodel-increments" && i + 1 < argc) {
            submodel_increment_steps_override = std::stoi(argv[++i]);
        } else if (arg == "--submodel-bisections" && i + 1 < argc) {
            submodel_max_bisections_override = std::stoi(argv[++i]);
        } else if (arg == "--submodel-adaptive-from-start") {
            submodel_enable_arc_length_from_start = true;
        } else if (arg == "--submodel-arc-threshold" && i + 1 < argc) {
            submodel_arc_length_threshold_override = std::stoi(argv[++i]);
        } else if (arg == "--submodel-adaptive-max-substeps" && i + 1 < argc) {
            submodel_adaptive_max_substeps_override = std::stoi(argv[++i]);
        } else if (arg == "--submodel-adaptive-max-bisections" && i + 1 < argc) {
            submodel_adaptive_max_bisections_override = std::stoi(argv[++i]);
        } else if (arg == "--submodel-output-interval" && i + 1 < argc) {
            submodel_output_interval_override = std::stoi(argv[++i]);
        } else if (arg == "--global-output-interval" && i + 1 < argc) {
            global_output_interval_override = std::stoi(argv[++i]);
        } else if (arg == "--submodel-snes-max-it" && i + 1 < argc) {
            submodel_snes_max_it_override = std::stoi(argv[++i]);
        } else if (arg == "--submodel-snes-atol" && i + 1 < argc) {
            submodel_snes_atol_override = std::stod(argv[++i]);
        } else if (arg == "--submodel-snes-rtol" && i + 1 < argc) {
            submodel_snes_rtol_override = std::stod(argv[++i]);
        } else if (arg == "--min-crack-opening" && i + 1 < argc) {
            min_crack_opening_override = std::stod(argv[++i]);
        } else if (arg == "--submodel-consistent-material-tangent") {
            submodel_use_consistent_material_tangent_override = 1;
        } else if (arg == "--submodel-secant-material-tangent") {
            submodel_use_consistent_material_tangent_override = 0;
        }
    }

    auto cfg = make_validation_config(protocol_preset);
    apply_execution_profile(cfg, execution_profile);
    if (steps_per_segment_override > 0) {
        cfg.steps_per_segment = steps_per_segment_override;
    }
    if (max_steps_override > 0) {
        cfg.max_steps = max_steps_override;
    }
    if (max_bisections_override >= 0) {
        cfg.max_bisections = max_bisections_override;
    }
    if (max_staggered_iterations_override > 0) {
        cfg.max_staggered_iterations = max_staggered_iterations_override;
    }
    if (staggered_tol_override > 0.0) {
        cfg.staggered_tol = staggered_tol_override;
    }
    if (staggered_relaxation_override >= 0.0) {
        cfg.staggered_relaxation = staggered_relaxation_override;
    }
    if (predictor_backtracking_attempts_override >= 0) {
        cfg.predictor_admissibility_backtrack_attempts =
            predictor_backtracking_attempts_override;
    }
    if (predictor_backtracking_factor_override >= 0.0) {
        cfg.predictor_admissibility_backtrack_factor =
            predictor_backtracking_factor_override;
    }
    if (std::isfinite(predictor_min_symmetric_eigenvalue_override)) {
        cfg.predictor_admissibility_min_symmetric_eigenvalue =
            predictor_min_symmetric_eigenvalue_override;
    }
    if (macro_step_cutback_attempts_override >= 0) {
        cfg.macro_step_cutback_attempts = macro_step_cutback_attempts_override;
    }
    if (macro_step_cutback_factor_override >= 0.0) {
        cfg.macro_step_cutback_factor = macro_step_cutback_factor_override;
    }
    if (macro_backtracking_attempts_override >= 0) {
        cfg.macro_failure_backtrack_attempts =
            macro_backtracking_attempts_override;
    }
    if (macro_backtracking_factor_override >= 0.0) {
        cfg.macro_failure_backtrack_factor = macro_backtracking_factor_override;
    }
    if (submodel_increment_steps_override > 0) {
        cfg.submodel_increment_steps = submodel_increment_steps_override;
    }
    if (submodel_max_bisections_override > 0) {
        cfg.submodel_max_bisections = submodel_max_bisections_override;
    }
    if (submodel_enable_arc_length_from_start) {
        cfg.submodel_enable_arc_length_from_start = true;
    }
    if (submodel_arc_length_threshold_override >= 0) {
        cfg.submodel_arc_length_threshold = submodel_arc_length_threshold_override;
    }
    if (submodel_adaptive_max_substeps_override > 0) {
        cfg.submodel_adaptive_max_substeps =
            submodel_adaptive_max_substeps_override;
    }
    if (submodel_adaptive_max_bisections_override > 0) {
        cfg.submodel_adaptive_max_bisections =
            submodel_adaptive_max_bisections_override;
    }
    if (submodel_output_interval_override > 0) {
        cfg.submodel_output_interval = submodel_output_interval_override;
    } else if (submodel_output_interval_override == 0) {
        cfg.submodel_output_interval = 0;
    }
    if (global_output_interval_override >= -1) {
        cfg.global_output_interval = std::max(global_output_interval_override, 0);
    }
    if (submodel_snes_max_it_override > 0) {
        cfg.submodel_snes_max_it = submodel_snes_max_it_override;
    }
    if (submodel_snes_atol_override > 0.0) {
        cfg.submodel_snes_atol = submodel_snes_atol_override;
    }
    if (submodel_snes_rtol_override > 0.0) {
        cfg.submodel_snes_rtol = submodel_snes_rtol_override;
    }
    if (min_crack_opening_override >= 0.0) {
        cfg.min_crack_opening = min_crack_opening_override;
    }
    if (submodel_use_consistent_material_tangent_override >= 0) {
        cfg.submodel_use_consistent_material_tangent =
            (submodel_use_consistent_material_tangent_override == 1);
    }

    const auto takes_value = [](const std::string& arg) {
        return arg == "--case"
            || arg == "--protocol"
            || arg == "--fe2-profile"
            || arg == "--steps-per-segment"
            || arg == "--max-steps"
            || arg == "--max-bisections"
            || arg == "--staggered-iters"
            || arg == "--staggered-tol"
            || arg == "--staggered-relaxation"
            || arg == "--predictor-backtracking-attempts"
            || arg == "--predictor-backtracking-factor"
            || arg == "--predictor-min-sym-eig"
            || arg == "--macro-step-cutback-attempts"
            || arg == "--macro-step-cutback-factor"
            || arg == "--macro-backtracking-attempts"
            || arg == "--macro-backtracking-factor"
            || arg == "--submodel-increments"
            || arg == "--submodel-bisections"
            || arg == "--submodel-arc-threshold"
            || arg == "--submodel-adaptive-max-substeps"
            || arg == "--submodel-adaptive-max-bisections"
            || arg == "--submodel-output-interval"
            || arg == "--global-output-interval"
            || arg == "--submodel-snes-max-it"
            || arg == "--submodel-snes-atol"
            || arg == "--submodel-snes-rtol"
            || arg == "--min-crack-opening";
    };

    std::vector<char*> petsc_args;
    petsc_args.reserve(static_cast<std::size_t>(argc) + 1);
    petsc_args.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--submodel-adaptive-from-start") {
            continue;
        }
        if (arg == "--submodel-consistent-material-tangent"
            || arg == "--submodel-secant-material-tangent")
        {
            continue;
        }
        if (takes_value(arg) && i + 1 < argc) {
            ++i;
            continue;
        }
        petsc_args.push_back(argv[i]);
    }
    petsc_args.push_back(nullptr);
    int petsc_argc = static_cast<int>(petsc_args.size()) - 1;
    char** petsc_argv = petsc_args.data();

    PetscInitialize(&petsc_argc, &petsc_argv, nullptr, nullptr);
    PetscOptionsSetValue(nullptr, "-snes_monitor_cancel", "");
    PetscOptionsSetValue(nullptr, "-ksp_monitor_cancel", "");

    int exit_code = 0;
    try {
        exit_code = [&]() -> int {
            sep('=');
            std::println("  fall_n — Cyclic Validation of FE² Pipeline");
            std::println("  Case: {}", case_id);
            std::println("  Active protocol: {} ({})",
                         format_protocol_amplitudes_mm(cfg),
                         cfg.protocol_name);
            std::println("  Execution profile: {}", cfg.execution_profile_name);
            std::println("  Steps: {}  |  Max bisection: {}",
                         cfg.total_steps(), cfg.max_bisections);
            if (cfg.max_steps > 0) {
                std::println("  Manual FE2 step cap: {} steps "
                             "(preserves protocol parametrization; only cases 4/5)",
                             cfg.max_steps);
            }
            std::println("  Sub-model ramp: {} increments  |  {} bisections  |  "
                         "arc-start={} threshold={} adaptive-budget={}/{}  |  "
                         "SNES max_it={} atol={:.2e} rtol={:.2e}  |  "
                         "material tangent={}",
                         cfg.submodel_increment_steps,
                         cfg.submodel_max_bisections,
                         cfg.submodel_enable_arc_length_from_start ? "on" : "off",
                         cfg.submodel_arc_length_threshold,
                         cfg.submodel_adaptive_max_substeps,
                         cfg.submodel_adaptive_max_bisections,
                         cfg.submodel_snes_max_it,
                         cfg.submodel_snes_atol,
                         cfg.submodel_snes_rtol,
                         cfg.submodel_use_consistent_material_tangent
                             ? "consistent"
                             : "fracture-sec");
            std::println("  FE2 outer loop: max-staggered={}  tol={:.2e}  relax={:.2f}",
                         cfg.max_staggered_iterations,
                         cfg.staggered_tol,
                         cfg.staggered_relaxation);
            std::println("  FE2 predictor filter: attempts={}  factor={:.2f}  min-sym-eig={:.2e}",
                         cfg.predictor_admissibility_backtrack_attempts,
                         cfg.predictor_admissibility_backtrack_factor,
                         cfg.predictor_admissibility_min_symmetric_eigenvalue);
            std::println("  FE2 macro continuation: cutback-attempts={}  cutback-factor={:.2f}  "
                         "backtracking-attempts={}  backtracking-factor={:.2f}",
                         cfg.macro_step_cutback_attempts,
                         cfg.macro_step_cutback_factor,
                         cfg.macro_failure_backtrack_attempts,
                         cfg.macro_failure_backtrack_factor);
            std::println("  Output cadence: global-vtk={}  |  submodel-vtk={}  "
                         "|  min-crack-opening={:.2e}",
                         cfg.global_output_interval,
                         cfg.submodel_output_interval,
                         cfg.min_crack_opening);
            sep('=');

            std::filesystem::create_directories(OUT_ROOT);

            std::map<std::string, std::vector<StepRecord>> results;
            const auto should_run = [&](const std::string& id) {
                return case_id == "all" || case_id == id;
            };

            if (should_run("0")) {
                const auto dir = OUT_ROOT + "case0";
                std::filesystem::create_directories(dir);
                results["0"] = run_case0(dir, cfg);
            }

            for (std::size_t nodes = 2; nodes <= 10; ++nodes) {
                const std::string id =
                    "1" + std::string(1, static_cast<char>('a' + (nodes - 2)));
                if (should_run(id)) {
                    const auto dir = OUT_ROOT + "case" + id;
                    std::filesystem::create_directories(dir);
                    results[id] = run_case1_by_nodes(nodes, dir, cfg);
                }
            }

            for (char variant : {'a', 'b', 'c'}) {
                const std::string id = "2" + std::string(1, variant);
                if (should_run(id)) {
                    const auto dir = OUT_ROOT + "case" + id;
                    std::filesystem::create_directories(dir);
                    results[id] = run_case2_variant(variant, dir, cfg);
                }
            }

            if (should_run("3")) {
                const auto dir = OUT_ROOT + "case3";
                std::filesystem::create_directories(dir);
                results["3"] = run_case3(dir, cfg);
            }
            if (should_run("4")) {
                const auto dir = OUT_ROOT + "case4";
                std::filesystem::create_directories(dir);
                results["4"] = run_case_fe2(false, dir, cfg);
            }
            if (should_run("5")) {
                const auto dir = OUT_ROOT + "case5";
                std::filesystem::create_directories(dir);
                results["5"] = run_case_fe2(true, dir, cfg);
            }

            sep('=');
            std::println("\n  SUMMARY");
            sep('-');
            std::println("  {:>6s}  {:>8s}  {:>12s}  {:>12s}",
                         "Case", "Records", "Max drift", "Max shear");
            sep('-');

            for (const auto& [id, recs] : results) {
                double max_d = 0.0;
                double max_v = 0.0;
                for (const auto& r : recs) {
                    max_d = std::max(max_d, std::abs(r.drift));
                    max_v = std::max(max_v, std::abs(r.base_shear));
                }
                std::println("  {:>6s}  {:8d}  {:12.4e}  {:12.4e}",
                             id, static_cast<int>(recs.size()), max_d, max_v);
            }

            sep('=');
            return 0;
        }();
    } catch (const std::exception& ex) {
        std::println(stderr, "ERROR: {}", ex.what());
        exit_code = 1;
    } catch (...) {
        std::println(stderr, "ERROR: unknown exception");
        exit_code = 1;
    }

    PetscFinalize();
    return exit_code;
}
