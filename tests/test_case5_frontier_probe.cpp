#include "../src/analysis/IncrementalControl.hh"
#include "../src/validation/TableCyclicValidationAPI.hh"
#include "../src/validation/TableCyclicValidationFE2Setup.hh"

#include <cassert>
#include <chrono>
#include <iostream>

#include <petsc.h>

using fall_n::CouplingTerminationReason;
using fall_n::table_cyclic_validation::CyclicValidationRunConfig;
using fall_n::table_cyclic_validation::ValidationProtocolPreset;
using fall_n::table_cyclic_validation::extract_base_shear_x;
using fall_n::table_cyclic_validation::make_validation_config;
using fall_n::table_cyclic_validation::apply_execution_profile;
using fall_n::table_cyclic_validation::build_fe2_case_context;

int main()
{
    int argc = 1;
    char arg0[] = "fall_n_case5_frontier_probe_test";
    char* argv_storage[] = {arg0, nullptr};
    char** argv = argv_storage;
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    auto cfg = make_validation_config(ValidationProtocolPreset::Extended50);
    apply_execution_profile(
        cfg,
        fall_n::table_cyclic_validation::ValidationExecutionProfile::
            FE2Crack50Exploratory);
    cfg.max_bisections = 2;

    {
        auto ctx = build_fe2_case_context(
            true,
            "data/output/cyclic_validation/case5_probe_test",
            cfg);

        auto& model = *ctx.model;
        auto& nl = *ctx.nl;
        auto& analysis = *ctx.analysis;
        const std::array<std::size_t, 4> slab_corners = {4, 7, 16, 19};

        auto scheme = make_control(
            [&slab_corners, &cfg]
            (double p, Vec /*f_full*/, Vec f_ext, auto* m) {
                VecSet(f_ext, 0.0);
                const double d = cfg.displacement(p);
                for (auto nid : slab_corners) {
                    m->update_imposed_value(nid, 0, d);
                }
            });

        nl.set_step_callback({});
        nl.begin_incremental(cfg.total_steps(), cfg.max_bisections, scheme);

        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = analysis.step();
        const auto t1 = std::chrono::steady_clock::now();

        const auto& report = analysis.last_report();
        const double p = report.attempted_state_valid
            ? report.attempted_macro_time
            : nl.current_time();
        const double drift = cfg.displacement(p);
        const double shear = extract_base_shear_x(model, ctx.base_nodes);

        std::cout << "[INFO] case5_frontier_probe step_ok=" << ok
                  << " termination=" << fall_n::to_string(report.termination_reason)
                  << " failed_submodels=" << report.failed_submodels
                  << " rollback=" << report.rollback_performed
                  << " iterations=" << report.iterations
                  << " macro_backtracking_attempts="
                  << report.macro_backtracking_attempts
                  << " macro_backtracking_succeeded="
                  << report.macro_backtracking_succeeded
                  << " macro_backtracking_last_alpha="
                  << report.macro_backtracking_last_alpha
                  << " macro_solver_reason="
                  << report.macro_solver_reason
                  << " macro_solver_iterations="
                  << report.macro_solver_iterations
                  << " macro_solver_function_norm="
                  << report.macro_solver_function_norm
                  << " max_force_residual_rel="
                  << report.max_force_residual_rel
                  << " max_tangent_residual_rel="
                  << report.max_tangent_residual_rel
                  << " p=" << p
                  << " drift=" << drift
                  << " base_shear=" << shear
                  << " wall_seconds="
                  << std::chrono::duration<double>(t1 - t0).count()
                  << "\n";

        assert(report.failed_submodels == 0);
        assert(report.termination_reason
               != CouplingTerminationReason::MicroSolveFailed);
    }

    PetscFinalize();
    std::cout << "[PASS] case5 frontier stays beyond local micro failure\n";
    return 0;
}
