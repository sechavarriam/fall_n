#include "../src/validation/TableCyclicValidationAPI.hh"
#include "../src/validation/TableCyclicValidationFE2Setup.hh"

#include <cassert>
#include <iostream>

#include <petsc.h>

using fall_n::table_cyclic_validation::CyclicValidationRunConfig;
using fall_n::table_cyclic_validation::ValidationExecutionProfile;
using fall_n::table_cyclic_validation::ValidationProtocolPreset;
using fall_n::table_cyclic_validation::apply_execution_profile;
using fall_n::table_cyclic_validation::build_fe2_case_context;
using fall_n::table_cyclic_validation::make_validation_config;

static void test_default_config_preserves_extended50_contract()
{
    const auto cfg = make_validation_config(ValidationProtocolPreset::Extended50);
    assert(cfg.protocol_name == "extended50");
    assert(cfg.execution_profile_name == "default");
    assert(cfg.steps_per_segment == 10);
    assert(cfg.enable_turning_point_checkpoints);
    assert(cfg.max_turning_point_restarts == 2);
    assert(cfg.predictor_admissibility_backtrack_attempts == 0);
    assert(cfg.predictor_admissibility_backtrack_factor == 0.5);
    assert(cfg.predictor_admissibility_min_symmetric_eigenvalue == 0.0);
    assert(cfg.macro_step_cutback_attempts == 0);
    assert(cfg.macro_step_cutback_factor == 0.5);
    assert(cfg.macro_failure_backtrack_attempts == 0);
    assert(!cfg.submodel_use_consistent_material_tangent);
    assert(cfg.global_output_interval == 1);
    assert(cfg.submodel_output_interval == 10);
    assert(cfg.submodel_tail_rescue_attempts == 0);
    assert(cfg.submodel_tail_rescue_progress_threshold == 0.75);
    assert(cfg.max_amplitude_m() == 0.050);
    assert(cfg.is_turning_point_step(10));
    assert(!cfg.is_turning_point_step(9));
    std::cout << "[PASS] test_default_config_preserves_extended50_contract\n";
}

static void test_fe2_crack50_profile_applies_runtime_tuning()
{
    auto cfg = make_validation_config(ValidationProtocolPreset::Extended50);
    apply_execution_profile(cfg,
                            ValidationExecutionProfile::FE2Crack50Exploratory);
    assert(cfg.execution_profile_name == "fe2_crack50");
    assert(cfg.steps_per_segment == 1);
    assert(cfg.predictor_admissibility_backtrack_attempts == 3);
    assert(cfg.predictor_admissibility_backtrack_factor == 0.5);
    assert(cfg.predictor_admissibility_min_symmetric_eigenvalue == 0.0);
    assert(cfg.macro_step_cutback_attempts == 2);
    assert(cfg.macro_step_cutback_factor == 0.5);
    assert(cfg.macro_failure_backtrack_attempts == 3);
    assert(cfg.macro_failure_backtrack_factor == 0.5);
    assert(cfg.submodel_increment_steps == 12);
    assert(cfg.submodel_max_bisections == 4);
    assert(cfg.submodel_enable_arc_length_from_start);
    assert(cfg.submodel_arc_length_threshold == 1);
    assert(cfg.submodel_adaptive_max_substeps == 48);
    assert(cfg.submodel_adaptive_max_bisections == 12);
    assert(cfg.submodel_tail_rescue_attempts == 2);
    assert(cfg.submodel_tail_rescue_progress_threshold == 0.75);
    assert(cfg.submodel_tail_rescue_substep_bonus == 24);
    assert(cfg.submodel_tail_rescue_bisection_bonus == 6);
    assert(cfg.submodel_tail_rescue_initial_fraction == 0.5);
    assert(cfg.submodel_snes_max_it == 150);
    assert(cfg.enable_turning_point_checkpoints);
    assert(cfg.max_turning_point_restarts == 3);
    assert(cfg.restart_snes_max_it_bonus == 50);
    assert(!cfg.submodel_use_consistent_material_tangent);
    assert(cfg.submodel_output_interval == 0);
    assert(cfg.global_output_interval == 0);
    assert(cfg.min_crack_opening == 0.0);
    assert(cfg.max_amplitude_m() == 0.050);
    std::cout << "[PASS] test_fe2_crack50_profile_applies_runtime_tuning\n";
}

static void test_fe2_frontier_profile_applies_fast_audit_tuning()
{
    auto cfg = make_validation_config(ValidationProtocolPreset::Extended50);
    apply_execution_profile(cfg,
                            ValidationExecutionProfile::FE2FrontierAudit);
    assert(cfg.execution_profile_name == "fe2_frontier_audit");
    assert(cfg.steps_per_segment == 1);
    assert(cfg.max_bisections == 1);
    assert(cfg.max_staggered_iterations == 2);
    assert(cfg.predictor_admissibility_backtrack_attempts == 1);
    assert(cfg.macro_step_cutback_attempts == 1);
    assert(cfg.macro_failure_backtrack_attempts == 1);
    assert(cfg.submodel_increment_steps == 4);
    assert(cfg.submodel_max_bisections == 1);
    assert(cfg.submodel_enable_arc_length_from_start);
    assert(cfg.submodel_arc_length_threshold == 1);
    assert(cfg.submodel_adaptive_max_substeps == 12);
    assert(cfg.submodel_adaptive_max_bisections == 4);
    assert(cfg.submodel_tail_rescue_attempts == 1);
    assert(cfg.submodel_tail_rescue_substep_bonus == 8);
    assert(cfg.submodel_tail_rescue_bisection_bonus == 2);
    assert(cfg.submodel_snes_max_it == 60);
    assert(!cfg.enable_turning_point_checkpoints);
    assert(cfg.max_turning_point_restarts == 0);
    assert(cfg.submodel_output_interval == 0);
    assert(cfg.global_output_interval == 0);
    assert(cfg.min_crack_opening == 0.0);
    std::cout << "[PASS] test_fe2_frontier_profile_applies_fast_audit_tuning\n";
}

static void test_fe2_setup_keeps_submodels_alive_after_context_return()
{
    int argc = 1;
    char arg0[] = "fall_n_cyclic_validation_api_test";
    char* argv_storage[] = {arg0, nullptr};
    char** argv = argv_storage;
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    auto cfg = make_validation_config(ValidationProtocolPreset::Extended50);
    apply_execution_profile(cfg,
                            ValidationExecutionProfile::FE2Crack50Exploratory);

    {
        const auto ctx = build_fe2_case_context(
            false,
            "data/output/cyclic_validation/test_context_lifetime",
            cfg);

        assert(ctx.coordinator);
        assert(ctx.analysis);
        assert(ctx.analysis->model().num_local_models() == 4);
        assert(ctx.coordinator->sub_models().size()
               == ctx.analysis->model().num_local_models());

        const auto& locals = ctx.analysis->model().local_models();
        for (std::size_t i = 0; i < locals.size(); ++i) {
            assert(&locals[i].sub_model() == &ctx.coordinator->sub_models()[i]);
            assert(!locals[i].sub_model().face_min_z_ids.empty());
            assert(!locals[i].sub_model().face_max_z_ids.empty());
        }
    }

    PetscFinalize();

    std::cout
        << "[PASS] test_fe2_setup_keeps_submodels_alive_after_context_return\n";
}

int main()
{
    test_default_config_preserves_extended50_contract();
    test_fe2_crack50_profile_applies_runtime_tuning();
    test_fe2_frontier_profile_applies_fast_audit_tuning();
    test_fe2_setup_keeps_submodels_alive_after_context_return();
    std::cout << "\n=== All cyclic validation API tests PASSED ===\n";
    return 0;
}
