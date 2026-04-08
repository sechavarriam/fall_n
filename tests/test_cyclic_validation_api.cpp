#include "../src/validation/TableCyclicValidationAPI.hh"

#include <cassert>
#include <iostream>

using fall_n::table_cyclic_validation::CyclicValidationRunConfig;
using fall_n::table_cyclic_validation::ValidationExecutionProfile;
using fall_n::table_cyclic_validation::ValidationProtocolPreset;
using fall_n::table_cyclic_validation::apply_execution_profile;
using fall_n::table_cyclic_validation::make_validation_config;

static void test_default_config_preserves_extended50_contract()
{
    const auto cfg = make_validation_config(ValidationProtocolPreset::Extended50);
    assert(cfg.protocol_name == "extended50");
    assert(cfg.execution_profile_name == "default");
    assert(cfg.steps_per_segment == 10);
    assert(cfg.global_output_interval == 1);
    assert(cfg.submodel_output_interval == 10);
    assert(cfg.max_amplitude_m() == 0.050);
    std::cout << "[PASS] test_default_config_preserves_extended50_contract\n";
}

static void test_fe2_crack50_profile_applies_runtime_tuning()
{
    auto cfg = make_validation_config(ValidationProtocolPreset::Extended50);
    apply_execution_profile(cfg,
                            ValidationExecutionProfile::FE2Crack50Exploratory);
    assert(cfg.execution_profile_name == "fe2_crack50");
    assert(cfg.steps_per_segment == 1);
    assert(cfg.submodel_increment_steps == 8);
    assert(cfg.submodel_max_bisections == 3);
    assert(cfg.submodel_enable_arc_length_from_start);
    assert(cfg.submodel_arc_length_threshold == 1);
    assert(cfg.submodel_adaptive_max_substeps == 48);
    assert(cfg.submodel_adaptive_max_bisections == 12);
    assert(cfg.submodel_output_interval == 0);
    assert(cfg.global_output_interval == 0);
    assert(cfg.min_crack_opening == 0.0);
    assert(cfg.max_amplitude_m() == 0.050);
    std::cout << "[PASS] test_fe2_crack50_profile_applies_runtime_tuning\n";
}

int main()
{
    test_default_config_preserves_extended50_contract();
    test_fe2_crack50_profile_applies_runtime_tuning();
    std::cout << "\n=== All cyclic validation API tests PASSED ===\n";
    return 0;
}
