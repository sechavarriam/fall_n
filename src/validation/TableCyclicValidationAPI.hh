#ifndef FALL_N_SRC_VALIDATION_TABLE_CYCLIC_VALIDATION_API_HH
#define FALL_N_SRC_VALIDATION_TABLE_CYCLIC_VALIDATION_API_HH

#include "src/utils/CyclicProtocol.hh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace fall_n::table_cyclic_validation {

#ifdef FALL_N_SOURCE_DIR
inline const std::string BASE = std::string(FALL_N_SOURCE_DIR) + "/";
#else
inline const std::string BASE = "./";
#endif

inline const std::string OUT_ROOT = BASE + "data/output/cyclic_validation/";

enum class ValidationProtocolPreset {
    Legacy20,
    Extended50
};

enum class ValidationExecutionProfile {
    Default,
    FE2Crack50Exploratory
};

struct CyclicValidationRunConfig {
    std::string protocol_name{};
    std::string execution_profile_name{"default"};
    std::vector<double> amplitudes_m{};
    int steps_per_segment{10};
    int max_steps{0};
    int max_bisections{8};
    int max_staggered_iterations{5};
    double staggered_tol{0.05};
    double staggered_relaxation{0.75};
    int submodel_increment_steps{30};
    int submodel_max_bisections{6};
    bool submodel_enable_arc_length_from_start{false};
    int submodel_arc_length_threshold{3};
    int submodel_adaptive_max_substeps{30};
    int submodel_adaptive_max_bisections{10};
    int submodel_snes_max_it{100};
    double submodel_snes_atol{2.0};
    double submodel_snes_rtol{1.0e-3};
    bool enable_turning_point_checkpoints{true};
    int max_turning_point_restarts{2};
    int restart_submodel_increment_step_bonus{4};
    int restart_submodel_bisection_bonus{2};
    int restart_adaptive_substep_bonus{12};
    int restart_adaptive_bisection_bonus{4};
    int restart_snes_max_it_bonus{25};
    int submodel_output_interval{10};
    int global_output_interval{1};
    double min_crack_opening{1.0e-4};

    [[nodiscard]] int total_steps() const noexcept {
        return fall_n::cyclic_step_count(amplitudes_m.size(), steps_per_segment);
    }

    [[nodiscard]] double displacement(double p) const {
        return fall_n::cyclic_displacement(
            p, std::span<const double>{amplitudes_m});
    }

    [[nodiscard]] double max_amplitude_m() const noexcept {
        return amplitudes_m.empty() ? 0.0 : amplitudes_m.back();
    }

    [[nodiscard]] bool is_turning_point_step(int step) const noexcept {
        return steps_per_segment > 0
            && step > 0
            && (step % steps_per_segment) == 0;
    }
};

[[nodiscard]] inline CyclicValidationRunConfig
make_validation_config(ValidationProtocolPreset preset)
{
    if (preset == ValidationProtocolPreset::Legacy20) {
        return CyclicValidationRunConfig{
            .protocol_name = "legacy20",
            .execution_profile_name = "default",
            .amplitudes_m = {fall_n::kLegacyCyclicAmplitudesM.begin(),
                             fall_n::kLegacyCyclicAmplitudesM.end()},
            .steps_per_segment = 10,
            .max_steps = 0,
            .max_bisections = 6,
            .max_staggered_iterations = 3,
            .staggered_tol = 0.05,
            .staggered_relaxation = 0.80,
            .submodel_increment_steps = 20,
            .submodel_max_bisections = 4,
            .submodel_enable_arc_length_from_start = false,
            .submodel_arc_length_threshold = 3,
            .submodel_adaptive_max_substeps = 30,
            .submodel_adaptive_max_bisections = 10,
            .submodel_snes_max_it = 100,
            .submodel_snes_atol = 2.0,
            .submodel_snes_rtol = 1.0e-3,
            .enable_turning_point_checkpoints = true,
            .max_turning_point_restarts = 1,
            .restart_submodel_increment_step_bonus = 4,
            .restart_submodel_bisection_bonus = 2,
            .restart_adaptive_substep_bonus = 12,
            .restart_adaptive_bisection_bonus = 4,
            .restart_snes_max_it_bonus = 25,
            .submodel_output_interval = 20,
            .global_output_interval = 1,
            .min_crack_opening = 5.0e-4
        };
    }

    return CyclicValidationRunConfig{
        .protocol_name = "extended50",
        .execution_profile_name = "default",
        .amplitudes_m = {fall_n::kExtendedValidationAmplitudesM.begin(),
                         fall_n::kExtendedValidationAmplitudesM.end()},
        .steps_per_segment = 10,
        .max_steps = 0,
        .max_bisections = 8,
        .max_staggered_iterations = 5,
        .staggered_tol = 0.05,
        .staggered_relaxation = 0.75,
        .submodel_increment_steps = 30,
        .submodel_max_bisections = 6,
        .submodel_enable_arc_length_from_start = false,
        .submodel_arc_length_threshold = 3,
        .submodel_adaptive_max_substeps = 30,
        .submodel_adaptive_max_bisections = 10,
        .submodel_snes_max_it = 100,
        .submodel_snes_atol = 2.0,
        .submodel_snes_rtol = 1.0e-3,
        .enable_turning_point_checkpoints = true,
        .max_turning_point_restarts = 2,
        .restart_submodel_increment_step_bonus = 4,
        .restart_submodel_bisection_bonus = 2,
        .restart_adaptive_substep_bonus = 16,
        .restart_adaptive_bisection_bonus = 4,
        .restart_snes_max_it_bonus = 25,
        .submodel_output_interval = 10,
        .global_output_interval = 1,
        .min_crack_opening = 1.0e-4
    };
}

inline void apply_execution_profile(CyclicValidationRunConfig& cfg,
                                    ValidationExecutionProfile profile)
{
    switch (profile) {
    case ValidationExecutionProfile::Default:
        cfg.execution_profile_name = "default";
        break;
    case ValidationExecutionProfile::FE2Crack50Exploratory:
        cfg.execution_profile_name = "fe2_crack50";
        cfg.steps_per_segment = 1;
        cfg.submodel_increment_steps = 8;
        cfg.submodel_max_bisections = 3;
        cfg.submodel_enable_arc_length_from_start = true;
        cfg.submodel_arc_length_threshold = 1;
        cfg.submodel_adaptive_max_substeps = 48;
        cfg.submodel_adaptive_max_bisections = 12;
        cfg.enable_turning_point_checkpoints = true;
        cfg.max_turning_point_restarts = 3;
        cfg.restart_submodel_increment_step_bonus = 6;
        cfg.restart_submodel_bisection_bonus = 3;
        cfg.restart_adaptive_substep_bonus = 24;
        cfg.restart_adaptive_bisection_bonus = 6;
        cfg.restart_snes_max_it_bonus = 50;
        cfg.submodel_output_interval = 0;
        cfg.global_output_interval = 0;
        cfg.min_crack_opening = 0.0;
        break;
    }
}

[[nodiscard]] inline std::string
describe_execution_profile(ValidationExecutionProfile profile)
{
    switch (profile) {
    case ValidationExecutionProfile::Default:
        return "default";
    case ValidationExecutionProfile::FE2Crack50Exploratory:
        return "fe2_crack50";
    }
    return "unknown";
}

[[nodiscard]] inline std::string
format_protocol_amplitudes_mm(const CyclicValidationRunConfig& cfg)
{
    std::ostringstream oss;
    oss << "\xC2\xB1";
    for (std::size_t i = 0; i < cfg.amplitudes_m.size(); ++i) {
        if (i > 0) oss << "/";
        const double mm = cfg.amplitudes_m[i] * 1.0e3;
        if (std::abs(mm - std::round(mm)) < 1.0e-12) {
            oss << static_cast<int>(std::lround(mm));
        } else {
            oss << std::fixed << std::setprecision(1) << mm;
        }
    }
    oss << " mm";
    return oss.str();
}

struct StepRecord {
    int    step;
    double p;
    double drift;
    double base_shear;
};

std::vector<StepRecord> run_case0(const std::string& out_dir,
                                  const CyclicValidationRunConfig& cfg);

std::vector<StepRecord> run_case1_by_nodes(std::size_t nodes,
                                           const std::string& out_dir,
                                           const CyclicValidationRunConfig& cfg);

std::vector<StepRecord> run_case2_variant(char variant,
                                          const std::string& out_dir,
                                          const CyclicValidationRunConfig& cfg);

std::vector<StepRecord> run_case3(const std::string& out_dir,
                                  const CyclicValidationRunConfig& cfg);

std::vector<StepRecord> run_case_fe2(bool two_way,
                                     const std::string& out_dir,
                                     const CyclicValidationRunConfig& cfg);

} // namespace fall_n::table_cyclic_validation

#endif
