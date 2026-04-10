#ifndef FALL_N_SRC_VALIDATION_TABLE_CYCLIC_VALIDATION_FE2_STEP_POSTPROCESS_HH
#define FALL_N_SRC_VALIDATION_TABLE_CYCLIC_VALIDATION_FE2_STEP_POSTPROCESS_HH

#include "src/validation/TableCyclicValidationFE2Setup.hh"
#include "src/validation/TableCyclicValidationRuntimeIO.hh"

#include <chrono>
#include <limits>
#include <vector>

namespace fall_n::table_cyclic_validation {

struct FE2StepDiagnostics {
    bool accepted{true};
    int total_cracked_gps{0};
    int total_cracks{0};
    int total_active_crack_history_points{0};
    int max_num_cracks_at_point{0};
    bool damage_scalar_available{false};
    double peak_submodel_damage_scalar{0.0};
    bool fracture_history_available{false};
    double most_compressive_submodel_sigma_o_max{0.0};
    double max_submodel_tau_o_max{0.0};
    double max_opening{0.0};
    double peak_damage{std::numeric_limits<double>::quiet_NaN()};
    std::vector<int> submodel_cracks{};
};

[[nodiscard]] FE2StepDiagnostics collect_fe2_step_diagnostics(
    StructModel& model,
    ValidationAnalysis& analysis,
    const DamageCriterion& damage_crit);

[[nodiscard]] FE2StepDiagnostics collect_fe2_failed_step_diagnostics(
    ValidationAnalysis& analysis);

void append_fe2_step_records(
    FE2RecorderBuffers& recorder_buffers,
    int step,
    double p,
    double d,
    double shear,
    const FE2StepDiagnostics& diagnostics,
    const ValidationAnalysis& analysis);

void print_fe2_step_progress(
    int step,
    int executed_steps,
    double p,
    double d,
    double shear,
    const FE2StepDiagnostics& diagnostics,
    int staggered_iterations,
    std::chrono::steady_clock::time_point start_time);

} // namespace fall_n::table_cyclic_validation

#endif // FALL_N_SRC_VALIDATION_TABLE_CYCLIC_VALIDATION_FE2_STEP_POSTPROCESS_HH
