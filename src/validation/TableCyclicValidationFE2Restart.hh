#ifndef FALL_N_SRC_VALIDATION_TABLE_CYCLIC_VALIDATION_FE2_RESTART_HH
#define FALL_N_SRC_VALIDATION_TABLE_CYCLIC_VALIDATION_FE2_RESTART_HH

#include "src/validation/TableCyclicValidationFE2Setup.hh"
#include "src/validation/TableCyclicValidationRuntimeIO.hh"
#include "src/validation/TableCyclicValidationSupport.hh"

#include <vector>

namespace fall_n::table_cyclic_validation {

void retune_fe2_local_models(
    ValidationAnalysis& analysis,
    const CyclicValidationRunConfig& cfg,
    int restart_attempt);

bool try_restart_from_turning_point(
    int& step,
    ValidationAnalysis& analysis,
    const CyclicValidationRunConfig& cfg,
    FE2TurningPointFrame<ValidationAnalysis::RestartBundle>& last_turning_point,
    std::vector<StepRecord>& records,
    FE2RecorderBuffers& recorder_buffers);

} // namespace fall_n::table_cyclic_validation

#endif // FALL_N_SRC_VALIDATION_TABLE_CYCLIC_VALIDATION_FE2_RESTART_HH
