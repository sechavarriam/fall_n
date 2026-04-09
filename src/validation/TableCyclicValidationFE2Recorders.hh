#ifndef FALL_N_SRC_VALIDATION_TABLE_CYCLIC_VALIDATION_FE2_RECORDERS_HH
#define FALL_N_SRC_VALIDATION_TABLE_CYCLIC_VALIDATION_FE2_RECORDERS_HH

#include "src/validation/TableCyclicValidationFE2Setup.hh"
#include "src/validation/TableCyclicValidationRuntimeIO.hh"

#include <string>

namespace fall_n::table_cyclic_validation {

[[nodiscard]] FE2RecorderBuffers initialize_fe2_recorders(
    const std::string& out_dir,
    const ValidationAnalysis& analysis);

} // namespace fall_n::table_cyclic_validation

#endif // FALL_N_SRC_VALIDATION_TABLE_CYCLIC_VALIDATION_FE2_RECORDERS_HH
