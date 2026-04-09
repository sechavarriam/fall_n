#ifndef FALL_N_SRC_VALIDATION_TABLE_CYCLIC_VALIDATION_SUBMODEL_FACTORIES_HH
#define FALL_N_SRC_VALIDATION_TABLE_CYCLIC_VALIDATION_SUBMODEL_FACTORIES_HH

#include "src/validation/TableCyclicValidationAPI.hh"

#include <memory>

namespace fall_n {

struct MultiscaleSubModel;
struct ConcreteMaterialFactory;
struct RebarMaterialFactory;

namespace table_cyclic_validation {

[[nodiscard]] std::unique_ptr<ConcreteMaterialFactory>
make_table_submodel_concrete_factory(const MultiscaleSubModel& sub,
                                     const CyclicValidationRunConfig& cfg);

[[nodiscard]] std::unique_ptr<RebarMaterialFactory>
make_table_submodel_rebar_factory();

} // namespace table_cyclic_validation
} // namespace fall_n

#endif // FALL_N_SRC_VALIDATION_TABLE_CYCLIC_VALIDATION_SUBMODEL_FACTORIES_HH
