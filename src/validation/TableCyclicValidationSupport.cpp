#include "src/validation/TableCyclicValidationSupport.hh"

namespace fall_n::table_cyclic_validation {

double extract_base_shear_x_struct_model(
    const StructModel& model,
    const std::vector<std::size_t>& base_nodes)
{
    return extract_base_shear_x(model, base_nodes);
}

} // namespace fall_n::table_cyclic_validation
