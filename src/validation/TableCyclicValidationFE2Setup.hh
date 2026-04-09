#ifndef FALL_N_SRC_VALIDATION_TABLE_CYCLIC_VALIDATION_FE2_SETUP_HH
#define FALL_N_SRC_VALIDATION_TABLE_CYCLIC_VALIDATION_FE2_SETUP_HH

#include "src/validation/TableCyclicValidationSupport.hh"

#include <memory>
#include <vector>

namespace fall_n::table_cyclic_validation {

using ValidationNlAnalysis = NonlinearAnalysis<
    TimoshenkoBeam3D, continuum::SmallStrain, NDOF, StructPolicy>;
using ValidationMacroBridge = BeamMacroBridge<StructModel, BeamElemT2>;
using ValidationMicroExecutor = SerialExecutor;
using ValidationAnalysis = MultiscaleAnalysis<
    ValidationNlAnalysis,
    ValidationMacroBridge,
    NonlinearSubModelEvolver,
    ValidationMicroExecutor>;

struct FE2CaseContext {
    std::unique_ptr<Domain<3>> domain{};
    std::unique_ptr<StructModel> model{};
    std::unique_ptr<MultiscaleCoordinator> coordinator{};
    std::unique_ptr<ValidationNlAnalysis> nl{};
    std::unique_ptr<ValidationAnalysis> analysis{};
    std::vector<std::size_t> base_nodes{};
};

[[nodiscard]] FE2CaseContext build_fe2_case_context(
    bool two_way,
    const std::string& out_dir,
    const CyclicValidationRunConfig& cfg);

} // namespace fall_n::table_cyclic_validation

#endif // FALL_N_SRC_VALIDATION_TABLE_CYCLIC_VALIDATION_FE2_SETUP_HH
