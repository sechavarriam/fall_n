#include "src/validation/TableCyclicValidationSubmodelFactories.hh"

#include "src/analysis/MultiscaleCoordinator.hh"
#include "src/materials/constitutive_models/non_lineal/KoBatheConcrete3D.hh"
#include "src/reconstruction/MaterialFactory.hh"
#include "src/validation/TableCyclicValidationSupport.hh"

#include <algorithm>

namespace fall_n::table_cyclic_validation {

std::unique_ptr<ConcreteMaterialFactory>
make_table_submodel_concrete_factory(const MultiscaleSubModel& sub,
                                     const CyclicValidationRunConfig& cfg)
{
    const double lb_mm =
        1.0e3 * std::max({sub.grid.dx, sub.grid.dy, sub.grid.dz});

    return std::make_unique<KoBatheConcreteMaterialFactory>(
        COL_FPC,
        lb_mm,
        0.06,
        0.0,
        KoBathe3DCrackStabilization::stabilized_default(),
        cfg.submodel_use_consistent_material_tangent);
}

std::unique_ptr<RebarMaterialFactory>
make_table_submodel_rebar_factory()
{
    return std::make_unique<MenegottoPintoRebarFactory>(
        STEEL_E, STEEL_FY, STEEL_B);
}

} // namespace fall_n::table_cyclic_validation
