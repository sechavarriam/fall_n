#ifndef FALL_N_SRC_MATERIALS_SUBMODEL_MATERIAL_FACTORY_HH
#define FALL_N_SRC_MATERIALS_SUBMODEL_MATERIAL_FACTORY_HH

// =============================================================================
//  SubmodelMaterialFactory -- abstract material providers for local sub-models
// =============================================================================
//
//  The multiscale/reconstruction layer should not depend on a single local
//  constitutive family.  This header defines the narrow contracts used by
//  local-model implementations to obtain continuum and embedded-line materials.
//
//  Reference implementations for Ko-Bathe concrete and Menegotto-Pinto steel
//  live in the materials module as well, but are intentionally split into a
//  second header so that the multiscale core can depend only on contracts.
//
// =============================================================================

#include <memory>

#include "Material.hh"
#include "MaterialPolicy.hh"

namespace fall_n {

struct ConcreteMaterialFactory {
    virtual ~ConcreteMaterialFactory() = default;
    virtual Material<ThreeDimensionalMaterial> create() const = 0;
    virtual std::unique_ptr<ConcreteMaterialFactory> clone() const = 0;
};

struct RebarMaterialFactory {
    virtual ~RebarMaterialFactory() = default;
    virtual Material<UniaxialMaterial> create() const = 0;
    virtual std::unique_ptr<RebarMaterialFactory> clone() const = 0;
};

using LocalContinuumMaterialFactory = ConcreteMaterialFactory;
using EmbeddedLineMaterialFactory = RebarMaterialFactory;

} // namespace fall_n

#endif // FALL_N_SRC_MATERIALS_SUBMODEL_MATERIAL_FACTORY_HH
