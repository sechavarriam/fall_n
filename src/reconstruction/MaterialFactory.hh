#ifndef FALL_N_SRC_RECONSTRUCTION_MATERIAL_FACTORY_HH
#define FALL_N_SRC_RECONSTRUCTION_MATERIAL_FACTORY_HH

// =============================================================================
//  MaterialFactory — Abstract factories for sub-model material creation
// =============================================================================
//
//  Decouples NonlinearSubModelEvolver from concrete constitutive models
//  (KoBatheConcrete3D, MenegottoPintoSteel).  Users inject factories at
//  construction time; the sub-model solver calls create() when building
//  its FE model. This follows the DamageCriterion ABC pattern.
//
// =============================================================================

#include <memory>
#include <utility>

#include "../materials/MaterialPolicy.hh"
#include "../materials/Material.hh"
#include "../materials/LinealElasticMaterial.hh"
#include "../materials/ConstitutiveIntegrator.hh"
#include "../materials/constitutive_models/non_lineal/KoBatheConcrete3D.hh"
#include "../materials/constitutive_models/non_lineal/MenegottoPintoSteel.hh"


namespace fall_n {


// =============================================================================
//  ConcreteMaterialFactory — creates Material<ThreeDimensionalMaterial>
// =============================================================================

struct ConcreteMaterialFactory {
    virtual ~ConcreteMaterialFactory() = default;
    virtual Material<ThreeDimensionalMaterial> create() const = 0;
    virtual std::unique_ptr<ConcreteMaterialFactory> clone() const = 0;
};


// =============================================================================
//  RebarMaterialFactory — creates Material<UniaxialMaterial>
// =============================================================================

struct RebarMaterialFactory {
    virtual ~RebarMaterialFactory() = default;
    virtual Material<UniaxialMaterial> create() const = 0;
    virtual std::unique_ptr<RebarMaterialFactory> clone() const = 0;
};


// =============================================================================
//  KoBatheConcreteMaterialFactory — default concrete factory
// =============================================================================

class KoBatheConcreteMaterialFactory final : public ConcreteMaterialFactory {
    double fc_;
public:
    explicit KoBatheConcreteMaterialFactory(double fc_MPa) : fc_{fc_MPa} {}

    Material<ThreeDimensionalMaterial> create() const override {
        InelasticMaterial<KoBatheConcrete3D> mat_inst{fc_};
        return Material<ThreeDimensionalMaterial>{mat_inst, InelasticUpdate{}};
    }

    std::unique_ptr<ConcreteMaterialFactory> clone() const override {
        return std::make_unique<KoBatheConcreteMaterialFactory>(*this);
    }
};


// =============================================================================
//  MenegottoPintoRebarFactory — default rebar factory
// =============================================================================

class MenegottoPintoRebarFactory final : public RebarMaterialFactory {
    double E_, fy_, b_;
public:
    MenegottoPintoRebarFactory(double E, double fy, double b)
        : E_{E}, fy_{fy}, b_{b} {}

    Material<UniaxialMaterial> create() const override {
        MenegottoPintoSteel steel_impl{E_, fy_, b_};
        InelasticMaterial<MenegottoPintoSteel> steel_inst{std::move(steel_impl)};
        return Material<UniaxialMaterial>{std::move(steel_inst), InelasticUpdate{}};
    }

    std::unique_ptr<RebarMaterialFactory> clone() const override {
        return std::make_unique<MenegottoPintoRebarFactory>(*this);
    }
};


}  // namespace fall_n

#endif // FALL_N_SRC_RECONSTRUCTION_MATERIAL_FACTORY_HH
