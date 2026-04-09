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
    double tp_ratio_;
    double fracture_energy_Nmm_;
    double lb_;
    KoBathe3DCrackStabilization crack_stabilization_{};
    KoBathe3DMaterialTangentMode material_tangent_mode_{
        KoBathe3DMaterialTangentMode::FractureSecant};
public:
    explicit KoBatheConcreteMaterialFactory(double fc_MPa,
                                            double lb_mm = 100.0,
                                            double fracture_energy_Nmm = 0.06,
                                            double tp_ratio = 0.0,
                                            KoBathe3DCrackStabilization
                                                crack_stabilization =
                                                    KoBathe3DCrackStabilization::
                                                        stabilized_default(),
                                            bool use_consistent_tangent = false)
        : fc_{fc_MPa}
        , tp_ratio_{tp_ratio}
        , fracture_energy_Nmm_{fracture_energy_Nmm}
        , lb_{lb_mm}
        , crack_stabilization_{crack_stabilization}
        , material_tangent_mode_{
              use_consistent_tangent
                  ? KoBathe3DMaterialTangentMode::
                        AdaptiveCentralDifferenceWithSecantFallback
                  : KoBathe3DMaterialTangentMode::FractureSecant}
    {}

    Material<ThreeDimensionalMaterial> create() const override {
        const KoBatheParameters params = parameters();
        KoBatheConcrete3D concrete_impl{params, crack_stabilization_};
        concrete_impl.set_material_tangent_mode(material_tangent_mode_);
        InelasticMaterial<KoBatheConcrete3D> mat_inst{
            std::move(concrete_impl)};
        return Material<ThreeDimensionalMaterial>{mat_inst, InelasticUpdate{}};
    }

    std::unique_ptr<ConcreteMaterialFactory> clone() const override {
        return std::make_unique<KoBatheConcreteMaterialFactory>(*this);
    }

    [[nodiscard]] double compressive_strength_MPa() const noexcept { return fc_; }
    [[nodiscard]] double tp_ratio() const noexcept { return tp_ratio_; }
    [[nodiscard]] double fracture_energy_Nmm() const noexcept
    {
        return fracture_energy_Nmm_;
    }
    [[nodiscard]] double length_scale_mm() const noexcept { return lb_; }
    [[nodiscard]] const KoBathe3DCrackStabilization&
    crack_stabilization() const noexcept
    {
        return crack_stabilization_;
    }
    [[nodiscard]] bool use_consistent_tangent() const noexcept
    {
        return material_tangent_mode_
            != KoBathe3DMaterialTangentMode::FractureSecant;
    }
    [[nodiscard]] KoBathe3DMaterialTangentMode material_tangent_mode() const noexcept
    {
        return material_tangent_mode_;
    }
    [[nodiscard]] KoBatheParameters parameters() const noexcept
    {
        // KoBatheParameters signature is
        //   (fc_MPa, tp_ratio, Gf_Nmm, lb_mm).
        // Keeping this mapping in a named helper makes the intended semantics
        // testable and avoids silent argument-order regressions.
        return KoBatheParameters{
            fc_, tp_ratio_, fracture_energy_Nmm_, lb_};
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
