#ifndef FALL_N_SRC_MATERIALS_SUBMODEL_MATERIAL_FACTORY_DEFAULTS_HH
#define FALL_N_SRC_MATERIALS_SUBMODEL_MATERIAL_FACTORY_DEFAULTS_HH

// =============================================================================
//  SubmodelMaterialFactoryDefaults -- reference material factories
// =============================================================================
//
//  These defaults implement the current repository reference path for local
//  continuum sub-models:
//    - Ko-Bathe 3D concrete for continuum cells
//    - Menegotto-Pinto steel for embedded rebars
//
//  They belong to the materials module rather than to reconstruction so that
//  the multiscale core can be agnostic to the specific local constitutive set.
//
// =============================================================================

#include <memory>
#include <utility>

#include "ConstitutiveIntegrator.hh"
#include "LinealElasticMaterial.hh"
#include "SubmodelMaterialFactory.hh"
#include "constitutive_models/non_lineal/KoBatheConcrete3D.hh"
#include "constitutive_models/non_lineal/MenegottoPintoSteel.hh"

namespace fall_n {

class KoBatheConcreteMaterialFactory final : public ConcreteMaterialFactory {
    double fc_;
    double tp_ratio_;
    double fracture_energy_Nmm_;
    double lb_;
    KoBathe3DCrackStabilization crack_stabilization_{};
    KoBathe3DMaterialTangentMode material_tangent_mode_{
        KoBathe3DMaterialTangentMode::FractureSecant};

public:
    explicit KoBatheConcreteMaterialFactory(
        double fc_MPa,
        double lb_mm = 100.0,
        double fracture_energy_Nmm = 0.06,
        double tp_ratio = 0.0,
        KoBathe3DCrackStabilization crack_stabilization =
            KoBathe3DCrackStabilization::stabilized_default(),
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

    Material<ThreeDimensionalMaterial> create() const override
    {
        const KoBatheParameters params = parameters();
        KoBatheConcrete3D concrete_impl{params, crack_stabilization_};
        concrete_impl.set_material_tangent_mode(material_tangent_mode_);
        InelasticMaterial<KoBatheConcrete3D> mat_inst{
            std::move(concrete_impl)};
        return Material<ThreeDimensionalMaterial>{mat_inst, InelasticUpdate{}};
    }

    std::unique_ptr<ConcreteMaterialFactory> clone() const override
    {
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
        return KoBatheParameters{
            fc_, tp_ratio_, fracture_energy_Nmm_, lb_};
    }
};

class MenegottoPintoRebarFactory final : public RebarMaterialFactory {
    double E_;
    double fy_;
    double b_;
    double R0_;
    double cR1_;
    double cR2_;

public:
    MenegottoPintoRebarFactory(double E,
                               double fy,
                               double b,
                               double R0 = 20.0,
                               double cR1 = 18.5,
                               double cR2 = 0.15)
        : E_{E}
        , fy_{fy}
        , b_{b}
        , R0_{R0}
        , cR1_{cR1}
        , cR2_{cR2}
    {}

    Material<UniaxialMaterial> create() const override
    {
        MenegottoPintoSteel steel_impl{E_, fy_, b_, R0_, cR1_, cR2_};
        InelasticMaterial<MenegottoPintoSteel> steel_inst{std::move(steel_impl)};
        return Material<UniaxialMaterial>{
            std::move(steel_inst), InelasticUpdate{}};
    }

    std::unique_ptr<RebarMaterialFactory> clone() const override
    {
        return std::make_unique<MenegottoPintoRebarFactory>(*this);
    }

    [[nodiscard]] double elastic_modulus_MPa() const noexcept { return E_; }
    [[nodiscard]] double yield_stress_MPa() const noexcept { return fy_; }
    [[nodiscard]] double hardening_ratio() const noexcept { return b_; }
    [[nodiscard]] double R0() const noexcept { return R0_; }
    [[nodiscard]] double cR1() const noexcept { return cR1_; }
    [[nodiscard]] double cR2() const noexcept { return cR2_; }
};

[[nodiscard]] inline std::unique_ptr<ConcreteMaterialFactory>
make_default_submodel_concrete_factory(double fc_MPa,
                                       double lb_mm = 100.0,
                                       bool use_consistent_tangent = false)
{
    return std::make_unique<KoBatheConcreteMaterialFactory>(
        fc_MPa,
        lb_mm,
        0.06,
        0.0,
        KoBathe3DCrackStabilization::stabilized_default(),
        use_consistent_tangent);
}

[[nodiscard]] inline std::unique_ptr<RebarMaterialFactory>
make_default_submodel_rebar_factory(double E = 200000.0,
                                    double fy = 420.0,
                                    double b = 0.01,
                                    double R0 = 20.0,
                                    double cR1 = 18.5,
                                    double cR2 = 0.15)
{
    return std::make_unique<MenegottoPintoRebarFactory>(
        E, fy, b, R0, cR1, cR2);
}

} // namespace fall_n

#endif // FALL_N_SRC_MATERIALS_SUBMODEL_MATERIAL_FACTORY_DEFAULTS_HH
