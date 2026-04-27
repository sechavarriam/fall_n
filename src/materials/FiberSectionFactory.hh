#ifndef FN_FIBER_SECTION_FACTORY_HH
#define FN_FIBER_SECTION_FACTORY_HH

// =============================================================================
//  FiberSectionFactory.hh — Generic fiber-mesh discretization utilities
// =============================================================================
//
//  Provides two building blocks for constructing fiber sections:
//    - add_patch_fibers():  regular grid subdivision of a rectangular region
//    - add_rebar_fibers():  discrete rebar placement at known (y,z) positions
//
//  These are material-agnostic: the caller supplies a factory callable that
//  returns Material<UniaxialMaterial> for each fiber.
//
//  Also includes convenience factories for common uniaxial constitutive
//  models (Menegotto-Pinto steel, Kent-Park concrete).
//

#include "Material.hh"
#include "LinealElasticMaterial.hh"
#include "update_strategy/IntegrationStrategy.hh"
#include "constitutive_models/non_lineal/FiberSection.hh"
#include "constitutive_models/non_lineal/KentParkConcrete.hh"
#include "constitutive_models/non_lineal/MenegottoPintoSteel.hh"

#include <array>
#include <utility>
#include <vector>

namespace fall_n {

// ═════════════════════════════════════════════════════════════════════════
//  Fiber mesh helpers
// ═════════════════════════════════════════════════════════════════════════

/// Discretize a rectangular concrete patch into an ny × nz grid of fibers.
///
/// Each fiber is centred at the midpoint of its sub-cell and has area
/// Δy·Δz.  The `material_factory` callable is invoked once per fiber
/// and must return `Material<UniaxialMaterial>`.
template <typename Factory>
void add_patch_fibers(
    std::vector<Fiber>& fibers,
    double y_min, double y_max, int ny,
    double z_min, double z_max, int nz,
    Factory&& material_factory)
{
    const double dy = (y_max - y_min) / static_cast<double>(ny);
    const double dz = (z_max - z_min) / static_cast<double>(nz);

    for (int iy = 0; iy < ny; ++iy) {
        for (int iz = 0; iz < nz; ++iz) {
            const double y = y_min + (static_cast<double>(iy) + 0.5) * dy;
            const double z = z_min + (static_cast<double>(iz) + 0.5) * dz;
            const double A = dy * dz;
            fibers.emplace_back(y, z, A, material_factory());
        }
    }
}

/// Place reinforcement fibers at known (y, z) positions.
///
/// Each rebar fiber gets the same tributary area and a freshly created
/// material from the factory.
template <std::size_t N, typename Factory>
void add_rebar_fibers(
    std::vector<Fiber>& fibers,
    const std::array<std::pair<double, double>, N>& positions,
    double area,
    Factory&& material_factory)
{
    for (const auto& [y, z] : positions) {
        fibers.emplace_back(y, z, area, material_factory());
    }
}


// ═════════════════════════════════════════════════════════════════════════
//  Uniaxial material convenience factories
// ═════════════════════════════════════════════════════════════════════════

/// Create a Menegotto-Pinto cyclic steel material.
inline Material<UniaxialMaterial>
make_steel_fiber_material(double E, double fy, double b) {
    return Material<UniaxialMaterial>{
        InelasticMaterial<MenegottoPintoSteel>{E, fy, b},
        InelasticUpdate{}
    };
}

/// Create a linear-elastic uniaxial fiber material.
inline Material<UniaxialMaterial>
make_elastic_uniaxial_material(double E) {
    MaterialInstance<ElasticRelation<UniaxialMaterial>> elastic_inst(E);
    return Material<UniaxialMaterial>{std::move(elastic_inst), ElasticUpdate{}};
}

/// Create an unconfined Kent-Park concrete material.
///
/// Residual strength = 10% of f'c.
inline Material<UniaxialMaterial>
make_unconfined_concrete(double fpc) {
    return Material<UniaxialMaterial>{
        InelasticMaterial<KentParkConcrete>{fpc, 0.10 * fpc},
        InelasticUpdate{}
    };
}

inline Material<UniaxialMaterial>
make_unconfined_concrete(double fpc, KentParkConcreteTensionConfig tension) {
    if (tension.tensile_strength <= 0.0) {
        tension.tensile_strength = 0.10 * fpc;
    }
    return Material<UniaxialMaterial>{
        InelasticMaterial<KentParkConcrete>{fpc, tension},
        InelasticUpdate{}
    };
}

/// Create a confined Kent-Park concrete material with confinement parameters.
///
/// @param fpc         Unconfined cylinder strength (MPa)
/// @param rho_s       Volumetric transverse reinforcement ratio
/// @param fyh         Yield strength of transverse reinforcement (MPa)
/// @param h_prime     Core dimension to centre-lines of perimeter hoops (m)
/// @param sh          Spacing of hoops (m)
inline Material<UniaxialMaterial>
make_confined_concrete(double fpc, double rho_s,
                       double fyh, double h_prime, double sh) {
    return Material<UniaxialMaterial>{
        InelasticMaterial<KentParkConcrete>{fpc, 0.10 * fpc, rho_s, fyh, h_prime, sh},
        InelasticUpdate{}
    };
}

inline Material<UniaxialMaterial>
make_confined_concrete(double fpc,
                       KentParkConcreteTensionConfig tension,
                       double rho_s,
                       double fyh,
                       double h_prime,
                       double sh) {
    if (tension.tensile_strength <= 0.0) {
        tension.tensile_strength = 0.10 * fpc;
    }
    return Material<UniaxialMaterial>{
        InelasticMaterial<KentParkConcrete>{
            fpc, tension, rho_s, fyh, h_prime, sh},
        InelasticUpdate{}
    };
}

} // namespace fall_n

#endif // FN_FIBER_SECTION_FACTORY_HH
