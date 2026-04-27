#ifndef FN_RC_SECTION_BUILDER_HH
#define FN_RC_SECTION_BUILDER_HH

// =============================================================================
//  RCSectionBuilder.hh — RC fiber-section material builders
// =============================================================================
//
//  High-level factories for constructing nonlinear fiber-discretized beam
//  cross-section materials from standard RC detail parameters.
//
//  Two builders are provided:
//    - make_rc_column_section():  square/rectangular column with 8 bars
//    - make_rc_beam_section():    rectangular beam with configurable bars
//
//  Both return Material<TimoshenkoBeam3D> ready for use in BeamElement.
//

#include "FiberSectionFactory.hh"
#include "RCSectionLayout.hh"
#include "../utils/SectionProperties.hh"

#include <utility>

namespace fall_n {

// ═════════════════════════════════════════════════════════════════════════
//  RC column section specification
// ═════════════════════════════════════════════════════════════════════════

template <typename UnconfinedFactory,
          typename ConfinedFactory,
          typename SteelFactory>
[[nodiscard]] inline std::vector<Fiber>
build_rc_column_fibers(const RCColumnSpec& s,
                       UnconfinedFactory&& make_unconfined,
                       ConfinedFactory&& make_confined,
                       SteelFactory&& make_steel)
{
    std::vector<Fiber> fibers;
    fibers.reserve(rc_column_fiber_count(s));

    for (const auto& patch : rc_column_patch_layout(s)) {
        add_patch_fibers(
            fibers,
            patch.y_min,
            patch.y_max,
            patch.ny,
            patch.z_min,
            patch.z_max,
            patch.nz,
            [&]() {
                return patch.material_role ==
                               RCSectionMaterialRole::confined_concrete
                           ? make_confined()
                           : make_unconfined();
            });
    }

    add_rebar_fibers(
        fibers,
        rc_column_longitudinal_bar_positions(s),
        rc_column_longitudinal_bar_area(s),
        make_steel);
    return fibers;
}

/// Build a nonlinear fiber-section material for a rectangular RC column.
///
/// Layout: 4 cover patches + 1 confined core + 8 longitudinal bars.
inline Material<TimoshenkoBeam3D>
make_rc_column_section(const RCColumnSpec& s) {
    const double Ec = concrete_initial_modulus(s.fpc);
    const double Gc = isotropic_shear_modulus(Ec, s.nu);
    const double J  = rectangular_torsion_constant(s.b, s.h);
    const double y_core = 0.5 * s.b - s.cover;
    const double z_core = 0.5 * s.h - s.cover;
    const KentParkConcreteTensionConfig concrete_tension{
        .tensile_strength = s.concrete_ft_ratio * s.fpc,
        .softening_multiplier = s.concrete_tension_softening_multiplier,
        .residual_tangent_ratio = s.concrete_tension_residual_tangent_ratio,
        .crack_transition_multiplier = s.concrete_tension_transition_multiplier,
    };

    auto fibers = build_rc_column_fibers(
        s,
        [&] { return make_unconfined_concrete(s.fpc, concrete_tension); },
        [&] {
            return make_confined_concrete(
                s.fpc,
                concrete_tension,
                s.rho_s,
                s.tie_fy,
                2.0 * std::min(y_core, z_core),
                s.tie_spacing);
        },
        [&] {
            return make_steel_fiber_material(
                s.steel_E, s.steel_fy, s.steel_b);
        });

    FiberSection3D section(Gc, s.kappa_y, s.kappa_z, J, std::move(fibers));
    return Material<TimoshenkoBeam3D>{
        InelasticMaterial<FiberSection3D>{std::move(section)},
        InelasticUpdate{}
    };
}


// ═════════════════════════════════════════════════════════════════════════
//  RC rectangular column section (12 bars)
// ═════════════════════════════════════════════════════════════════════════

/// Build an elasticized control section over the same RC fiber layout.
///
/// The geometry, patch subdivision, and rebar placement match the nonlinear
/// column section exactly; only the constitutive fibers collapse to linear
/// elastic laws. This is useful as a parity slice when comparing section-level
/// observables against external tools.
inline Material<TimoshenkoBeam3D>
make_rc_column_section_elasticized(const RCColumnSpec& s) {
    const double Ec = concrete_initial_modulus(s.fpc);
    const double Gc = isotropic_shear_modulus(Ec, s.nu);
    const double J  = rectangular_torsion_constant(s.b, s.h);

    auto fibers = build_rc_column_fibers(
        s,
        [&] { return make_elastic_uniaxial_material(Ec); },
        [&] { return make_elastic_uniaxial_material(Ec); },
        [&] { return make_elastic_uniaxial_material(s.steel_E); });

    FiberSection3D section(Gc, s.kappa_y, s.kappa_z, J, std::move(fibers));
    return Material<TimoshenkoBeam3D>{
        InelasticMaterial<FiberSection3D>{std::move(section)},
        InelasticUpdate{}
    };
}

struct RCRectColumnSpec {
    // Section geometry
    double bx;              // width  along local y (m)
    double by;              // depth  along local z (m)
    double cover;           // clear cover to stirrup centre (m)
    double bar_diameter;    // longitudinal bar diameter (m)
    double tie_spacing;     // transverse tie spacing (m)

    // Concrete
    double fpc;             // unconfined cylinder strength (MPa)
    double nu;              // Poisson's ratio (for shear modulus)

    // Steel
    double steel_E;         // Young's modulus (MPa)
    double steel_fy;        // yield strength (MPa)
    double steel_b;         // strain-hardening ratio

    // Transverse reinforcement
    double tie_fy;          // tie yield strength (MPa)
    double rho_s = 0.015;   // volumetric transverse reinforcement ratio

    // Shear correction factors
    double kappa_y = 5.0/6.0;
    double kappa_z = 5.0/6.0;
};

/// Build a nonlinear fiber-section material for a rectangular RC column
/// with 12 longitudinal bars (4 corner + 4 per long face).
///
/// Layout: 4 cover patches + 1 confined core + 12 longitudinal bars.
/// Bars are placed at the corners and at 2 intermediate locations along
/// each long face (bx direction).
inline Material<TimoshenkoBeam3D>
make_rc_rect_column_section(const RCRectColumnSpec& s) {
    const double Ec = concrete_initial_modulus(s.fpc);
    const double Gc = isotropic_shear_modulus(Ec, s.nu);
    const double J  = rectangular_torsion_constant(s.bx, s.by);

    std::vector<Fiber> fibers;
    fibers.reserve(80);

    const double y_edge = 0.5 * s.bx;
    const double z_edge = 0.5 * s.by;
    const double y_core = y_edge - s.cover;
    const double z_core = z_edge - s.cover;

    // Cover concrete (4 patches)
    auto unconfined = [&] { return make_unconfined_concrete(s.fpc); };

    add_patch_fibers(fibers, -y_edge, y_edge, 10, -z_edge, -z_core, 2, unconfined);
    add_patch_fibers(fibers, -y_edge, y_edge, 10,  z_core,  z_edge, 2, unconfined);
    add_patch_fibers(fibers, -y_edge, -y_core, 2, -z_core, z_core, 4, unconfined);
    add_patch_fibers(fibers,  y_core,  y_edge, 2, -z_core, z_core, 4, unconfined);

    // Confined core
    add_patch_fibers(
        fibers, -y_core, y_core, 8, -z_core, z_core, 6,
        [&] {
            return make_confined_concrete(
                s.fpc, s.rho_s, s.tie_fy,
                2.0 * std::min(y_core, z_core),
                s.tie_spacing);
        });

    // 12-bar reinforcement pattern:
    //   4 corners
    //   4 intermediate along top/bottom faces (bx direction)
    const double y_bar = y_edge - s.cover;
    const double z_bar = z_edge - s.cover;
    const double A_bar = bar_area(s.bar_diameter);

    // Intermediate spacing: 2 bars per long face → at ±y_bar/3
    const double y_mid1 = -y_bar / 3.0;
    const double y_mid2 =  y_bar / 3.0;

    const std::array<std::pair<double, double>, 12> bars = {{
        // corners
        {-y_bar, -z_bar}, { y_bar, -z_bar},
        {-y_bar,  z_bar}, { y_bar,  z_bar},
        // bottom face intermediates
        { y_mid1, -z_bar}, { y_mid2, -z_bar},
        // top face intermediates
        { y_mid1,  z_bar}, { y_mid2,  z_bar},
        // left face intermediates
        {-y_bar, 0.0},
        // right face intermediates
        { y_bar, 0.0},
        // short face (by) intermediates — centre of each short face
        { 0.0, -z_bar}, { 0.0, z_bar}
    }};

    add_rebar_fibers(
        fibers, bars, A_bar,
        [&] { return make_steel_fiber_material(s.steel_E, s.steel_fy, s.steel_b); });

    FiberSection3D section(Gc, s.kappa_y, s.kappa_z, J, std::move(fibers));
    return Material<TimoshenkoBeam3D>{
        InelasticMaterial<FiberSection3D>{std::move(section)},
        InelasticUpdate{}
    };
}


// ═════════════════════════════════════════════════════════════════════════
//  RC beam section specification
// ═════════════════════════════════════════════════════════════════════════

struct RCBeamSpec {
    // Section geometry
    double b;               // width  (m)
    double h;               // height (m)
    double cover;           // clear cover to stirrup centre (m)
    double bar_diameter;    // longitudinal bar diameter (m)

    // Concrete
    double fpc;             // unconfined cylinder strength (MPa)
    double nu;              // Poisson's ratio

    // Steel
    double steel_E;         // Young's modulus (MPa)
    double steel_fy;        // yield strength (MPa)
    double steel_b;         // strain-hardening ratio

    // Shear correction
    double kappa_y = 5.0/6.0;
    double kappa_z = 5.0/6.0;
};

/// Build a nonlinear fiber-section material for a rectangular RC beam.
///
/// Layout: 4 cover patches + 1 unconfined core + 6 longitudinal bars
/// (3 top + 3 bottom).
inline Material<TimoshenkoBeam3D>
make_rc_beam_section(const RCBeamSpec& s) {
    const double Ec = concrete_initial_modulus(s.fpc);
    const double Gc = isotropic_shear_modulus(Ec, s.nu);
    const double J  = rectangular_torsion_constant(s.b, s.h);

    std::vector<Fiber> fibers;
    fibers.reserve(42);

    const double y_edge = 0.5 * s.b;
    const double z_edge = 0.5 * s.h;
    const double y_core = y_edge - s.cover;
    const double z_core = z_edge - s.cover;

    auto unconfined = [&] { return make_unconfined_concrete(s.fpc); };

    // Cover and core — all unconfined for beams
    add_patch_fibers(fibers, -y_edge, y_edge, 6, -z_edge, -z_core, 2, unconfined);
    add_patch_fibers(fibers, -y_edge, y_edge, 6,  z_core,  z_edge, 2, unconfined);
    add_patch_fibers(fibers, -y_edge, -y_core, 2, -z_core, z_core, 6, unconfined);
    add_patch_fibers(fibers,  y_core,  y_edge, 2, -z_core, z_core, 6, unconfined);
    add_patch_fibers(fibers, -y_core,  y_core, 4, -z_core, z_core, 6, unconfined);

    // 6-bar reinforcement: 3 bottom + 3 top
    const double y_bar = y_edge - s.cover;
    const double z_bar = z_edge - s.cover;
    const double A_bar = bar_area(s.bar_diameter);

    const std::array<std::pair<double, double>, 6> bars = {{
        {-y_bar, -z_bar}, {0.0, -z_bar}, { y_bar, -z_bar},
        {-y_bar,  z_bar}, {0.0,  z_bar}, { y_bar,  z_bar}
    }};

    add_rebar_fibers(
        fibers, bars, A_bar,
        [&] { return make_steel_fiber_material(s.steel_E, s.steel_fy, s.steel_b); });

    FiberSection3D section(Gc, s.kappa_y, s.kappa_z, J, std::move(fibers));
    return Material<TimoshenkoBeam3D>{
        InelasticMaterial<FiberSection3D>{std::move(section)},
        InelasticUpdate{}
    };
}

} // namespace fall_n

#endif // FN_RC_SECTION_BUILDER_HH
