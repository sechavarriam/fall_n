#ifndef FALL_N_REDUCED_RC_COLUMN_REFERENCE_SPEC_HH
#define FALL_N_REDUCED_RC_COLUMN_REFERENCE_SPEC_HH

// =============================================================================
//  ReducedRCColumnReferenceSpec.hh
// =============================================================================
//
//  Canonical geometric / material specification for the reduced reinforced-
//  concrete column reboot.
//
//  This header exists to decouple the new validation reboot from the older
//  table-cyclic support constants. The rebooted reduced-column campaign should
//  be able to reference one normative section/column definition without pulling
//  legacy driver support surfaces into every new artifact.
//
// =============================================================================

#include "../materials/RCSectionBuilder.hh"

#include <cstddef>

namespace fall_n::validation_reboot {

struct ReducedRCColumnReferenceSpec {
    double column_height_m{3.2};
    double section_b_m{0.25};
    double section_h_m{0.25};
    double cover_m{0.03};
    double longitudinal_bar_diameter_m{0.016};
    double tie_spacing_m{0.08};

    double concrete_fpc_mpa{30.0};
    double concrete_nu{0.20};
    double concrete_ft_ratio{0.02};
    double concrete_tension_softening_multiplier{0.50};
    double concrete_tension_residual_tangent_ratio{1.0e-6};
    double concrete_tension_transition_multiplier{0.50};

    double steel_E_mpa{200000.0};
    double steel_fy_mpa{420.0};
    double steel_b{0.01};

    double tie_fy_mpa{420.0};
    double rho_s{0.015};

    double kappa_y{5.0 / 6.0};
    double kappa_z{5.0 / 6.0};

    RCColumnSectionMeshSpec section_mesh{};
};

inline constexpr ReducedRCColumnReferenceSpec
default_reduced_rc_column_reference_spec_v{};

struct ReducedRCColumnSteelAreaSummary {
    std::size_t longitudinal_bar_count{0};
    double single_bar_area_m2{0.0};
    double total_longitudinal_steel_area_m2{0.0};
    double gross_section_area_m2{0.0};
    double longitudinal_steel_ratio{0.0};
};

[[nodiscard]] constexpr RCColumnSpec
to_rc_column_section_spec(const ReducedRCColumnReferenceSpec& spec) noexcept
{
    return RCColumnSpec{
        .b = spec.section_b_m,
        .h = spec.section_h_m,
        .cover = spec.cover_m,
        .bar_diameter = spec.longitudinal_bar_diameter_m,
        .tie_spacing = spec.tie_spacing_m,
        .fpc = spec.concrete_fpc_mpa,
        .nu = spec.concrete_nu,
        .concrete_ft_ratio = spec.concrete_ft_ratio,
        .concrete_tension_softening_multiplier =
            spec.concrete_tension_softening_multiplier,
        .concrete_tension_residual_tangent_ratio =
            spec.concrete_tension_residual_tangent_ratio,
        .concrete_tension_transition_multiplier =
            spec.concrete_tension_transition_multiplier,
        .steel_E = spec.steel_E_mpa,
        .steel_fy = spec.steel_fy_mpa,
        .steel_b = spec.steel_b,
        .tie_fy = spec.tie_fy_mpa,
        .rho_s = spec.rho_s,
        .kappa_y = spec.kappa_y,
        .kappa_z = spec.kappa_z,
        .section_mesh = spec.section_mesh,
    };
}

[[nodiscard]] inline ReducedRCColumnSteelAreaSummary
describe_reduced_rc_column_structural_steel_area(
    const ReducedRCColumnReferenceSpec& spec =
        default_reduced_rc_column_reference_spec_v) noexcept
{
    const auto section_spec = to_rc_column_section_spec(spec);
    const auto positions = rc_column_longitudinal_bar_positions(section_spec);
    const double single_bar_area =
        rc_column_longitudinal_bar_area(section_spec);
    const double total_steel_area =
        static_cast<double>(positions.size()) * single_bar_area;
    const double gross_area = spec.section_b_m * spec.section_h_m;

    return {
        .longitudinal_bar_count = positions.size(),
        .single_bar_area_m2 = single_bar_area,
        .total_longitudinal_steel_area_m2 = total_steel_area,
        .gross_section_area_m2 = gross_area,
        .longitudinal_steel_ratio =
            gross_area > 0.0 ? total_steel_area / gross_area : 0.0,
    };
}

[[nodiscard]] inline Material<TimoshenkoBeam3D>
make_default_reduced_rc_column_section_material()
{
    return make_rc_column_section(
        to_rc_column_section_spec(default_reduced_rc_column_reference_spec_v));
}

} // namespace fall_n::validation_reboot

#endif // FALL_N_REDUCED_RC_COLUMN_REFERENCE_SPEC_HH
