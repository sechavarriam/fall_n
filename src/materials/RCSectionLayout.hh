#ifndef FALL_N_RC_SECTION_LAYOUT_HH
#define FALL_N_RC_SECTION_LAYOUT_HH

#include "../utils/SectionProperties.hh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace fall_n {

struct RCColumnSectionMeshSpec {
    int cover_top_bottom_ny{8};
    int cover_top_bottom_nz{2};
    int cover_side_ny{2};
    int cover_side_nz{4};
    int core_ny{6};
    int core_nz{6};
};

[[nodiscard]] constexpr RCColumnSectionMeshSpec
coarse_rc_column_section_mesh() noexcept
{
    return {
        .cover_top_bottom_ny = 4,
        .cover_top_bottom_nz = 1,
        .cover_side_ny = 1,
        .cover_side_nz = 2,
        .core_ny = 3,
        .core_nz = 3,
    };
}

[[nodiscard]] constexpr RCColumnSectionMeshSpec
canonical_rc_column_section_mesh() noexcept
{
    return {};
}

[[nodiscard]] constexpr RCColumnSectionMeshSpec
fine_rc_column_section_mesh() noexcept
{
    return {
        .cover_top_bottom_ny = 16,
        .cover_top_bottom_nz = 4,
        .cover_side_ny = 4,
        .cover_side_nz = 8,
        .core_ny = 12,
        .core_nz = 12,
    };
}

[[nodiscard]] constexpr RCColumnSectionMeshSpec
ultra_fine_rc_column_section_mesh() noexcept
{
    return {
        .cover_top_bottom_ny = 24,
        .cover_top_bottom_nz = 6,
        .cover_side_ny = 6,
        .cover_side_nz = 12,
        .core_ny = 18,
        .core_nz = 18,
    };
}

[[nodiscard]] constexpr bool
is_valid_rc_column_section_mesh(const RCColumnSectionMeshSpec& mesh) noexcept
{
    return mesh.cover_top_bottom_ny > 0 &&
           mesh.cover_top_bottom_nz > 0 &&
           mesh.cover_side_ny > 0 &&
           mesh.cover_side_nz > 0 &&
           mesh.core_ny > 0 &&
           mesh.core_nz > 0;
}

struct RCColumnSpec {
    double b;
    double h;
    double cover;
    double bar_diameter;
    double tie_spacing;

    double fpc;
    double nu;
    double concrete_ft_ratio = 0.10;
    double concrete_tension_softening_multiplier = 0.0;
    double concrete_tension_residual_tangent_ratio = 1.0e-6;
    double concrete_tension_transition_multiplier = 0.0;

    double steel_E;
    double steel_fy;
    double steel_b;

    double tie_fy;
    double rho_s = 0.015;

    double kappa_y = 5.0 / 6.0;
    double kappa_z = 5.0 / 6.0;

    RCColumnSectionMeshSpec section_mesh{};
};

enum class RCSectionZoneKind {
    cover_top,
    cover_bottom,
    cover_left,
    cover_right,
    confined_core,
    longitudinal_steel,
};

[[nodiscard]] constexpr std::string_view
to_string(RCSectionZoneKind zone) noexcept
{
    switch (zone) {
        case RCSectionZoneKind::cover_top:
            return "cover_top";
        case RCSectionZoneKind::cover_bottom:
            return "cover_bottom";
        case RCSectionZoneKind::cover_left:
            return "cover_left";
        case RCSectionZoneKind::cover_right:
            return "cover_right";
        case RCSectionZoneKind::confined_core:
            return "confined_core";
        case RCSectionZoneKind::longitudinal_steel:
            return "longitudinal_steel";
    }
    return "unknown_rc_section_zone";
}

enum class RCSectionMaterialRole {
    unconfined_concrete,
    confined_concrete,
    reinforcing_steel,
};

[[nodiscard]] constexpr std::string_view
to_string(RCSectionMaterialRole role) noexcept
{
    switch (role) {
        case RCSectionMaterialRole::unconfined_concrete:
            return "unconfined_concrete";
        case RCSectionMaterialRole::confined_concrete:
            return "confined_concrete";
        case RCSectionMaterialRole::reinforcing_steel:
            return "reinforcing_steel";
    }
    return "unknown_rc_section_material_role";
}

struct RCSectionPatchLayout {
    double y_min;
    double y_max;
    int ny;
    double z_min;
    double z_max;
    int nz;
    RCSectionZoneKind zone;
    RCSectionMaterialRole material_role;
};

struct RCSectionFiberLayoutRecord {
    std::size_t fiber_index;
    double y;
    double z;
    double area;
    RCSectionZoneKind zone;
    RCSectionMaterialRole material_role;
};

[[nodiscard]] inline auto
rc_column_patch_layout(const RCColumnSpec& s)
{
    if (!is_valid_rc_column_section_mesh(s.section_mesh)) {
        throw std::invalid_argument(
            "RC column section mesh requires strictly positive subdivisions.");
    }

    const double y_edge = 0.5 * s.b;
    const double z_edge = 0.5 * s.h;
    const double y_core = y_edge - s.cover;
    const double z_core = z_edge - s.cover;

    return std::array<RCSectionPatchLayout, 5>{{
        {
            .y_min = -y_edge,
            .y_max = y_edge,
            .ny = s.section_mesh.cover_top_bottom_ny,
            .z_min = -z_edge,
            .z_max = -z_core,
            .nz = s.section_mesh.cover_top_bottom_nz,
            .zone = RCSectionZoneKind::cover_bottom,
            .material_role = RCSectionMaterialRole::unconfined_concrete,
        },
        {
            .y_min = -y_edge,
            .y_max = y_edge,
            .ny = s.section_mesh.cover_top_bottom_ny,
            .z_min = z_core,
            .z_max = z_edge,
            .nz = s.section_mesh.cover_top_bottom_nz,
            .zone = RCSectionZoneKind::cover_top,
            .material_role = RCSectionMaterialRole::unconfined_concrete,
        },
        {
            .y_min = -y_edge,
            .y_max = -y_core,
            .ny = s.section_mesh.cover_side_ny,
            .z_min = -z_core,
            .z_max = z_core,
            .nz = s.section_mesh.cover_side_nz,
            .zone = RCSectionZoneKind::cover_left,
            .material_role = RCSectionMaterialRole::unconfined_concrete,
        },
        {
            .y_min = y_core,
            .y_max = y_edge,
            .ny = s.section_mesh.cover_side_ny,
            .z_min = -z_core,
            .z_max = z_core,
            .nz = s.section_mesh.cover_side_nz,
            .zone = RCSectionZoneKind::cover_right,
            .material_role = RCSectionMaterialRole::unconfined_concrete,
        },
        {
            .y_min = -y_core,
            .y_max = y_core,
            .ny = s.section_mesh.core_ny,
            .z_min = -z_core,
            .z_max = z_core,
            .nz = s.section_mesh.core_nz,
            .zone = RCSectionZoneKind::confined_core,
            .material_role = RCSectionMaterialRole::confined_concrete,
        },
    }};
}

[[nodiscard]] constexpr auto
rc_column_longitudinal_bar_positions(const RCColumnSpec& s) noexcept
{
    const double y_bar = 0.5 * s.b - s.cover;
    const double z_bar = 0.5 * s.h - s.cover;

    return std::array<std::pair<double, double>, 8>{{
        {-y_bar, -z_bar}, { y_bar, -z_bar},
        {-y_bar,  z_bar}, { y_bar,  z_bar},
        { 0.0,   -z_bar}, { 0.0,    z_bar},
        {-y_bar,  0.0  }, { y_bar,   0.0  },
    }};
}

[[nodiscard]] constexpr double
rc_column_longitudinal_bar_area(const RCColumnSpec& s) noexcept
{
    return bar_area(s.bar_diameter);
}

[[nodiscard]] constexpr std::size_t
canonical_rc_column_fiber_count() noexcept
{
    return 92;
}

[[nodiscard]] constexpr std::size_t
rc_column_concrete_fiber_count(const RCColumnSectionMeshSpec& mesh) noexcept
{
    return 2u * static_cast<std::size_t>(
                    mesh.cover_top_bottom_ny * mesh.cover_top_bottom_nz) +
           2u * static_cast<std::size_t>(
                    mesh.cover_side_ny * mesh.cover_side_nz) +
           static_cast<std::size_t>(mesh.core_ny * mesh.core_nz);
}

[[nodiscard]] constexpr std::size_t
rc_column_fiber_count(const RCColumnSpec& s) noexcept
{
    return rc_column_concrete_fiber_count(s.section_mesh) + 8u;
}

[[nodiscard]] inline std::vector<RCSectionFiberLayoutRecord>
build_rc_column_fiber_layout(const RCColumnSpec& s)
{
    const auto patches = rc_column_patch_layout(s);
    const auto bars = rc_column_longitudinal_bar_positions(s);
    const auto area_bar = rc_column_longitudinal_bar_area(s);

    std::vector<RCSectionFiberLayoutRecord> layout;
    layout.reserve(rc_column_fiber_count(s));

    const auto append_patch = [&](const RCSectionPatchLayout& patch) {
        const double dy =
            (patch.y_max - patch.y_min) / static_cast<double>(patch.ny);
        const double dz =
            (patch.z_max - patch.z_min) / static_cast<double>(patch.nz);
        const double area = dy * dz;

        for (int iy = 0; iy < patch.ny; ++iy) {
            for (int iz = 0; iz < patch.nz; ++iz) {
                layout.push_back({
                    .fiber_index = layout.size(),
                    .y = patch.y_min + (static_cast<double>(iy) + 0.5) * dy,
                    .z = patch.z_min + (static_cast<double>(iz) + 0.5) * dz,
                    .area = area,
                    .zone = patch.zone,
                    .material_role = patch.material_role,
                });
            }
        }
    };

    for (const auto& patch : patches) {
        append_patch(patch);
    }

    for (const auto& [y, z] : bars) {
        layout.push_back({
            .fiber_index = layout.size(),
            .y = y,
            .z = z,
            .area = area_bar,
            .zone = RCSectionZoneKind::longitudinal_steel,
            .material_role = RCSectionMaterialRole::reinforcing_steel,
        });
    }

    return layout;
}

} // namespace fall_n

#endif // FALL_N_RC_SECTION_LAYOUT_HH
