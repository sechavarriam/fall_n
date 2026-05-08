#ifndef PRISMATIC_DOMAIN_BUILDER_HH
#define PRISMATIC_DOMAIN_BUILDER_HH

// ═══════════════════════════════════════════════════════════════════════
//  PrismaticDomainBuilder — Hex8/Hex20/Hex27 prismatic mesh for
//  sub-model volumes
// ═══════════════════════════════════════════════════════════════════════
//
//  Generates a structured hexahedral mesh for a rectangular prism
//  defined by its width (X), height (Y) and length (Z), optionally
//  rotated and translated to align with a beam element's local frame.
//
//  Supports three element types via the `hex_order` field:
//
//    HexOrder::Linear      →  Hex8   (LagrangeElement3D<2,2,2>)
//    HexOrder::Serendipity →  Hex20  (SerendipityElement3D<2>)
//    HexOrder::Quadratic   →  Hex27  (LagrangeElement3D<3,3,3>)
//
//  Usage:
//    auto [domain, grid] = fall_n::make_prismatic_domain({
//        .width  = 0.40,
//        .height = 0.60,
//        .length = 3.20,
//        .nx = 4, .ny = 6, .nz = 8,
//        .hex_order = fall_n::HexOrder::Serendipity,
//    });
//
//  The returned PrismaticGrid provides O(1) node_id() lookups and
//  face extraction for boundary condition application.  For quadratic
//  meshes the grid has (2·n+1) nodes per direction; face extraction
//  returns ALL nodes on the face (corners + mid-edge + mid-face).
//
// ═══════════════════════════════════════════════════════════════════════

#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <petsc.h>

#include "../domain/Domain.hh"
#include "../elements/element_geometry/LagrangeElement.hh"
#include "../elements/element_geometry/SerendipityElement.hh"
#include "../numerics/numerical_integration/GaussLegendreCellIntegrator.hh"

namespace fall_n {

// ─────────────────────────────────────────────────────────────────────
//  HexOrder — element polynomial order for prismatic meshes
// ─────────────────────────────────────────────────────────────────────
enum class HexOrder {
    Linear,       // Hex8  — 8-node trilinear
    Serendipity,  // Hex20 — 20-node serendipity (quadratic, no interior nodes)
    Quadratic     // Hex27 — 27-node triquadratic
};

enum class LongitudinalBiasLocation {
    FixedEnd,
    LoadedEnd,
    BothEnds
};

[[nodiscard]] constexpr const char*
to_string(LongitudinalBiasLocation location) noexcept
{
    switch (location) {
        case LongitudinalBiasLocation::FixedEnd:
            return "fixed_end";
        case LongitudinalBiasLocation::LoadedEnd:
            return "loaded_end";
        case LongitudinalBiasLocation::BothEnds:
            return "both_ends";
    }
    return "unknown_longitudinal_bias_location";
}

// ─────────────────────────────────────────────────────────────────────
//  Face identifiers for the six faces of the prism
// ─────────────────────────────────────────────────────────────────────
enum class PrismFace { MinX, MaxX, MinY, MaxY, MinZ, MaxZ };

// ─────────────────────────────────────────────────────────────────────
//  PrismaticGrid — metadata for a structured hex mesh (any order)
// ─────────────────────────────────────────────────────────────────────
//
//  For linear meshes (Hex8):     nodes_per_dir  = n + 1      (step = 1)
//  For quadratic meshes (Hex20/27): nodes_per_dir = 2·n + 1  (step = 2)
//
//  The grid always stores the FULL quadratic-level grid of node IDs.
//  For Hex20 (serendipity), mid-face and body-centre nodes simply
//  do not exist in the Domain — they are never created.  The grid's
//  node_id() still provides valid indices for the nodes that DO exist;
//  the caller is responsible for only querying corner + mid-edge nodes
//  in the serendipity case.
//
struct PrismaticGrid {
    int nx, ny, nz;          // number of elements per direction
    int step;                // 1 for linear, 2 for quadratic
    double dx, dy, dz;       // nominal element sizes
    double width, height, length;
    HexOrder hex_order;
    double longitudinal_bias_power{1.0};
    LongitudinalBiasLocation longitudinal_bias_location{
        LongitudinalBiasLocation::FixedEnd};
    std::vector<double> x_coordinates{};
    std::vector<double> y_coordinates{};
    std::vector<double> z_coordinates{};

    int nodes_x() const noexcept { return step * nx + 1; }
    int nodes_y() const noexcept { return step * ny + 1; }
    int nodes_z() const noexcept { return step * nz + 1; }
    int total_nodes()    const noexcept { return nodes_x() * nodes_y() * nodes_z(); }
    int total_elements() const noexcept { return nx * ny * nz; }

    double x_coordinate(int ix) const noexcept {
        if (!x_coordinates.empty() &&
            ix >= 0 &&
            ix < static_cast<int>(x_coordinates.size())) {
            return x_coordinates[static_cast<std::size_t>(ix)];
        }
        return -0.5 * width +
               static_cast<double>(ix) * (dx / static_cast<double>(step));
    }

    double y_coordinate(int iy) const noexcept {
        if (!y_coordinates.empty() &&
            iy >= 0 &&
            iy < static_cast<int>(y_coordinates.size())) {
            return y_coordinates[static_cast<std::size_t>(iy)];
        }
        return -0.5 * height +
               static_cast<double>(iy) * (dy / static_cast<double>(step));
    }

    double z_coordinate(int iz) const noexcept {
        if (!z_coordinates.empty() &&
            iz >= 0 &&
            iz < static_cast<int>(z_coordinates.size())) {
            return z_coordinates[static_cast<std::size_t>(iz)];
        }
        return static_cast<double>(iz) * (dz / static_cast<double>(step));
    }

    /// O(1) node ID from structured grid indices.
    ///  For linear:    ix ∈ [0, nx],  iy ∈ [0, ny],  iz ∈ [0, nz]
    ///  For quadratic: ix ∈ [0, 2·nx], iy ∈ [0, 2·ny], iz ∈ [0, 2·nz]
    PetscInt node_id(int ix, int iy, int iz) const noexcept {
        return static_cast<PetscInt>(
            iz * nodes_x() * nodes_y() + iy * nodes_x() + ix);
    }

    [[nodiscard]] bool active_node_for_order(int ix,
                                             int iy,
                                             int iz) const noexcept
    {
        if (hex_order != HexOrder::Serendipity) {
            return true;
        }

        // Hex20 uses corner and edge-midpoint nodes only. The full quadratic
        // grid still contains face-centre/body-centre coordinates for Hex27
        // compatibility, but those nodes do not receive elemental DOFs.
        const int odd_count = (std::abs(ix) % 2) +
                              (std::abs(iy) % 2) +
                              (std::abs(iz) % 2);
        return odd_count <= 1;
    }

    /// Collect all node IDs on a given prism face.
    ///  For linear meshes: returns corner nodes only.
    ///  For quadratic meshes: returns ALL grid nodes on the face
    ///  (corners + mid-edge + mid-face).  For Hex20 one should
    ///  filter out the mid-face nodes that don't exist, but in
    ///  practice all face nodes are needed for BC application.
    std::vector<PetscInt> nodes_on_face(PrismFace face) const {
        std::vector<PetscInt> ids;
        const auto push_if_active = [&](int ix, int iy, int iz) {
            if (active_node_for_order(ix, iy, iz)) {
                ids.push_back(node_id(ix, iy, iz));
            }
        };
        const int gnx = nodes_x() - 1;  // max index in x
        const int gny = nodes_y() - 1;
        const int gnz = nodes_z() - 1;
        switch (face) {
        case PrismFace::MinX:
            ids.reserve(static_cast<std::size_t>(nodes_y() * nodes_z()));
            for (int iz = 0; iz <= gnz; ++iz)
                for (int iy = 0; iy <= gny; ++iy)
                    push_if_active(0, iy, iz);
            break;
        case PrismFace::MaxX:
            ids.reserve(static_cast<std::size_t>(nodes_y() * nodes_z()));
            for (int iz = 0; iz <= gnz; ++iz)
                for (int iy = 0; iy <= gny; ++iy)
                    push_if_active(gnx, iy, iz);
            break;
        case PrismFace::MinY:
            ids.reserve(static_cast<std::size_t>(nodes_x() * nodes_z()));
            for (int iz = 0; iz <= gnz; ++iz)
                for (int ix = 0; ix <= gnx; ++ix)
                    push_if_active(ix, 0, iz);
            break;
        case PrismFace::MaxY:
            ids.reserve(static_cast<std::size_t>(nodes_x() * nodes_z()));
            for (int iz = 0; iz <= gnz; ++iz)
                for (int ix = 0; ix <= gnx; ++ix)
                    push_if_active(ix, gny, iz);
            break;
        case PrismFace::MinZ:
            ids.reserve(static_cast<std::size_t>(nodes_x() * nodes_y()));
            for (int iy = 0; iy <= gny; ++iy)
                for (int ix = 0; ix <= gnx; ++ix)
                    push_if_active(ix, iy, 0);
            break;
        case PrismFace::MaxZ:
            ids.reserve(static_cast<std::size_t>(nodes_x() * nodes_y()));
            for (int iy = 0; iy <= gny; ++iy)
                for (int ix = 0; ix <= gnx; ++ix)
                    push_if_active(ix, iy, gnz);
            break;
        }
        return ids;
    }
};

// ─────────────────────────────────────────────────────────────────────
//  PrismaticSpec — user-facing specification
// ─────────────────────────────────────────────────────────────────────
struct PrismaticSpec {
    double width;    // X-extent
    double height;   // Y-extent
    double length;   // Z-extent (beam axis direction)
    int    nx;       // elements in X
    int    ny;       // elements in Y
    int    nz;       // elements in Z

    // Element polynomial order (default: linear Hex8)
    HexOrder hex_order = HexOrder::Linear;

    // Bias power for the longitudinal grid. 1.0 is uniform. Values > 1.0
    // cluster nodes near the selected longitudinal end(s) without changing
    // connectivity.
    double longitudinal_bias_power = 1.0;
    LongitudinalBiasLocation longitudinal_bias_location =
        LongitudinalBiasLocation::FixedEnd;

    // Optional user-supplied local corner levels in the cross-section.
    // When empty, the builder falls back to a uniform centred axis.
    std::vector<double> x_corner_levels_local{};
    std::vector<double> y_corner_levels_local{};

    // Optional origin and rotation (default: centred at origin, aligned
    // with global axes: X = width, Y = height, Z = length).
    std::array<double, 3> origin = {0.0, 0.0, 0.0};

    // Rotation matrix columns (identity = no rotation).
    // e_x, e_y, e_z define the local frame in global coordinates.
    std::array<double, 3> e_x = {1.0, 0.0, 0.0};
    std::array<double, 3> e_y = {0.0, 1.0, 0.0};
    std::array<double, 3> e_z = {0.0, 0.0, 1.0};

    std::string physical_group = "Solid";
};

inline std::vector<double>
build_prismatic_corner_axis_levels(
    double length,
    int subdivisions,
    double bias_power,
    LongitudinalBiasLocation bias_location = LongitudinalBiasLocation::FixedEnd)
{
    if (subdivisions <= 0) {
        throw std::invalid_argument(
            "PrismaticDomainBuilder requires positive longitudinal subdivisions.");
    }
    if (!(bias_power > 0.0)) {
        throw std::invalid_argument(
            "PrismaticDomainBuilder requires longitudinal_bias_power > 0.");
    }

    std::vector<double> levels(static_cast<std::size_t>(subdivisions + 1), 0.0);
    for (int i = 0; i <= subdivisions; ++i) {
        const double s =
            static_cast<double>(i) / static_cast<double>(subdivisions);
        double biased_s = s;
        switch (bias_location) {
            case LongitudinalBiasLocation::FixedEnd:
                biased_s = std::pow(s, bias_power);
                break;
            case LongitudinalBiasLocation::LoadedEnd:
                biased_s = 1.0 - std::pow(1.0 - s, bias_power);
                break;
            case LongitudinalBiasLocation::BothEnds:
                biased_s = s <= 0.5
                    ? 0.5 * std::pow(2.0 * s, bias_power)
                    : 1.0 - 0.5 * std::pow(2.0 * (1.0 - s), bias_power);
                break;
        }
        levels[static_cast<std::size_t>(i)] =
            length * biased_s;
    }
    levels.front() = 0.0;
    levels.back() = length;
    return levels;
}

inline std::vector<double>
build_prismatic_uniform_corner_axis_levels(
    double min_value,
    double max_value,
    int subdivisions)
{
    if (subdivisions <= 0) {
        throw std::invalid_argument(
            "PrismaticDomainBuilder requires positive subdivisions.");
    }
    if (!(max_value > min_value)) {
        throw std::invalid_argument(
            "PrismaticDomainBuilder requires increasing axis bounds.");
    }

    std::vector<double> levels(static_cast<std::size_t>(subdivisions + 1), 0.0);
    const double delta =
        (max_value - min_value) / static_cast<double>(subdivisions);
    for (int i = 0; i <= subdivisions; ++i) {
        levels[static_cast<std::size_t>(i)] =
            min_value + delta * static_cast<double>(i);
    }
    levels.front() = min_value;
    levels.back() = max_value;
    return levels;
}

inline void validate_prismatic_corner_axis_levels(
    const std::vector<double>& levels,
    int subdivisions,
    std::string_view axis_name)
{
    if (levels.size() != static_cast<std::size_t>(subdivisions + 1)) {
        throw std::invalid_argument(
            std::string{"PrismaticDomainBuilder requires "} +
            std::string{axis_name} +
            " corner levels to match the number of elements + 1.");
    }
    for (std::size_t i = 1; i < levels.size(); ++i) {
        if (!(levels[i] > levels[i - 1])) {
            throw std::invalid_argument(
                std::string{"PrismaticDomainBuilder requires strictly "}
                + "increasing " + std::string{axis_name} + " corner levels.");
        }
    }
}

inline std::vector<double> expand_prismatic_axis_levels(
    const std::vector<double>& corner_levels,
    int step)
{
    if (corner_levels.size() < 2) {
        throw std::invalid_argument(
            "PrismaticDomainBuilder requires at least two corner levels.");
    }
    if (step != 1 && step != 2) {
        throw std::invalid_argument(
            "PrismaticDomainBuilder supports only linear or quadratic z-level expansion.");
    }

    if (step == 1) {
        return corner_levels;
    }

    const auto element_count = corner_levels.size() - 1;
    std::vector<double> levels(2 * element_count + 1, 0.0);
    for (std::size_t e = 0; e < element_count; ++e) {
        const auto i0 = static_cast<std::size_t>(2 * e);
        const auto i1 = i0 + 1;
        const auto i2 = i0 + 2;
        levels[i0] = corner_levels[e];
        levels[i1] = 0.5 * (corner_levels[e] + corner_levels[e + 1]);
        levels[i2] = corner_levels[e + 1];
    }
    levels.back() = corner_levels.back();
    return levels;
}

struct PrismaticAxisLocation {
    int element_index{0};
    double parent_coordinate{0.0};
};

inline PrismaticAxisLocation locate_prismatic_axis_interval(
    const std::vector<double>& corner_levels,
    double coordinate)
{
    if (corner_levels.size() < 2) {
        throw std::invalid_argument(
            "PrismaticDomainBuilder requires at least two axis levels.");
    }

    if (coordinate <= corner_levels.front()) {
        return {0, -1.0};
    }
    if (coordinate >= corner_levels.back()) {
        return {static_cast<int>(corner_levels.size()) - 2, 1.0};
    }

    const auto upper =
        std::upper_bound(corner_levels.begin(), corner_levels.end(), coordinate);
    const auto element_index =
        static_cast<int>(std::distance(corner_levels.begin(), upper) - 1);
    const double x0 = corner_levels[static_cast<std::size_t>(element_index)];
    const double x1 = corner_levels[static_cast<std::size_t>(element_index + 1)];
    const double local = (coordinate - x0) / (x1 - x0);
    return {
        element_index,
        std::clamp(2.0 * local - 1.0, -1.0, 1.0),
    };
}

// ─────────────────────────────────────────────────────────────────────
//  make_prismatic_domain — construct a structured hex mesh (Hex8/20/27)
// ─────────────────────────────────────────────────────────────────────
inline std::pair<Domain<3>, PrismaticGrid>
make_prismatic_domain(const PrismaticSpec& spec)
{
    const int step = (spec.hex_order == HexOrder::Linear) ? 1 : 2;
    const double dx = spec.width  / static_cast<double>(spec.nx);
    const double dy = spec.height / static_cast<double>(spec.ny);
    const double dz = spec.length / static_cast<double>(spec.nz);
    const auto x_corner_levels = spec.x_corner_levels_local.empty()
                                     ? build_prismatic_uniform_corner_axis_levels(
                                           -0.5 * spec.width,
                                           0.5 * spec.width,
                                           spec.nx)
                                     : spec.x_corner_levels_local;
    validate_prismatic_corner_axis_levels(x_corner_levels, spec.nx, "x");
    const auto y_corner_levels = spec.y_corner_levels_local.empty()
                                     ? build_prismatic_uniform_corner_axis_levels(
                                           -0.5 * spec.height,
                                           0.5 * spec.height,
                                           spec.ny)
                                     : spec.y_corner_levels_local;
    validate_prismatic_corner_axis_levels(y_corner_levels, spec.ny, "y");
    const auto z_corner_levels = build_prismatic_corner_axis_levels(
        spec.length,
        spec.nz,
        spec.longitudinal_bias_power,
        spec.longitudinal_bias_location);
    const auto x_coordinates =
        expand_prismatic_axis_levels(x_corner_levels, step);
    const auto y_coordinates =
        expand_prismatic_axis_levels(y_corner_levels, step);
    const auto z_coordinates =
        expand_prismatic_axis_levels(z_corner_levels, step);

    PrismaticGrid grid{
        spec.nx, spec.ny, spec.nz,
        step,
        dx, dy, dz,
        spec.width, spec.height, spec.length,
        spec.hex_order,
        spec.longitudinal_bias_power,
        spec.longitudinal_bias_location,
        x_coordinates,
        y_coordinates,
        z_coordinates
    };

    Domain<3> domain;

    // ── Nodes ─────────────────────────────────────────────────────
    //  For linear meshes:    (nx+1)×(ny+1)×(nz+1) nodes.
    //  For quadratic meshes: (2·nx+1)×(2·ny+1)×(2·nz+1) nodes.
    //
    //  All grid nodes are created, even for Hex20 where face-centre
    //  and body-centre positions are unused — they are harmless.
    const auto total = static_cast<std::size_t>(grid.total_nodes());
    domain.preallocate_node_capacity(total);

    // Local coordinates: x ∈ [-w/2, w/2], y ∈ [-h/2, h/2], z ∈ [0, L]
    for (int iz = 0; iz < grid.nodes_z(); ++iz) {
        for (int iy = 0; iy < grid.nodes_y(); ++iy) {
            for (int ix = 0; ix < grid.nodes_x(); ++ix) {
                const double lx = grid.x_coordinate(ix);
                const double ly = grid.y_coordinate(iy);
                const double lz = grid.z_coordinate(iz);

                const double gx = spec.origin[0]
                    + lx * spec.e_x[0] + ly * spec.e_y[0] + lz * spec.e_z[0];
                const double gy = spec.origin[1]
                    + lx * spec.e_x[1] + ly * spec.e_y[1] + lz * spec.e_z[1];
                const double gz = spec.origin[2]
                    + lx * spec.e_x[2] + ly * spec.e_y[2] + lz * spec.e_z[2];

                domain.add_node(grid.node_id(ix, iy, iz), gx, gy, gz);
            }
        }
    }

    // ── Elements ──────────────────────────────────────────────────
    std::size_t tag = 0;

    if (spec.hex_order == HexOrder::Linear) {
        // ── Hex8 — LagrangeElement3D<2,2,2> ──────────────────────
        for (int iz = 0; iz < spec.nz; ++iz) {
            for (int iy = 0; iy < spec.ny; ++iy) {
                for (int ix = 0; ix < spec.nx; ++ix) {
                    PetscInt conn[8] = {
                        grid.node_id(ix,     iy,     iz),
                        grid.node_id(ix + 1, iy,     iz),
                        grid.node_id(ix,     iy + 1, iz),
                        grid.node_id(ix + 1, iy + 1, iz),
                        grid.node_id(ix,     iy,     iz + 1),
                        grid.node_id(ix + 1, iy,     iz + 1),
                        grid.node_id(ix,     iy + 1, iz + 1),
                        grid.node_id(ix + 1, iy + 1, iz + 1),
                    };
                    auto& geom = domain.make_element<LagrangeElement3D<2, 2, 2>>(
                        GaussLegendreCellIntegrator<2, 2, 2>{}, tag++, conn);
                    geom.set_physical_group(spec.physical_group);
                }
            }
        }
    } else if (spec.hex_order == HexOrder::Serendipity) {
        // ── Hex20 — SerendipityElement<3,3,2> ────────────────────
        //  Connectivity follows SerendipityCell<3,2> node ordering:
        //    0–7   : corners   (even, even, even)
        //    8–11  : bottom-face edge midpoints
        //    12–15 : top-face edge midpoints
        //    16–19 : vertical edge midpoints
        for (int iz = 0; iz < spec.nz; ++iz) {
            for (int iy = 0; iy < spec.ny; ++iy) {
                for (int ix = 0; ix < spec.nx; ++ix) {
                    const int bx = 2 * ix, by = 2 * iy, bz = 2 * iz;
                    PetscInt conn[20] = {
                        grid.node_id(bx,   by,   bz  ),   //  0 (-1,-1,-1)
                        grid.node_id(bx+2, by,   bz  ),   //  1 (+1,-1,-1)
                        grid.node_id(bx+2, by+2, bz  ),   //  2 (+1,+1,-1)
                        grid.node_id(bx,   by+2, bz  ),   //  3 (-1,+1,-1)
                        grid.node_id(bx,   by,   bz+2),   //  4 (-1,-1,+1)
                        grid.node_id(bx+2, by,   bz+2),   //  5 (+1,-1,+1)
                        grid.node_id(bx+2, by+2, bz+2),   //  6 (+1,+1,+1)
                        grid.node_id(bx,   by+2, bz+2),   //  7 (-1,+1,+1)
                        grid.node_id(bx+1, by,   bz  ),   //  8 ( 0,-1,-1)
                        grid.node_id(bx+2, by+1, bz  ),   //  9 (+1, 0,-1)
                        grid.node_id(bx+1, by+2, bz  ),   // 10 ( 0,+1,-1)
                        grid.node_id(bx,   by+1, bz  ),   // 11 (-1, 0,-1)
                        grid.node_id(bx+1, by,   bz+2),   // 12 ( 0,-1,+1)
                        grid.node_id(bx+2, by+1, bz+2),   // 13 (+1, 0,+1)
                        grid.node_id(bx+1, by+2, bz+2),   // 14 ( 0,+1,+1)
                        grid.node_id(bx,   by+1, bz+2),   // 15 (-1, 0,+1)
                        grid.node_id(bx,   by,   bz+1),   // 16 (-1,-1, 0)
                        grid.node_id(bx+2, by,   bz+1),   // 17 (+1,-1, 0)
                        grid.node_id(bx+2, by+2, bz+1),   // 18 (+1,+1, 0)
                        grid.node_id(bx,   by+2, bz+1),   // 19 (-1,+1, 0)
                    };
                    auto& geom = domain.make_element<SerendipityElement<3, 3, 2>>(
                        GaussLegendreCellIntegrator<3, 3, 3>{}, tag++, conn);
                    geom.set_physical_group(spec.physical_group);
                }
            }
        }
    } else {
        // ── Hex27 — LagrangeElement3D<3,3,3> ─────────────────────
        //  Column-major ordering: conn[i0 + 3·i1 + 9·i2]
        for (int iz = 0; iz < spec.nz; ++iz) {
            for (int iy = 0; iy < spec.ny; ++iy) {
                for (int ix = 0; ix < spec.nx; ++ix) {
                    const int bx = 2 * ix, by = 2 * iy, bz = 2 * iz;
                    PetscInt conn[27];
                    for (int i2 = 0; i2 < 3; ++i2)
                        for (int i1 = 0; i1 < 3; ++i1)
                            for (int i0 = 0; i0 < 3; ++i0)
                                conn[i0 + 3 * i1 + 9 * i2] =
                                    grid.node_id(bx + i0, by + i1, bz + i2);
                    auto& geom = domain.make_element<LagrangeElement3D<3, 3, 3>>(
                        GaussLegendreCellIntegrator<3, 3, 3>{}, tag++, conn);
                    geom.set_physical_group(spec.physical_group);
                }
            }
        }
    }

    domain.assemble_sieve();

    return {std::move(domain), std::move(grid)};
}


// ─────────────────────────────────────────────────────────────────────
//  align_to_beam — build a PrismaticSpec aligned to a beam element
// ─────────────────────────────────────────────────────────────────────
//
//  Given two endpoint coordinates (node A and node B of a beam) and a
//  reference "up" vector, returns a PrismaticSpec whose Z-axis is
//  aligned with the beam axis, Y with the strong-axis (up), and X
//  with the cross product.
//
//  width/height define the cross-section extents; length is set to
//  the distance |B − A|.
//
inline PrismaticSpec align_to_beam(
    std::array<double, 3> A,
    std::array<double, 3> B,
    std::array<double, 3> up,
    double width, double height,
    int nx, int ny, int nz,
    const std::string& group = "Solid",
    HexOrder hex_order = HexOrder::Linear)
{
    // e_z = beam axis (A → B normalised)
    std::array<double, 3> ab{B[0]-A[0], B[1]-A[1], B[2]-A[2]};
    double L = std::sqrt(ab[0]*ab[0] + ab[1]*ab[1] + ab[2]*ab[2]);
    std::array<double, 3> ez{ab[0]/L, ab[1]/L, ab[2]/L};

    // e_x = up × e_z (weak axis)
    std::array<double, 3> ex{
        up[1]*ez[2] - up[2]*ez[1],
        up[2]*ez[0] - up[0]*ez[2],
        up[0]*ez[1] - up[1]*ez[0]
    };
    double nex = std::sqrt(ex[0]*ex[0] + ex[1]*ex[1] + ex[2]*ex[2]);
    ex = {ex[0]/nex, ex[1]/nex, ex[2]/nex};

    // e_y = e_z × e_x (strong axis, ≈ up direction)
    std::array<double, 3> ey{
        ez[1]*ex[2] - ez[2]*ex[1],
        ez[2]*ex[0] - ez[0]*ex[2],
        ez[0]*ex[1] - ez[1]*ex[0]
    };

    // The prismatic mesh uses local z in [0,L], so the placement
    // origin must be face A. Using the midpoint shifts every local
    // continuous column by L/2 along the macro element axis.
    std::array<double, 3> origin{A[0], A[1], A[2]};

    return PrismaticSpec{
        .width  = width,
        .height = height,
        .length = L,
        .nx = nx, .ny = ny, .nz = nz,
        .hex_order = hex_order,
        .origin = origin,
        .e_x = ex, .e_y = ey, .e_z = ez,
        .physical_group = group
    };
}

} // namespace fall_n


// ═══════════════════════════════════════════════════════════════════════
//  Reinforced Prismatic Domain — hex8 mesh + embedded rebar line elements
// ═══════════════════════════════════════════════════════════════════════

namespace fall_n {

// ── Rebar bar specification (single longitudinal bar) ────────────────
//
//  Bars are placed at their exact physical (y, z) coordinates relative
//  to the section centre — NOT snapped to grid corners.  This gives
//  physically correct positioning inside the concrete cover.
//
//  The PrismaticDomainBuilder creates independent rebar nodes at these
//  positions (one per z-level of the grid).  The EmbeddingInfo records
//  which host hex element contains each rebar node and the parent
//  coordinates (ξ, η) on the cross-section, enabling shape-function
//  interpolation for the Master-Slave coupling.
//
struct RebarBar {
    double ly{0};             ///< Local y coordinate [m] (section width dir)
    double lz{0};             ///< Local z coordinate [m] (section height dir)
    double area{0};           ///< Bar cross-sectional area [length²]
    double diameter{0};       ///< Bar diameter [m] (for VTK visualisation)
    std::string group = "Rebar";  ///< Physical group label
};

// ── Embedding info for a single rebar node ───────────────────────────
//  Maps a rebar-only node to its host hex element so that the
//  displacement can be interpolated as  u_rebar = Σ N_i(ξ,η,ζ) · u_i.
struct RebarNodeEmbedding {
    PetscInt  rebar_node_id{-1};    ///< Node ID in the Domain
    int       host_elem_ix{0};      ///< Host hex element index (ix in grid)
    int       host_elem_iy{0};      ///< Host hex element index (iy in grid)
    int       host_elem_iz{0};      ///< Host hex element index (iz in grid)
    double    xi{0};                ///< Parent coordinate ξ ∈ [-1,1] in host
    double    eta{0};               ///< Parent coordinate η ∈ [-1,1] in host
    double    zeta{0};              ///< Parent coordinate ζ ∈ [-1,1] in host
};

// ── Rebar specification for a prismatic section ──────────────────────
struct RebarSpec {
    std::vector<RebarBar> bars;
};

// Generic embedded one-dimensional reinforcement paths.
//
// Longitudinal bars are still represented by RebarSpec because they have a
// structured relationship with the prism axis and produce rich validation
// histories. These polylines cover the complementary case: transverse ties,
// stirrup legs, boundary bars, and future arbitrary embedded fibers whose
// nodes must be placed at exact physical coordinates and tied to the host by
// the same interpolation/penalty machinery.
struct EmbeddedRebarPolyline {
    std::vector<std::array<double, 3>> local_points{};
    bool closed{false};
    double area{0.0};
    double diameter{0.0};
    std::string group{"EmbeddedRebar"};
};

struct EmbeddedRebarSpec {
    std::vector<EmbeddedRebarPolyline> polylines{};
};

struct EmbeddedRebarElementMetadata {
    std::size_t polyline_index{0};
    std::size_t segment_index{0};
    double area{0.0};
    double diameter{0.0};
    std::string group{"EmbeddedRebar"};
};

enum class RebarLineInterpolation {
    automatic,
    two_node_linear,
    three_node_quadratic
};

[[nodiscard]] constexpr const char*
to_string(RebarLineInterpolation interpolation) noexcept
{
    switch (interpolation) {
        case RebarLineInterpolation::automatic:
            return "automatic";
        case RebarLineInterpolation::two_node_linear:
            return "two_node_linear";
        case RebarLineInterpolation::three_node_quadratic:
            return "three_node_quadratic";
    }
    return "unknown_rebar_line_interpolation";
}

[[nodiscard]] inline std::size_t resolve_rebar_line_num_nodes(
    HexOrder hex_order,
    RebarLineInterpolation interpolation)
{
    switch (interpolation) {
        case RebarLineInterpolation::automatic:
            // The validated default keeps 2-node bars for Hex8/Hex20 and
            // only promotes to 3-node interpolation for the full triquadratic
            // Hex27 host. Serendipity Hex20 keeps an explicit override path
            // for research probes, but not as the canonical baseline.
            return hex_order == HexOrder::Quadratic ? 3u : 2u;
        case RebarLineInterpolation::two_node_linear:
            return 2u;
        case RebarLineInterpolation::three_node_quadratic:
            if (hex_order == HexOrder::Linear) {
                throw std::invalid_argument(
                    "Three-node embedded rebar interpolation requires a "
                    "quadratic host mesh.");
            }
            return 3u;
    }
    throw std::invalid_argument("Unsupported rebar line interpolation.");
}

// ─────────────────────────────────────────────────────────────────────
//  make_reinforced_prismatic_domain — hex mesh with embedded rebar
// ─────────────────────────────────────────────────────────────────────
//
//  Generates a prismatic hex mesh identical to make_prismatic_domain,
//  then adds embedded rebar bars at their EXACT physical positions
//  within the cross-section.
//
//  Each rebar bar at (ly, lz) produces:
//    - (nz + 1) × step  additional nodes at the physical bar position
//      along each z-level of the grid.
//    - nz line elements connecting consecutive rebar nodes.
//
//  The rebar nodes do NOT coincide with hex nodes (unless the bar
//  position happens to match a grid node).  The coupling between
//  rebar and continuum is provided by the caller via penalty springs
//  or MPC, using the RebarNodeEmbedding data which records:
//    - host hex element indices (ix, iy) in the cross-section
//    - parent coordinates (ξ, η) within the host element
//
//  For visualisation, each RebarBar stores its diameter so that VTK
//  can render tubes with physically correct cross-section size.
//
struct RebarElementRange {
    std::size_t first;  ///< First rebar element index in Domain::elements()
    std::size_t last;   ///< One past last rebar element index
};

struct ReinforcedDomainResult {
    Domain<3>          domain;
    PrismaticGrid      grid;
    RebarElementRange  rebar_range;
    RebarElementRange  embedded_rebar_range{0, 0};
    RebarLineInterpolation rebar_line_interpolation{
        RebarLineInterpolation::automatic};
    std::size_t rebar_line_num_nodes{2};

    /// Per-bar embedding info: host hex element + parent coordinates.
    /// Size = num_bars × (nz + 1) rebar nodes.
    std::vector<RebarNodeEmbedding> embeddings;

    /// Per-bar diameters: one per bar (same order as RebarSpec::bars).
    std::vector<double> bar_diameters;

    /// Arbitrary embedded polyline nodes/elements, e.g. stirrup loops.
    std::vector<RebarNodeEmbedding> embedded_rebar_embeddings;
    std::vector<EmbeddedRebarElementMetadata> embedded_rebar_elements;
};

inline std::array<double, 3> prismatic_local_to_global(
    const PrismaticSpec& spec,
    const std::array<double, 3>& local) noexcept
{
    return {
        spec.origin[0] + local[0] * spec.e_x[0] +
            local[1] * spec.e_y[0] + local[2] * spec.e_z[0],
        spec.origin[1] + local[0] * spec.e_x[1] +
            local[1] * spec.e_y[1] + local[2] * spec.e_z[1],
        spec.origin[2] + local[0] * spec.e_x[2] +
            local[1] * spec.e_y[2] + local[2] * spec.e_z[2],
    };
}

inline RebarNodeEmbedding make_prismatic_point_embedding(
    const std::vector<double>& x_corner_levels,
    const std::vector<double>& y_corner_levels,
    const std::vector<double>& z_corner_levels,
    PetscInt node_id,
    const std::array<double, 3>& local)
{
    const auto x_location =
        locate_prismatic_axis_interval(x_corner_levels, local[0]);
    const auto y_location =
        locate_prismatic_axis_interval(y_corner_levels, local[1]);
    const auto z_location =
        locate_prismatic_axis_interval(z_corner_levels, local[2]);
    return RebarNodeEmbedding{
        node_id,
        x_location.element_index,
        y_location.element_index,
        z_location.element_index,
        x_location.parent_coordinate,
        y_location.parent_coordinate,
        z_location.parent_coordinate};
}

inline ReinforcedDomainResult
make_reinforced_prismatic_domain(const PrismaticSpec& spec,
                                 const RebarSpec& rebar,
                                 RebarLineInterpolation rebar_line_interpolation =
                                     RebarLineInterpolation::automatic,
                                 const EmbeddedRebarSpec& embedded_rebar = {})
{
    const int step = (spec.hex_order == HexOrder::Linear) ? 1 : 2;
    const double dx = spec.width  / static_cast<double>(spec.nx);
    const double dy = spec.height / static_cast<double>(spec.ny);
    const double dz = spec.length / static_cast<double>(spec.nz);
    const auto x_corner_levels = spec.x_corner_levels_local.empty()
                                     ? build_prismatic_uniform_corner_axis_levels(
                                           -0.5 * spec.width,
                                           0.5 * spec.width,
                                           spec.nx)
                                     : spec.x_corner_levels_local;
    validate_prismatic_corner_axis_levels(x_corner_levels, spec.nx, "x");
    const auto y_corner_levels = spec.y_corner_levels_local.empty()
                                     ? build_prismatic_uniform_corner_axis_levels(
                                           -0.5 * spec.height,
                                           0.5 * spec.height,
                                           spec.ny)
                                     : spec.y_corner_levels_local;
    validate_prismatic_corner_axis_levels(y_corner_levels, spec.ny, "y");
    const auto z_corner_levels = build_prismatic_corner_axis_levels(
        spec.length,
        spec.nz,
        spec.longitudinal_bias_power,
        spec.longitudinal_bias_location);
    const auto x_coordinates =
        expand_prismatic_axis_levels(x_corner_levels, step);
    const auto y_coordinates =
        expand_prismatic_axis_levels(y_corner_levels, step);
    const auto z_coordinates =
        expand_prismatic_axis_levels(z_corner_levels, step);

    PrismaticGrid grid{
        spec.nx, spec.ny, spec.nz,
        step,
        dx, dy, dz,
        spec.width, spec.height, spec.length,
        spec.hex_order,
        spec.longitudinal_bias_power,
        spec.longitudinal_bias_location,
        x_coordinates,
        y_coordinates,
        z_coordinates
    };

    Domain<3> domain;

    // ── Hex nodes (same scheme as make_prismatic_domain) ──────────
    const auto total_hex_nodes = static_cast<std::size_t>(grid.total_nodes());
    const auto num_bars = rebar.bars.size();
    const auto rebar_nodes_per_bar =
        static_cast<std::size_t>(step * spec.nz + 1);
    const auto total_rebar_nodes = num_bars * rebar_nodes_per_bar;
    std::size_t total_embedded_rebar_nodes = 0;
    for (const auto& polyline : embedded_rebar.polylines) {
        total_embedded_rebar_nodes += polyline.local_points.size();
    }
    const auto rebar_line_num_nodes =
        resolve_rebar_line_num_nodes(spec.hex_order, rebar_line_interpolation);

    domain.preallocate_node_capacity(
        total_hex_nodes + total_rebar_nodes + total_embedded_rebar_nodes);

    for (int iz = 0; iz < grid.nodes_z(); ++iz) {
        for (int iy = 0; iy < grid.nodes_y(); ++iy) {
            for (int ix = 0; ix < grid.nodes_x(); ++ix) {
                const double lx = grid.x_coordinate(ix);
                const double ly = grid.y_coordinate(iy);
                const double lz = grid.z_coordinate(iz);

                const double gx = spec.origin[0]
                    + lx * spec.e_x[0] + ly * spec.e_y[0] + lz * spec.e_z[0];
                const double gy = spec.origin[1]
                    + lx * spec.e_x[1] + ly * spec.e_y[1] + lz * spec.e_z[1];
                const double gz = spec.origin[2]
                    + lx * spec.e_x[2] + ly * spec.e_y[2] + lz * spec.e_z[2];

                domain.add_node(grid.node_id(ix, iy, iz), gx, gy, gz);
            }
        }
    }

    // ── Hex elements (dispatched by order) ────────────────────────
    std::size_t tag = 0;

    if (spec.hex_order == HexOrder::Linear) {
        for (int iz = 0; iz < spec.nz; ++iz) {
            for (int iy = 0; iy < spec.ny; ++iy) {
                for (int ix = 0; ix < spec.nx; ++ix) {
                    PetscInt conn[8] = {
                        grid.node_id(ix,     iy,     iz),
                        grid.node_id(ix + 1, iy,     iz),
                        grid.node_id(ix,     iy + 1, iz),
                        grid.node_id(ix + 1, iy + 1, iz),
                        grid.node_id(ix,     iy,     iz + 1),
                        grid.node_id(ix + 1, iy,     iz + 1),
                        grid.node_id(ix,     iy + 1, iz + 1),
                        grid.node_id(ix + 1, iy + 1, iz + 1),
                    };
                    auto& geom = domain.make_element<LagrangeElement3D<2, 2, 2>>(
                        GaussLegendreCellIntegrator<2, 2, 2>{}, tag++, conn);
                    geom.set_physical_group(spec.physical_group);
                }
            }
        }
    } else if (spec.hex_order == HexOrder::Serendipity) {
        for (int iz = 0; iz < spec.nz; ++iz) {
            for (int iy = 0; iy < spec.ny; ++iy) {
                for (int ix = 0; ix < spec.nx; ++ix) {
                    const int bx = 2 * ix, by = 2 * iy, bz = 2 * iz;
                    PetscInt conn[20] = {
                        grid.node_id(bx,   by,   bz  ),
                        grid.node_id(bx+2, by,   bz  ),
                        grid.node_id(bx+2, by+2, bz  ),
                        grid.node_id(bx,   by+2, bz  ),
                        grid.node_id(bx,   by,   bz+2),
                        grid.node_id(bx+2, by,   bz+2),
                        grid.node_id(bx+2, by+2, bz+2),
                        grid.node_id(bx,   by+2, bz+2),
                        grid.node_id(bx+1, by,   bz  ),
                        grid.node_id(bx+2, by+1, bz  ),
                        grid.node_id(bx+1, by+2, bz  ),
                        grid.node_id(bx,   by+1, bz  ),
                        grid.node_id(bx+1, by,   bz+2),
                        grid.node_id(bx+2, by+1, bz+2),
                        grid.node_id(bx+1, by+2, bz+2),
                        grid.node_id(bx,   by+1, bz+2),
                        grid.node_id(bx,   by,   bz+1),
                        grid.node_id(bx+2, by,   bz+1),
                        grid.node_id(bx+2, by+2, bz+1),
                        grid.node_id(bx,   by+2, bz+1),
                    };
                    auto& geom = domain.make_element<SerendipityElement<3, 3, 2>>(
                        GaussLegendreCellIntegrator<3, 3, 3>{}, tag++, conn);
                    geom.set_physical_group(spec.physical_group);
                }
            }
        }
    } else {
        for (int iz = 0; iz < spec.nz; ++iz) {
            for (int iy = 0; iy < spec.ny; ++iy) {
                for (int ix = 0; ix < spec.nx; ++ix) {
                    const int bx = 2 * ix, by = 2 * iy, bz = 2 * iz;
                    PetscInt conn[27];
                    for (int i2 = 0; i2 < 3; ++i2)
                        for (int i1 = 0; i1 < 3; ++i1)
                            for (int i0 = 0; i0 < 3; ++i0)
                                conn[i0 + 3 * i1 + 9 * i2] =
                                    grid.node_id(bx + i0, by + i1, bz + i2);
                    auto& geom = domain.make_element<LagrangeElement3D<3, 3, 3>>(
                        GaussLegendreCellIntegrator<3, 3, 3>{}, tag++, conn);
                    geom.set_physical_group(spec.physical_group);
                }
            }
        }
    }

    // ── Rebar: create independent nodes at exact physical positions ──
    //
    //  Each bar at local (ly, lz) gets rebar_nodes_per_bar nodes along
    //  the z-axis.  Node IDs start after the hex grid nodes.
    //  For each rebar node, we compute:
    //    - The host hex element (ix_host, iy_host) that contains (ly, lz)
    //    - The parent coordinates (ξ, η) within that element
    //
    PetscInt rebar_base_id = static_cast<PetscInt>(total_hex_nodes);
    std::vector<RebarNodeEmbedding> embeddings;
    embeddings.reserve(total_rebar_nodes);
    std::vector<double> bar_diameters;
    bar_diameters.reserve(num_bars);

    for (std::size_t b = 0; b < num_bars; ++b) {
        const auto& bar = rebar.bars[b];
        bar_diameters.push_back(bar.diameter);

        // Map physical (ly, lz) → host hex element in cross-section
        //   ly ∈ [-width/2, width/2] → X direction
        //   lz ∈ [-height/2, height/2] → Y direction
        const auto x_location =
            locate_prismatic_axis_interval(x_corner_levels, bar.ly);
        const auto y_location =
            locate_prismatic_axis_interval(y_corner_levels, bar.lz);
        const int ix_host = x_location.element_index;
        const int iy_host = y_location.element_index;
        const double xi = x_location.parent_coordinate;
        const double eta = y_location.parent_coordinate;

        // Create rebar nodes along the z-axis at each z sub-level
        for (std::size_t iz = 0; iz < rebar_nodes_per_bar; ++iz) {
            const PetscInt nid = rebar_base_id++;
            const double lz_pos = grid.z_coordinate(static_cast<int>(iz));

            const double gx = spec.origin[0]
                + bar.ly * spec.e_x[0] + bar.lz * spec.e_y[0]
                + lz_pos * spec.e_z[0];
            const double gy = spec.origin[1]
                + bar.ly * spec.e_x[1] + bar.lz * spec.e_y[1]
                + lz_pos * spec.e_z[1];
            const double gz = spec.origin[2]
                + bar.ly * spec.e_x[2] + bar.lz * spec.e_y[2]
                + lz_pos * spec.e_z[2];

            domain.add_node(nid, gx, gy, gz);

            // Host hex element in the z-direction and parent ζ
            int iz_elem = static_cast<int>(iz) / step;
            if (iz_elem >= spec.nz) iz_elem = spec.nz - 1;
            const double zeta_r = 2.0
                * static_cast<double>(static_cast<int>(iz) - iz_elem * step)
                / static_cast<double>(step) - 1.0;

            embeddings.push_back(RebarNodeEmbedding{
                nid, ix_host, iy_host, iz_elem,
                xi, eta, zeta_r});
        }
    }

    // ── Rebar line elements ───────────────────────────────────────
    //  Each bar produces nz line elements connecting consecutive
    //  rebar nodes.  For quadratic (step=2) meshes, line elements
    //  connect every-other rebar node (corner nodes of each z-layer).
    const std::size_t rebar_first = domain.num_elements();
    const PetscInt rebar_nodes_base =
        static_cast<PetscInt>(total_hex_nodes);

    for (std::size_t b = 0; b < num_bars; ++b) {
        const PetscInt bar_start =
            rebar_nodes_base + static_cast<PetscInt>(b * rebar_nodes_per_bar);

        for (int iz = 0; iz < spec.nz; ++iz) {
            if (rebar_line_num_nodes == 2) {
                PetscInt conn[2] = {
                    bar_start + static_cast<PetscInt>(step * iz),
                    bar_start + static_cast<PetscInt>(step * (iz + 1)),
                };
                auto& geom = domain.make_element<LagrangeElement3D<2>>(
                    GaussLegendreCellIntegrator<2>{}, tag++, conn);
                geom.set_physical_group(rebar.bars[b].group);
                continue;
            }

            if (step != 2) {
                throw std::invalid_argument(
                    "Three-node embedded rebar interpolation requires "
                    "quadratic z-level subdivision.");
            }

            PetscInt conn[3] = {
                bar_start + static_cast<PetscInt>(2 * iz),
                bar_start + static_cast<PetscInt>(2 * iz + 1),
                bar_start + static_cast<PetscInt>(2 * iz + 2),
            };
            auto& geom = domain.make_element<LagrangeElement3D<3>>(
                GaussLegendreCellIntegrator<3>{}, tag++, conn);
            geom.set_physical_group(rebar.bars[b].group);
        }
    }

    const std::size_t rebar_last = domain.num_elements();

    const std::size_t embedded_rebar_first = domain.num_elements();
    std::vector<RebarNodeEmbedding> embedded_rebar_embeddings;
    embedded_rebar_embeddings.reserve(total_embedded_rebar_nodes);
    std::vector<EmbeddedRebarElementMetadata> embedded_rebar_elements;

    PetscInt embedded_rebar_base_id = rebar_base_id;
    for (std::size_t p = 0; p < embedded_rebar.polylines.size(); ++p) {
        const auto& polyline = embedded_rebar.polylines[p];
        const auto point_count = polyline.local_points.size();
        if (point_count < 2) {
            throw std::invalid_argument(
                "Embedded rebar polylines require at least two local points.");
        }
        if (!(polyline.area > 0.0)) {
            throw std::invalid_argument(
                "Embedded rebar polylines require a positive cross-section area.");
        }

        std::vector<PetscInt> polyline_node_ids;
        polyline_node_ids.reserve(point_count);
        for (const auto& local : polyline.local_points) {
            const PetscInt nid = embedded_rebar_base_id++;
            const auto global = prismatic_local_to_global(spec, local);
            domain.add_node(nid, global[0], global[1], global[2]);
            polyline_node_ids.push_back(nid);
            embedded_rebar_embeddings.push_back(
                make_prismatic_point_embedding(
                    x_corner_levels,
                    y_corner_levels,
                    z_corner_levels,
                    nid,
                    local));
        }

        const auto segment_count =
            polyline.closed ? point_count : point_count - 1;
        embedded_rebar_elements.reserve(
            embedded_rebar_elements.size() + segment_count);
        for (std::size_t s = 0; s < segment_count; ++s) {
            const auto next = (s + 1) % point_count;
            PetscInt conn[2] = {
                polyline_node_ids[s],
                polyline_node_ids[next],
            };
            auto& geom = domain.make_element<LagrangeElement3D<2>>(
                GaussLegendreCellIntegrator<2>{}, tag++, conn);
            geom.set_physical_group(polyline.group);
            embedded_rebar_elements.push_back({
                .polyline_index = p,
                .segment_index = s,
                .area = polyline.area,
                .diameter = polyline.diameter,
                .group = polyline.group});
        }
    }

    const std::size_t embedded_rebar_last = domain.num_elements();

    domain.assemble_sieve();

    return {std::move(domain),
            std::move(grid),
            RebarElementRange{rebar_first, rebar_last},
            RebarElementRange{embedded_rebar_first, embedded_rebar_last},
            rebar_line_interpolation,
            rebar_line_num_nodes,
            std::move(embeddings),
            std::move(bar_diameters),
            std::move(embedded_rebar_embeddings),
            std::move(embedded_rebar_elements)};
}

} // namespace fall_n

#endif // PRISMATIC_DOMAIN_BUILDER_HH
