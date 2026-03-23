#ifndef PRISMATIC_DOMAIN_BUILDER_HH
#define PRISMATIC_DOMAIN_BUILDER_HH

// ═══════════════════════════════════════════════════════════════════════
//  PrismaticDomainBuilder — Hex8 prismatic mesh for sub-model volumes
// ═══════════════════════════════════════════════════════════════════════
//
//  Generates a structured hex8 mesh for a rectangular prism defined by
//  its width (X), height (Y) and length (Z), optionally rotated and
//  translated to align with a beam element's local frame:
//
//    auto [domain, grid] = fall_n::make_prismatic_domain({
//        .width  = 0.40,   // X-extent
//        .height = 0.60,   // Y-extent
//        .length = 3.20,   // Z-extent (beam axis)
//        .nx = 4, .ny = 6, .nz = 8,
//    });
//
//  The returned PrismaticGrid provides O(1) node_id() lookups and
//  face extraction for boundary condition application.
//
// ═══════════════════════════════════════════════════════════════════════

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include <petsc.h>

#include "../domain/Domain.hh"
#include "../elements/element_geometry/LagrangeElement.hh"
#include "../numerics/numerical_integration/GaussLegendreCellIntegrator.hh"

namespace fall_n {

// ─────────────────────────────────────────────────────────────────────
//  Face identifiers for the six faces of the prism
// ─────────────────────────────────────────────────────────────────────
enum class PrismFace { MinX, MaxX, MinY, MaxY, MinZ, MaxZ };

// ─────────────────────────────────────────────────────────────────────
//  PrismaticGrid — metadata for a structured hex mesh
// ─────────────────────────────────────────────────────────────────────
struct PrismaticGrid {
    int nx, ny, nz;          // number of elements per direction
    double dx, dy, dz;       // element sizes
    double width, height, length;

    int nodes_x() const noexcept { return nx + 1; }
    int nodes_y() const noexcept { return ny + 1; }
    int nodes_z() const noexcept { return nz + 1; }
    int total_nodes()    const noexcept { return nodes_x() * nodes_y() * nodes_z(); }
    int total_elements() const noexcept { return nx * ny * nz; }

    /// O(1) node ID from structured grid indices.
    PetscInt node_id(int ix, int iy, int iz) const noexcept {
        return static_cast<PetscInt>(
            iz * nodes_x() * nodes_y() + iy * nodes_x() + ix);
    }

    /// Collect all node IDs on a given prism face.
    std::vector<PetscInt> nodes_on_face(PrismFace face) const {
        std::vector<PetscInt> ids;
        switch (face) {
        case PrismFace::MinX:
            ids.reserve(static_cast<std::size_t>(nodes_y() * nodes_z()));
            for (int iz = 0; iz <= nz; ++iz)
                for (int iy = 0; iy <= ny; ++iy)
                    ids.push_back(node_id(0, iy, iz));
            break;
        case PrismFace::MaxX:
            ids.reserve(static_cast<std::size_t>(nodes_y() * nodes_z()));
            for (int iz = 0; iz <= nz; ++iz)
                for (int iy = 0; iy <= ny; ++iy)
                    ids.push_back(node_id(nx, iy, iz));
            break;
        case PrismFace::MinY:
            ids.reserve(static_cast<std::size_t>(nodes_x() * nodes_z()));
            for (int iz = 0; iz <= nz; ++iz)
                for (int ix = 0; ix <= nx; ++ix)
                    ids.push_back(node_id(ix, 0, iz));
            break;
        case PrismFace::MaxY:
            ids.reserve(static_cast<std::size_t>(nodes_x() * nodes_z()));
            for (int iz = 0; iz <= nz; ++iz)
                for (int ix = 0; ix <= nx; ++ix)
                    ids.push_back(node_id(ix, ny, iz));
            break;
        case PrismFace::MinZ:
            ids.reserve(static_cast<std::size_t>(nodes_x() * nodes_y()));
            for (int iy = 0; iy <= ny; ++iy)
                for (int ix = 0; ix <= nx; ++ix)
                    ids.push_back(node_id(ix, iy, 0));
            break;
        case PrismFace::MaxZ:
            ids.reserve(static_cast<std::size_t>(nodes_x() * nodes_y()));
            for (int iy = 0; iy <= ny; ++iy)
                for (int ix = 0; ix <= nx; ++ix)
                    ids.push_back(node_id(ix, iy, nz));
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

// ─────────────────────────────────────────────────────────────────────
//  make_prismatic_domain — construct a structured hex8 mesh
// ─────────────────────────────────────────────────────────────────────
inline std::pair<Domain<3>, PrismaticGrid>
make_prismatic_domain(const PrismaticSpec& spec)
{
    const double dx = spec.width  / static_cast<double>(spec.nx);
    const double dy = spec.height / static_cast<double>(spec.ny);
    const double dz = spec.length / static_cast<double>(spec.nz);

    PrismaticGrid grid{
        spec.nx, spec.ny, spec.nz,
        dx, dy, dz,
        spec.width, spec.height, spec.length
    };

    Domain<3> domain;

    // ── Nodes ─────────────────────────────────────────────────────
    const auto total = static_cast<std::size_t>(grid.total_nodes());
    domain.preallocate_node_capacity(total);

    // Local coordinates: x ∈ [-w/2, w/2], y ∈ [-h/2, h/2], z ∈ [0, L]
    const double x0 = -spec.width  / 2.0;
    const double y0 = -spec.height / 2.0;

    for (int iz = 0; iz <= spec.nz; ++iz) {
        for (int iy = 0; iy <= spec.ny; ++iy) {
            for (int ix = 0; ix <= spec.nx; ++ix) {
                const double lx = x0 + static_cast<double>(ix) * dx;
                const double ly = y0 + static_cast<double>(iy) * dy;
                const double lz =      static_cast<double>(iz) * dz;

                // Rotate + translate to global frame
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

    // ── Hex8 elements ─────────────────────────────────────────────
    std::size_t tag = 0;
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
    const std::string& group = "Solid")
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

    // Origin = midpoint of A and B
    std::array<double, 3> origin{
        (A[0]+B[0])/2.0, (A[1]+B[1])/2.0, (A[2]+B[2])/2.0
    };

    return PrismaticSpec{
        .width  = width,
        .height = height,
        .length = L,
        .nx = nx, .ny = ny, .nz = nz,
        .origin = origin,
        .e_x = ex, .e_y = ey, .e_z = ez,
        .physical_group = group
    };
}

} // namespace fall_n

#endif // PRISMATIC_DOMAIN_BUILDER_HH
