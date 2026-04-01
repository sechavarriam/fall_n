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

#include <array>
#include <cstddef>
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
    double dx, dy, dz;       // element sizes
    double width, height, length;
    HexOrder hex_order;

    int nodes_x() const noexcept { return step * nx + 1; }
    int nodes_y() const noexcept { return step * ny + 1; }
    int nodes_z() const noexcept { return step * nz + 1; }
    int total_nodes()    const noexcept { return nodes_x() * nodes_y() * nodes_z(); }
    int total_elements() const noexcept { return nx * ny * nz; }

    /// O(1) node ID from structured grid indices.
    ///  For linear:    ix ∈ [0, nx],  iy ∈ [0, ny],  iz ∈ [0, nz]
    ///  For quadratic: ix ∈ [0, 2·nx], iy ∈ [0, 2·ny], iz ∈ [0, 2·nz]
    PetscInt node_id(int ix, int iy, int iz) const noexcept {
        return static_cast<PetscInt>(
            iz * nodes_x() * nodes_y() + iy * nodes_x() + ix);
    }

    /// Collect all node IDs on a given prism face.
    ///  For linear meshes: returns corner nodes only.
    ///  For quadratic meshes: returns ALL grid nodes on the face
    ///  (corners + mid-edge + mid-face).  For Hex20 one should
    ///  filter out the mid-face nodes that don't exist, but in
    ///  practice all face nodes are needed for BC application.
    std::vector<PetscInt> nodes_on_face(PrismFace face) const {
        std::vector<PetscInt> ids;
        const int gnx = nodes_x() - 1;  // max index in x
        const int gny = nodes_y() - 1;
        const int gnz = nodes_z() - 1;
        switch (face) {
        case PrismFace::MinX:
            ids.reserve(static_cast<std::size_t>(nodes_y() * nodes_z()));
            for (int iz = 0; iz <= gnz; ++iz)
                for (int iy = 0; iy <= gny; ++iy)
                    ids.push_back(node_id(0, iy, iz));
            break;
        case PrismFace::MaxX:
            ids.reserve(static_cast<std::size_t>(nodes_y() * nodes_z()));
            for (int iz = 0; iz <= gnz; ++iz)
                for (int iy = 0; iy <= gny; ++iy)
                    ids.push_back(node_id(gnx, iy, iz));
            break;
        case PrismFace::MinY:
            ids.reserve(static_cast<std::size_t>(nodes_x() * nodes_z()));
            for (int iz = 0; iz <= gnz; ++iz)
                for (int ix = 0; ix <= gnx; ++ix)
                    ids.push_back(node_id(ix, 0, iz));
            break;
        case PrismFace::MaxY:
            ids.reserve(static_cast<std::size_t>(nodes_x() * nodes_z()));
            for (int iz = 0; iz <= gnz; ++iz)
                for (int ix = 0; ix <= gnx; ++ix)
                    ids.push_back(node_id(ix, gny, iz));
            break;
        case PrismFace::MinZ:
            ids.reserve(static_cast<std::size_t>(nodes_x() * nodes_y()));
            for (int iy = 0; iy <= gny; ++iy)
                for (int ix = 0; ix <= gnx; ++ix)
                    ids.push_back(node_id(ix, iy, 0));
            break;
        case PrismFace::MaxZ:
            ids.reserve(static_cast<std::size_t>(nodes_x() * nodes_y()));
            for (int iy = 0; iy <= gny; ++iy)
                for (int ix = 0; ix <= gnx; ++ix)
                    ids.push_back(node_id(ix, iy, gnz));
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
//  make_prismatic_domain — construct a structured hex mesh (Hex8/20/27)
// ─────────────────────────────────────────────────────────────────────
inline std::pair<Domain<3>, PrismaticGrid>
make_prismatic_domain(const PrismaticSpec& spec)
{
    const int step = (spec.hex_order == HexOrder::Linear) ? 1 : 2;
    const double dx = spec.width  / static_cast<double>(spec.nx);
    const double dy = spec.height / static_cast<double>(spec.ny);
    const double dz = spec.length / static_cast<double>(spec.nz);

    PrismaticGrid grid{
        spec.nx, spec.ny, spec.nz,
        step,
        dx, dy, dz,
        spec.width, spec.height, spec.length,
        spec.hex_order
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
    const double x0 = -spec.width  / 2.0;
    const double y0 = -spec.height / 2.0;
    const double sub_dx = dx / static_cast<double>(step);
    const double sub_dy = dy / static_cast<double>(step);
    const double sub_dz = dz / static_cast<double>(step);

    for (int iz = 0; iz < grid.nodes_z(); ++iz) {
        for (int iy = 0; iy < grid.nodes_y(); ++iy) {
            for (int ix = 0; ix < grid.nodes_x(); ++ix) {
                const double lx = x0 + static_cast<double>(ix) * sub_dx;
                const double ly = y0 + static_cast<double>(iy) * sub_dy;
                const double lz =      static_cast<double>(iz) * sub_dz;

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

    // Origin = midpoint of A and B
    std::array<double, 3> origin{
        (A[0]+B[0])/2.0, (A[1]+B[1])/2.0, (A[2]+B[2])/2.0
    };

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

    /// Per-bar embedding info: host hex element + parent coordinates.
    /// Size = num_bars × (nz + 1) rebar nodes.
    std::vector<RebarNodeEmbedding> embeddings;

    /// Per-bar diameters: one per bar (same order as RebarSpec::bars).
    std::vector<double> bar_diameters;
};

inline ReinforcedDomainResult
make_reinforced_prismatic_domain(const PrismaticSpec& spec,
                                 const RebarSpec& rebar)
{
    const int step = (spec.hex_order == HexOrder::Linear) ? 1 : 2;
    const double dx = spec.width  / static_cast<double>(spec.nx);
    const double dy = spec.height / static_cast<double>(spec.ny);
    const double dz = spec.length / static_cast<double>(spec.nz);

    PrismaticGrid grid{
        spec.nx, spec.ny, spec.nz,
        step,
        dx, dy, dz,
        spec.width, spec.height, spec.length,
        spec.hex_order
    };

    Domain<3> domain;

    // ── Hex nodes (same scheme as make_prismatic_domain) ──────────
    const auto total_hex_nodes = static_cast<std::size_t>(grid.total_nodes());
    const auto num_bars = rebar.bars.size();
    const auto rebar_nodes_per_bar =
        static_cast<std::size_t>(step * spec.nz + 1);
    const auto total_rebar_nodes = num_bars * rebar_nodes_per_bar;

    domain.preallocate_node_capacity(total_hex_nodes + total_rebar_nodes);

    const double x0 = -spec.width  / 2.0;
    const double y0 = -spec.height / 2.0;
    const double sub_dx = dx / static_cast<double>(step);
    const double sub_dy = dy / static_cast<double>(step);
    const double sub_dz = dz / static_cast<double>(step);

    for (int iz = 0; iz < grid.nodes_z(); ++iz) {
        for (int iy = 0; iy < grid.nodes_y(); ++iy) {
            for (int ix = 0; ix < grid.nodes_x(); ++ix) {
                const double lx = x0 + static_cast<double>(ix) * sub_dx;
                const double ly = y0 + static_cast<double>(iy) * sub_dy;
                const double lz =      static_cast<double>(iz) * sub_dz;

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
        const double fx = (bar.ly - x0) / dx;  // fractional element index
        const double fy = (bar.lz - y0) / dy;

        int ix_host = static_cast<int>(std::floor(fx));
        int iy_host = static_cast<int>(std::floor(fy));
        ix_host = std::clamp(ix_host, 0, spec.nx - 1);
        iy_host = std::clamp(iy_host, 0, spec.ny - 1);

        // Parent coordinates (ξ, η) ∈ [-1, 1] within the host element
        const double xi  = 2.0 * (fx - static_cast<double>(ix_host)) - 1.0;
        const double eta = 2.0 * (fy - static_cast<double>(iy_host)) - 1.0;

        // Create rebar nodes along the z-axis at each z sub-level
        for (std::size_t iz = 0; iz < rebar_nodes_per_bar; ++iz) {
            const PetscInt nid = rebar_base_id++;
            const double lz_pos = static_cast<double>(iz) * sub_dz;

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
            rebar_nodes_base
            + static_cast<PetscInt>(b * rebar_nodes_per_bar);

        for (int iz = 0; iz < spec.nz; ++iz) {
            PetscInt conn[2] = {
                bar_start + static_cast<PetscInt>(step * iz),
                bar_start + static_cast<PetscInt>(step * (iz + 1)),
            };
            auto& geom = domain.make_element<LagrangeElement3D<2>>(
                GaussLegendreCellIntegrator<2>{}, tag++, conn);
            geom.set_physical_group(rebar.bars[b].group);
        }
    }

    const std::size_t rebar_last = domain.num_elements();

    domain.assemble_sieve();

    return {std::move(domain), std::move(grid),
            RebarElementRange{rebar_first, rebar_last},
            std::move(embeddings),
            std::move(bar_diameters)};
}

} // namespace fall_n

#endif // PRISMATIC_DOMAIN_BUILDER_HH
