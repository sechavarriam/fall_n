#ifndef BUILDING_DOMAIN_BUILDER_HH
#define BUILDING_DOMAIN_BUILDER_HH

// ═══════════════════════════════════════════════════════════════════════
//  BuildingDomainBuilder — Regular framed building domain construction
// ═══════════════════════════════════════════════════════════════════════
//
//  Provides a declarative, spec-driven API for constructing the
//  geometric domain of a regular framed building:
//
//    auto [domain, grid] = fall_n::make_building_domain({
//        .x_axes       = {0.0, 6.0, 12.0, 18.0},
//        .y_axes       = {0.0, 5.0, 10.0},
//        .num_stories  = 5,
//        .story_height = 3.20,
//    });
//
//  The returned BuildingGrid holds immutable topology metadata and
//  provides O(1) node_id() lookups for subsequent load application.
//
//  Design notes (scripting-friendliness):
//  - All parameters are POD-like structs with designated initialisers.
//  - No template parameters in the public API.
//  - Physical group names are std::string (not enum), matching the
//    Domain's string-based physical_group() tags.
//  - return types are concrete (no type-erasure, no CRTP).
//
// ═══════════════════════════════════════════════════════════════════════

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <petsc.h>

#include "../domain/Domain.hh"
#include "../elements/element_geometry/LagrangeElement.hh"
#include "../numerics/numerical_integration/GaussLegendreCellIntegrator.hh"

namespace fall_n {

// ─────────────────────────────────────────────────────────────────────
//  BuildingGrid — immutable topology metadata for a regular grid
// ─────────────────────────────────────────────────────────────────────
struct BuildingGrid {
    std::vector<double> x_axes;
    std::vector<double> y_axes;
    std::vector<double> z_levels;
    int num_axes_x;
    int num_axes_y;
    int num_stories;
    double story_height;

    // O(1) node ID from grid indices.
    PetscInt node_id(int ix, int iy, int level) const noexcept {
        return static_cast<PetscInt>(
            level * num_axes_x * num_axes_y + iy * num_axes_x + ix);
    }

    int num_plan_nodes()  const noexcept { return num_axes_x * num_axes_y; }
    int num_levels()      const noexcept { return num_stories + 1; }
    int total_nodes()     const noexcept { return num_levels() * num_plan_nodes(); }

    std::size_t num_columns() const noexcept {
        return static_cast<std::size_t>(num_stories * num_axes_x * num_axes_y);
    }
    std::size_t num_beams() const noexcept {
        return static_cast<std::size_t>(
            num_stories * ((num_axes_x - 1) * num_axes_y
                         + (num_axes_y - 1) * num_axes_x));
    }
    std::size_t num_slabs() const noexcept {
        return static_cast<std::size_t>(
            num_stories * (num_axes_x - 1) * (num_axes_y - 1));
    }
};

// ─────────────────────────────────────────────────────────────────────
//  FramedBuildingSpec — user-facing specification
// ─────────────────────────────────────────────────────────────────────
struct FramedBuildingSpec {
    std::vector<double> x_axes;
    std::vector<double> y_axes;
    int    num_stories;
    double story_height;

    // Physical group names (defaults match common convention).
    std::string column_group = "Columns";
    std::string beam_group   = "Beams";
    std::string slab_group   = "Slabs";
};

// ─────────────────────────────────────────────────────────────────────
//  make_building_domain — construct an assembled Domain<3> + grid
// ─────────────────────────────────────────────────────────────────────
inline std::pair<Domain<3>, BuildingGrid>
make_building_domain(const FramedBuildingSpec& spec)
{
    const int nx = static_cast<int>(spec.x_axes.size());
    const int ny = static_cast<int>(spec.y_axes.size());
    const int ns = spec.num_stories;
    const int nl = ns + 1;

    // Build z-levels
    std::vector<double> z_levels(static_cast<std::size_t>(nl));
    for (int k = 0; k < nl; ++k)
        z_levels[static_cast<std::size_t>(k)] = spec.story_height * static_cast<double>(k);

    BuildingGrid grid{
        spec.x_axes, spec.y_axes, z_levels,
        nx, ny, ns, spec.story_height
    };

    Domain<3> domain;

    // ── Nodes ─────────────────────────────────────────────────────
    const auto total = static_cast<std::size_t>(grid.total_nodes());
    domain.preallocate_node_capacity(total);

    for (int level = 0; level < nl; ++level) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                domain.add_node(
                    grid.node_id(ix, iy, level),
                    spec.x_axes[static_cast<std::size_t>(ix)],
                    spec.y_axes[static_cast<std::size_t>(iy)],
                    z_levels[static_cast<std::size_t>(level)]);
            }
        }
    }

    std::size_t tag = 0;

    // ── Columns (vertical 2-node line elements) ──────────────────
    for (int level = 0; level < ns; ++level) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                PetscInt conn[2] = {
                    grid.node_id(ix, iy, level),
                    grid.node_id(ix, iy, level + 1)
                };
                auto& geom = domain.make_element<LagrangeElement3D<2>>(
                    GaussLegendreCellIntegrator<2>{}, tag++, conn);
                geom.set_physical_group(spec.column_group);
            }
        }
    }

    // ── Beams (horizontal 2-node line elements, X then Y) ────────
    for (int level = 1; level <= ns; ++level) {
        // X-direction beams
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx - 1; ++ix) {
                PetscInt conn[2] = {
                    grid.node_id(ix, iy, level),
                    grid.node_id(ix + 1, iy, level)
                };
                auto& geom = domain.make_element<LagrangeElement3D<2>>(
                    GaussLegendreCellIntegrator<2>{}, tag++, conn);
                geom.set_physical_group(spec.beam_group);
            }
        }
        // Y-direction beams
        for (int ix = 0; ix < nx; ++ix) {
            for (int iy = 0; iy < ny - 1; ++iy) {
                PetscInt conn[2] = {
                    grid.node_id(ix, iy, level),
                    grid.node_id(ix, iy + 1, level)
                };
                auto& geom = domain.make_element<LagrangeElement3D<2>>(
                    GaussLegendreCellIntegrator<2>{}, tag++, conn);
                geom.set_physical_group(spec.beam_group);
            }
        }
    }

    // ── Slabs (4-node quad elements) ─────────────────────────────
    for (int level = 1; level <= ns; ++level) {
        for (int iy = 0; iy < ny - 1; ++iy) {
            for (int ix = 0; ix < nx - 1; ++ix) {
                PetscInt conn[4] = {
                    grid.node_id(ix,     iy,     level),
                    grid.node_id(ix + 1, iy,     level),
                    grid.node_id(ix,     iy + 1, level),
                    grid.node_id(ix + 1, iy + 1, level)
                };
                auto& geom = domain.make_element<LagrangeElement3D<2, 2>>(
                    GaussLegendreCellIntegrator<2, 2>{}, tag++, conn);
                geom.set_physical_group(spec.slab_group);
            }
        }
    }

    domain.assemble_sieve();

    return {std::move(domain), std::move(grid)};
}


// ─────────────────────────────────────────────────────────────────────
//  SelfWeightSpec — parameters for gravity self-weight calculation
// ─────────────────────────────────────────────────────────────────────
struct SelfWeightSpec {
    double density;        // MN·s²/m⁴  (e.g. 2.4e-3 for RC)
    double gravity;        // m/s²       (9.81)
    double column_area;    // m²
    double beam_area;      // m²
    double slab_thickness; // m
};

// ─────────────────────────────────────────────────────────────────────
//  apply_building_self_weight — nodal lumped gravity from tributary
// ─────────────────────────────────────────────────────────────────────
//
//  Applies gravity self-weight as equivalent nodal forces (–Z):
//    Columns : ½·ρ·g·A_col·H   at each end node
//    Beams   : ½·ρ·g·A_beam·L  at each end node
//    Slabs   : ¼·ρ·g·t·Lx·Ly   at each corner node
//
template <typename ModelT>
void apply_building_self_weight(
    ModelT& model,
    const BuildingGrid& grid,
    const SelfWeightSpec& sw)
{
    const int nx = grid.num_axes_x;
    const int ny = grid.num_axes_y;
    const int ns = grid.num_stories;

    // Column self-weight
    for (int level = 0; level < ns; ++level) {
        const double w_half = 0.5 * sw.density * sw.gravity
                            * sw.column_area * grid.story_height;
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                model.apply_node_force(
                    grid.node_id(ix, iy, level),
                    0.0, 0.0, -w_half, 0.0, 0.0, 0.0);
                model.apply_node_force(
                    grid.node_id(ix, iy, level + 1),
                    0.0, 0.0, -w_half, 0.0, 0.0, 0.0);
            }
        }
    }

    // Beam self-weight (X-direction)
    for (int level = 1; level <= ns; ++level) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx - 1; ++ix) {
                const double Lx = grid.x_axes[static_cast<std::size_t>(ix + 1)]
                                - grid.x_axes[static_cast<std::size_t>(ix)];
                const double w_half = 0.5 * sw.density * sw.gravity * sw.beam_area * Lx;
                model.apply_node_force(
                    grid.node_id(ix, iy, level),
                    0.0, 0.0, -w_half, 0.0, 0.0, 0.0);
                model.apply_node_force(
                    grid.node_id(ix + 1, iy, level),
                    0.0, 0.0, -w_half, 0.0, 0.0, 0.0);
            }
        }

        // Beam self-weight (Y-direction)
        for (int ix = 0; ix < nx; ++ix) {
            for (int iy = 0; iy < ny - 1; ++iy) {
                const double Ly = grid.y_axes[static_cast<std::size_t>(iy + 1)]
                                - grid.y_axes[static_cast<std::size_t>(iy)];
                const double w_half = 0.5 * sw.density * sw.gravity * sw.beam_area * Ly;
                model.apply_node_force(
                    grid.node_id(ix, iy, level),
                    0.0, 0.0, -w_half, 0.0, 0.0, 0.0);
                model.apply_node_force(
                    grid.node_id(ix, iy + 1, level),
                    0.0, 0.0, -w_half, 0.0, 0.0, 0.0);
            }
        }
    }

    // Slab tributary self-weight (¼ of panel weight at each corner)
    for (int level = 1; level <= ns; ++level) {
        for (int iy = 0; iy < ny - 1; ++iy) {
            for (int ix = 0; ix < nx - 1; ++ix) {
                const double Lx = grid.x_axes[static_cast<std::size_t>(ix + 1)]
                                - grid.x_axes[static_cast<std::size_t>(ix)];
                const double Ly = grid.y_axes[static_cast<std::size_t>(iy + 1)]
                                - grid.y_axes[static_cast<std::size_t>(iy)];
                const double w_quarter = 0.25 * sw.density * sw.gravity
                                       * sw.slab_thickness * Lx * Ly;
                model.apply_node_force(
                    grid.node_id(ix,     iy,     level),
                    0.0, 0.0, -w_quarter, 0.0, 0.0, 0.0);
                model.apply_node_force(
                    grid.node_id(ix + 1, iy,     level),
                    0.0, 0.0, -w_quarter, 0.0, 0.0, 0.0);
                model.apply_node_force(
                    grid.node_id(ix,     iy + 1, level),
                    0.0, 0.0, -w_quarter, 0.0, 0.0, 0.0);
                model.apply_node_force(
                    grid.node_id(ix + 1, iy + 1, level),
                    0.0, 0.0, -w_quarter, 0.0, 0.0, 0.0);
            }
        }
    }
}


// ─────────────────────────────────────────────────────────────────────
//  apply_triangular_lateral_load — height-proportional lateral forces
// ─────────────────────────────────────────────────────────────────────
//
//  Applies lateral force at each floor node, weighted linearly by
//  story level:  F_node = amplitude * (level / num_stories)
//  in the specified direction (0=X, 1=Y).
//
template <typename ModelT>
void apply_triangular_lateral_load(
    ModelT& model,
    const BuildingGrid& grid,
    int direction,         // 0 = X, 1 = Y
    double amplitude)
{
    const int nx = grid.num_axes_x;
    const int ny = grid.num_axes_y;
    const int ns = grid.num_stories;

    for (int level = 1; level <= ns; ++level) {
        const double w = amplitude
                       * static_cast<double>(level) / static_cast<double>(ns);
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                const double fx = (direction == 0) ? w : 0.0;
                const double fy = (direction == 1) ? w : 0.0;
                model.apply_node_force(
                    grid.node_id(ix, iy, level),
                    fx, fy, 0.0, 0.0, 0.0, 0.0);
            }
        }
    }
}

} // namespace fall_n

#endif // BUILDING_DOMAIN_BUILDER_HH
