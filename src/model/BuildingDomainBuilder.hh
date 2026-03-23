#ifndef BUILDING_DOMAIN_BUILDER_HH
#define BUILDING_DOMAIN_BUILDER_HH

// ═══════════════════════════════════════════════════════════════════════
//  BuildingDomainBuilder — Framed building domain construction
// ═══════════════════════════════════════════════════════════════════════
//
//  Provides a declarative, spec-driven API for constructing the
//  geometric domain of a framed building, including irregular
//  floor plans with rectangular cutouts (L-shape, setbacks):
//
//    auto [domain, grid] = fall_n::make_building_domain({
//        .x_axes       = {0.0, 6.0, 12.0, 18.0},
//        .y_axes       = {0.0, 5.0, 10.0},
//        .num_stories  = 5,
//        .story_height = 3.20,
//    });
//
//  L-shaped / setback example (cutout upper-right corner above story 0):
//
//    auto [domain, grid] = fall_n::make_building_domain({
//        .x_axes       = {0.0, 6.0, 12.0, 18.0},
//        .y_axes       = {0.0, 5.0, 10.0},
//        .num_stories  = 5,
//        .story_height = 3.20,
//        .cutout_x_start = 2, .cutout_x_end = 3,
//        .cutout_y_start = 1, .cutout_y_end = 2,
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
//  BuildingGrid — topology metadata for a (possibly L-shaped) grid
// ─────────────────────────────────────────────────────────────────────
struct BuildingGrid {
    std::vector<double> x_axes;
    std::vector<double> y_axes;
    std::vector<double> z_levels;
    int num_axes_x;
    int num_axes_y;
    int num_stories;
    double story_height;

    // Cutout specification (bay indices, 0-based).
    // When cutout_x_start == cutout_x_end, no cutout (full rectangle).
    int cutout_x_start = 0, cutout_x_end = 0;
    int cutout_y_start = 0, cutout_y_end = 0;
    int cutout_above_story = 0;

    // O(1) node ID from grid indices (dense numbering, independent of activity).
    PetscInt node_id(int ix, int iy, int level) const noexcept {
        return static_cast<PetscInt>(
            level * num_axes_x * num_axes_y + iy * num_axes_x + ix);
    }

    /// Is bay (ix, iy) active at floor level (1-based)?
    bool is_bay_active(int ix, int iy, int level) const noexcept {
        if (cutout_x_start >= cutout_x_end) return true;
        if (level <= cutout_above_story) return true;
        return !(ix >= cutout_x_start && ix < cutout_x_end
              && iy >= cutout_y_start && iy < cutout_y_end);
    }

    /// Is grid node (ix, iy) active at level (0-based)?
    /// Level 0 mirrors level 1 (columns must reach the ground).
    bool is_node_active(int ix, int iy, int level) const noexcept {
        if (cutout_x_start >= cutout_x_end) return true;
        const int eff = (level == 0) ? 1 : level;
        for (int bx : {ix - 1, ix})
            for (int by : {iy - 1, iy})
                if (bx >= 0 && bx < num_axes_x - 1
                    && by >= 0 && by < num_axes_y - 1
                    && is_bay_active(bx, by, eff))
                    return true;
        return false;
    }

    bool has_cutout() const noexcept {
        return cutout_x_start < cutout_x_end;
    }

    int num_plan_nodes()  const noexcept { return num_axes_x * num_axes_y; }
    int num_levels()      const noexcept { return num_stories + 1; }

    int total_nodes() const noexcept {
        if (!has_cutout())
            return num_levels() * num_plan_nodes();
        int c = 0;
        for (int level = 0; level < num_levels(); ++level)
            for (int iy = 0; iy < num_axes_y; ++iy)
                for (int ix = 0; ix < num_axes_x; ++ix)
                    c += is_node_active(ix, iy, level) ? 1 : 0;
        return c;
    }

    std::size_t num_columns() const noexcept {
        if (!has_cutout())
            return static_cast<std::size_t>(num_stories * num_axes_x * num_axes_y);
        std::size_t c = 0;
        for (int level = 0; level < num_stories; ++level)
            for (int iy = 0; iy < num_axes_y; ++iy)
                for (int ix = 0; ix < num_axes_x; ++ix)
                    if (is_node_active(ix, iy, level)
                        && is_node_active(ix, iy, level + 1))
                        ++c;
        return c;
    }

    std::size_t num_beams() const noexcept {
        if (!has_cutout())
            return static_cast<std::size_t>(
                num_stories * ((num_axes_x - 1) * num_axes_y
                             + (num_axes_y - 1) * num_axes_x));
        std::size_t c = 0;
        for (int level = 1; level <= num_stories; ++level) {
            for (int iy = 0; iy < num_axes_y; ++iy)
                for (int ix = 0; ix < num_axes_x - 1; ++ix)
                    if (is_node_active(ix, iy, level)
                        && is_node_active(ix + 1, iy, level))
                        ++c;
            for (int ix = 0; ix < num_axes_x; ++ix)
                for (int iy = 0; iy < num_axes_y - 1; ++iy)
                    if (is_node_active(ix, iy, level)
                        && is_node_active(ix, iy + 1, level))
                        ++c;
        }
        return c;
    }

    std::size_t num_slabs() const noexcept {
        if (!has_cutout())
            return static_cast<std::size_t>(
                num_stories * (num_axes_x - 1) * (num_axes_y - 1));
        std::size_t c = 0;
        for (int level = 1; level <= num_stories; ++level)
            for (int iy = 0; iy < num_axes_y - 1; ++iy)
                for (int ix = 0; ix < num_axes_x - 1; ++ix)
                    if (is_bay_active(ix, iy, level))
                        ++c;
        return c;
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

    // Optional cutout for L-shape / setback floor plans.
    // Bays in [cutout_x_start, cutout_x_end) × [cutout_y_start, cutout_y_end)
    // are removed at floor levels above cutout_above_story.
    // Default: no cutout (start == end).
    int cutout_x_start = 0, cutout_x_end = 0;
    int cutout_y_start = 0, cutout_y_end = 0;
    int cutout_above_story = 0;
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
        nx, ny, ns, spec.story_height,
        spec.cutout_x_start, spec.cutout_x_end,
        spec.cutout_y_start, spec.cutout_y_end,
        spec.cutout_above_story
    };

    Domain<3> domain;

    // ── Nodes ─────────────────────────────────────────────────────
    const auto total = static_cast<std::size_t>(grid.total_nodes());
    domain.preallocate_node_capacity(total);

    for (int level = 0; level < nl; ++level) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                if (!grid.is_node_active(ix, iy, level)) continue;
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
                if (!grid.is_node_active(ix, iy, level)
                    || !grid.is_node_active(ix, iy, level + 1))
                    continue;
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
                if (!grid.is_node_active(ix, iy, level)
                    || !grid.is_node_active(ix + 1, iy, level))
                    continue;
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
                if (!grid.is_node_active(ix, iy, level)
                    || !grid.is_node_active(ix, iy + 1, level))
                    continue;
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
                if (!grid.is_bay_active(ix, iy, level)) continue;
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
                if (!grid.is_node_active(ix, iy, level)
                    || !grid.is_node_active(ix, iy, level + 1))
                    continue;
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
                if (!grid.is_node_active(ix, iy, level)
                    || !grid.is_node_active(ix + 1, iy, level))
                    continue;
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
                if (!grid.is_node_active(ix, iy, level)
                    || !grid.is_node_active(ix, iy + 1, level))
                    continue;
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
                if (!grid.is_bay_active(ix, iy, level)) continue;
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
                if (!grid.is_node_active(ix, iy, level)) continue;
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
