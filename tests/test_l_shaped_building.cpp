// ═══════════════════════════════════════════════════════════════════════
//  test_l_shaped_building.cpp — Phase 1 tests for L-shaped building
//  domain construction with rectangular cutouts
// ═══════════════════════════════════════════════════════════════════════

#include <print>
#include <cmath>

#include <petsc.h>

#include "../src/model/BuildingDomainBuilder.hh"

namespace {

int g_pass = 0, g_fail = 0;

void report(const char* label, bool ok) {
    if (ok) { ++g_pass; std::println("  PASS  {}", label); }
    else    { ++g_fail; std::println("  FAIL  {}", label); }
}

// ─────────────────────────────────────────────────────────────────────
//  1. Rectangular building (no cutout) — baseline regression
// ─────────────────────────────────────────────────────────────────────
void test_rectangular_baseline() {
    std::println("\n── Test: Rectangular baseline (4×3, 3 stories) ──────────────");

    auto [domain, grid] = fall_n::make_building_domain({
        .x_axes       = {0.0, 6.0, 12.0, 18.0},
        .y_axes       = {0.0, 5.0, 10.0},
        .num_stories  = 3,
        .story_height = 3.2,
    });

    // 4×3 grid, 3 stories → 4 levels × 12 nodes/level = 48 nodes
    report("total_nodes = 48",  grid.total_nodes() == 48);

    // Columns: 3 stories × 12 intersections = 36
    report("num_columns = 36",  grid.num_columns() == 36);

    // Beams per level: (3 X-beams × 3 Y-rows) + (2 Y-beams × 4 X-cols) = 9 + 8 = 17
    // × 3 stories = 51
    report("num_beams = 51",    grid.num_beams() == 51);

    // Slabs: 3 stories × 3 bays_x × 2 bays_y = 18
    report("num_slabs = 18",    grid.num_slabs() == 18);

    // All nodes should be active
    report("no cutout",         !grid.has_cutout());

    // Domain element count = columns + beams + slabs
    std::size_t expected_elements = 36 + 51 + 18;
    report("domain elements",
           domain.num_elements() == expected_elements);
}

// ─────────────────────────────────────────────────────────────────────
//  2. L-shaped building — cutout upper-right corner
// ─────────────────────────────────────────────────────────────────────
void test_l_shaped_cutout() {
    std::println("\n── Test: L-shaped (4×3, cutout bay [2,3)×[1,2), 3 stories) ─");

    // Grid: 4 x-axes × 3 y-axes = 3 bays_x × 2 bays_y per floor
    // Cutout: bay (2, 1) removed at all stories (above_story = 0)
    //
    //   Plan view (bays):
    //     y=1 ┌──┬──┐XX│    ← bay (2,1) removed
    //     y=0 ├──┼──┼──┤
    //         └──┴──┴──┘
    //        x=0  1  2
    //
    auto [domain, grid] = fall_n::make_building_domain({
        .x_axes       = {0.0, 6.0, 12.0, 18.0},
        .y_axes       = {0.0, 5.0, 10.0},
        .num_stories  = 3,
        .story_height = 3.2,
        .cutout_x_start = 2, .cutout_x_end = 3,
        .cutout_y_start = 1, .cutout_y_end = 2,
    });

    report("has_cutout",        grid.has_cutout());

    // Bay activity
    report("bay(0,0,1) active", grid.is_bay_active(0, 0, 1));
    report("bay(2,0,1) active", grid.is_bay_active(2, 0, 1));
    report("bay(0,1,1) active", grid.is_bay_active(0, 1, 1));
    report("bay(1,1,1) active", grid.is_bay_active(1, 1, 1));
    report("bay(2,1,1) cut",   !grid.is_bay_active(2, 1, 1));

    // Node (3, 2) = top-right corner of removed bay.
    // Adjacent bays: (2,1) = CUT, (2,0)? no — (3-1=2, 2-1=1) = CUT,
    // (3,1)? out of range, (3,2)? out of range, (2,2)? out of range
    // Wait: bx ∈ {ix-1, ix} = {2, 3}, by ∈ {iy-1, iy} = {1, 2}
    // (2,1): CUT; (2,2): out of range; (3,1): out of range; (3,2): out of range
    report("node(3,2,1) inactive", !grid.is_node_active(3, 2, 1));

    // Node (2, 2): bx ∈ {1, 2}, by ∈ {1, 2}
    // (1,1): ACTIVE; stop → node active
    report("node(2,2,1) active",   grid.is_node_active(2, 2, 1));

    // Node (2, 1): bx ∈ {1, 2}, by ∈ {0, 1}
    // (1,0): ACTIVE → node active
    report("node(2,1,1) active",   grid.is_node_active(2, 1, 1));

    // Node (3, 1): bx ∈ {2, 3}, by ∈ {0, 1}
    // (2,0): ACTIVE → node active
    report("node(3,1,1) active",   grid.is_node_active(3, 1, 1));

    // Ground-level nodes mirror level 1
    report("node(3,2,0) inactive", !grid.is_node_active(3, 2, 0));
    report("node(2,2,0) active",    grid.is_node_active(2, 2, 0));

    // With 1 bay removed, 5 active bays per floor:
    // Slabs: 3 stories × 5 bays = 15
    report("num_slabs = 15",    grid.num_slabs() == 15);

    // Active nodes per floor: 12 - 1 (node 3,2) = 11
    // Levels: 4 × 11 = 44
    report("total_nodes = 44",  grid.total_nodes() == 44);

    // Columns: 3 stories × 11 active node positions = 33
    report("num_columns = 33",  grid.num_columns() == 33);

    // Total elements in domain = columns + beams + slabs
    std::size_t expected = grid.num_columns() + grid.num_beams() + grid.num_slabs();
    std::println("    Elements: {} cols + {} beams + {} slabs = {}",
                 grid.num_columns(), grid.num_beams(), grid.num_slabs(), expected);
    report("domain elements match",
           domain.num_elements() == expected);

    // Verify the removed node ID doesn't appear in any element
    PetscInt removed_id = grid.node_id(3, 2, 1);
    bool found_removed = false;
    for (std::size_t i = 0; i < domain.num_elements(); ++i) {
        const auto& elem = domain.element(i);
        for (std::size_t j = 0; j < elem.num_nodes(); ++j)
            if (elem.node(j) == removed_id) found_removed = true;
    }
    report("removed node not in connectivity", !found_removed);
}

// ─────────────────────────────────────────────────────────────────────
//  3. Setback building — cutout only above story 2
// ─────────────────────────────────────────────────────────────────────
void test_setback() {
    std::println("\n── Test: Setback (3×2, cutout bay[1,2)×[0,1) above story 2) ");

    // 3 x-axes × 2 y-axes → 2 bays_x × 1 bay_y, 4 stories
    // Below story 2: full rectangle (2 bays), above: 1 bay removed
    //
    //   Stories 3-4:   ┌──┐    (1 bay)
    //   Stories 1-2:   ├──┼──┤  (2 bays)
    //
    auto [domain, grid] = fall_n::make_building_domain({
        .x_axes       = {0.0, 5.0, 10.0},
        .y_axes       = {0.0, 6.0},
        .num_stories  = 4,
        .story_height = 3.0,
        .cutout_x_start = 1, .cutout_x_end = 2,
        .cutout_y_start = 0, .cutout_y_end = 1,
        .cutout_above_story = 2,
    });

    // Levels 0-2: full rectangle (6 nodes each); levels 3-4: 4 nodes each
    // Total: 3×6 + 2×4 = 18 + 8 = 26
    report("total_nodes = 26",  grid.total_nodes() == 26);

    // Bay activity at level 2 (within threshold)
    report("bay(1,0,2) active", grid.is_bay_active(1, 0, 2));
    // Bay activity at level 3 (above threshold)
    report("bay(1,0,3) cut",   !grid.is_bay_active(1, 0, 3));
    report("bay(0,0,3) active", grid.is_bay_active(0, 0, 3));

    // Slabs: stories 1-2 have 2 bays each, stories 3-4 have 1 bay each
    // = 2×2 + 2×1 = 6
    report("num_slabs = 6",    grid.num_slabs() == 6);

    std::println("    Elements: {} cols + {} beams + {} slabs",
                 grid.num_columns(), grid.num_beams(), grid.num_slabs());

    std::size_t expected = grid.num_columns() + grid.num_beams() + grid.num_slabs();
    report("domain elements match",
           domain.num_elements() == expected);
}

// ─────────────────────────────────────────────────────────────────────
//  4. Larger L-shape — 2×2 cutout in a 5×4 grid
// ─────────────────────────────────────────────────────────────────────
void test_large_l_shape() {
    std::println("\n── Test: Large L-shape (5×4, cutout [2,4)×[2,3), 2 stories) ");

    // 5×4 grid → 4 bays_x × 3 bays_y = 12 bays per floor
    // Cutout: bays [2,4) × [2,3) = 2 bays removed per floor
    // Active bays per floor: 12 - 2 = 10
    //
    //   y=2  ┌──┬──┐XX│XX│    ← bays (2,2), (3,2) removed
    //   y=1  ├──┼──┼──┼──┤
    //   y=0  ├──┼──┼──┼──┤
    //        └──┴──┴──┴──┘
    //       x=0  1  2  3
    //
    auto [domain, grid] = fall_n::make_building_domain({
        .x_axes       = {0.0, 5.0, 10.0, 15.0, 20.0},
        .y_axes       = {0.0, 5.0, 10.0, 15.0},
        .num_stories  = 2,
        .story_height = 3.5,
        .cutout_x_start = 2, .cutout_x_end = 4,
        .cutout_y_start = 2, .cutout_y_end = 3,
    });

    // Slabs: 2 stories × 10 active bays = 20
    report("num_slabs = 20",    grid.num_slabs() == 20);

    // Node (4, 3): upper-right corner, bx ∈ {3,4}, by ∈ {2,3}
    // (3,2)=CUT, others out of range → inactive
    report("corner(4,3) inactive", !grid.is_node_active(4, 3, 1));

    // But node (2, 3): bx ∈ {1,2}, by ∈ {2,3}
    // (1,2)=ACTIVE → active
    report("corner(2,3) active",    grid.is_node_active(2, 3, 1));

    std::size_t expected = grid.num_columns() + grid.num_beams() + grid.num_slabs();
    std::println("    Elements: {} cols + {} beams + {} slabs = {}",
                 grid.num_columns(), grid.num_beams(), grid.num_slabs(), expected);
    report("domain elements match",
           domain.num_elements() == expected);
}

// ─────────────────────────────────────────────────────────────────────
//  5. Beam count verification for L-shape
// ─────────────────────────────────────────────────────────────────────
void test_beam_count_l_shape() {
    std::println("\n── Test: Beam count verification (3×3, cutout [1,2)×[1,2)) ─");

    // 3×3 grid → 2 bays_x × 2 bays_y = 4 bays, remove 1 → 3 active
    //
    //   y=1  ┌──┐XX│    (bay(1,1) cut)
    //   y=0  ├──┼──┤
    //        └──┴──┘
    //       x=0  1
    //
    auto [domain, grid] = fall_n::make_building_domain({
        .x_axes       = {0.0, 5.0, 10.0},
        .y_axes       = {0.0, 5.0, 10.0},
        .num_stories  = 1,
        .story_height = 3.0,
        .cutout_x_start = 1, .cutout_x_end = 2,
        .cutout_y_start = 1, .cutout_y_end = 2,
    });

    // Node (2,2) is inactive (bx ∈ {1,2}, by ∈ {1,2}: (1,1)=CUT, rest OOB)
    report("node(2,2) inactive", !grid.is_node_active(2, 2, 1));

    // Active nodes at level 1: all 9 except (2,2) = 8
    // But also check level 0 mirrors level 1
    report("total_nodes = 16",   grid.total_nodes() == 16);

    // Slabs = 1 × 3 = 3
    report("num_slabs = 3",      grid.num_slabs() == 3);

    // Beams: manually count X-beams at level 1
    //   iy=0: (0,0)→(1,0) ✓, (1,0)→(2,0) ✓  → 2
    //   iy=1: (0,1)→(1,1) ✓ (both active), (1,1)→(2,1) ✓  → 2
    //   iy=2: (0,2)→(1,2) ✓ (both active), (1,2)→(2,2)? (2,2) inactive → skip → 1
    // X-beams: 2 + 2 + 1 = 5
    //
    // Y-beams at level 1:
    //   ix=0: (0,0)→(0,1) ✓, (0,1)→(0,2) ✓  → 2
    //   ix=1: (1,0)→(1,1) ✓, (1,1)→(1,2) ✓  → 2
    //   ix=2: (2,0)→(2,1) ✓, (2,1)→(2,2)? (2,2) inactive → skip → 1
    // Y-beams: 2 + 2 + 1 = 5
    //
    // Total beams = 10
    report("num_beams = 10",     grid.num_beams() == 10);

    // Columns: only 8 active nodes → 8 columns (1 story)
    report("num_columns = 8",    grid.num_columns() == 8);

    std::size_t expected = 8 + 10 + 3;
    report("domain elements = 21",
           domain.num_elements() == expected);
}

void test_timoshenko_n4_lobatto_domain() {
    std::println("\n-- Test: Timoshenko N=4 + Lobatto frame domain --");

    auto [domain, grid] = fall_n::make_building_domain_timoshenko_n4_lobatto({
        .x_axes       = {0.0, 6.0},
        .y_axes       = {0.0, 5.0},
        .num_stories  = 1,
        .story_height = 3.2,
        .include_slabs = false,
    });

    const std::size_t frame_elements =
        grid.num_columns() + grid.num_beams();
    report("N4 domain element count",
           domain.num_elements() == frame_elements);
    report("N4 domain owns two interior nodes per frame element",
           domain.num_nodes() ==
               static_cast<std::size_t>(grid.total_nodes()) +
               2U * frame_elements);

    bool all_frame_elements_have_four_nodes = true;
    bool all_frame_elements_have_three_ips = true;
    for (const auto& elem : domain.elements()) {
        all_frame_elements_have_four_nodes =
            all_frame_elements_have_four_nodes && elem.num_nodes() == 4;
        all_frame_elements_have_three_ips =
            all_frame_elements_have_three_ips &&
            elem.num_integration_points() == 3;
    }
    report("N4 frame geometries have four nodes",
           all_frame_elements_have_four_nodes);
    report("N4 frame geometries use three Lobatto stations",
           all_frame_elements_have_three_ips);
}


} // namespace


int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    std::println("================================================================");
    std::println("  L-Shaped Building Tests (Phase 1 — ch64)");
    std::println("================================================================");

    test_rectangular_baseline();
    test_l_shaped_cutout();
    test_setback();
    test_large_l_shape();
    test_beam_count_l_shape();
    test_timoshenko_n4_lobatto_domain();

    std::println("\n=== {} PASSED, {} FAILED ===\n", g_pass, g_fail);

    PetscFinalize();
    return g_fail > 0 ? 1 : 0;
}
