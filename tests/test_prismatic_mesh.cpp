// ═══════════════════════════════════════════════════════════════════════
//  test_prismatic_mesh.cpp — Phase 3 tests for prismatic hex8 mesh
// ═══════════════════════════════════════════════════════════════════════

#include <print>
#include <cmath>
#include <algorithm>

#include <petsc.h>

#include "../src/model/PrismaticDomainBuilder.hh"

namespace {

int g_pass = 0, g_fail = 0;

void report(const char* label, bool ok) {
    if (ok) { ++g_pass; std::println("  PASS  {}", label); }
    else    { ++g_fail; std::println("  FAIL  {}", label); }
}

constexpr double tol = 1e-12;

// ─────────────────────────────────────────────────────────────────────
//  1. Basic prismatic mesh: 4×6×8 elements
// ─────────────────────────────────────────────────────────────────────
void test_basic_prism() {
    std::println("\n── Test: Basic prismatic mesh (4×6×8) ──────────────────────");

    auto [domain, grid] = fall_n::make_prismatic_domain({
        .width  = 0.40,
        .height = 0.60,
        .length = 3.20,
        .nx = 4, .ny = 6, .nz = 8,
    });

    // Nodes: (4+1)×(6+1)×(8+1) = 5×7×9 = 315
    report("total_nodes = 315",  grid.total_nodes() == 315);

    // Elements: 4×6×8 = 192
    report("total_elements = 192", grid.total_elements() == 192);
    report("domain matches",
           domain.num_elements() == static_cast<std::size_t>(grid.total_elements()));

    // Element sizes
    report("dx = 0.10", std::abs(grid.dx - 0.10) < tol);
    report("dy = 0.10", std::abs(grid.dy - 0.10) < tol);
    report("dz = 0.40", std::abs(grid.dz - 0.40) < tol);

    // Node ID at corners
    report("node_id(0,0,0) = 0",  grid.node_id(0, 0, 0) == 0);
    report("node_id(4,6,8) = 314", grid.node_id(4, 6, 8) == 314);
}

// ─────────────────────────────────────────────────────────────────────
//  2. Face node extraction
// ─────────────────────────────────────────────────────────────────────
void test_face_nodes() {
    std::println("\n── Test: Face node extraction ────────────────────────────");

    auto [domain, grid] = fall_n::make_prismatic_domain({
        .width  = 1.0,
        .height = 1.0,
        .length = 2.0,
        .nx = 2, .ny = 3, .nz = 4,
    });

    // MinX face: nodes with ix=0 → (ny+1)×(nz+1) = 4×5 = 20
    auto minx = grid.nodes_on_face(fall_n::PrismFace::MinX);
    report("MinX count = 20", minx.size() == 20);

    // MaxX face: nodes with ix=nx → same count
    auto maxx = grid.nodes_on_face(fall_n::PrismFace::MaxX);
    report("MaxX count = 20", maxx.size() == 20);

    // MinY face: (nx+1)×(nz+1) = 3×5 = 15
    auto miny = grid.nodes_on_face(fall_n::PrismFace::MinY);
    report("MinY count = 15", miny.size() == 15);

    // MaxZ face: (nx+1)×(ny+1) = 3×4 = 12
    auto maxz = grid.nodes_on_face(fall_n::PrismFace::MaxZ);
    report("MaxZ count = 12", maxz.size() == 12);

    // MinZ face should contain node(0,0,0)
    auto minz = grid.nodes_on_face(fall_n::PrismFace::MinZ);
    report("MinZ contains origin",
           std::find(minz.begin(), minz.end(), grid.node_id(0, 0, 0))
               != minz.end());

    // MaxZ should contain node(nx,ny,nz)
    report("MaxZ contains corner",
           std::find(maxz.begin(), maxz.end(), grid.node_id(2, 3, 4))
               != maxz.end());

    // MinX and MaxX should be disjoint
    bool overlap = false;
    for (auto id : minx)
        if (std::find(maxx.begin(), maxx.end(), id) != maxx.end())
            overlap = true;
    report("MinX/MaxX disjoint", !overlap);
}

// ─────────────────────────────────────────────────────────────────────
//  3. Coordinate verification (centred at origin, no rotation)
// ─────────────────────────────────────────────────────────────────────
void test_coordinates() {
    std::println("\n── Test: Coordinate system verification ─────────────────────");

    auto [domain, grid] = fall_n::make_prismatic_domain({
        .width  = 2.0,
        .height = 1.0,
        .length = 3.0,
        .nx = 2, .ny = 1, .nz = 3,
    });

    // Node (0,0,0): x = -1.0, y = -0.5, z = 0.0
    auto& n000 = domain.node(grid.node_id(0, 0, 0));
    report("n(0,0,0).x = -1.0", std::abs(n000.coord(0) - (-1.0)) < tol);
    report("n(0,0,0).y = -0.5", std::abs(n000.coord(1) - (-0.5)) < tol);
    report("n(0,0,0).z = 0.0",  std::abs(n000.coord(2) -   0.0)  < tol);

    // Node (2,1,3): x = 1.0, y = 0.5, z = 3.0
    auto& n213 = domain.node(grid.node_id(2, 1, 3));
    report("n(2,1,3).x = 1.0",  std::abs(n213.coord(0) - 1.0) < tol);
    report("n(2,1,3).y = 0.5",  std::abs(n213.coord(1) - 0.5) < tol);
    report("n(2,1,3).z = 3.0",  std::abs(n213.coord(2) - 3.0) < tol);
}

// ─────────────────────────────────────────────────────────────────────
//  4. Alignment to a beam (vertical beam along Z)
// ─────────────────────────────────────────────────────────────────────
void test_align_to_beam_vertical() {
    std::println("\n── Test: align_to_beam — vertical beam ──────────────────────");

    // Beam from (0,0,0) to (0,0,3.2), up = {0,1,0}
    auto spec = fall_n::align_to_beam(
        {0.0, 0.0, 0.0}, {0.0, 0.0, 3.2},
        {0.0, 1.0, 0.0},
        0.40, 0.60,
        2, 3, 4
    );

    report("length ≈ 3.2",  std::abs(spec.length - 3.2)  < 1e-10);
    report("width = 0.4",   std::abs(spec.width  - 0.4)  < tol);
    report("origin.z = 1.6", std::abs(spec.origin[2] - 1.6) < 1e-10);

    // e_z should be {0,0,1}
    report("e_z ≈ {0,0,1}",
           std::abs(spec.e_z[0]) < tol
        && std::abs(spec.e_z[1]) < tol
        && std::abs(spec.e_z[2] - 1.0) < tol);

    auto [domain, grid] = fall_n::make_prismatic_domain(spec);
    report("elements = 24", domain.num_elements() == 24);
}

// ─────────────────────────────────────────────────────────────────────
//  5. Alignment to a horizontal beam (along X)
// ─────────────────────────────────────────────────────────────────────
void test_align_to_beam_horizontal() {
    std::println("\n── Test: align_to_beam — horizontal beam (X-axis) ───────────");

    // Beam from (2,0,5) to (8,0,5), length = 6.0
    auto spec = fall_n::align_to_beam(
        {2.0, 0.0, 5.0}, {8.0, 0.0, 5.0},
        {0.0, 0.0, 1.0},   // up = Z
        0.30, 0.50,
        3, 5, 6
    );

    report("length ≈ 6.0",  std::abs(spec.length - 6.0) < 1e-10);

    // e_z should be {1,0,0} (beam axis along X)
    report("e_z ≈ {1,0,0}",
           std::abs(spec.e_z[0] - 1.0) < tol
        && std::abs(spec.e_z[1]) < tol
        && std::abs(spec.e_z[2]) < tol);

    // Origin at midpoint (5, 0, 5)
    report("origin ≈ (5,0,5)",
           std::abs(spec.origin[0] - 5.0) < tol
        && std::abs(spec.origin[1]) < tol
        && std::abs(spec.origin[2] - 5.0) < tol);

    auto [domain, grid] = fall_n::make_prismatic_domain(spec);
    report("elements = 90",  domain.num_elements() == 90);
    report("nodes = 168",    grid.total_nodes() == (3+1)*(5+1)*(6+1));
}

// ─────────────────────────────────────────────────────────────────────
//  6. Single-element mesh (1×1×1)
// ─────────────────────────────────────────────────────────────────────
void test_single_element() {
    std::println("\n── Test: Single hex8 element (1×1×1) ────────────────────────");

    auto [domain, grid] = fall_n::make_prismatic_domain({
        .width  = 2.0,
        .height = 2.0,
        .length = 2.0,
        .nx = 1, .ny = 1, .nz = 1,
    });

    report("nodes = 8",     grid.total_nodes() == 8);
    report("elements = 1",  domain.num_elements() == 1);

    // All 6 faces should have 4 nodes each
    for (auto face : {fall_n::PrismFace::MinX, fall_n::PrismFace::MaxX,
                      fall_n::PrismFace::MinY, fall_n::PrismFace::MaxY,
                      fall_n::PrismFace::MinZ, fall_n::PrismFace::MaxZ})
    {
        auto ids = grid.nodes_on_face(face);
        if (ids.size() != 4) {
            report("face has 4 nodes", false);
            return;
        }
    }
    report("all faces have 4 nodes", true);
}


} // namespace


int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    std::println("================================================================");
    std::println("  Prismatic Hex Mesh Tests (Phase 3 — ch64)");
    std::println("================================================================");

    test_basic_prism();
    test_face_nodes();
    test_coordinates();
    test_align_to_beam_vertical();
    test_align_to_beam_horizontal();
    test_single_element();

    std::println("\n=== {} PASSED, {} FAILED ===\n", g_pass, g_fail);

    PetscFinalize();
    return g_fail > 0 ? 1 : 0;
}
