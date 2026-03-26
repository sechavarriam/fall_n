// =============================================================================
//  test_reinforced_submodel.cpp
//
//  Integration test for SubModelSolver::solve_reinforced()
//  Validates:
//    - make_reinforced_prismatic_domain creates mixed hex + line Domain
//    - SubModelSolver::solve_reinforced() converges with MultiElementPolicy
//    - Reinforced model is stiffer than unreinforced
//    - Reaction-force-based E_eff is positive and reasonable
//
// =============================================================================

#include "header_files.hh"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>

using namespace fall_n;

// ── Helpers ──────────────────────────────────────────────────────────────────

static int passed = 0;
static int total  = 0;

static void check(bool cond, const char* msg) {
    ++total;
    if (cond) {
        ++passed;
        std::cout << "  [PASS] " << msg << "\n";
    } else {
        std::cout << "  [FAIL] " << msg << "\n";
    }
}

// =============================================================================
//  Test 1: Reinforced domain builder
// =============================================================================

static void test_reinforced_domain() {
    std::cout << "\n── Test 1: Reinforced domain builder ──\n";

    PrismaticSpec spec{
        .width  = 0.40,
        .height = 0.60,
        .length = 1.00,
        .nx = 2, .ny = 3, .nz = 4,
    };

    // 4 corner rebar bars at grid positions
    RebarSpec rebar;
    rebar.bars = {
        {0, 0, 0.0005, "Rebar"},   // corner (0,0)
        {2, 0, 0.0005, "Rebar"},   // corner (nx,0)
        {0, 3, 0.0005, "Rebar"},   // corner (0,ny)
        {2, 3, 0.0005, "Rebar"},   // corner (nx,ny)
    };

    auto result = make_reinforced_prismatic_domain(spec, rebar);

    auto& domain = result.domain;
    auto& grid   = result.grid;
    auto& range  = result.rebar_range;

    // Hex elements: 2*3*4 = 24
    const std::size_t num_hex = static_cast<std::size_t>(spec.nx * spec.ny * spec.nz);
    // Rebar elements: 4 bars * 4 z-segments = 16
    const std::size_t num_rebar = 4 * static_cast<std::size_t>(spec.nz);
    const std::size_t num_total = num_hex + num_rebar;

    check(domain.num_elements() == num_total,
          ("total elements = " + std::to_string(num_total)).c_str());
    check(range.first == num_hex, "rebar_range.first = num_hex");
    check(range.last == num_total, "rebar_range.last = num_total");

    // Check that rebar elements are 2-node line elements
    check(domain.element(range.first).num_nodes() == 2,
          "rebar element has 2 nodes");
    check(domain.element(0).num_nodes() == 8,
          "hex element has 8 nodes");

    // Grid metadata
    check(grid.nx == spec.nx && grid.ny == spec.ny && grid.nz == spec.nz,
          "grid dimensions match spec");

    // Nodes: (nx+1)*(ny+1)*(nz+1) = 3*4*5 = 60
    check(domain.num_nodes() == 60, "60 nodes");
}

// =============================================================================
//  Test 2: Solve reinforced vs unreinforced
// =============================================================================

static void test_reinforced_solve() {
    std::cout << "\n── Test 2: Reinforced solve (convergence + stiffness) ──\n";

    // Small prismatic domain
    PrismaticSpec spec{
        .width  = 0.30,
        .height = 0.30,
        .length = 1.00,
        .nx = 2, .ny = 2, .nz = 4,
    };

    // ── Unreinforced solve (via solve_reinforced with no rebar) ─────
    //
    // Use solve_reinforced for both cases so E_eff computation is
    // consistent (reaction-force-based).

    constexpr double imposed_disp = 0.0001;  // 0.1 mm → ε = 0.01%

    RebarSpec no_rebar;  // empty bar list
    auto unr_rd = make_reinforced_prismatic_domain(spec, no_rebar);

    MultiscaleSubModel unr_sub;
    unr_sub.domain = std::move(unr_rd.domain);
    unr_sub.grid   = std::move(unr_rd.grid);

    auto face_min = unr_sub.grid.nodes_on_face(PrismFace::MinZ);
    auto face_max = unr_sub.grid.nodes_on_face(PrismFace::MaxZ);

    for (auto nid : face_min)
        unr_sub.bc_min_z.emplace_back(static_cast<std::size_t>(nid),
            Eigen::Vector3d{0.0, 0.0, 0.0});
    for (auto nid : face_max)
        unr_sub.bc_max_z.emplace_back(static_cast<std::size_t>(nid),
            Eigen::Vector3d{0.0, 0.0, imposed_disp});

    SubModelSolver solver(30.0);  // f'c = 30 MPa
    RebarSteelConfig steel{200000.0, 420.0, 0.01};

    auto unr_result = solver.solve_reinforced(
        unr_sub, steel, unr_rd.rebar_range, {}, spec.nz);

    check(unr_result.converged, "unreinforced: converged");
    check(unr_result.E_eff > 0.0, "unreinforced: E_eff > 0");

    std::cout << "    unreinforced E_eff = " << std::fixed << std::setprecision(1)
              << unr_result.E_eff << " MPa\n";
    std::cout << "    unreinforced stress_zz = " << std::scientific << std::setprecision(4)
              << unr_result.avg_stress[2] << " MPa\n";
    std::cout << "    unreinforced strain_zz = " << unr_result.avg_strain[2] << "\n";

    // ── Reinforced solve ────────────────────────────────────────────
    RebarSpec rebar;
    rebar.bars = {
        {1, 1, 0.0002, "Rebar"},   // center bar
    };

    auto reinforced = make_reinforced_prismatic_domain(spec, rebar);

    MultiscaleSubModel rc_sub;
    rc_sub.domain = std::move(reinforced.domain);
    rc_sub.grid   = std::move(reinforced.grid);

    face_min = rc_sub.grid.nodes_on_face(PrismFace::MinZ);
    face_max = rc_sub.grid.nodes_on_face(PrismFace::MaxZ);

    for (auto nid : face_min)
        rc_sub.bc_min_z.emplace_back(static_cast<std::size_t>(nid),
            Eigen::Vector3d{0.0, 0.0, 0.0});
    for (auto nid : face_max)
        rc_sub.bc_max_z.emplace_back(static_cast<std::size_t>(nid),
            Eigen::Vector3d{0.0, 0.0, imposed_disp});

    std::vector<double> rebar_areas = {0.0002};

    auto rc_result = solver.solve_reinforced(
        rc_sub, steel, reinforced.rebar_range, rebar_areas, spec.nz);

    check(rc_result.converged, "reinforced: converged");
    check(rc_result.E_eff > 0.0, "reinforced: E_eff > 0");

    std::cout << "    reinforced E_eff = " << std::fixed << std::setprecision(1)
              << rc_result.E_eff << " MPa\n";
    std::cout << "    reinforced stress_zz = " << std::scientific << std::setprecision(4)
              << rc_result.avg_stress[2] << " MPa\n";
    std::cout << "    reinforced strain_zz = " << rc_result.avg_strain[2] << "\n";

    // Reinforced should be stiffer: concrete + rebar composite stiffness
    check(rc_result.E_eff > unr_result.E_eff,
          "reinforced E_eff > unreinforced (rebar adds stiffness)");

    // Sanity bounds: E_c ≈ 4700√30 ≈ 25700 MPa, with rule-of-mixtures + small
    // rebar area the increase should be modest
    check(unr_result.E_eff > 10000.0 && unr_result.E_eff < 50000.0,
          "unreinforced E_eff in [10k, 50k] MPa range");
    check(rc_result.E_eff < unr_result.E_eff * 2.0,
          "reinforced E_eff < 2x unreinforced (small rho)");
}


// =============================================================================
//  Main
// =============================================================================

int main(int argc, char* argv[]) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    std::cout << "╔══════════════════════════════════════════════════╗\n";
    std::cout << "║  Reinforced SubModel — Integration Tests        ║\n";
    std::cout << "╚══════════════════════════════════════════════════╝\n";

    test_reinforced_domain();
    test_reinforced_solve();

    std::cout << "\n══════════════════════════════════════════════════\n";
    std::cout << "  Results: " << passed << "/" << total << " passed\n";
    std::cout << "══════════════════════════════════════════════════\n";

    PetscFinalize();
    return (passed == total) ? 0 : 1;
}
