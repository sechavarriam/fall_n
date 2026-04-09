// =============================================================================
//  test_reinforced_submodel.cpp
//
//  Integration test for SubModelSolver::solve_reinforced()
//  Validates:
//    - make_reinforced_prismatic_domain creates mixed hex + line Domain
//    - SubModelSolver::solve_reinforced() converges with MultiElementPolicy
//    - Reinforced model is stiffer than unreinforced
//    - Reaction-force-based E_eff is positive and reasonable
//    - VTK Gauss-point export contains stress, strain, and crack fields
//
// =============================================================================

#include "header_files.hh"

#include <cmath>
#include <filesystem>
#include <cstdlib>
#include <iostream>
#include <iomanip>

#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridReader.h>

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

    // 4 corner rebar bars at section corner positions
    RebarSpec rebar;
    rebar.bars = {
        {-0.20, -0.30, 0.0005, 0.025, "Rebar"},   // corner (-w/2, -h/2)
        { 0.20, -0.30, 0.0005, 0.025, "Rebar"},   // corner (+w/2, -h/2)
        {-0.20,  0.30, 0.0005, 0.025, "Rebar"},   // corner (-w/2, +h/2)
        { 0.20,  0.30, 0.0005, 0.025, "Rebar"},   // corner (+w/2, +h/2)
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

    // Nodes: hex (nx+1)(ny+1)(nz+1) = 60 + 4 bars × (nz+1) rebar = 80
    const std::size_t num_hex_nodes = 60;
    const std::size_t num_rebar_nodes = 4 * static_cast<std::size_t>(spec.nz + 1);
    check(domain.num_nodes() == num_hex_nodes + num_rebar_nodes,
          (std::to_string(num_hex_nodes + num_rebar_nodes) + " nodes").c_str());
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
        {0.0, 0.0, 0.0002, 0.016, "Rebar"},   // center bar
    };

    auto reinforced = make_reinforced_prismatic_domain(spec, rebar);

    MultiscaleSubModel rc_sub;
    rc_sub.domain = std::move(reinforced.domain);
    rc_sub.grid   = std::move(reinforced.grid);
    rc_sub.rebar_range      = reinforced.rebar_range;
    rc_sub.rebar_embeddings = std::move(reinforced.embeddings);
    rc_sub.rebar_diameters  = std::move(reinforced.bar_diameters);

    face_min = rc_sub.grid.nodes_on_face(PrismFace::MinZ);
    face_max = rc_sub.grid.nodes_on_face(PrismFace::MaxZ);

    // Append rebar face-end nodes to BC lists
    {
        const std::size_t rpb = static_cast<std::size_t>(
            rc_sub.grid.step * rc_sub.grid.nz + 1);
        for (std::size_t b = 0; b < rc_sub.rebar_diameters.size(); ++b) {
            face_min.push_back(rc_sub.rebar_embeddings[b * rpb].rebar_node_id);
            face_max.push_back(rc_sub.rebar_embeddings[b * rpb + rpb - 1].rebar_node_id);
        }
    }

    for (auto nid : face_min)
        rc_sub.bc_min_z.emplace_back(static_cast<std::size_t>(nid),
            Eigen::Vector3d{0.0, 0.0, 0.0});
    for (auto nid : face_max)
        rc_sub.bc_max_z.emplace_back(static_cast<std::size_t>(nid),
            Eigen::Vector3d{0.0, 0.0, imposed_disp});

    std::vector<double> rebar_areas = {0.0002};
    rc_sub.rebar_areas = rebar_areas;

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

    SubModelSolver explicit_alpha_solver(30.0);
    EmbeddedLinePenaltyCouplingConfig explicit_alpha_cfg;
    explicit_alpha_cfg.mode = EmbeddedLinePenaltyMode::ExplicitAlpha;
    explicit_alpha_cfg.alpha_penalty = 1.0e4 * 4700.0 * std::sqrt(30.0);
    explicit_alpha_solver.set_embedded_line_coupling_config(explicit_alpha_cfg);

    auto rc_result_explicit_alpha = explicit_alpha_solver.solve_reinforced(
        rc_sub, steel, reinforced.rebar_range, rebar_areas, spec.nz);

    check(rc_result_explicit_alpha.converged,
          "reinforced: converged with explicit penalty alpha");
    check(std::abs(rc_result_explicit_alpha.E_eff - rc_result.E_eff)
              / std::max(1.0, std::abs(rc_result.E_eff))
              < 5.0e-2,
          "explicit penalty alpha preserves reinforced effective stiffness");
}


// =============================================================================
//  Test 3: VTK Gauss-point export with crack fields
// =============================================================================
//
//  Solve a reinforced sub-model under tension large enough to initiate
//  cracking in the Ko-Bathe concrete model.  Write VTK Gauss cloud output,
//  read it back, and verify that stress/strain tensors and crack-related
//  point-data arrays are present.

static void test_vtk_crack_export() {
    std::cout << "\n── Test 3: VTK Gauss-point crack field export ──\n";

    PrismaticSpec spec{
        .width  = 0.30,
        .height = 0.30,
        .length = 1.00,
        .nx = 2, .ny = 2, .nz = 4,
    };

    // One center rebar
    RebarSpec rebar;
    rebar.bars = { {0.0, 0.0, 0.0002, 0.016, "Rebar"} };

    auto rd = make_reinforced_prismatic_domain(spec, rebar);

    MultiscaleSubModel sub;
    sub.domain = std::move(rd.domain);
    sub.grid   = std::move(rd.grid);
    sub.rebar_range      = rd.rebar_range;
    sub.rebar_embeddings = std::move(rd.embeddings);
    sub.rebar_diameters  = std::move(rd.bar_diameters);

    // Tension: fix min-z face, pull max-z face with ε ≈ 5×10⁻⁴
    // This should exceed the cracking threshold (ft/Ee ≈ 5.6×10⁻⁵)
    constexpr double imposed_disp = 0.0005;  // 0.5 mm over 1 m → ε = 5×10⁻⁴

    auto face_min = sub.grid.nodes_on_face(PrismFace::MinZ);
    auto face_max = sub.grid.nodes_on_face(PrismFace::MaxZ);

    // Append rebar face-end nodes to BC lists
    {
        const std::size_t rpb = static_cast<std::size_t>(
            sub.grid.step * sub.grid.nz + 1);
        for (std::size_t b = 0; b < sub.rebar_diameters.size(); ++b) {
            face_min.push_back(sub.rebar_embeddings[b * rpb].rebar_node_id);
            face_max.push_back(sub.rebar_embeddings[b * rpb + rpb - 1].rebar_node_id);
        }
    }

    for (auto nid : face_min)
        sub.bc_min_z.emplace_back(static_cast<std::size_t>(nid),
            Eigen::Vector3d(0.0, 0.0, 0.0));
    for (auto nid : face_max)
        sub.bc_max_z.emplace_back(static_cast<std::size_t>(nid),
            Eigen::Vector3d(0.0, 0.0, imposed_disp));

    SubModelSolver solver(30.0);
    RebarSteelConfig steel{200000.0, 420.0, 0.01};
    std::vector<double> rebar_areas = {0.0002};
    sub.rebar_areas = rebar_areas;

    // Write VTK to temp directory
    auto vtk_prefix = (std::filesystem::temp_directory_path()
                        / "test_reinforced_vtk_crack").string();

    auto result = solver.solve_reinforced(
        sub, steel, rd.rebar_range, rebar_areas, spec.nz, vtk_prefix);

    check(result.converged, "tension solve converged");

    // ── Verify Gauss cloud VTU file exists and has content ──────────
    std::string gauss_path = vtk_prefix + "_gauss.vtu";
    check(std::filesystem::exists(gauss_path),
          "gauss VTU file exists");

    vtkNew<vtkXMLUnstructuredGridReader> reader;
    reader->SetFileName(gauss_path.c_str());
    reader->Update();

    auto* grid = reader->GetOutput();
    check(grid != nullptr, "gauss VTU parseable");

    if (grid == nullptr) return;

    auto* pd = grid->GetPointData();
    const auto npts = grid->GetNumberOfPoints();

    // Must have Gauss points (2×2×2 per hex element, 16 hex → 128 QPs,
    // plus truss QPs)
    check(npts > 0, "gauss cloud has points");
    std::cout << "    gauss points = " << npts << "\n";

    // ── Stress / strain tensor fields must be present ────────────────
    check(pd->GetArray("qp_stress_xx") != nullptr,
          "qp_stress_xx present");
    check(pd->GetArray("qp_strain_xx") != nullptr,
          "qp_strain_xx present");
    check(pd->GetArray("qp_stress_von_mises") != nullptr,
          "qp_stress_von_mises present");

    // ── Crack fields should be present (tension exceeds crack threshold)
    auto* num_cracks = pd->GetArray("qp_num_cracks");
    check(num_cracks != nullptr, "qp_num_cracks present");

    if (num_cracks != nullptr) {
        // At least some Gauss points should have cracks
        int cracked_count = 0;
        for (vtkIdType i = 0; i < num_cracks->GetNumberOfTuples(); ++i) {
            if (num_cracks->GetComponent(i, 0) > 0.5)
                ++cracked_count;
        }
        check(cracked_count > 0,
              "at least one GP is cracked");
        std::cout << "    cracked GPs = " << cracked_count
                  << " / " << num_cracks->GetNumberOfTuples() << "\n";
    }

    check(pd->GetArray("qp_crack_normal_1") != nullptr,
          "qp_crack_normal_1 present");
    check(pd->GetArray("qp_crack_strain_1") != nullptr,
          "qp_crack_strain_1 present");
    check(pd->GetArray("qp_crack_closed_1") != nullptr,
          "qp_crack_closed_1 present");

    // ── Fracture history fields ─────────────────────────────────────
    check(pd->GetArray("qp_sigma_o_max") != nullptr,
          "qp_sigma_o_max present");
    check(pd->GetArray("qp_tau_o_max") != nullptr,
          "qp_tau_o_max present");

    // ── Plastic strain / equivalent plastic strain ──────────────────
    check(pd->GetArray("qp_equivalent_plastic_strain") != nullptr,
          "qp_equivalent_plastic_strain present");

    // ── Mesh export should also exist ───────────────────────────────
    std::string mesh_path = vtk_prefix + "_mesh.vtu";
    check(std::filesystem::exists(mesh_path),
          "mesh VTU file exists");

    // Cleanup temp files
    std::filesystem::remove(gauss_path);
    std::filesystem::remove(mesh_path);
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
    test_vtk_crack_export();

    std::cout << "\n══════════════════════════════════════════════════\n";
    std::cout << "  Results: " << passed << "/" << total << " passed\n";
    std::cout << "══════════════════════════════════════════════════\n";

    PetscFinalize();
    return (passed == total) ? 0 : 1;
}
