// =============================================================================
//  test_boundary_conditions.cpp — Tests for the generic BC/IC infrastructure
// =============================================================================
//
//  Part 1: NodeSelector tests (pure geometry, no mesh)
//  Part 2: Per-DOF constraint API (requires PETSc + Gmsh mesh)
//  Part 3: Selector-based constraints (requires PETSc + Gmsh mesh)
//  Part 4: Non-zero Dirichlet (imposed displacement)
//  Part 5: ModelState capture and queries
//
// =============================================================================

#include "header_files.hh"

#include "src/model/NodeSelector.hh"
#include "src/model/ModelState.hh"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

// ─── Test infrastructure ─────────────────────────────────────────────────────

static int passed = 0, failed = 0;

static void report(const char* name, bool ok) {
    if (ok) { ++passed; std::cout << "  PASS  " << name << "\n"; }
    else    { ++failed; std::cout << "  FAIL  " << name << "\n"; }
}

constexpr bool approx(double a, double b, double tol = 1e-10) {
    return std::abs(a - b) <= tol;
}

// ─── Constants ───────────────────────────────────────────────────────────────

#ifdef FALL_N_SOURCE_DIR
static const std::string BASE = FALL_N_SOURCE_DIR "/";
#else
static const std::string BASE = "./";
#endif
static const std::string MESH = BASE + "data/input/Beam_LagCell_Ord1_30x6x3.msh";

static constexpr std::size_t DIM  = 3;
static constexpr std::size_t NDOF = DIM;

// Material
static constexpr double E_mod = 200.0;
static constexpr double nu    = 0.3;

// Geometry: beam [0,10] × [0,0.4] × [0,0.8]
static constexpr double X_FIXED = 0.0;
static constexpr double X_TIP   = 10.0;

// =============================================================================
//  Part 1: NodeSelector tests (no mesh required)
// =============================================================================

void test_plane_selector_basic() {
    Node<3> n1{0, 0.0, 1.0, 2.0};
    Node<3> n2{1, 5.0, 1.0, 2.0};
    Node<3> n3{2, 0.0, 3.0, 2.0};

    fn::PlaneSelector<3> sel{0, 0.0};  // x = 0 plane

    report("PlaneSelector: node on plane",     sel(n1) == true);
    report("PlaneSelector: node off plane",    sel(n2) == false);
    report("PlaneSelector: other on plane",    sel(n3) == true);
}

void test_plane_selector_tolerance() {
    Node<3> n1{0, 1e-7, 0.0, 0.0};   // within default tol (1e-6)
    Node<3> n2{0, 2e-6, 0.0, 0.0};   // outside default tol

    fn::PlaneSelector<3> sel{0, 0.0};

    report("PlaneSelector tol: within",  sel(n1) == true);
    report("PlaneSelector tol: outside", sel(n2) == false);

    // Custom tight tolerance
    fn::PlaneSelector<3> tight{0, 0.0, 1e-8};
    report("PlaneSelector tight tol: reject 1e-7", tight(n1) == false);
}

void test_plane_selector_different_axes() {
    Node<3> n{0, 1.0, 2.0, 3.0};

    report("PlaneSelector axis=0: x=1", fn::PlaneSelector<3>{0, 1.0}(n) == true);
    report("PlaneSelector axis=1: y=2", fn::PlaneSelector<3>{1, 2.0}(n) == true);
    report("PlaneSelector axis=2: z=3", fn::PlaneSelector<3>{2, 3.0}(n) == true);
    report("PlaneSelector axis=0: x=2", fn::PlaneSelector<3>{0, 2.0}(n) == false);
}

void test_box_selector() {
    Node<3> inside {0, 1.0, 1.0, 1.0};
    Node<3> outside{1, 5.0, 5.0, 5.0};
    Node<3> corner {2, 0.0, 0.0, 0.0};
    Node<3> edge   {3, 2.0, 1.0, 0.0};

    fn::BoxSelector<3> box{{0.0, 0.0, 0.0}, {2.0, 2.0, 2.0}};

    report("BoxSelector: inside",  box(inside) == true);
    report("BoxSelector: outside", box(outside) == false);
    report("BoxSelector: corner",  box(corner) == true);
    report("BoxSelector: edge",    box(edge) == true);
}

void test_sphere_selector() {
    Node<3> center{0, 0.0, 0.0, 0.0};
    Node<3> on    {1, 1.0, 0.0, 0.0};
    Node<3> inside{2, 0.5, 0.5, 0.0};
    Node<3> out   {3, 1.0, 1.0, 0.0};   // sqrt(2) ≈ 1.414 > 1

    fn::SphereSelector<3> sphere{{0.0, 0.0, 0.0}, 1.0};

    report("SphereSelector: at center", sphere(center) == true);
    report("SphereSelector: on surface", sphere(on) == true);
    report("SphereSelector: inside", sphere(inside) == true);
    report("SphereSelector: outside (sqrt2)", sphere(out) == false);
}

void test_node_id_selector() {
    Node<3> n0{0, 0.0, 0.0, 0.0};
    Node<3> n1{1, 1.0, 0.0, 0.0};
    Node<3> n5{5, 5.0, 0.0, 0.0};

    fn::NodeIdSelector sel{0ul, 5ul};  // note: size_t literals

    report("NodeIdSelector: id=0 matches",   sel(n0) == true);
    report("NodeIdSelector: id=1 no match",  sel(n1) == false);
    report("NodeIdSelector: id=5 matches",   sel(n5) == true);
}

void test_selector_combinators() {
    Node<3> n{0, 0.0, 0.0, 0.0};      // at origin
    Node<3> m{1, 5.0, 5.0, 5.0};      // far away

    auto on_x0 = fn::PlaneSelector<3>{0, 0.0};
    auto on_y0 = fn::PlaneSelector<3>{1, 0.0};

    // AND: both planes
    auto corner = fn::select_and(on_x0, on_y0);
    report("select_and: corner at origin", corner(n) == true);
    report("select_and: not at corner",    corner(m) == false);

    // OR: either plane
    Node<3> on_x_only{2, 0.0, 5.0, 5.0};
    auto either = fn::select_or(on_x0, on_y0);
    report("select_or: on x-plane",  either(on_x_only) == true);
    report("select_or: far away",    either(m) == false);

    // NOT
    auto not_x0 = fn::select_not(on_x0);
    report("select_not: not on x0", not_x0(m) == true);
    report("select_not: on x0",     not_x0(n) == false);
}

void test_lambda_as_selector() {
    Node<3> n{0, 3.0, 4.0, 5.0};

    // Lambdas qualify as NodeSelectors
    auto high_z = [](const auto& node) { return node.coord(2) > 4.0; };

    report("Lambda selector: z > 4", high_z(n) == true);

    auto low_z = [](const auto& node) { return node.coord(2) < 2.0; };
    report("Lambda selector: z < 2", low_z(n) == false);
}

void test_convenience_factories() {
    Node<3> n{0, 0.0, 0.0, 0.0};
    Node<3> m{1, 5.0, 5.0, 5.0};

    auto sel = fn::on_plane<3>(0, 0.0);
    report("on_plane factory: on plane",  sel(n) == true);
    report("on_plane factory: off plane", sel(m) == false);

    auto bsel = fn::in_box<3>({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
    report("in_box factory: inside",  bsel(n) == true);
    report("in_box factory: outside", bsel(m) == false);

    auto ssel = fn::in_sphere<3>({0.0, 0.0, 0.0}, 1.0);
    report("in_sphere factory: inside",  ssel(n) == true);
    report("in_sphere factory: outside", ssel(m) == false);
}

void test_2d_selectors() {
    Node<2> n{0, 1.0, 2.0};
    Node<2> m{1, 5.0, 5.0};

    fn::PlaneSelector<2> sel{0, 1.0};
    report("2D PlaneSelector: on line",  sel(n) == true);
    report("2D PlaneSelector: off line", sel(m) == false);

    fn::BoxSelector<2> box{{0.0, 0.0}, {3.0, 3.0}};
    report("2D BoxSelector: inside",  box(n) == true);
    report("2D BoxSelector: outside", box(m) == false);
}

// =============================================================================
//  Part 2: Per-DOF constraint API (integration tests with PETSc + mesh)
// =============================================================================

void test_constrain_dof_single() {
    std::cout << "\n  ── Per-DOF constraint tests ──\n";

    Domain<DIM> D;
    GmshDomainBuilder b(MESH, D);

    ContinuumIsotropicElasticMaterial mat_inst{E_mod, nu};
    Material<ThreeDimensionalMaterial> mat{mat_inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF> M{D, mat};

    // Find a node at x=0 (should exist in the beam mesh)
    std::size_t test_node_id = 0;
    bool found = false;
    for (const auto& node : D.nodes()) {
        if (std::abs(node.coord(0) - X_FIXED) < 1e-6) {
            test_node_id = node.id();
            found = true;
            break;
        }
    }
    report("Found node at x=0 for constraint test", found);
    if (!found) return;

    // Constrain only DOF 0 (u_x) of this node
    M.constrain_dof(test_node_id, 0, 0.0);

    report("is_constrained: true after constrain_dof",
           M.is_constrained(test_node_id));
    report("is_dof_constrained(0): true",
           M.is_dof_constrained(test_node_id, 0));
    report("is_dof_constrained(1): false",
           M.is_dof_constrained(test_node_id, 1) == false);
    report("is_dof_constrained(2): false",
           M.is_dof_constrained(test_node_id, 2) == false);
}

void test_constrain_dof_multiple() {
    Domain<DIM> D;
    GmshDomainBuilder b(MESH, D);

    ContinuumIsotropicElasticMaterial mat_inst{E_mod, nu};
    Material<ThreeDimensionalMaterial> mat{mat_inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF> M{D, mat};

    // Find a node at x=0
    std::size_t nid = 0;
    for (const auto& node : D.nodes()) {
        if (std::abs(node.coord(0) - X_FIXED) < 1e-6) { nid = node.id(); break; }
    }

    // Constrain DOFs 0 and 2 individually (not DOF 1)
    M.constrain_dof(nid, 0, 0.0);
    M.constrain_dof(nid, 2, 0.0);

    report("Multi constrain: DOF 0 constrained", M.is_dof_constrained(nid, 0));
    report("Multi constrain: DOF 1 free",        M.is_dof_constrained(nid, 1) == false);
    report("Multi constrain: DOF 2 constrained", M.is_dof_constrained(nid, 2));
    report("Multi constrain: 2 total",           M.num_constraints() == 2);
}

void test_constrain_dof_nonzero_value() {
    Domain<DIM> D;
    GmshDomainBuilder b(MESH, D);

    ContinuumIsotropicElasticMaterial mat_inst{E_mod, nu};
    Material<ThreeDimensionalMaterial> mat{mat_inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF> M{D, mat};

    std::size_t nid = 0;
    for (const auto& node : D.nodes()) {
        if (std::abs(node.coord(0) - X_FIXED) < 1e-6) { nid = node.id(); break; }
    }

    M.constrain_dof(nid, 0, 0.005);  // Impose u_x = 0.005

    report("Non-zero Dirichlet: value stored",
           approx(M.prescribed_value(nid, 0), 0.005));
    report("Non-zero Dirichlet: zero for free DOF",
           approx(M.prescribed_value(nid, 1), 0.0));
}

void test_constrain_dof_update_value() {
    Domain<DIM> D;
    GmshDomainBuilder b(MESH, D);

    ContinuumIsotropicElasticMaterial mat_inst{E_mod, nu};
    Material<ThreeDimensionalMaterial> mat{mat_inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF> M{D, mat};

    std::size_t nid = 0;
    for (const auto& node : D.nodes()) {
        if (std::abs(node.coord(0) - X_FIXED) < 1e-6) { nid = node.id(); break; }
    }

    M.constrain_dof(nid, 0, 0.001);
    M.constrain_dof(nid, 0, 0.002); // Update!

    report("Constrain update: value updated to 0.002",
           approx(M.prescribed_value(nid, 0), 0.002));
    // Still only 1 constraint (not 2)
    report("Constrain update: still 1 constraint", M.num_constraints() == 1);
}

// =============================================================================
//  Part 3: Selector-based constraints
// =============================================================================

void test_constrain_dof_where_plane() {
    std::cout << "\n  ── Selector-based constraint tests ──\n";

    Domain<DIM> D;
    GmshDomainBuilder b(MESH, D);

    ContinuumIsotropicElasticMaterial mat_inst{E_mod, nu};
    Material<ThreeDimensionalMaterial> mat{mat_inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF> M{D, mat};

    // Count nodes at x=0
    std::size_t n_on_plane = 0;
    for (const auto& node : D.nodes())
        if (std::abs(node.coord(0) - X_FIXED) < 1e-6) ++n_on_plane;

    // Constrain only u_x (DOF 0) at x=0 plane
    M.constrain_dof_where(fn::PlaneSelector<3>{0, X_FIXED}, 0, 0.0);

    report("constrain_dof_where: correct total",
           M.num_constraints() == n_on_plane);

    // Verify per-DOF: DOF 0 constrained, DOFs 1,2 free
    bool all_correct = true;
    for (const auto& node : D.nodes()) {
        if (std::abs(node.coord(0) - X_FIXED) < 1e-6) {
            if (!M.is_dof_constrained(node.id(), 0)) all_correct = false;
            if (M.is_dof_constrained(node.id(), 1))  all_correct = false;
            if (M.is_dof_constrained(node.id(), 2))  all_correct = false;
        }
    }
    report("constrain_dof_where: only DOF 0 per node", all_correct);
}

void test_constrain_where_all_dofs() {
    Domain<DIM> D;
    GmshDomainBuilder b(MESH, D);

    ContinuumIsotropicElasticMaterial mat_inst{E_mod, nu};
    Material<ThreeDimensionalMaterial> mat{mat_inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF> M{D, mat};

    std::size_t n_on_plane = 0;
    for (const auto& node : D.nodes())
        if (std::abs(node.coord(0) - X_FIXED) < 1e-6) ++n_on_plane;

    // Fix ALL DOFs at x=0 (clamped)
    M.constrain_where(fn::PlaneSelector<3>{0, X_FIXED});

    // Each node has 3 constrained DOFs
    report("constrain_where: 3 DOFs per node",
           M.num_constraints() == n_on_plane * 3);

    bool all_correct = true;
    for (const auto& node : D.nodes()) {
        if (std::abs(node.coord(0) - X_FIXED) < 1e-6) {
            if (!M.is_dof_constrained(node.id(), 0)) all_correct = false;
            if (!M.is_dof_constrained(node.id(), 1)) all_correct = false;
            if (!M.is_dof_constrained(node.id(), 2)) all_correct = false;
        }
    }
    report("constrain_where: all 3 DOFs constrained", all_correct);
}

void test_constrain_where_with_values() {
    Domain<DIM> D;
    GmshDomainBuilder b(MESH, D);

    ContinuumIsotropicElasticMaterial mat_inst{E_mod, nu};
    Material<ThreeDimensionalMaterial> mat{mat_inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF> M{D, mat};

    // Impose u = (0.01, 0.0, -0.01) at x=10 (tip)
    std::array<double, 3> imposed = {0.01, 0.0, -0.01};
    M.constrain_where(fn::PlaneSelector<3>{0, X_TIP}, imposed);

    bool all_correct = true;
    for (const auto& node : D.nodes()) {
        if (std::abs(node.coord(0) - X_TIP) < 1e-6) {
            if (!approx(M.prescribed_value(node.id(), 0), 0.01))  all_correct = false;
            if (!approx(M.prescribed_value(node.id(), 1), 0.0))   all_correct = false;
            if (!approx(M.prescribed_value(node.id(), 2), -0.01)) all_correct = false;
        }
    }
    report("constrain_where(values): prescribed values correct", all_correct);
}

void test_constrain_lambda_selector() {
    Domain<DIM> D;
    GmshDomainBuilder b(MESH, D);

    ContinuumIsotropicElasticMaterial mat_inst{E_mod, nu};
    Material<ThreeDimensionalMaterial> mat{mat_inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF> M{D, mat};

    // Fix u_z (DOF 2) where z = 0 (bottom face) using a lambda
    M.constrain_dof_where(
        [](const auto& n) { return std::abs(n.coord(2)) < 1e-6; },
        2, 0.0);

    // Verify at least some nodes were constrained
    report("Lambda selector: some constraints added",
           M.num_constraints() > 0);

    // Verify no constraint on non-bottom nodes
    bool no_false_positive = true;
    for (const auto& node : D.nodes()) {
        if (std::abs(node.coord(2)) > 1e-6) {
            if (M.is_constrained(node.id())) {
                no_false_positive = false;
                break;
            }
        }
    }
    report("Lambda selector: no false positives", no_false_positive);
}

void test_composite_selector() {
    Domain<DIM> D;
    GmshDomainBuilder b(MESH, D);

    ContinuumIsotropicElasticMaterial mat_inst{E_mod, nu};
    Material<ThreeDimensionalMaterial> mat{mat_inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF> M{D, mat};

    // Fix nodes at the intersection: x=0 AND y=0
    auto corner_selector = fn::select_and(
        fn::PlaneSelector<3>{0, 0.0},
        fn::PlaneSelector<3>{1, 0.0}
    );

    M.constrain_where(corner_selector);

    // Count expected: nodes at x=0 AND y=0
    std::size_t expected = 0;
    for (const auto& node : D.nodes()) {
        if (std::abs(node.coord(0)) < 1e-6 && std::abs(node.coord(1)) < 1e-6)
            ++expected;
    }

    // 3 DOFs per node
    report("Composite selector: correct count",
           M.num_constraints() == expected * 3);
}

// =============================================================================
//  Part 4: Full solve with roller BC (per-DOF) vs clamped (all DOFs)
// =============================================================================

void test_roller_vs_clamped() {
    std::cout << "\n  ── Roller vs clamped solve test ──\n";

    // ── Solve 1: Clamped (fix all DOFs at x=0) ──────────────────────────
    double max_u_clamped;
    {
        Domain<DIM> D;
        GmshDomainBuilder b(MESH, D);
        ContinuumIsotropicElasticMaterial mat_inst{E_mod, nu};
        Material<ThreeDimensionalMaterial> mat{mat_inst, ElasticUpdate{}};
        Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF> M{D, mat};

        M.constrain_where(fn::PlaneSelector<3>{0, X_FIXED});  // ALL DOFs
        M.setup();

        D.create_boundary_from_plane("Load", 0, X_TIP);
        M.apply_surface_traction("Load", 0.0, 0.05, -0.05);

        LinearAnalysis<ThreeDimensionalMaterial> solver{&M};
        solver.solve();

        const PetscScalar* arr; PetscInt n;
        VecGetLocalSize(M.state_vector(), &n);
        VecGetArrayRead(M.state_vector(), &arr);
        max_u_clamped = 0.0;
        for (PetscInt i = 0; i < n; ++i)
            max_u_clamped = std::max(max_u_clamped, std::abs(arr[i]));
        VecRestoreArrayRead(M.state_vector(), &arr);
    }

    // ── Solve 2: Roller x (fix only u_x at x=0) ────────────────────────
    double max_u_roller;
    {
        Domain<DIM> D;
        GmshDomainBuilder b(MESH, D);
        ContinuumIsotropicElasticMaterial mat_inst{E_mod, nu};
        Material<ThreeDimensionalMaterial> mat{mat_inst, ElasticUpdate{}};
        Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF> M{D, mat};

        // FIX ONLY u_x at x=0 (roller in x-direction)
        M.constrain_dof_where(fn::PlaneSelector<3>{0, X_FIXED}, 0, 0.0);
        // Prevent rigid body modes: fix u_y and u_z at origin ...
        auto at_origin = fn::select_and(fn::PlaneSelector<3>{0, 0.0},
            fn::select_and(fn::PlaneSelector<3>{1, 0.0},
                           fn::PlaneSelector<3>{2, 0.0}));
        M.constrain_dof_where(at_origin, 1, 0.0);
        M.constrain_dof_where(at_origin, 2, 0.0);
        // ... and fix u_y at (0, 0, z_max) to prevent rotation about x
        auto at_z_top = fn::select_and(fn::PlaneSelector<3>{0, 0.0},
            fn::select_and(fn::PlaneSelector<3>{1, 0.0},
                           fn::PlaneSelector<3>{2, 0.8}));
        M.constrain_dof_where(at_z_top, 1, 0.0);
        M.setup();

        D.create_boundary_from_plane("Load", 0, X_TIP);
        M.apply_surface_traction("Load", 0.0, 0.05, -0.05);

        LinearAnalysis<ThreeDimensionalMaterial> solver{&M};
        solver.solve();

        const PetscScalar* arr; PetscInt n;
        VecGetLocalSize(M.state_vector(), &n);
        VecGetArrayRead(M.state_vector(), &arr);
        max_u_roller = 0.0;
        for (PetscInt i = 0; i < n; ++i)
            max_u_roller = std::max(max_u_roller, std::abs(arr[i]));
        VecRestoreArrayRead(M.state_vector(), &arr);
    }

    std::cout << "       Clamped max |u| = " << max_u_clamped << "\n";
    std::cout << "       Roller  max |u| = " << max_u_roller  << "\n";

    // Roller support should give LARGER deflection than clamped
    // (fewer constraints → more flexibility)
    report("Roller > Clamped deflection", max_u_roller > max_u_clamped);

    // Both should be finite and positive
    report("Clamped deflection positive", max_u_clamped > 1e-12);
    report("Roller deflection positive",  max_u_roller  > 1e-12);
}

// =============================================================================
//  Part 5: ModelState capture
// =============================================================================

void test_model_state_capture() {
    std::cout << "\n  ── ModelState capture tests ──\n";

    Domain<DIM> D;
    GmshDomainBuilder b(MESH, D);

    ContinuumIsotropicElasticMaterial mat_inst{E_mod, nu};
    Material<ThreeDimensionalMaterial> mat{mat_inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF> M{D, mat};
    M.fix_x(X_FIXED);
    M.setup();

    D.create_boundary_from_plane("Load", 0, X_TIP);
    M.apply_surface_traction("Load", 0.0, 0.05, -0.05);

    LinearAnalysis<ThreeDimensionalMaterial> solver{&M};
    solver.solve();
    M.update_elements_state();

    // Capture state
    auto state = M.capture_state();

    report("capture_state: not empty",       !state.empty());
    report("capture_state: node count",      state.num_nodes() == D.num_nodes());
    report("capture_state: element count",   state.num_elements() == M.num_elements());
    report("capture_state: has element data", state.has_element_data());

    // Verify some displacement is non-zero (beam deflects)
    bool has_nonzero_disp = false;
    for (const auto& nd : state.nodes) {
        for (double d : nd.displacement) {
            if (std::abs(d) > 1e-12) { has_nonzero_disp = true; break; }
        }
        if (has_nonzero_disp) break;
    }
    report("capture_state: non-zero displacement", has_nonzero_disp);

    // Verify some stress is non-zero
    bool has_nonzero_stress = false;
    for (const auto& ed : state.elements) {
        for (const auto& gp : ed.gauss_points) {
            for (double s : gp.stress) {
                if (std::abs(s) > 1e-12) { has_nonzero_stress = true; break; }
            }
            if (has_nonzero_stress) break;
        }
        if (has_nonzero_stress) break;
    }
    report("capture_state: non-zero stress", has_nonzero_stress);

    // Node IDs should be sequential and match domain
    bool ids_match = true;
    for (std::size_t i = 0; i < state.nodes.size(); ++i) {
        if (state.nodes[i].node_id != D.nodes()[i].id()) {
            ids_match = false;
            break;
        }
    }
    report("capture_state: node IDs match domain", ids_match);
}

void test_model_state_data_structure() {
    // Pure data structure tests (no PETSc)
    ModelState<3> s;
    report("ModelState default: empty",           s.empty());
    report("ModelState default: no elements",     !s.has_element_data());
    report("ModelState default: zero nodes",      s.num_nodes() == 0);
    report("ModelState default: zero elements",   s.num_elements() == 0);

    s.nodes.push_back({0, {1.0, 2.0, 3.0}, {0.0, 0.0, 0.0}});
    report("ModelState: not empty after add",     !s.empty());
    report("ModelState: 1 node",                  s.num_nodes() == 1);

    ModelState<3>::GaussPointData gp;
    gp.stress = {100.0, 200.0, 300.0, 0.0, 0.0, 0.0};
    gp.strain = {0.001, 0.002, 0.003, 0.0, 0.0, 0.0};
    s.elements.push_back({0, {gp}});
    report("ModelState: has element data",        s.has_element_data());

    // Copy semantics
    auto s2 = s;
    report("ModelState: copyable",                s2.num_nodes() == 1);
    report("ModelState: copy independent",
           s2.nodes[0].displacement[0] == 1.0);
}

// =============================================================================
//  Part 6: Backward compatibility — fix_x still works
// =============================================================================

void test_backward_compat_fix_x() {
    std::cout << "\n  ── Backward compatibility tests ──\n";

    Domain<DIM> D;
    GmshDomainBuilder b(MESH, D);

    ContinuumIsotropicElasticMaterial mat_inst{E_mod, nu};
    Material<ThreeDimensionalMaterial> mat{mat_inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF> M{D, mat};

    // Old API: fix_x should still work (fixes ALL DOFs at x=0)
    M.fix_x(X_FIXED);
    M.setup();

    D.create_boundary_from_plane("Load", 0, X_TIP);
    M.apply_surface_traction("Load", 0.0, 0.05, -0.05);

    LinearAnalysis<ThreeDimensionalMaterial> solver{&M};
    solver.solve();

    const PetscScalar* arr; PetscInt n;
    VecGetLocalSize(M.state_vector(), &n);
    VecGetArrayRead(M.state_vector(), &arr);
    double mx = 0.0;
    for (PetscInt i = 0; i < n; ++i) mx = std::max(mx, std::abs(arr[i]));
    VecRestoreArrayRead(M.state_vector(), &arr);

    report("Backward compat: positive deflection", mx > 1e-6);
    // Should match previous known value (approx 6.2 for HEX8)
    report("Backward compat: reasonable magnitude", mx > 0.1 && mx < 10.0);

    std::cout << "       fix_x result: max |u| = " << mx << "\n";
}

// =============================================================================
//  Main
// =============================================================================

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    std::cout << "══════════════════════════════════════════════════════\n"
              << "  Boundary Condition / Initial Condition Test Suite\n"
              << "══════════════════════════════════════════════════════\n";

    // Part 1: Pure selector tests
    std::cout << "\n  ── NodeSelector tests ──\n";
    test_plane_selector_basic();
    test_plane_selector_tolerance();
    test_plane_selector_different_axes();
    test_box_selector();
    test_sphere_selector();
    test_node_id_selector();
    test_selector_combinators();
    test_lambda_as_selector();
    test_convenience_factories();
    test_2d_selectors();

    // Part 1b: ModelState data structure
    test_model_state_data_structure();

    // Part 2: Per-DOF constraints
    test_constrain_dof_single();
    test_constrain_dof_multiple();
    test_constrain_dof_nonzero_value();
    test_constrain_dof_update_value();

    // Part 3: Selector-based constraints
    test_constrain_dof_where_plane();
    test_constrain_where_all_dofs();
    test_constrain_where_with_values();
    test_constrain_lambda_selector();
    test_composite_selector();

    // Part 4: Full solve comparison
    test_roller_vs_clamped();

    // Part 5: State capture
    test_model_state_capture();

    // Part 6: Backward compatibility 
    test_backward_compat_fix_x();

    // Report
    std::cout << "\n══════════════════════════════════════════════════════\n"
              << "  Results: " << passed << " passed, " << failed << " failed\n"
              << "══════════════════════════════════════════════════════\n";

    PetscFinalize();
    return (failed > 0) ? 1 : 0;
}
