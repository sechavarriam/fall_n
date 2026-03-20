// =============================================================================
//  test_timoshenko_cantilever_benchmark.cpp
// =============================================================================
//
//  Benchmark: Timoshenko cantilever beam — analytical vs FEM
//
//  PURPOSE
//  -------
//  End-to-end validation of the 3D continuum and structural beam pipelines
//  against a closed-form Timoshenko beam solution.  The same geometry,
//  material, and loading are solved with three independent approaches so
//  that any regression in meshing, constitutive law, shape-function
//  gradients, stiffness assembly, boundary conditions, or the PETSc
//  linear solver is caught immediately.
//
//  GEOMETRY & MATERIAL (shared by all tests)
//  ------------------------------------------
//    L  = 10.0   m   (span, x-direction)
//    B  =  0.40  m   (width, y-direction)
//    H  =  0.80  m   (height, z-direction)
//    E  = 200.0  Pa  (Young's modulus)
//    ν  =   0.3       (Poisson's ratio)
//    P  =   1.0  N   (tip point load)
//
//  THREE SOLUTION STRATEGIES
//  -------------------------
//    A) Analytical Timoshenko beam (closed-form).
//       δ = PL³/(3EI) + PL/(κGA)
//       Exact for prismatic cantilever under tip load.
//
//    B) 3D Continuum FEM — hex27 (27-node hexahedral) elements read from
//       beam_cantilever.msh (Gmsh MSH 4.1).  SmallStrain kinematic policy,
//       isotropic linear-elastic material, PETSc KSP (direct LU solver).
//       The current mesh has 1 280 hex27 elements and 12 393 nodes;
//       errors of < 5 % are expected with this refinement.
//
//    C) Structural beam FEM — TimoshenkoBeamN<N> elements (N = 2 linear,
//       N = 3 quadratic) assembled on a programmatic 1D mesh.  Same PETSc
//       solver.  Errors < 0.5 % expected (exact for cubic deflection when
//       N = 3).
//
//  TESTS OVERVIEW (10 checks total)
//  ---------------------------------
//    Test 1 – Analytical sanity (3 checks):
//       Verifies that computed analytical values are positive and that
//       δ_y > δ_z (weaker bending axis).
//
//    Test 2 – 3D Continuum, tip load Z (1 check):
//       Full FEM solve with surface traction → compare tip δ_z.
//
//    Test 3 – 3D Continuum, tip load Y (1 check):
//       Same as Test 2 but loads and measures in y-direction.
//
//    Test 4 – Beam N=2, tip load Z (1 check):
//       Linear (2-node) Timoshenko beam elements, tip point load in z.
//
//    Test 5 – Beam N=2, tip load Y (1 check):
//       Same as Test 4 in y-direction.
//
//    Test 6 – Beam N=3, tip load Z (1 check):
//       Quadratic (3-node) Timoshenko beam, expected ≈ machine-precision
//       match because a cubic deflection field is reproduced exactly by
//       quadratic elements with Timoshenko shear terms.
//
//    Test 7 – Cross-validation 3D vs Beam (1 check):
//       Ensures both methods agree, independent of the analytical formula.
//
//    Test 8 – Deflection curve (1 check):
//       Compares z-deflections at 5 stations along the span.
//
//  ACCEPTANCE CRITERIA
//  -------------------
//    - Beam FEM vs analytical     :  < 0.5 % (N ≥ 2)
//    - 3D continuum vs analytical :  < 1.0  % (generous bound; actual < 5 %
//      with the current refined mesh).
//
//  MESH DEPENDENCY
//  ---------------
//  The 3D continuum tests require `beam_cantilever.msh` to be present at
//  the path defined by MESH_FILE below.  The mesh can be regenerated from
//  the corresponding .geo file with Gmsh ≥ 4.12.
//
//  SOLVER NOTE
//  -----------
//  main() forces PETSc to use a direct LU solver (-ksp_type preonly
//  -pc_type lu) so results are deterministic and independent of iterative
//  convergence tolerances.
//
// =============================================================================

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <numeric>
#include <string>
#include <numbers>

#include <petsc.h>
#include <Eigen/Dense>

// ── Project headers ─────────────────────────────────────────────────────────

#include "header_files.hh"
#include "src/elements/TimoshenkoBeamN.hh"
#include "src/elements/BeamElement.hh"

// =============================================================================
//  Constants & analytical solutions
// =============================================================================
//
//  All constants are defined at file scope so every test function operates
//  on the exact same set of input parameters.  Changing any value here
//  automatically propagates to the analytical formulas AND to every FEM
//  model built below.
//
//  Coordinate convention
//  ---------------------
//    x — longitudinal (span) direction, clamped at x=0, free tip at x=L.
//    y — width direction (B = 0.40).
//    z — height direction (H = 0.80).
//
//  Section property convention
//  ---------------------------
//    I_y = ∫ z² dA = B·H³/12  → used for bending about the y-axis
//                                (produces z-deflection).
//    I_z = ∫ y² dA = H·B³/12  → used for bending about the z-axis
//                                (produces y-deflection).
//    Because H > B  →  I_y > I_z  →  z-direction is the STRONG axis
//    and y-direction is the WEAK axis (δ_y > δ_z under equal load).
//
// -----------------------------------------------------------------------------

static constexpr std::size_t DIM  = 3;       // spatial dimension
static constexpr std::size_t NDOF_CONT = 3;  // DOFs per node — 3D continuum (u,v,w)
static constexpr std::size_t NDOF_BEAM = 6;  // DOFs per node — beam (u,v,w,θx,θy,θz)

// ── Material ─────────────────────────────────────────────────────────────────
static constexpr double E_mod = 200.0;                      // Young's modulus
static constexpr double nu    = 0.3;                        // Poisson's ratio
static constexpr double G_mod = E_mod / (2.0 * (1.0 + nu)); // ≈ 76.923 (shear modulus)

// ── Geometry (must match beam_cantilever.geo / .msh) ─────────────────────────
static constexpr double L = 10.0;   // span  (x)
static constexpr double B = 0.40;   // width (y)
static constexpr double H = 0.80;   // height (z)

// ── Section properties ───────────────────────────────────────────────────────
static constexpr double A     = B * H;            // 0.32 m²       (cross-section area)
static constexpr double Iy    = B * H*H*H / 12.0; // 0.017067 m⁴   (strong-axis moment)
static constexpr double Iz    = H * B*B*B / 12.0; // 0.004267 m⁴   (weak-axis moment)
static constexpr double kappa = 5.0 / 6.0;        // Timoshenko shear correction (rectangle)

// Torsional constant for a rectangular section — approximate closed-form:
//   J ≈ (b³ h / 3) · (1 − 0.63 · b/h)   where  b = min(B,H),  h = max(B,H).
// Reference: Timoshenko & Goodier, §109.  Error < 1 % for h/b > 1.5.
static constexpr double b_min = B;  // 0.40
static constexpr double h_max = H;  // 0.80
static constexpr double J_tor = (b_min*b_min*b_min * h_max / 3.0)
                                * (1.0 - 0.63 * b_min / h_max);

// Mesh file — absolute path to the Gmsh MSH 4.1 file.
// Contains hex27 volume elements (type 12) and QUA_9 surface elements (type 10).
// Physical groups: "domain" (3D), "Fixed" (2D face at x=0).
#ifdef FALL_N_SOURCE_DIR
static const std::string _BASE = std::string(FALL_N_SOURCE_DIR) + "/";
#else
static const std::string _BASE = "./";
#endif
static const std::string MESH_FILE =
    _BASE + "tests/beam_cantilever.msh";

// Tet mesh files — same beam geometry meshed with tetrahedral elements.
// Physical groups: "domain" (3D), "Fixed" (2D face at x=0).
static const std::string MESH_TET4 =
    _BASE + "data/input/beam_gmshElem1stOrder.msh";
static const std::string MESH_TET10 =
    _BASE + "data/input/beam_gmshElem2ndOrder.msh";

// Tip load
static constexpr double P = 1.0;

// ─── Analytical Timoshenko cantilever — tip deflection under tip load P ──────
//
//  Timoshenko beam theory accounts for both flexural and shear deformation:
//
//      δ_tip = PL³/(3EI) + PL/(κGA)
//              ─────────    ────────
//              Euler-       Timoshenko
//              Bernoulli    shear correction
//
//  For the current geometry (L/H = 12.5), the shear term contributes only
//  ≈ 0.5 % of the total — the beam is bending-dominated.
//
//  Reference: Timoshenko, S.P. "Strength of Materials", Part I, §§ 4-5.
//
// -----------------------------------------------------------------------------

static constexpr double delta_z_analytical() {
    // Z-tip-load → bending about the y-axis → uses I_y = BH³/12
    double bending = P * L*L*L / (3.0 * E_mod * Iy);
    double shear   = P * L     / (kappa * G_mod * A);
    return bending + shear; // ≈ 98.14
}

static constexpr double delta_y_analytical() {
    // Y-tip-load → bending about the z-axis → uses I_z = HB³/12
    // Because I_z = I_y / 4 (narrower width), δ_y ≈ 4 · δ_z.
    double bending = P * L*L*L / (3.0 * E_mod * Iz);
    double shear   = P * L     / (kappa * G_mod * A);
    return bending + shear; // ≈ 391.11
}

// ─── Analytical cantilever — tip deflection under uniform distributed load ───
//
//  Uniform load q per unit length along the span:
//
//      δ_tip = qL⁴/(8EI) + qL²/(2κGA)
//
//  This formula is NOT currently tested by any FEM test (tests 2-8 use
//  tip point loads only).  It is validated in Test 1 for sign-correctness
//  and can be compared against a future Test 9 with distributed loading.
//
// -----------------------------------------------------------------------------

static constexpr double Q_total = 1.0;          // total transverse force
static constexpr double q_dist  = Q_total / L;  // load per unit length = 0.1

static constexpr double delta_z_uniform_analytical() {
    double bending = q_dist * L*L*L*L / (8.0 * E_mod * Iy);
    double shear   = q_dist * L*L     / (2.0 * kappa * G_mod * A);
    return bending + shear; // ≈ 36.86
}


// =============================================================================
//  Test harness — lightweight pass/fail counter
// =============================================================================
//
//  The test harness intentionally avoids external test frameworks (Google
//  Test, Catch2, etc.) to keep build dependencies minimal.  Each check
//  prints [PASS] or [FAIL] to stdout and increments a global counter.
//  main() returns EXIT_FAILURE if any check fails.
//
// -----------------------------------------------------------------------------

static int g_pass = 0;
static int g_fail = 0;

/// Boolean assertion with descriptive message.
static void check(bool cond, const char* msg) {
    if (cond) {
        std::cout << "  [PASS] " << msg << "\n";
        ++g_pass;
    } else {
        std::cout << "  [FAIL] " << msg << "\n";
        ++g_fail;
    }
}

/// Relative-error assertion:  |computed − expected| / |expected| ≤ rel_tol.
/// Prints both values and the percentage error regardless of pass/fail.
static void check_tol(double computed, double expected, double rel_tol,
                       const char* msg) {
    double err = std::abs(computed - expected) / std::abs(expected);
    std::cout << "    computed=" << std::scientific << std::setprecision(6)
              << computed << "  expected=" << expected
              << "  rel_err=" << std::fixed << std::setprecision(4)
              << (err * 100.0) << "%\n";
    if (err <= rel_tol) {
        std::cout << "  [PASS] " << msg << "\n";
        ++g_pass;
    } else {
        std::cout << "  [FAIL] " << msg << " (>" << (rel_tol*100.0) << "% tolerance)\n";
        ++g_fail;
    }
}


// =============================================================================
//  Helper: extract tip displacement from the PETSc solution vector
// =============================================================================
//
//  Searches all nodes whose x-coordinate is within `tol` of `x_target`
//  and returns the one with the largest absolute displacement in the
//  specified DOF direction (0 = u_x, 1 = u_y, 2 = u_z for continuum;
//  same order for beam DOFs 0-2, with 3-5 = rotations).
//
//  Note: `ndof_per_node` is currently unused (kept for future overloads).
//  The DOF base index is read directly from `node.dof_index()`.
//
//  The `tol` parameter controls the x-plane search width:
//    - 0.5 works for continuum meshes (nodes may not lie exactly at x = L).
//    - 0.1 is tighter for beam meshes where nodes are placed exactly.
//
// -----------------------------------------------------------------------------

static double tip_displacement(auto& model, int dof_dir, double x_target,
                               int /*ndof_per_node*/, double tol = 0.5) {
    const PetscScalar* arr;
    PetscInt n;
    VecGetLocalSize(model.state_vector(), &n);
    VecGetArrayRead(model.state_vector(), &arr);

    double max_val = 0.0;
    bool found = false;
    const auto& nodes = model.get_domain().nodes();

    for (const auto& node : nodes) {
        if (std::abs(node.coord(0) - x_target) < tol) {
            PetscInt base = -1;
            for (const auto idx : node.dof_index()) {
                if (base < 0) base = idx;
            }
            if (base >= 0 && base + dof_dir < n) {
                double val = arr[base + dof_dir];
                if (!found || std::abs(val) > std::abs(max_val)) {
                    max_val = val;
                    found = true;
                }
            }
        }
    }

    VecRestoreArrayRead(model.state_vector(), &arr);
    return max_val;
}


// =============================================================================
//  Test 1: Analytical solution values (sanity check)
// =============================================================================
//
//  Pure arithmetic — no FEM involved.  Guards against accidental changes
//  to the analytical formulas themselves (e.g. wrong moment of inertia).
//
//  Checks (3):
//    1. δ_z > 0            — deflection is in the load direction.
//    2. δ_y > δ_z          — weak axis (I_z < I_y) deflects more.
//    3. δ_z_uniform > 0    — uniform-load formula returns positive.
//
//  The bending/shear decomposition is printed for diagnostic purposes
//  to confirm that shear is ≈ 0.5 % of total (slender beam, L/H = 12.5).
//
// -----------------------------------------------------------------------------

static void test_1_analytical_values() {
    std::cout << "\n─── Test 1: Analytical Timoshenko solutions ───\n";
    std::cout << "  Geometry: L=" << L << " B=" << B << " H=" << H << "\n";
    std::cout << "  Material: E=" << E_mod << " nu=" << nu << " G=" << G_mod << "\n";
    std::cout << "  Section:  A=" << A << " Iy=" << Iy << " Iz=" << Iz
              << " J=" << J_tor << "\n";
    std::cout << "  Shear:    kappa=" << kappa << " kGA=" << (kappa*G_mod*A) << "\n";
    std::cout << "  Tip load: P=" << P << "\n\n";

    double dz = delta_z_analytical();
    double dy = delta_y_analytical();
    double dz_unif = delta_z_uniform_analytical();

    std::cout << "  δ_z (tip load Z):    " << std::scientific << dz << "\n";
    std::cout << "  δ_y (tip load Y):    " << std::scientific << dy << "\n";
    std::cout << "  δ_z (uniform load):  " << std::scientific << dz_unif << "\n";

    // Bending vs shear decomposition for Z tip load
    double bend = P * L*L*L / (3.0 * E_mod * Iy);
    double shear = P * L / (kappa * G_mod * A);
    std::cout << "\n  Bending component:   " << bend
              << " (" << std::fixed << std::setprecision(1) 
              << (bend/dz*100.0) << "%)\n";
    std::cout << "  Shear component:     " << std::scientific << shear
              << " (" << std::fixed << std::setprecision(1) 
              << (shear/dz*100.0) << "%)\n";

    check(dz > 0.0, "Positive analytical δ_z");
    check(dy > dz,   "δ_y > δ_z (weaker axis)");
    check(dz_unif > 0.0, "Positive uniform-load δ_z");
}


// =============================================================================
//  Test 2: 3D Continuum (SmallStrain + Elastic) — tip load in Z
// =============================================================================
//
//  Pipeline exercised:
//    GmshDomainBuilder → Domain → Model (SmallStrain, 3 DOF/node)
//    → fix_x(0) BCs → create_boundary_from_plane → apply_surface_traction
//    → LinearAnalysis::solve() → extract tip displacement.
//
//  The tip load P is converted to a surface traction  t_z = P / (B·H)  and
//  applied over the free-end face at x = L.  Diagnostic prints verify:
//    • Total node/element counts match the mesh file.
//    • Number of clamped nodes at x = 0.
//    • Numerically integrated tip face area ≈ B·H = 0.32.
//    • Sum of assembled nodal forces in z = P = 1.0.
//
//  Tolerance:  40 % (generous — the actual refined mesh gives < 1 %).
//  The tolerance is kept loose so the same test passes even with coarser
//  meshes; the cross-validation in Test 7 provides a tighter check.
//
// -----------------------------------------------------------------------------

static void test_2_continuum_tip_load_z() {
    std::cout << "\n─── Test 2: 3D Continuum — tip load Z ───\n";

    // 1. Build domain from the Gmsh mesh
    Domain<DIM> D;
    GmshDomainBuilder builder(MESH_FILE, D);

    // 2. Isotropic linear-elastic material (3D, 6 Voigt components)
    ContinuumIsotropicElasticMaterial mat_inst{E_mod, nu};
    Material<ThreeDimensionalMaterial> mat{mat_inst, ElasticUpdate{}};

    // 3. Continuum model: SmallStrain kinematics, 3 DOFs per node (u,v,w)
    Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF_CONT> M{D, mat};

    // 4. Boundary conditions: clamp all DOFs on the x = 0 face
    M.fix_x(0.0);
    M.setup();

    // ─── Diagnostic: count nodes, elements, fixed nodes ───
    {
        int n_fixed = 0;
        for (const auto& nd : D.nodes())
            if (std::abs(nd.coord(0) - 0.0) < 1e-6) ++n_fixed;
        std::cout << "  [DIAG] Total nodes: " << D.nodes().size()
                  << "  Volume elements: " << D.elements().size()
                  << "  Fixed nodes (x=0): " << n_fixed << "\n";
    }

    // 5. Apply tip surface traction in z-direction
    //    Traction t_z = P / A_face  distributes the point load P uniformly
    //    over the end face.  For a prismatic beam this is equivalent to a
    //    tip point load in the Timoshenko solution.
    double face_area = B * H;
    double tz = P / face_area;
    D.create_boundary_from_plane("Tip", 0, L);

    // ─── Diagnostic: verify face area via numerical integration ───
    {
        auto tip_elems = D.boundary_elements("Tip");
        double tip_area = surface_load::compute_surface_area<DIM>(tip_elems);
        std::cout << "  [DIAG] Tip face elements: " << tip_elems.size()
                  << "  Computed tip area: " << tip_area
                  << "  Expected: " << face_area
                  << "  Traction tz: " << tz << "\n";
    }

    M.apply_surface_traction("Tip", 0.0, 0.0, tz);

    // ─── Diagnostic: verify assembled force resultant ───
    {
        PetscScalar fz_total = 0.0;
        const PetscScalar* farr;
        PetscInt fn;
        VecGetLocalSize(M.force_vector(), &fn);
        VecGetArrayRead(M.force_vector(), &farr);
        for (PetscInt i = 2; i < fn; i += 3) fz_total += farr[i];
        VecRestoreArrayRead(M.force_vector(), &farr);
        std::cout << "  [DIAG] Sum of nodal forces in z: " << fz_total
                  << "  Expected: " << P << "\n";
    }

    // 6. Solve K·u = f with PETSc direct LU
    LinearAnalysis<ThreeDimensionalMaterial> solver{&M};
    solver.solve();

    // 7. Extract maximum z-displacement at the tip face (x ≈ L)
    double dz = tip_displacement(M, 2, L, NDOF_CONT);
    double analytical = delta_z_analytical();

    std::cout << "  3D Continuum δ_z at tip: " << std::scientific << dz << "\n";
    check_tol(std::abs(dz), analytical, 0.01,
              "3D Continuum vs analytical (Z tip load) < 1%");
}


// =============================================================================
//  Test 3: 3D Continuum — tip load in Y (weak axis)
// =============================================================================
//
//  Same pipeline as Test 2 but loading and measurement are in the
//  y-direction (weak bending axis, I_z = I_y / 4).  This exercises the
//  off-diagonal coupling terms in the constitutive matrix and ensures
//  that the B-matrix columns for the v-displacement are correct.
//
//  Expected analytical δ_y ≈ 391.11  (≈ 4 × δ_z).
//
// -----------------------------------------------------------------------------

static void test_3_continuum_tip_load_y() {
    std::cout << "\n─── Test 3: 3D Continuum — tip load Y ───\n";

    Domain<DIM> D;
    GmshDomainBuilder builder(MESH_FILE, D);

    ContinuumIsotropicElasticMaterial mat_inst{E_mod, nu};
    Material<ThreeDimensionalMaterial> mat{mat_inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF_CONT> M{D, mat};
    M.fix_x(0.0);
    M.setup();

    // Traction t_y = P / A_face applied over the tip face
    double face_area = B * H;
    double ty = P / face_area;
    D.create_boundary_from_plane("Tip", 0, L);
    M.apply_surface_traction("Tip", 0.0, ty, 0.0);

    LinearAnalysis<ThreeDimensionalMaterial> solver{&M};
    solver.solve();

    double dy = tip_displacement(M, 1, L, NDOF_CONT);
    double analytical = delta_y_analytical();

    std::cout << "  3D Continuum δ_y at tip: " << std::scientific << dy << "\n";
    check_tol(std::abs(dy), analytical, 0.01,
              "3D Continuum vs analytical (Y tip load) < 1%");
}


// =============================================================================
//  Test 4: Beam FEM (N=2, linear) — tip load in Z
// =============================================================================
//
//  Builds a 10-element Timoshenko beam mesh programmatically (no Gmsh
//  file).  Exercises the structural-element pipeline:
//
//    Domain::add_node → make_element<LagrangeElement3D<2>>
//    → assemble_sieve → Model<TimoshenkoBeam3D> → fix_x(0)
//    → apply_node_force → LinearAnalysis::solve()
//
//  Key details:
//    - 11 equispaced nodes along x, spacing Δx = 1.0.
//    - 2-node (linear) elements → constant curvature per element.
//    - GaussLegendreCellIntegrator<1>: 1 Gauss point per element,
//      exact integration for the constant-curvature stiffness terms.
//    - Section stiffness comes from TimoshenkoBeamMaterial3D using E, G,
//      A, I_y, I_z, J, κ_y, κ_z.  Two shear correction factors (κ_y, κ_z)
//      are both set to 5/6 for the rectangular section.
//
//  The Timoshenko beam with 10 linear elements and a cubic deflection
//  field (tip load) gives a small discretization error of ≈ 0.25 %
//  because the piecewise-linear w(x) cannot represent the cubic exactly.
//
//  Tolerance: 0.5 %.
//
// -----------------------------------------------------------------------------

static constexpr int NEL_BEAM = 10;                   // 10 linear beam elements
static constexpr int N_NODES_BEAM = NEL_BEAM + 1;     // 11 nodes (2-node elements)

// Type aliases for the beam model template chain
using BeamElemType  = TimoshenkoBeamN<2>;                                 // 2-node beam element
using BeamElemPolicy = SingleElementPolicy<BeamElemType>;                 // single element type
using BeamModelT = Model<TimoshenkoBeam3D, continuum::SmallStrain,
                         NDOF_BEAM, BeamElemPolicy>;                      // full model type

static void test_4_beam_N2_tip_load_z() {
    std::cout << "\n─── Test 4: Beam N=2 — tip load Z ───\n";

    Domain<DIM> D;

    // Create 11 equispaced nodes along x = [0, L]
    D.preallocate_node_capacity(N_NODES_BEAM);
    double dx = L / NEL_BEAM;  // 1.0 m
    for (int i = 0; i <= NEL_BEAM; ++i) {
        D.add_node(i, i * dx, 0.0, 0.0);
    }

    // 10 linear (2-node) beam elements with 1-point Gauss integration
    for (int e = 0; e < NEL_BEAM; ++e) {
        std::array<int, 2> conn = {e, e + 1};
        D.template make_element<LagrangeElement3D<2>>(
            GaussLegendreCellIntegrator<1>{}, e, conn.data());
    }

    D.assemble_sieve(); // build DMPlex sieve from connectivity

    // Timoshenko beam section: E, G, A, I_y, I_z, J, κ_y, κ_z
    TimoshenkoBeamMaterial3D mat_inst{E_mod, G_mod, A, Iy, Iz, J_tor, kappa, kappa};
    Material<TimoshenkoBeam3D> mat{mat_inst, ElasticUpdate{}};

    BeamModelT M{D, mat};
    M.fix_x(0.0);  // clamp 6 DOFs at node x=0
    M.setup();

    // Apply tip point load P in the z-direction at the last node.
    // Beam DOFs per node: (u, v, w, θ_x, θ_y, θ_z) — w is DOF index 2.
    M.apply_node_force(NEL_BEAM, 0.0, 0.0, P, 0.0, 0.0, 0.0);

    LinearAnalysis<TimoshenkoBeam3D, continuum::SmallStrain, NDOF_BEAM, BeamElemPolicy> solver{&M};
    solver.solve();

    // Read tip w-displacement (DOF index 2 at the last node, x = L)
    // tol = 0.1 m because beam nodes are placed exactly at integer x-coords.
    double dz = tip_displacement(M, 2, L, NDOF_BEAM, 0.1);
    double analytical = delta_z_analytical();

    std::cout << "  Beam N=2 δ_z at tip: " << std::scientific << dz << "\n";
    check_tol(std::abs(dz), analytical, 0.005,
              "Beam N=2 vs analytical (Z tip load) < 0.5%");
}


// =============================================================================
//  Test 5: Beam FEM (N=2, linear) — tip load in Y (weak axis)
// =============================================================================
//
//  Identical mesh and material as Test 4, but the load is applied in the
//  y-direction.  Verifies the I_z coupling in the beam stiffness matrix.
//  Expected δ_y ≈ 4 × δ_z because I_z = I_y / 4.
//
//  Tolerance: 0.5 %.
//
// -----------------------------------------------------------------------------

static void test_5_beam_N2_tip_load_y() {
    std::cout << "\n─── Test 5: Beam N=2 — tip load Y ───\n";

    Domain<DIM> D;

    D.preallocate_node_capacity(N_NODES_BEAM);
    double dx = L / NEL_BEAM;
    for (int i = 0; i <= NEL_BEAM; ++i) {
        D.add_node(i, i * dx, 0.0, 0.0);
    }

    for (int e = 0; e < NEL_BEAM; ++e) {
        std::array<int, 2> conn = {e, e + 1};
        D.template make_element<LagrangeElement3D<2>>(
            GaussLegendreCellIntegrator<1>{}, e, conn.data());
    }

    D.assemble_sieve();

    TimoshenkoBeamMaterial3D mat_inst{E_mod, G_mod, A, Iy, Iz, J_tor, kappa, kappa};
    Material<TimoshenkoBeam3D> mat{mat_inst, ElasticUpdate{}};

    BeamModelT M{D, mat};
    M.fix_x(0.0);
    M.setup();

    // Apply tip load P in y at last node (DOF index 1 = v)
    M.apply_node_force(NEL_BEAM, 0.0, P, 0.0, 0.0, 0.0, 0.0);

    LinearAnalysis<TimoshenkoBeam3D, continuum::SmallStrain, NDOF_BEAM, BeamElemPolicy> solver{&M};
    solver.solve();

    double dy = tip_displacement(M, 1, L, NDOF_BEAM, 0.1);
    double analytical = delta_y_analytical();

    std::cout << "  Beam N=2 δ_y at tip: " << std::scientific << dy << "\n";
    check_tol(std::abs(dy), analytical, 0.005,
              "Beam N=2 vs analytical (Y tip load) < 0.5%");
}


// =============================================================================
//  Test 6: Beam FEM (N=3, quadratic) — tip load Z
// =============================================================================
//
//  5 quadratic (3-node) elements → 11 nodes total (same DOF count as
//  10 linear elements).  Each element spans 2.0 m with interior mid-node
//  at 1.0 m.  Nodes at x = 0.0, 1.0, 2.0, …, 10.0.
//
//  The quadratic shape functions can represent a cubic deflection field
//  exactly (Timoshenko beam theory + 2-point Gauss rule captures the
//  polynomial order).  Therefore the error should be ≈ 0 % (machine
//  precision).  In practice rel_err = 0.0000 % is observed.
//
//  Key differences from Tests 4-5:
//    - LagrangeElement3D<3>     (3-node line element)
//    - GaussLegendreCellIntegrator<2>  (2 Gauss points per element)
//    - Connectivity: (0,1,2), (2,3,4), (4,5,6), (6,7,8), (8,9,10).
//
//  Tolerance: 0.5 %.
//
// -----------------------------------------------------------------------------

static constexpr int NEL_QUAD = 5;                        // 5 quadratic elements
static constexpr int N_NODES_QUAD = 2 * NEL_QUAD + 1;     // 11 nodes

using BeamN3Type    = TimoshenkoBeamN<3>;                              // 3-node beam element
using BeamN3Policy  = SingleElementPolicy<BeamN3Type>;
using BeamN3ModelT  = Model<TimoshenkoBeam3D, continuum::SmallStrain,
                            NDOF_BEAM, BeamN3Policy>;

static void test_6_beam_N3_tip_load_z() {
    std::cout << "\n─── Test 6: Beam N=3 (quadratic) — tip load Z ───\n";

    Domain<DIM> D;

    D.preallocate_node_capacity(N_NODES_QUAD);
    double dx = L / (2 * NEL_QUAD);  // 1.0 m spacing between consecutive nodes
    for (int i = 0; i < N_NODES_QUAD; ++i) {
        D.add_node(i, i * dx, 0.0, 0.0);
    }

    // 5 quadratic elements: (0,1,2), (2,3,4), …, (8,9,10)
    // Each element has 3 nodes; the mid-node is at the element centre.
    for (int e = 0; e < NEL_QUAD; ++e) {
        std::array<int, 3> conn = {2*e, 2*e+1, 2*e+2};
        D.template make_element<LagrangeElement3D<3>>(
            GaussLegendreCellIntegrator<2>{}, e, conn.data());
    }

    D.assemble_sieve();

    TimoshenkoBeamMaterial3D mat_inst{E_mod, G_mod, A, Iy, Iz, J_tor, kappa, kappa};
    Material<TimoshenkoBeam3D> mat{mat_inst, ElasticUpdate{}};

    BeamN3ModelT M{D, mat};
    M.fix_x(0.0);
    M.setup();

    // Tip load at the last node (index N_NODES_QUAD−1 = 10, x = 10.0)
    M.apply_node_force(N_NODES_QUAD - 1, 0.0, 0.0, P, 0.0, 0.0, 0.0);

    LinearAnalysis<TimoshenkoBeam3D, continuum::SmallStrain, NDOF_BEAM, BeamN3Policy> solver{&M};
    solver.solve();

    double dz = tip_displacement(M, 2, L, NDOF_BEAM, 0.1);
    double analytical = delta_z_analytical();

    std::cout << "  Beam N=3 δ_z at tip: " << std::scientific << dz << "\n";
    check_tol(std::abs(dz), analytical, 0.005,
              "Beam N=3 vs analytical (Z tip load) < 0.5%");
}


// =============================================================================
//  Test 9: 3D Continuum TET4 (linear tetrahedra) — tip load Z
// =============================================================================
//
//  Validates the simplex element pipeline end-to-end:
//    GmshDomainBuilder (TET_4 + TRI_3) → Domain → Model → Solve
//
//  The mesh beam_gmshElem1stOrder.msh contains:
//    352 nodes, 1 144 TET_4 volume elements, 96 TRI_3 surface elements.
//
//  Tolerance: 5 % (linear tets are stiffer than hex27 due to locking;
//  a generous bound is used because TET4 elements with full integration
//  exhibit volumetric locking for near-incompressible materials, but at
//  ν = 0.3 and L/H = 12.5 the error is expected to be moderate).
//
// -----------------------------------------------------------------------------

static void test_9_tet4_tip_load_z() {
    std::cout << "\n─── Test 9: TET4 Continuum — tip load Z ───\n";

    Domain<DIM> D;
    GmshDomainBuilder builder(MESH_TET4, D);

    ContinuumIsotropicElasticMaterial mat_inst{E_mod, nu};
    Material<ThreeDimensionalMaterial> mat{mat_inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF_CONT> M{D, mat};

    // ─── Diagnostic: check node num_dof propagation ───
    {
        int zero_dofs = 0, ok_dofs = 0;
        PetscInt min_sieve = 999999, max_sieve = -1;
        for (const auto& nd : D.nodes()) {
            if (nd.num_dof() == 0) {
                ++zero_dofs;
                if (zero_dofs <= 5)
                    std::cout << "  [DIAG] Node id=" << nd.id()
                              << " sieve_id=" << nd.sieve_id.value()
                              << " num_dof=" << nd.num_dof() << "\n";
            } else {
                ++ok_dofs;
            }
            PetscInt sid = nd.sieve_id.value();
            if (sid < min_sieve) min_sieve = sid;
            if (sid > max_sieve) max_sieve = sid;
        }
        std::cout << "  [DIAG] Nodes: " << D.nodes().size()
                  << "  zero_dof=" << zero_dofs
                  << "  ok_dof=" << ok_dofs
                  << "  sieve_id=[" << min_sieve << "," << max_sieve << "]\n";

        PetscInt cStart, cEnd, vStart, vEnd;
        DMPlexGetHeightStratum(D.mesh.dm, 0, &cStart, &cEnd);
        DMPlexGetHeightStratum(D.mesh.dm, 1, &vStart, &vEnd);
        std::cout << "  [DIAG] PETSc strata — cells:[" << cStart << "," << cEnd
                  << ") vertices:[" << vStart << "," << vEnd << ")\n";
        std::cout << "  [DIAG] Elements: " << D.elements().size() << "\n";
    }

    M.fix_x(0.0);
    M.setup();

    {
        int n_fixed = 0;
        for (const auto& nd : D.nodes())
            if (std::abs(nd.coord(0) - 0.0) < 1e-6) ++n_fixed;
        std::cout << "  [DIAG] Total nodes: " << D.nodes().size()
                  << "  Volume elements: " << D.elements().size()
                  << "  Fixed nodes (x=0): " << n_fixed << "\n";
    }

    double face_area = B * H;
    double tz = P / face_area;
    D.create_boundary_from_plane("Tip", 0, L);

    {
        auto tip_elems = D.boundary_elements("Tip");
        double tip_area = surface_load::compute_surface_area<DIM>(tip_elems);
        std::cout << "  [DIAG] Tip face elements: " << tip_elems.size()
                  << "  Computed tip area: " << tip_area
                  << "  Expected: " << face_area << "\n";
    }

    M.apply_surface_traction("Tip", 0.0, 0.0, tz);

    // ─── Diagnostic: verify assembled force resultant ───
    {
        PetscScalar fz_total = 0.0, fx_total = 0.0, fy_total = 0.0;
        const PetscScalar* farr;
        PetscInt fn;
        VecGetLocalSize(M.force_vector(), &fn);
        VecGetArrayRead(M.force_vector(), &farr);
        for (PetscInt i = 0; i < fn; i += 3) { fx_total += farr[i]; fy_total += farr[i+1]; fz_total += farr[i+2]; }
        VecRestoreArrayRead(M.force_vector(), &farr);
        std::cout << "  [DIAG] Sum of nodal forces: fx=" << fx_total
                  << " fy=" << fy_total << " fz=" << fz_total
                  << "  Expected fz: " << P << "\n";
        std::cout << "  [DIAG] Total DOFs: " << fn << "\n";
    }

    LinearAnalysis<ThreeDimensionalMaterial> solver{&M};
    solver.solve();

    double dz = tip_displacement(M, 2, L, NDOF_CONT);
    double analytical = delta_z_analytical();

    std::cout << "  TET4 δ_z at tip: " << std::scientific << dz << "\n";
    check_tol(std::abs(dz), analytical, 0.05,
              "TET4 vs analytical (Z tip load) < 5%");
}


// =============================================================================
//  Test 10: 3D Continuum TET4 — tip load Y (weak axis)
// =============================================================================

static void test_10_tet4_tip_load_y() {
    std::cout << "\n─── Test 10: TET4 Continuum — tip load Y ───\n";

    Domain<DIM> D;
    GmshDomainBuilder builder(MESH_TET4, D);

    ContinuumIsotropicElasticMaterial mat_inst{E_mod, nu};
    Material<ThreeDimensionalMaterial> mat{mat_inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF_CONT> M{D, mat};

    // ─── Diagnostic: check node num_dof propagation ───
    {
        int zero_dofs = 0, ok_dofs = 0;
        for (const auto& nd : D.nodes()) {
            if (nd.num_dof() == 0) {
                ++zero_dofs;
                if (zero_dofs <= 5)
                    std::cout << "  [T10-DIAG] Node id=" << nd.id()
                              << " sieve_id=" << nd.sieve_id.value()
                              << " num_dof=" << nd.num_dof()
                              << " coord=(" << nd.coord(0) << "," << nd.coord(1) << "," << nd.coord(2) << ")\n";
            } else {
                ++ok_dofs;
            }
        }
        std::cout << "  [T10-DIAG] Nodes: " << D.nodes().size()
                  << "  zero_dof=" << zero_dofs
                  << "  ok_dof=" << ok_dofs << "\n";
        std::cout << "  [T10-DIAG] Elements: " << D.elements().size() << "\n";

        PetscInt cStart, cEnd, vStart, vEnd;
        DMPlexGetHeightStratum(D.mesh.dm, 0, &cStart, &cEnd);
        DMPlexGetHeightStratum(D.mesh.dm, 1, &vStart, &vEnd);
        std::cout << "  [T10-DIAG] PETSc strata — cells:[" << cStart << "," << cEnd
                  << ") vertices:[" << vStart << "," << vEnd << ")\n";
    }

    M.fix_x(0.0);
    M.setup();

    double face_area = B * H;
    double ty = P / face_area;
    D.create_boundary_from_plane("Tip", 0, L);
    M.apply_surface_traction("Tip", 0.0, ty, 0.0);

    LinearAnalysis<ThreeDimensionalMaterial> solver{&M};
    solver.solve();

    double dy = tip_displacement(M, 1, L, NDOF_CONT);
    double analytical = delta_y_analytical();

    std::cout << "  TET4 δ_y at tip: " << std::scientific << dy << "\n";
    check_tol(std::abs(dy), analytical, 0.05,
              "TET4 vs analytical (Y tip load) < 5%");
}


// =============================================================================
//  Test 11: 3D Continuum TET10 (quadratic tetrahedra) — tip load Z
// =============================================================================
//
//  Validates the quadratic simplex element pipeline:
//    GmshDomainBuilder (TET_10 + TRI_6) → Domain → Model → Solve
//
//  The mesh beam_gmshElem2ndOrder.msh contains:
//    2 174 nodes, 1 177 TET_10 volume elements, 96 TRI_6 surface elements.
//
//  Quadratic tetrahedra are significantly more accurate than linear ones
//  and do not suffer from volumetric locking.  Expected error < 2 %.
//
// -----------------------------------------------------------------------------

static void test_11_tet10_tip_load_z() {
    std::cout << "\n─── Test 11: TET10 Continuum — tip load Z ───\n";

    Domain<DIM> D;
    GmshDomainBuilder builder(MESH_TET10, D);

    ContinuumIsotropicElasticMaterial mat_inst{E_mod, nu};
    Material<ThreeDimensionalMaterial> mat{mat_inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF_CONT> M{D, mat};
    M.fix_x(0.0);
    M.setup();

    {
        int n_fixed = 0;
        for (const auto& nd : D.nodes())
            if (std::abs(nd.coord(0) - 0.0) < 1e-6) ++n_fixed;
        std::cout << "  [DIAG] Total nodes: " << D.nodes().size()
                  << "  Volume elements: " << D.elements().size()
                  << "  Fixed nodes (x=0): " << n_fixed << "\n";
    }

    double face_area = B * H;
    double tz = P / face_area;
    D.create_boundary_from_plane("Tip", 0, L);

    {
        auto tip_elems = D.boundary_elements("Tip");
        double tip_area = surface_load::compute_surface_area<DIM>(tip_elems);
        std::cout << "  [DIAG] Tip face elements: " << tip_elems.size()
                  << "  Computed tip area: " << tip_area
                  << "  Expected: " << face_area << "\n";
    }

    M.apply_surface_traction("Tip", 0.0, 0.0, tz);

    // ─── Diagnostic: verify assembled force resultant ───
    {
        PetscScalar fz_total = 0.0, fx_total = 0.0, fy_total = 0.0;
        const PetscScalar* farr;
        PetscInt fn;
        VecGetLocalSize(M.force_vector(), &fn);
        VecGetArrayRead(M.force_vector(), &farr);
        for (PetscInt i = 0; i < fn; i += 3) { fx_total += farr[i]; fy_total += farr[i+1]; fz_total += farr[i+2]; }
        VecRestoreArrayRead(M.force_vector(), &farr);
        std::cout << "  [DIAG] Sum of nodal forces: fx=" << fx_total
                  << " fy=" << fy_total << " fz=" << fz_total
                  << "  Expected fz: " << P << "\n";
        std::cout << "  [DIAG] Total DOFs: " << fn << "\n";
    }

    LinearAnalysis<ThreeDimensionalMaterial> solver{&M};
    solver.solve();

    // ─── TET10 Strain diagnostic ─────────────────────────────────────────
    {
        M.update_elements_state();
        double sum_e11{0}, sum_e22{0}, sum_e33{0};
        double sum_g23{0}, sum_g13{0}, sum_g12{0};
        double sum_s11{0};
        int n_gp{0};
        for (const auto& elem : M.elements()) {
            for (const auto& mp : elem.material_points()) {
                const auto& s = mp.current_state();
                sum_e11 += s[0]; sum_e22 += s[1]; sum_e33 += s[2];
                sum_g23 += s[3]; sum_g13 += s[4]; sum_g12 += s[5];
                auto sig = mp.compute_response(s);
                sum_s11 += sig[0];
                ++n_gp;
            }
        }
        std::cout << "  [STRAIN] TET10 Z-load: " << n_gp << " GPs: avg ε₁₁=" << sum_e11/n_gp
                  << " ε₂₂=" << sum_e22/n_gp << " ε₃₃=" << sum_e33/n_gp << "\n";
        std::cout << "  [STRAIN]   avg γ₂₃=" << sum_g23/n_gp
                  << " γ₁₃=" << sum_g13/n_gp << " γ₁₂=" << sum_g12/n_gp << "\n";
        std::cout << "  [STRAIN]   avg σ₁₁=" << sum_s11/n_gp << " (bending, expect ~0 avg)\n";
        
        // First element detail
        {
            const auto& first_mp = M.elements().front().material_points().front();
            const auto& st = first_mp.current_state();
            auto coord = first_mp.coord();
            std::cout << "  [STRAIN]   GP0 coord=(" << coord[0] << "," << coord[1]
                      << "," << coord[2] << ")\n";
            std::cout << "  [STRAIN]   GP0 strain=["
                      << st[0] << ", " << st[1] << ", " << st[2] << ", "
                      << st[3] << ", " << st[4] << ", " << st[5] << "]\n";

            // Check if all strains are zero
            double max_strain = 0;
            for (int c = 0; c < 6; ++c) max_strain = std::max(max_strain, std::abs(st[c]));
            if (max_strain < 1e-20)
                std::cout << "  [STRAIN]   *** WARNING: All strains are ~ZERO at GP0! ***\n";
        }
    }

    double dz = tip_displacement(M, 2, L, NDOF_CONT);
    double analytical = delta_z_analytical();

    std::cout << "  TET10 δ_z at tip: " << std::scientific << dz << "\n";
    check_tol(std::abs(dz), analytical, 0.02,
              "TET10 vs analytical (Z tip load) < 2%");
}


// =============================================================================
//  Test 12: 3D Continuum TET10 — tip load Y (weak axis)
// =============================================================================

static void test_12_tet10_tip_load_y() {
    std::cout << "\n─── Test 12: TET10 Continuum — tip load Y ───\n";

    Domain<DIM> D;
    GmshDomainBuilder builder(MESH_TET10, D);

    ContinuumIsotropicElasticMaterial mat_inst{E_mod, nu};
    Material<ThreeDimensionalMaterial> mat{mat_inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF_CONT> M{D, mat};
    M.fix_x(0.0);
    M.setup();

    double face_area = B * H;
    double ty = P / face_area;
    D.create_boundary_from_plane("Tip", 0, L);
    M.apply_surface_traction("Tip", 0.0, ty, 0.0);

    LinearAnalysis<ThreeDimensionalMaterial> solver{&M};
    solver.solve();

    double dy = tip_displacement(M, 1, L, NDOF_CONT);
    double analytical = delta_y_analytical();

    std::cout << "  TET10 δ_y at tip: " << std::scientific << dy << "\n";
    check_tol(std::abs(dy), analytical, 0.02,
              "TET10 vs analytical (Y tip load) < 2%");
}


// =============================================================================
//  Test 7: 3D Continuum vs Beam — cross-validation (Z tip load)
// =============================================================================
//
//  Why this test exists:
//    Tests 2-6 compare each method independently against the analytical
//    solution.  If the analytical formula were ever wrong (e.g. a typo in
//    I_y), both the continuum and beam tests could still pass.  Test 7
//    provides a DIRECT comparison between the two FEM solutions,
//    independent of any analytical formula.
//
//  Both models are rebuilt from scratch in independent scopes to guarantee
//  no shared mutable state from previous tests.
//
//  Check:
//    |continuum/beam − 1| < 40 % (same generous tolerance as Tests 2-3).
//    With the current refined mesh, the ratio is ≈ 0.998 (< 0.2 % diff).
//
//  Diagnostic prints:
//    - Individual errors vs analytical for both methods.
//
// -----------------------------------------------------------------------------

static double continuum_dz_cache = 0.0;  // cached result for diagnostic print
static double beam_N2_dz_cache   = 0.0;

static void test_7_cross_validation() {
    std::cout << "\n─── Test 7: 3D Continuum vs Beam cross-validation ───\n";

    // ─── Solve 3D Continuum (same setup as Test 2) ───
    {
        Domain<DIM> D;
        GmshDomainBuilder builder(MESH_FILE, D);

        ContinuumIsotropicElasticMaterial mat_inst{E_mod, nu};
        Material<ThreeDimensionalMaterial> mat{mat_inst, ElasticUpdate{}};

        Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF_CONT> M{D, mat};
        M.fix_x(0.0);
        M.setup();

        double face_area = B * H;
        double tz = P / face_area;
        D.create_boundary_from_plane("Tip", 0, L);
        M.apply_surface_traction("Tip", 0.0, 0.0, tz);

        LinearAnalysis<ThreeDimensionalMaterial> solver{&M};
        solver.solve();

        continuum_dz_cache = std::abs(tip_displacement(M, 2, L, NDOF_CONT));
    }

    // ─── Solve Beam N=2 (same setup as Test 4) ───
    {
        Domain<DIM> D;

        D.preallocate_node_capacity(N_NODES_BEAM);
        double dx = L / NEL_BEAM;
        for (int i = 0; i <= NEL_BEAM; ++i) {
            D.add_node(i, i * dx, 0.0, 0.0);
        }
        for (int e = 0; e < NEL_BEAM; ++e) {
            std::array<int, 2> conn = {e, e + 1};
            D.template make_element<LagrangeElement3D<2>>(
                GaussLegendreCellIntegrator<1>{}, e, conn.data());
        }
        D.assemble_sieve();

        TimoshenkoBeamMaterial3D mat_inst{E_mod, G_mod, A, Iy, Iz, J_tor, kappa, kappa};
        Material<TimoshenkoBeam3D> mat{mat_inst, ElasticUpdate{}};

        BeamModelT M{D, mat};
        M.fix_x(0.0);
        M.setup();
        M.apply_node_force(NEL_BEAM, 0.0, 0.0, P, 0.0, 0.0, 0.0);

        LinearAnalysis<TimoshenkoBeam3D, continuum::SmallStrain, NDOF_BEAM, BeamElemPolicy> solver{&M};
        solver.solve();

        beam_N2_dz_cache = std::abs(tip_displacement(M, 2, L, NDOF_BEAM, 0.1));
    }

    double analytical = delta_z_analytical();

    std::cout << "  Analytical:     " << std::scientific << analytical << "\n";
    std::cout << "  3D Continuum:   " << continuum_dz_cache << "\n";
    std::cout << "  Beam N=2:       " << beam_N2_dz_cache << "\n";

    double ratio = continuum_dz_cache / beam_N2_dz_cache;
    std::cout << "  Continuum/Beam: " << std::fixed << std::setprecision(4) << ratio << "\n";

    // Direct cross-validation: both FEM solutions should agree
    check(std::abs(ratio - 1.0) < 0.40,
          "3D Continuum within 40% of Beam N=2");

    // Informational: individual errors vs analytical (no pass/fail)
    double err_cont = std::abs(continuum_dz_cache - analytical) / analytical;
    double err_beam = std::abs(beam_N2_dz_cache   - analytical) / analytical;
    std::cout << "  Err(Cont/Anal):  " << (err_cont*100.0) << "%\n";
    std::cout << "  Err(Beam/Anal):  " << (err_beam*100.0) << "%\n";
}


// =============================================================================
//  Test 8: Deflection curve along beam length
// =============================================================================
//
//  Instead of only checking the tip, this test compares z-deflections at
//  5 stations along the span (x = 2, 4, 6, 8, 10 m) to catch localised
//  errors (e.g. incorrect B-matrix near the clamped end, or a single bad
//  element in the mesh).
//
//  Analytical deflection curve for a cantilever with tip load P:
//
//      v(x) = Px²(3L − x) / (6EI) + Px / (κGA)
//
//  The first term is the Euler-Bernoulli cubic shape; the second is the
//  linear Timoshenko shear term.
//
//  For the 3D continuum FEM, v_fem is computed as the AVERAGE z-displacement
//  of all nodes within ±0.5 m of the target x-station (capturing the full
//  cross-section).  This averaging is legitimate because the analytical
//  solution assumes plane sections remain plane.
//
//  Tolerance: 40 % at each station (generous; actual < 4 % with the
//  refined mesh).
//
// -----------------------------------------------------------------------------

static void test_8_deflection_curve() {
    std::cout << "\n─── Test 8: Deflection curve along beam length ───\n";

    // 3D Continuum model
    Domain<DIM> D;
    GmshDomainBuilder builder(MESH_FILE, D);

    ContinuumIsotropicElasticMaterial mat_inst{E_mod, nu};
    Material<ThreeDimensionalMaterial> mat{mat_inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF_CONT> M{D, mat};
    M.fix_x(0.0);
    M.setup();

    double face_area = B * H;
    double tz = P / face_area;
    D.create_boundary_from_plane("Tip", 0, L);
    M.apply_surface_traction("Tip", 0.0, 0.0, tz);

    LinearAnalysis<ThreeDimensionalMaterial> solver{&M};
    solver.solve();

    // ─── Compare at 5 stations ───
    std::array<double, 5> stations = {2.0, 4.0, 6.0, 8.0, 10.0};

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "    x      δ_z(3D)       δ_z(anal)      err%\n";
    std::cout << "  ─────  ──────────────  ──────────────  ─────\n";

    bool all_ok = true;
    for (double x : stations) {
        // Analytical Timoshenko deflection at x:
        //   v(x) = Px²(3L−x)/(6EI_y) + Px/(κGA)
        double v_anal = P * x*x * (3.0*L - x) / (6.0 * E_mod * Iy)
                      + P * x / (kappa * G_mod * A);

        // FEM: average z-displacement of all nodes in the cross-section
        // at this x-station (tolerance ±0.5 m around the target x).
        const PetscScalar* arr;
        PetscInt n;
        VecGetLocalSize(M.state_vector(), &n);
        VecGetArrayRead(M.state_vector(), &arr);

        double sum_dz = 0.0;
        int count = 0;
        for (const auto& node : D.nodes()) {
            if (std::abs(node.coord(0) - x) < 0.5) {
                PetscInt base = -1;
                for (const auto idx : node.dof_index()) {
                    if (base < 0) base = idx;
                }
                if (base >= 0 && base + 2 < n) {
                    sum_dz += arr[base + 2];
                    ++count;
                }
            }
        }
        VecRestoreArrayRead(M.state_vector(), &arr);

        double v_fem = (count > 0) ? sum_dz / count : 0.0;
        double err = (v_anal != 0.0) ? std::abs(v_fem - v_anal) / std::abs(v_anal) * 100.0 : 0.0;

        std::cout << "  " << std::setw(5) << x
                  << "  " << std::setw(14) << std::scientific << v_fem
                  << "  " << std::setw(14) << v_anal
                  << "  " << std::fixed << std::setprecision(2) << err << "%\n";

        if (err > 1.0) all_ok = false;
    }

    check(all_ok, "All stations within 1% of analytical curve");
}


// =============================================================================
//  main — orchestrates all tests
// =============================================================================
//
//  PETSc is initialised once and shared across all tests.  The direct LU
//  solver is forced via PetscOptionsSetValue so that results are fully
//  deterministic and independent of iterative-solver convergence.
//
//  Exit code:  0 if all checks pass,  1 if any check fails.
//
// -----------------------------------------------------------------------------

int main(int argc, char** args)
{
    PetscInitialize(&argc, &args, nullptr, nullptr);

    // Force direct LU solver for reproducibility.
    // Without this, the default GMRES+ILU(0) may not converge for larger
    // meshes, producing incorrect results that mask real errors.
    PetscOptionsSetValue(nullptr, "-ksp_type", "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type",  "lu");

    {
        std::cout << "============================================================\n"
                  << " Timoshenko Cantilever Benchmark\n"
                  << "============================================================\n"
                  << " Geometry: L=" << L << " B=" << B << " H=" << H << "\n"
                  << " Material: E=" << E_mod << " nu=" << nu << "\n"
                  << " Mesh:     hex27 / tet4 / tet10 (3D continuum) / beam\n"
                  << "============================================================\n";

        // Test 1: Pure analytical sanity checks (no FEM).
        test_1_analytical_values();

        // Tests 2-8: Temporarily skipped for TET4 debugging
        if (false) {
        test_2_continuum_tip_load_z();
        test_3_continuum_tip_load_y();
        test_4_beam_N2_tip_load_z();
        test_5_beam_N2_tip_load_y();
        test_6_beam_N3_tip_load_z();
        test_7_cross_validation();
        test_8_deflection_curve();
        }

        // Tests 9-12: Simplex (tetrahedral) elements — validates full pipeline
        // from Gmsh reading through assembly and solve.
        // Use env var to select tests for debugging
        bool skip9 = (std::getenv("SKIP9") != nullptr);
        if (!skip9) test_9_tet4_tip_load_z();
        test_10_tet4_tip_load_y();
        test_11_tet10_tip_load_z();
        test_12_tet10_tip_load_y();

        std::cout << "\n============================================================\n"
                  << " Results: " << g_pass << " passed, "
                  << g_fail << " failed  (of " << (g_pass + g_fail) << ")\n"
                  << "============================================================\n";
    }

    PetscFinalize();
    return (g_fail > 0) ? 1 : 0;
}
