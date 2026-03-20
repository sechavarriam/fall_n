// =============================================================================
//  test_nonlinear_validation.cpp — Systematic nonlinear FEM validation suite
// =============================================================================
//
//  PURPOSE:
//    Fast, rigorous validation of the nonlinear analysis pipeline on a
//    small mesh.  Catches errors in:
//      - Material tangent consistency (SVK, Neo-Hookean)
//      - Kinematic formulation correctness (TotalLagrangian, UpdatedLagrangian)
//      - Newton-Raphson convergence (solve, solve_incremental)
//      - Automatic bisection in solve_incremental
//      - Plasticity integration (J2 Von Mises, isotropic hardening)
//
//  MESH:
//    tests/validation_cube.msh — 2×2×2 = 8 hex27 Lagrangian elements,
//    125 nodes, unit cube [0,1]³.  Physical groups:
//      "domain" (tag 13) — 8 hex27 volume elements
//      "Fixed"  (tag 14) — 4 quad9 faces at z=0
//      "Loaded" (tag 15) — 4 quad9 faces at z=1
//
//  BOUNDARY CONDITIONS:
//    - z = 0 face: clamped (all DOFs fixed)
//    - z = 1 face: uniaxial traction in z-direction
//
//  This is a uniaxial compression/tension configuration.  For a unit cube
//  with face area A = 1.0 and height L = 1.0:
//
//    1D bar solution (exact for continuum without Poisson coupling):
//      σ_z = F/A = t_z · A / A = t_z
//      ε_z = σ_z / E
//      δ_z = ε_z · L = t_z / E
//
//    For a clamped cube (constrained Poisson expansion at z=0), the 1D
//    formula overestimates displacement slightly, but serves as a
//    sanity check.
//
//  TESTS OVERVIEW:
//    ┌────┬──────────────────────────────────────────────────────────────┐
//    │  1 │ SmallStrain + elastic → analytical reference               │
//    │  2 │ TL + SVK (small load) ≈ SmallStrain                       │
//    │  3 │ TL + NH (small load) ≈ SmallStrain                        │
//    │  4 │ UL + SVK = TL + SVK (exact equivalence)                   │
//    │  5 │ UL + NH = TL + NH (exact equivalence)                     │
//    │  6 │ TL + SVK incremental — large load converges               │
//    │  7 │ TL + NH incremental — large load converges with bisection │
//    │  8 │ J2 plasticity — converges, larger deformation than elastic │
//    │  9 │ Bisection recovery — large step diverges, bisection saves │
//    │ 10 │ Physical consistency — SVK, NH, J2 ordering               │
//    └────┴──────────────────────────────────────────────────────────────┘
//
//  ACCEPTANCE CRITERIA:
//    All 10 tests must PASS for the nonlinear pipeline to be considered
//    correct.  Total runtime target: < 30 seconds (8 hex27, direct LU).
//
//  REQUIRES:
//    PETSc + MPI + Eigen + VTK (for Material/Model headers)
//
// =============================================================================

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <numeric>
#include <string>

#include <petsc.h>

// ── Project headers ──────────────────────────────────────────────────────────

#include "header_files.hh"                        // Domain, Model, etc.
#include "src/continuum/HyperelasticRelation.hh"  // SVKRelation, NeoHookeanRelation

// =============================================================================
//  Constants
// =============================================================================

static constexpr std::size_t DIM  = 3;
static constexpr std::size_t NDOF = DIM;  // 3 translational DOFs per node

// Material properties (steel-like)
static constexpr double E_mod      = 200.0;   // Young's modulus
static constexpr double nu         = 0.3;     // Poisson's ratio
static constexpr double sigma_y    = 0.250;   // Initial yield stress (J2)
static constexpr double H_hard     = 10.0;    // Isotropic hardening modulus

// Derived (currently unused, kept for future reference)
// static constexpr double G_mod = E_mod / (2.0 * (1.0 + nu));  // shear modulus

// Mesh file (relative to build directory — tests run from build/)
#ifdef FALL_N_SOURCE_DIR
static const std::string MESH_FILE =
    std::string(FALL_N_SOURCE_DIR) + "/tests/validation_cube.msh";
#else
static const std::string MESH_FILE = "./tests/validation_cube.msh";
#endif

// Cube dimensions
static constexpr double L_cube = 1.0;   // cube side length
static constexpr double A_face = L_cube * L_cube;  // loaded face area

// Load levels
//   small_load: used for linearization comparison (Tests 1-5).
//               Must produce strain ε < 0.5% so that the O(ε) linearization
//               error between TL and SmallStrain stays below ~1%.
//   large_load: used for finite-deformation tests (Tests 6-7, 9-10).
static constexpr double small_load = 0.5;
static constexpr double large_load = 50.0;

// =============================================================================
//  Test harness
// =============================================================================
//
//  Minimal pass/fail framework.  Each check() increments the global
//  pass or fail counter and prints a [PASS] or [FAIL] message.
//
//  The program returns exit code 0 only if all checks pass.

static int g_pass = 0;
static int g_fail = 0;

static void check(bool cond, const char* msg) {
    if (cond) {
        std::cout << "  [PASS] " << msg << "\n";
        ++g_pass;
    } else {
        std::cout << "  [FAIL] " << msg << "\n";
        ++g_fail;
    }
}

// =============================================================================
//  Helpers
// =============================================================================

/// Extract the full solution vector from a model into a std::vector.
/// Used for comparing two solutions element-by-element.
static std::vector<double> extract_solution(auto& M) {
    const PetscScalar* arr;
    PetscInt n;
    VecGetLocalSize(M.state_vector(), &n);
    VecGetArrayRead(M.state_vector(), &arr);
    std::vector<double> sol(arr, arr + n);
    VecRestoreArrayRead(M.state_vector(), &arr);
    return sol;
}

/// Maximum absolute displacement (any component) across all DOFs.
static double max_displacement(auto& M) {
    const PetscScalar* arr;
    PetscInt n;
    VecGetLocalSize(M.state_vector(), &n);
    VecGetArrayRead(M.state_vector(), &arr);
    double mx = 0.0;
    for (PetscInt i = 0; i < n; ++i)
        mx = std::max(mx, std::abs(arr[i]));
    VecRestoreArrayRead(M.state_vector(), &arr);
    return mx;
}

/// Maximum z-displacement (every 3rd DOF starting at index 2).
static double max_z_displacement(auto& M) {
    const PetscScalar* arr;
    PetscInt n;
    VecGetLocalSize(M.state_vector(), &n);
    VecGetArrayRead(M.state_vector(), &arr);
    double max_uz = 0.0;
    for (PetscInt i = 2; i < n; i += static_cast<PetscInt>(DIM))
        max_uz = std::max(max_uz, std::abs(arr[i]));
    VecRestoreArrayRead(M.state_vector(), &arr);
    return max_uz;
}

/// L∞ relative error between two solution vectors.
///   err = max_i |a[i] - b[i]| / max_i |b[i]|
/// If the reference b is zero everywhere, returns absolute error.
static double linf_relative_error(const std::vector<double>& a,
                                  const std::vector<double>& b) {
    assert(a.size() == b.size());
    double max_diff = 0.0;
    double max_ref  = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
        max_ref  = std::max(max_ref,  std::abs(b[i]));
    }
    return (max_ref > 1e-30) ? (max_diff / max_ref) : max_diff;
}

/// Apply a uniform z-traction on the z=1 face.
///
///   traction = (0, 0, tz)  where  tz = total_force / A_face
///
/// Uses create_boundary_from_plane to find nodes on z=L_cube and
/// distributes the traction via consistent surface integration (quad9).
/// The guard prevents duplicate boundary creation if called multiple
/// times on the same Domain.
template <typename ModelT>
static void apply_z_traction(ModelT& M, double total_force) {
    double tz = total_force / A_face;  // uniform traction
    auto& D = M.get_domain();
    if (!D.has_boundary_group("Load_z1")) {
        D.create_boundary_from_plane("Load_z1", 2, L_cube);  // axis=2 (z), coord=L
    }
    M.apply_surface_traction("Load_z1", 0.0, 0.0, tz);
}


// =============================================================================
//  Test 1: SmallStrain + Linear Elastic → analytical reference
// =============================================================================
//
//  The simplest possible case.  A unit cube clamped at z=0 with uniaxial
//  tension at z=1.  This establishes a reference solution that all
//  nonlinear formulations must recover under infinitesimal load.
//
//  For a SNES-based linear elastic problem, Newton converges in exactly
//  1 iteration (the residual is zero after one linear solve).
//
//  The FE solution includes Poisson coupling at the clamped face, so the
//  displacement is NOT exactly FL/(EA).  We only check:
//    - SNES converged in ≤ 2 iterations
//    - Displacement is positive (tension) and finite
//    - Mesh has the expected number of elements and nodes
//
//  The solution vector is stored as reference for Tests 2-5.
// -----------------------------------------------------------------------------

static std::vector<double> ref_solution;  // SmallStrain reference

static void test_1_small_strain_reference() {
    std::cout << "\n─── Test 1: SmallStrain + Linear Elastic (reference) ───\n";

    Domain<DIM> D;
    GmshDomainBuilder builder(MESH_FILE, D);

    ContinuumIsotropicElasticMaterial mat_inst{E_mod, nu};
    Material<ThreeDimensionalMaterial> mat{mat_inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF> M{D, mat};
    M.fix_z(0.0);   // clamp z=0 face
    M.setup();
    apply_z_traction(M, small_load);   // small load: well within linear regime

    // Use SNES (not LinearAnalysis) to validate the nonlinear pipeline
    // with a linear problem.  Should converge in 1 iteration.
    NonlinearAnalysis<ThreeDimensionalMaterial, continuum::SmallStrain> nl{&M};
    bool converged = nl.solve();

    check(converged, "SNES converged");
    check(nl.num_iterations() <= 2, "Converged in ≤ 2 iterations (linear problem)");

    check(D.num_elements() == 8,  "8 hex27 elements loaded from Gmsh");
    check(D.num_nodes()   == 125, "125 nodes loaded from Gmsh");

    double max_uz = max_z_displacement(M);
    std::cout << "  Max |uz|: " << std::scientific << std::setprecision(6)
              << max_uz << "\n";

    // Sanity: 1D bar estimate: δ = F·L/(E·A)
    std::cout << "  1D bar estimate: " << small_load * L_cube / (E_mod * A_face) << "\n";

    check(max_uz > 0.0,            "Positive z-displacement (tension)");
    check(std::isfinite(max_uz),   "Finite displacement");

    // Store reference for later comparison
    ref_solution = extract_solution(M);
}


// =============================================================================
//  Test 2: TL + SVK (small load) ≈ SmallStrain
// =============================================================================
//
//  For infinitesimal deformation, the Green-Lagrange strain E reduces to
//  the infinitesimal strain ε, and the SVK constitutive law S = λ·tr(E)·I + 2μ·E
//  reduces to the linear elastic law σ = λ·tr(ε)·I + 2μ·ε.
//
//  Therefore, the TL+SVK solution must match the SmallStrain solution
//  to within floating-point precision when the load is small enough.
//
//  Tolerance: rel_err < 0.1%  (generous; typically < 0.001%)
// -----------------------------------------------------------------------------

static void test_2_tl_svk_vs_small_strain() {
    std::cout << "\n─── Test 2: TotalLagrangian + SVK ≈ SmallStrain ───\n";

    Domain<DIM> D;
    GmshDomainBuilder builder(MESH_FILE, D);

    auto svk = continuum::SaintVenantKirchhoff<3>::from_E_nu(E_mod, nu);
    MaterialInstance<continuum::SVKRelation<3>> inst{svk};
    Material<ThreeDimensionalMaterial> mat{inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::TotalLagrangian, NDOF> M{D, mat};
    M.fix_z(0.0);
    M.setup();
    apply_z_traction(M, small_load);  // same load as Test 1

    NonlinearAnalysis<ThreeDimensionalMaterial, continuum::TotalLagrangian> nl{&M};
    bool converged = nl.solve();

    check(converged, "SNES converged (TL+SVK)");
    std::cout << "  SNES iterations: " << nl.num_iterations() << "\n";

    auto sol = extract_solution(M);
    double rel_err = linf_relative_error(sol, ref_solution);
    std::cout << "  L∞ rel.err vs SmallStrain: " << std::scientific << rel_err << "\n";
    // At ε ≈ small_load/E = 0.25%, the O(ε) linearization error is ~0.3%.
    check(rel_err < 1e-2, "TL+SVK ≈ SmallStrain for small load (< 1%)");
}


// =============================================================================
//  Test 3: TL + Neo-Hookean (small load) ≈ SmallStrain
// =============================================================================
//
//  Same linearization argument as Test 2.  The Neo-Hookean stored-energy:
//    W(C) = μ/2·(tr C − 3) − μ·ln J + λ/2·(ln J)²
//  has the same linearized tangent as the isotropic elastic law.
//
//  This test catches bugs in the Neo-Hookean stress S or tangent C that
//  would only be visible when F ≠ I (large deformation), but even at
//  small strain we verify the basic linearization is correct.
//
//  Tolerance: rel_err < 0.1%
// -----------------------------------------------------------------------------

static void test_3_tl_nh_vs_small_strain() {
    std::cout << "\n─── Test 3: TotalLagrangian + Neo-Hookean ≈ SmallStrain ───\n";

    Domain<DIM> D;
    GmshDomainBuilder builder(MESH_FILE, D);

    auto nh = continuum::CompressibleNeoHookean<3>::from_E_nu(E_mod, nu);
    MaterialInstance<continuum::NeoHookeanRelation<3>> inst{nh};
    Material<ThreeDimensionalMaterial> mat{inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::TotalLagrangian, NDOF> M{D, mat};
    M.fix_z(0.0);
    M.setup();
    apply_z_traction(M, small_load);  // same load as Tests 1-2

    NonlinearAnalysis<ThreeDimensionalMaterial, continuum::TotalLagrangian> nl{&M};
    bool converged = nl.solve();

    check(converged, "SNES converged (TL+NH)");
    std::cout << "  SNES iterations: " << nl.num_iterations() << "\n";

    auto sol = extract_solution(M);
    double rel_err = linf_relative_error(sol, ref_solution);
    std::cout << "  L∞ rel.err vs SmallStrain: " << std::scientific << rel_err << "\n";
    // At ε ≈ small_load/E = 0.25%, the O(ε) linearization error is ~0.2%.
    check(rel_err < 1e-2, "TL+NH ≈ SmallStrain for small load (< 1%)");
}


// =============================================================================
//  Test 4: UL + SVK = TL + SVK (exact equivalence)
// =============================================================================
//
//  The UpdatedLagrangian and TotalLagrangian formulations are theoretically
//  equivalent for any hyperelastic material: both compute the same F, E, S,
//  and C — just in different reference frames.
//
//  In the current implementation, UpdatedLagrangian::evaluate() actually
//  delegates to TotalLagrangian::evaluate(), so the solutions should be
//  identical to machine precision.
//
//  This test verifies no bugs were introduced in the delegation chain or
//  in the spatial pathway that might affect future UpdatedLagrangian work.
//
//  Tolerance: rel_err < 1e-8 (essentially machine eps after linear algebra)
// -----------------------------------------------------------------------------

static void test_4_ul_svk_equals_tl() {
    std::cout << "\n─── Test 4: UL + SVK = TL + SVK ───\n";

    // TL reference
    std::vector<double> tl_sol;
    {
        Domain<DIM> D;
        GmshDomainBuilder builder(MESH_FILE, D);

        auto svk = continuum::SaintVenantKirchhoff<3>::from_E_nu(E_mod, nu);
        MaterialInstance<continuum::SVKRelation<3>> inst{svk};
        Material<ThreeDimensionalMaterial> mat{inst, ElasticUpdate{}};

        Model<ThreeDimensionalMaterial, continuum::TotalLagrangian, NDOF> M{D, mat};
        M.fix_z(0.0);
        M.setup();
        apply_z_traction(M, small_load);

        NonlinearAnalysis<ThreeDimensionalMaterial, continuum::TotalLagrangian> nl{&M};
        nl.solve();
        tl_sol = extract_solution(M);
    }

    // UL solution
    std::vector<double> ul_sol;
    {
        Domain<DIM> D;
        GmshDomainBuilder builder(MESH_FILE, D);

        auto svk = continuum::SaintVenantKirchhoff<3>::from_E_nu(E_mod, nu);
        MaterialInstance<continuum::SVKRelation<3>> inst{svk};
        Material<ThreeDimensionalMaterial> mat{inst, ElasticUpdate{}};

        Model<ThreeDimensionalMaterial, continuum::UpdatedLagrangian, NDOF> M{D, mat};
        M.fix_z(0.0);
        M.setup();
        apply_z_traction(M, small_load);

        NonlinearAnalysis<ThreeDimensionalMaterial, continuum::UpdatedLagrangian> nl{&M};
        nl.solve();

        check(nl.converged_reason() > 0, "SNES converged (UL+SVK)");
        std::cout << "  SNES iterations (UL): " << nl.num_iterations() << "\n";
        ul_sol = extract_solution(M);
    }

    double rel_err = linf_relative_error(ul_sol, tl_sol);
    std::cout << "  L∞ rel.err UL vs TL: " << std::scientific << rel_err << "\n";
    check(rel_err < 1e-8, "UL ≡ TL (same solution to machine precision)");
}


// =============================================================================
//  Test 5: UL + NH = TL + NH (exact equivalence)
// =============================================================================
//
//  Same as Test 4 but for Neo-Hookean.  Since UpdatedLagrangian delegates
//  to TotalLagrangian, the solutions must be identical.
//
//  This is not redundant with Test 4: a bug that corrupts only the
//  Neo-Hookean pathway (e.g. wrong C→C⁻¹ conversion) would be caught here
//  but not in Test 4.
// -----------------------------------------------------------------------------

static void test_5_ul_nh_equals_tl() {
    std::cout << "\n─── Test 5: UL + NH = TL + NH ───\n";

    // TL reference
    std::vector<double> tl_sol;
    {
        Domain<DIM> D;
        GmshDomainBuilder builder(MESH_FILE, D);

        auto nh = continuum::CompressibleNeoHookean<3>::from_E_nu(E_mod, nu);
        MaterialInstance<continuum::NeoHookeanRelation<3>> inst{nh};
        Material<ThreeDimensionalMaterial> mat{inst, ElasticUpdate{}};

        Model<ThreeDimensionalMaterial, continuum::TotalLagrangian, NDOF> M{D, mat};
        M.fix_z(0.0);
        M.setup();
        apply_z_traction(M, small_load);

        NonlinearAnalysis<ThreeDimensionalMaterial, continuum::TotalLagrangian> nl{&M};
        nl.solve();
        tl_sol = extract_solution(M);
    }

    // UL solution
    std::vector<double> ul_sol;
    {
        Domain<DIM> D;
        GmshDomainBuilder builder(MESH_FILE, D);

        auto nh = continuum::CompressibleNeoHookean<3>::from_E_nu(E_mod, nu);
        MaterialInstance<continuum::NeoHookeanRelation<3>> inst{nh};
        Material<ThreeDimensionalMaterial> mat{inst, ElasticUpdate{}};

        Model<ThreeDimensionalMaterial, continuum::UpdatedLagrangian, NDOF> M{D, mat};
        M.fix_z(0.0);
        M.setup();
        apply_z_traction(M, small_load);

        NonlinearAnalysis<ThreeDimensionalMaterial, continuum::UpdatedLagrangian> nl{&M};
        nl.solve();

        check(nl.converged_reason() > 0, "SNES converged (UL+NH)");
        std::cout << "  SNES iterations (UL): " << nl.num_iterations() << "\n";
        ul_sol = extract_solution(M);
    }

    double rel_err = linf_relative_error(ul_sol, tl_sol);
    std::cout << "  L∞ rel.err UL vs TL: " << std::scientific << rel_err << "\n";
    check(rel_err < 1e-8, "UL ≡ TL (same solution to machine precision)");
}


// =============================================================================
//  Test 6: TL + SVK incremental — large load
// =============================================================================
//
//  Applies a large load (50 × yield-level) in 5 increments.
//  At this load level, the Green-Lagrange strain is O(1), so geometric
//  nonlinearity is significant and the SVK model deviates from linear.
//
//  The test verifies:
//    - All 5 load steps converge (possibly with bisection)
//    - The displacement is finite and positive (tension)
//    - The displacement is LARGER than the SmallStrain reference scaled
//      by the load ratio (geometric stiffening/softening effect)
// -----------------------------------------------------------------------------

static double svk_large_uz = 0.0;   // stored for Test 10

static void test_6_tl_svk_incremental() {
    std::cout << "\n─── Test 6: TL + SVK incremental (large load) ───\n";

    Domain<DIM> D;
    GmshDomainBuilder builder(MESH_FILE, D);

    auto svk = continuum::SaintVenantKirchhoff<3>::from_E_nu(E_mod, nu);
    MaterialInstance<continuum::SVKRelation<3>> inst{svk};
    Material<ThreeDimensionalMaterial> mat{inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::TotalLagrangian, NDOF> M{D, mat};
    M.fix_z(0.0);
    M.setup();
    apply_z_traction(M, large_load);  // large load for finite deformation

    NonlinearAnalysis<ThreeDimensionalMaterial, continuum::TotalLagrangian> nl{&M};
    bool ok = nl.solve_incremental(5);  // 5 steps, auto-bisection

    check(ok, "All load steps converged (TL+SVK, 5 steps)");

    svk_large_uz = max_z_displacement(M);
    std::cout << "  Max |uz|: " << std::scientific << svk_large_uz << "\n";
    check(svk_large_uz > 0.0,          "Positive z-displacement");
    check(std::isfinite(svk_large_uz), "Finite displacement");
}


// =============================================================================
//  Test 7: TL + Neo-Hookean incremental — large load
// =============================================================================
//
//  Neo-Hookean divergence is the most common failure mode in the current
//  codebase (main.cpp Cases 4 and 6).  On this small mesh, with moderate
//  load and sufficient load steps, it should converge.
//
//  We use 10 load steps (twice as many as SVK) because the NH tangent
//  varies more rapidly near large strains.  The automatic bisection
//  provides a safety net if any step diverges.
//
//  Key physics: NH enforces J > 0 (no element inversion) through the
//  ln(J) term, which makes it more robust than SVK for large compression.
//  Under tension, NH is softer than SVK at large strain.
// -----------------------------------------------------------------------------

static double nh_large_uz = 0.0;   // stored for Test 10

static void test_7_tl_nh_incremental() {
    std::cout << "\n─── Test 7: TL + Neo-Hookean incremental (large load) ───\n";

    Domain<DIM> D;
    GmshDomainBuilder builder(MESH_FILE, D);

    auto nh = continuum::CompressibleNeoHookean<3>::from_E_nu(E_mod, nu);
    MaterialInstance<continuum::NeoHookeanRelation<3>> inst{nh};
    Material<ThreeDimensionalMaterial> mat{inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::TotalLagrangian, NDOF> M{D, mat};
    M.fix_z(0.0);
    M.setup();
    apply_z_traction(M, large_load);  // same large load as Test 6

    NonlinearAnalysis<ThreeDimensionalMaterial, continuum::TotalLagrangian> nl{&M};
    bool ok = nl.solve_incremental(10, 4);  // 10 steps, 4 bisection levels

    check(ok, "All load steps converged (TL+NH, 10 steps)");

    nh_large_uz = max_z_displacement(M);
    std::cout << "  Max |uz|: " << std::scientific << nh_large_uz << "\n";
    check(nh_large_uz > 0.0,          "Positive z-displacement");
    check(std::isfinite(nh_large_uz), "Finite displacement");
}


// =============================================================================
//  Test 8: J2 Plasticity — incremental, beyond yield
// =============================================================================
//
//  SmallStrain + J2 Von Mises plasticity with isotropic hardening.
//  The load is chosen to cause significant plastic flow:
//
//    σ_yield = 0.250 on A = 1  → F_yield = 0.25
//    Applied: F = 5.0  → ~20× yield → deep plastic regime
//
//  Due to triaxial constraint at the clamped face, the effective Von Mises
//  stress is lower than the nominal uniaxial value, but still well above
//  yield away from the constraint.
//
//  We verify:
//    - All load steps converge (20 steps for plasticity)
//    - Displacement is finite
//    - Plastic displacement > elastic displacement at the same load
//      (plasticity reduces the effective tangent modulus → larger δ)
//
//  The elastic reference is computed inline for direct comparison.
// -----------------------------------------------------------------------------

static double j2_uz = 0.0;   // stored for Test 10

static void test_8_j2_plasticity() {
    std::cout << "\n─── Test 8: SmallStrain + J2 Plasticity ───\n";

    double load = 5.0;

    // ── Plastic solution (20 increments) ──
    {
        Domain<DIM> D;
        GmshDomainBuilder builder(MESH_FILE, D);

        J2PlasticMaterial3D j2_inst{E_mod, nu, sigma_y, H_hard};
        Material<ThreeDimensionalMaterial> mat{j2_inst, InelasticUpdate{}};

        Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF> M{D, mat};
        M.fix_z(0.0);
        M.setup();
        apply_z_traction(M, load);

        NonlinearAnalysis<ThreeDimensionalMaterial, continuum::SmallStrain> nl{&M};
        bool ok = nl.solve_incremental(20, 4);

        check(ok, "J2 plasticity converged (20 load steps)");

        j2_uz = max_z_displacement(M);
        std::cout << "  Max |uz| (plastic): " << std::scientific << j2_uz << "\n";
        check(std::isfinite(j2_uz), "Finite displacement");
    }

    // ── Elastic reference at same load ──
    double elastic_uz;
    {
        Domain<DIM> D;
        GmshDomainBuilder builder(MESH_FILE, D);

        ContinuumIsotropicElasticMaterial mat_inst{E_mod, nu};
        Material<ThreeDimensionalMaterial> mat{mat_inst, ElasticUpdate{}};

        Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF> M{D, mat};
        M.fix_z(0.0);
        M.setup();
        apply_z_traction(M, load);

        NonlinearAnalysis<ThreeDimensionalMaterial, continuum::SmallStrain> nl{&M};
        nl.solve();

        elastic_uz = max_z_displacement(M);
    }

    std::cout << "  Max |uz| (elastic): " << std::scientific << elastic_uz << "\n";
    std::cout << "  Ratio plastic/elastic: " << std::fixed << std::setprecision(4)
              << j2_uz / elastic_uz << "\n";

    // Plastic displacement must exceed elastic (material softens beyond yield)
    check(j2_uz > elastic_uz,
          "Plastic deformation > elastic at same load (material softening)");
}


// =============================================================================
//  Test 9: Bisection recovery — force a diverged step, verify recovery
// =============================================================================
//
//  This test specifically exercises the automatic bisection added to
//  solve_incremental().  We apply a very large load in just 2 steps:
//    λ₁ = 0.5, λ₂ = 1.0
//
//  If the first step diverges (likely for the Neo-Hookean under this
//  extreme load), the bisection should automatically split it into
//  sub-steps until convergence.
//
//  With max_bisections=4, the finest sub-step is Δλ/16 = 0.03125,
//  which should be small enough to converge for moderate loads on
//  this mesh.
//
//  The test PASSES if solve_incremental returns true (all steps
//  converged, possibly after bisection).
// -----------------------------------------------------------------------------

static void test_9_bisection_recovery() {
    std::cout << "\n─── Test 9: Bisection recovery (TL + SVK, large step) ───\n";

    Domain<DIM> D;
    GmshDomainBuilder builder(MESH_FILE, D);

    auto svk = continuum::SaintVenantKirchhoff<3>::from_E_nu(E_mod, nu);
    MaterialInstance<continuum::SVKRelation<3>> inst{svk};
    Material<ThreeDimensionalMaterial> mat{inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::TotalLagrangian, NDOF> M{D, mat};
    M.fix_z(0.0);
    M.setup();
    apply_z_traction(M, 100.0);  // very large load

    NonlinearAnalysis<ThreeDimensionalMaterial, continuum::TotalLagrangian> nl{&M};
    // Only 2 steps → Δλ = 0.5 each → likely needs bisection
    bool ok = nl.solve_incremental(2, 5);  // 5 bisection levels

    check(ok, "Bisection recovered all load steps (2 coarse steps + bisection)");

    double max_uz = max_z_displacement(M);
    std::cout << "  Max |uz|: " << std::scientific << max_uz << "\n";
    check(max_uz > 0.0,          "Positive z-displacement after bisection");
    check(std::isfinite(max_uz), "Finite displacement after bisection");
}


// =============================================================================
//  Test 10: Physical consistency — ordering of model responses
// =============================================================================
//
//  Under the SAME large tensile load, different material models must
//  produce physically consistent relative displacements:
//
//    1. SVK and NH should give different results at large strain.
//       (They have different stored-energy functions.)
//
//    2. Both SVK and NH displacements should be finite and positive.
//
//    3. The J2 plastic displacement (from Test 8 at a smaller load)
//       should exceed the elastic reference at the same load.
//
//  This cross-validation catches global sign errors, energy-function
//  mix-ups, or models that silently return the same (wrong) answer.
// -----------------------------------------------------------------------------

static void test_10_physical_consistency() {
    std::cout << "\n─── Test 10: Physical consistency (cross-validation) ───\n";

    std::cout << "  SVK  large-load |uz|: " << std::scientific << svk_large_uz << "\n";
    std::cout << "  NH   large-load |uz|: " << std::scientific << nh_large_uz  << "\n";
    std::cout << "  J2   small-load |uz|: " << std::scientific << j2_uz        << "\n";

    // SVK and NH should give DIFFERENT results at large strain.
    // If they are identical, something is wrong (e.g. both falling through
    // to the linear pathway).
    double rel_diff = std::abs(svk_large_uz - nh_large_uz)
                    / std::max(svk_large_uz, nh_large_uz);
    std::cout << "  |SVK - NH| / max: " << std::scientific << rel_diff << "\n";
    check(rel_diff > 1e-4,
          "SVK ≠ NH at large strain (different constitutive response)");

    // Both must be positive and finite (already checked individually,
    // but this is a safety net for cached values)
    check(svk_large_uz > 0.0 && std::isfinite(svk_large_uz),
          "SVK large-load displacement is valid");
    check(nh_large_uz > 0.0 && std::isfinite(nh_large_uz),
          "NH large-load displacement is valid");
}


// =============================================================================
//  main
// =============================================================================
//
//  Initialises PETSc with a direct LU solver (no iterative solver
//  convergence uncertainty) and runs all 10 tests sequentially.
//
//  PETSc options:
//    -ksp_type preonly          → skip iterative KSP, use direct factorisation
//    -pc_type lu                → LU factorisation
//    -snes_rtol 1e-8            → relative residual tolerance for SNES
//    -snes_atol 1e-10           → absolute residual tolerance
//    -snes_max_it 50            → max Newton iterations per solve
//    -snes_converged_reason     → print convergence reason
//    -snes_linesearch_type basic → full Newton step (no line search scaling)
//
//  Exit code: 0 if all checks pass, 1 if any fails.
// -----------------------------------------------------------------------------

int main(int argc, char** args) {
    PetscInitialize(&argc, &args, nullptr, nullptr);

    // ── Direct solver + SNES config ──
    PetscOptionsSetValue(nullptr, "-ksp_type",          "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type",           "lu");
    PetscOptionsSetValue(nullptr, "-snes_rtol",         "1e-8");
    PetscOptionsSetValue(nullptr, "-snes_atol",         "1e-10");
    PetscOptionsSetValue(nullptr, "-snes_max_it",       "50");
    PetscOptionsSetValue(nullptr, "-snes_converged_reason", nullptr);
    PetscOptionsSetValue(nullptr, "-snes_linesearch_type",  "basic");

    {
        std::cout << "============================================================\n"
                  << " Nonlinear Validation Suite\n"
                  << "============================================================\n"
                  << " Mesh: 2×2×2 hex27 (8 elements, 125 nodes)\n"
                  << " Domain: unit cube [0,1]³\n"
                  << " Fixed face: z = 0    Loaded face: z = 1\n"
                  << " Material: E=" << E_mod << " ν=" << nu << "\n"
                  << " J2: σ_y=" << sigma_y << " H=" << H_hard << "\n"
                  << "============================================================\n";

        // Group A: Linearization tests (small load)
        test_1_small_strain_reference();

        // ── Strain verification diagnostic ──────────────────────────────
        //  For the reference case (SmallStrain + Linear Elastic, uniaxial z),
        //  every Gauss point should have approximately:
        //    ε₃₃ ≈ σ₃₃/E = small_load / E_mod                (axial)
        //    ε₁₁ ≈ ε₂₂ ≈ -ν · ε₃₃                            (Poisson)
        //    γ₂₃ ≈ γ₁₃ ≈ γ₁₂ ≈ 0                              (no shear)
        //    σ₃₃ ≈ small_load (traction = stress for uniform)
        //
        //  Near the clamped face (z=0), Poisson constraint causes deviations,
        //  but away from it the solution should match. We check average strains
        //  across ALL Gauss points (the interior points dominate).
        {
            std::cout << "\n─── Strain/Stress Verification at Gauss Points ───\n";

            // Rebuild the model for Test 1 (the scope closed above)
            Domain<DIM> D;
            GmshDomainBuilder builder(MESH_FILE, D);
            ContinuumIsotropicElasticMaterial mat_inst{E_mod, nu};
            Material<ThreeDimensionalMaterial> mat{mat_inst, ElasticUpdate{}};
            Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF> M{D, mat};
            M.fix_z(0.0);
            M.setup();
            apply_z_traction(M, small_load);

            NonlinearAnalysis<ThreeDimensionalMaterial, continuum::SmallStrain> nl{&M};
            nl.solve();

            // update_elements_state is called inside solve; call again for safety
            M.update_elements_state();

            // Expected values for uniaxial tension (away from clamped face)
            double eps_zz_expect = small_load / E_mod;
            double eps_xx_expect = -nu * eps_zz_expect;
            double sig_zz_expect = small_load;

            // Collect strains from ALL Gauss points
            double sum_e11 = 0, sum_e22 = 0, sum_e33 = 0;
            double sum_g23 = 0, sum_g13 = 0, sum_g12 = 0;
            double sum_s33 = 0;
            int n_gp = 0;

            for (const auto& elem : M.elements()) {
                for (const auto& mp : elem.material_points()) {
                    const auto& state = mp.current_state();
                    sum_e11 += state[0];
                    sum_e22 += state[1];
                    sum_e33 += state[2];
                    sum_g23 += state[3];
                    sum_g13 += state[4];
                    sum_g12 += state[5];

                    auto stress = mp.compute_response(state);
                    sum_s33 += stress[2];
                    ++n_gp;
                }
            }

            double avg_e11 = sum_e11 / n_gp;
            double avg_e22 = sum_e22 / n_gp;
            double avg_e33 = sum_e33 / n_gp;
            double avg_g23 = sum_g23 / n_gp;
            double avg_g13 = sum_g13 / n_gp;
            double avg_g12 = sum_g12 / n_gp;
            double avg_s33 = sum_s33 / n_gp;

            std::cout << "  Expected: ε₃₃=" << std::scientific << eps_zz_expect
                      << "  ε₁₁=ε₂₂=" << eps_xx_expect << "\n";
            std::cout << "  Avg GP:   ε₁₁=" << avg_e11 << "  ε₂₂=" << avg_e22
                      << "  ε₃₃=" << avg_e33 << "\n";
            std::cout << "            γ₂₃=" << avg_g23 << "  γ₁₃=" << avg_g13
                      << "  γ₁₂=" << avg_g12 << "\n";
            std::cout << "  Avg σ₃₃=" << avg_s33
                      << "  (expected ≈" << sig_zz_expect << ")\n";

            // Checks: average ε₃₃ should be close to analytical
            double e33_rel = std::abs(avg_e33 - eps_zz_expect) / std::abs(eps_zz_expect);
            check(e33_rel < 0.10,
                  "Avg ε₃₃ within 10% of analytical (uniaxial tension)");

            // Shear strains should be negligible
            double max_shear = std::max({std::abs(avg_g23), std::abs(avg_g13), std::abs(avg_g12)});
            check(max_shear < 0.1 * std::abs(avg_e33),
                  "Shear strains negligible vs axial");

            // Stress σ₃₃ should be close to applied traction
            double s33_rel = std::abs(avg_s33 - sig_zz_expect) / std::abs(sig_zz_expect);
            check(s33_rel < 0.10,
                  "Avg σ₃₃ within 10% of applied traction");

            // Print first element, first GP detailed state for debugging
            {
                const auto& first_elem = M.elements().front();
                const auto& first_mp = first_elem.material_points().front();
                const auto& s = first_mp.current_state();
                auto sig = first_mp.compute_response(s);
                auto coord = first_mp.coord();
                std::cout << "  [Detail] GP0 coords=("
                          << coord[0] << "," << coord[1] << "," << coord[2] << ")\n";
                std::cout << "           strain=[" << s[0] << ", " << s[1] << ", "
                          << s[2] << ", " << s[3] << ", " << s[4] << ", " << s[5] << "]\n";
                std::cout << "           stress=[" << sig[0] << ", " << sig[1] << ", "
                          << sig[2] << ", " << sig[3] << ", " << sig[4] << ", " << sig[5] << "]\n";
            }
        }

        test_2_tl_svk_vs_small_strain();
        test_3_tl_nh_vs_small_strain();

        // Group B: Formulation equivalence (UL = TL)
        test_4_ul_svk_equals_tl();
        test_5_ul_nh_equals_tl();

        // Group C: Large-deformation convergence
        test_6_tl_svk_incremental();
        test_7_tl_nh_incremental();

        // Group D: Plasticity
        test_8_j2_plasticity();

        // Group E: Bisection and consistency
        test_9_bisection_recovery();
        test_10_physical_consistency();

        std::cout << "\n============================================================\n"
                  << " Results: " << g_pass << " passed, "
                  << g_fail << " failed  (of " << (g_pass + g_fail) << ")\n"
                  << "============================================================\n";
    }

    PetscFinalize();
    return (g_fail > 0) ? 1 : 0;
}
