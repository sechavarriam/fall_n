// =============================================================================
//  test_snes_gmsh_pipeline.cpp — Phase 7B: Multi-element SNES pipeline tests
// =============================================================================
//
//  End-to-end tests on a real Gmsh mesh (4×4×4 = 64 hex27 Lagrangian elements,
//  729 nodes, cube [0,10]³).  Validates the full production FEM pipeline:
//
//    GmshDomainBuilder → Model → ContinuumElement → KinematicPolicy
//    → Material<> (type-erasure) → NonlinearAnalysis (SNES, parallel assembly)
//
//  ==========================================================================
//  Step 1: Multi-element Gmsh + SNES (hyperelastic)
//  ==========================================================================
//    Test 1:  SmallStrain + linear elastic on 64 hex27 — SNES converges
//    Test 2:  TotalLagrangian + SVK on 64 hex27 — matches SmallStrain
//    Test 3:  TL + SVK incremental on 64 hex27 — converges under large load
//    Test 4:  TL + Neo-Hookean incremental on 64 hex27 — converges
//
//  ==========================================================================
//  Step 2: UpdatedLagrangian → SNES (spatial formulation end-to-end)
//  ==========================================================================
//    Test 5:  UL + SVK on 64 hex27 — matches TL for small load
//    Test 6:  UL + NH incremental on 64 hex27 — converges under large load
//
//  ==========================================================================
//  Step 3: Plasticity + SNES incremental
//  ==========================================================================
//    Test 7:  SmallStrain + J2 Von Mises + InelasticUpdate — beyond yield
//    Test 8:  TotalLagrangian + J2 Von Mises incremental — plasticity + geom NL
//    Test 9:  Residual plasticity: unload and check permanent deformation
//
//  Mesh:  data/input/Test_BoxSide10_LagrangianElements27Nodes.msh
//         64 hex27 elements (3×3×3 GP each = 1728 Gauss points total)
//
//  BCs:   z = 0 face clamped (all DOFs)
//         Uniaxial tension in z-direction on z = 10 face
//
//  Requires: PETSc + MPI + Eigen + VTK (for Material/Model headers)
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

// ── Project headers ───────────────────────────────────────────────────────────

#include "header_files.hh"                        // Domain, Model, NLAnalysis, etc.
#include "src/continuum/HyperelasticRelation.hh"  // SVKRelation, NeoHookeanRelation

// ── Constants ─────────────────────────────────────────────────────────────────

static constexpr std::size_t DIM  = 3;
static constexpr std::size_t NDOF = DIM;

static constexpr double E_modulus   = 200.0;    // Young's modulus
static constexpr double nu_poisson  = 0.3;      // Poisson's ratio
static constexpr double sigma_y0    = 0.250;    // Initial yield stress (J2)
static constexpr double H_hardening = 10.0;     // Isotropic hardening modulus

static const std::string MESH_FILE =
    "/home/sechavarriam/MyLibs/fall_n/data/input/"
    "Test_BoxSide10_LagrangianElements27Nodes.msh";

static int passed = 0;
static int failed = 0;

// ── Helpers ───────────────────────────────────────────────────────────────────

static void check(bool cond, const char* msg) {
    if (cond) {
        std::cout << "  [PASS] " << msg << "\n";
        ++passed;
    } else {
        std::cout << "  [FAIL] " << msg << "\n";
        ++failed;
    }
}

/// Extract full local solution vector from model's current_state.
static std::vector<double> extract_solution(auto& model) {
    const PetscScalar* arr;
    PetscInt n;
    VecGetLocalSize(model.current_state, &n);
    VecGetArrayRead(model.current_state, &arr);
    std::vector<double> sol(arr, arr + n);
    VecRestoreArrayRead(model.current_state, &arr);
    return sol;
}

/// L∞ relative error between two solution vectors.
static double linf_relative_error(const std::vector<double>& a,
                                  const std::vector<double>& b) {
    assert(a.size() == b.size());
    double max_err = 0.0;
    double max_val = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        max_err = std::max(max_err, std::abs(a[i] - b[i]));
        max_val = std::max(max_val, std::max(std::abs(a[i]), std::abs(b[i])));
    }
    return (max_val > 0.0) ? max_err / max_val : max_err;
}

/// Cube face area (side = 10).
static constexpr double FACE_AREA = 10.0 * 10.0;  // = 100

/// Apply distributed uniaxial tension (in z-direction) on z = 10 face.
/// Uses consistent nodal forces from surface Gauss quadrature:
///   f_I = ∫_Γ N_I(ξ,η) · t · dA
/// The surface elements are created at runtime via create_boundary_from_plane
/// because the mesh only has surface elements on z = 0 ("Fixed").
template <typename ModelT>
static void apply_z_traction(ModelT& M, double total_force) {
    double traction_z = total_force / FACE_AREA;   // uniform traction
    auto& D = M.get_domain();
    if (!D.has_boundary_group("Load_z10")) {
        D.create_boundary_from_plane("Load_z10", 2, 10.0);
    }
    M.apply_surface_traction("Load_z10", 0.0, 0.0, traction_z);
}

/// Old lumped approach — kept for comparison / back-compat tests.
template <typename ModelT>
static void apply_z_tension_lumped(ModelT& M, double total_force) {
    M._force_orthogonal_plane(2, 10.0, 0.0, 0.0, total_force);
}

/// Compute L∞ norm of a PETSc vector.
[[maybe_unused]]
static double vec_linf_norm(Vec v) {
    PetscReal val;
    VecNorm(v, NORM_INFINITY, &val);
    return static_cast<double>(val);
}

/// Max z-displacement among all DOFs in the model.
static double max_z_displacement(auto& model) {
    const PetscScalar* arr;
    PetscInt n;
    VecGetLocalSize(model.current_state, &n);
    VecGetArrayRead(model.current_state, &arr);
    double max_uz = 0.0;
    for (PetscInt i = 2; i < n; i += DIM)  // z-component every 3 DOFs
        max_uz = std::max(max_uz, std::abs(arr[i]));
    VecRestoreArrayRead(model.current_state, &arr);
    return max_uz;
}


// =============================================================================
//  STEP 1: Multi-element Gmsh + SNES (hyperelastic)
// =============================================================================

// ─── Reference solution (SmallStrain + linear elastic) ─────────────────────

static std::vector<double> ref_solution_gmsh;

static void test_1_gmsh_small_strain_linear() {
    std::cout << "\n─── Test 1: Gmsh 64×hex27 — SmallStrain + Linear Elastic ───\n";

    Domain<DIM> D;
    GmshDomainBuilder builder(MESH_FILE, D);

    ContinuumIsotropicElasticMaterial linear_mat{E_modulus, nu_poisson};
    Material<ThreeDimensionalMaterial> mat{linear_mat, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF> M{D, mat};
    M.fix_z(0.0);
    M.setup();
    apply_z_traction(M, 5.0);

    NonlinearAnalysis<ThreeDimensionalMaterial, continuum::SmallStrain> nl{&M};
    nl.solve();

    auto reason = nl.converged_reason();
    auto its    = nl.num_iterations();

    check(reason > 0, "SNES converged");
    check(its <= 2,   "Converged in ≤ 2 iterations (linear)");

    check(D.num_elements() == 64,     "64 hex27 elements loaded from Gmsh");
    check(D.num_nodes()    == 729,    "729 nodes loaded from Gmsh");

    double max_uz = max_z_displacement(M);
    std::cout << "  Max |uz|: " << std::scientific << std::setprecision(6)
              << max_uz << "\n";
    check(max_uz > 0.0, "Positive z-displacement (tension)");
    check(std::isfinite(max_uz), "Finite displacement");

    ref_solution_gmsh = extract_solution(M);
}


// ─── TotalLagrangian + SVK on Gmsh mesh ────────────────────────────────────

static void test_2_gmsh_tl_svk() {
    std::cout << "\n─── Test 2: Gmsh 64×hex27 — TL + SVK ───\n";

    Domain<DIM> D;
    GmshDomainBuilder builder(MESH_FILE, D);

    auto svk = continuum::SaintVenantKirchhoff<3>::from_E_nu(E_modulus, nu_poisson);
    MaterialInstance<continuum::SVKRelation<3>> inst{svk};
    Material<ThreeDimensionalMaterial> mat{inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::TotalLagrangian, NDOF> M{D, mat};
    M.fix_z(0.0);
    M.setup();
    apply_z_traction(M, 5.0);

    NonlinearAnalysis<ThreeDimensionalMaterial, continuum::TotalLagrangian> nl{&M};
    nl.solve();

    auto reason = nl.converged_reason();
    auto its    = nl.num_iterations();

    check(reason > 0, "SNES converged");
    std::cout << "  SNES iterations: " << its << "\n";

    auto sol = extract_solution(M);
    double rel_err = linf_relative_error(sol, ref_solution_gmsh);
    std::cout << "  L∞ rel.err vs SmallStrain: " << std::scientific << rel_err << "\n";
    check(rel_err < 1e-2, "TL+SVK ≈ SmallStrain for small load (< 1%)");
}


// ─── TL + SVK incremental (larger deformation) ────────────────────────────

static void test_3_gmsh_tl_svk_incremental() {
    std::cout << "\n─── Test 3: Gmsh 64×hex27 — TL + SVK incremental ───\n";

    Domain<DIM> D;
    GmshDomainBuilder builder(MESH_FILE, D);

    auto svk = continuum::SaintVenantKirchhoff<3>::from_E_nu(E_modulus, nu_poisson);
    MaterialInstance<continuum::SVKRelation<3>> inst{svk};
    Material<ThreeDimensionalMaterial> mat{inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::TotalLagrangian, NDOF> M{D, mat};
    M.fix_z(0.0);
    M.setup();
    apply_z_traction(M, 50.0);  // 10× larger load → geometric NL matters

    NonlinearAnalysis<ThreeDimensionalMaterial, continuum::TotalLagrangian> nl{&M};
    nl.solve_incremental(5);

    auto reason = nl.converged_reason();
    check(reason > 0, "SNES converged (all 5 load steps)");

    double max_uz = max_z_displacement(M);
    std::cout << "  Max |uz|: " << std::scientific << max_uz << "\n";
    check(max_uz > 0.0, "Positive z-displacement");
    check(std::isfinite(max_uz), "Finite displacement");
}


// ─── TL + Neo-Hookean incremental ─────────────────────────────────────────

static void test_4_gmsh_tl_nh_incremental() {
    std::cout << "\n─── Test 4: Gmsh 64×hex27 — TL + Neo-Hookean incremental ───\n";

    Domain<DIM> D;
    GmshDomainBuilder builder(MESH_FILE, D);

    auto nh = continuum::CompressibleNeoHookean<3>::from_E_nu(E_modulus, nu_poisson);
    MaterialInstance<continuum::NeoHookeanRelation<3>> inst{nh};
    Material<ThreeDimensionalMaterial> mat{inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::TotalLagrangian, NDOF> M{D, mat};
    M.fix_z(0.0);
    M.setup();
    apply_z_traction(M, 50.0);

    NonlinearAnalysis<ThreeDimensionalMaterial, continuum::TotalLagrangian> nl{&M};
    nl.solve_incremental(5);

    auto reason = nl.converged_reason();
    check(reason > 0, "SNES converged (all 5 load steps)");

    double max_uz = max_z_displacement(M);
    std::cout << "  Max |uz|: " << std::scientific << max_uz << "\n";
    check(max_uz > 0.0, "Positive z-displacement");
    check(std::isfinite(max_uz), "Finite displacement");
}


// =============================================================================
//  STEP 2: UpdatedLagrangian → SNES (spatial formulation end-to-end)
// =============================================================================

// ─── UL + SVK — matches TL for small load ──────────────────────────────────

static void test_5_gmsh_ul_svk() {
    std::cout << "\n─── Test 5: Gmsh 64×hex27 — UL + SVK ───\n";

    // TL reference
    std::vector<double> tl_sol;
    {
        Domain<DIM> D;
        GmshDomainBuilder builder(MESH_FILE, D);

        auto svk = continuum::SaintVenantKirchhoff<3>::from_E_nu(E_modulus, nu_poisson);
        MaterialInstance<continuum::SVKRelation<3>> inst{svk};
        Material<ThreeDimensionalMaterial> mat{inst, ElasticUpdate{}};

        Model<ThreeDimensionalMaterial, continuum::TotalLagrangian, NDOF> M{D, mat};
        M.fix_z(0.0);
        M.setup();
        apply_z_traction(M, 5.0);

        NonlinearAnalysis<ThreeDimensionalMaterial, continuum::TotalLagrangian> nl{&M};
        nl.solve();
        tl_sol = extract_solution(M);
    }

    // UL solution
    std::vector<double> ul_sol;
    {
        Domain<DIM> D;
        GmshDomainBuilder builder(MESH_FILE, D);

        auto svk = continuum::SaintVenantKirchhoff<3>::from_E_nu(E_modulus, nu_poisson);
        MaterialInstance<continuum::SVKRelation<3>> inst{svk};
        Material<ThreeDimensionalMaterial> mat{inst, ElasticUpdate{}};

        Model<ThreeDimensionalMaterial, continuum::UpdatedLagrangian, NDOF> M{D, mat};
        M.fix_z(0.0);
        M.setup();
        apply_z_traction(M, 5.0);

        NonlinearAnalysis<ThreeDimensionalMaterial, continuum::UpdatedLagrangian> nl{&M};
        nl.solve();

        check(nl.converged_reason() > 0, "SNES converged (UL)");
        std::cout << "  SNES iterations (UL): " << nl.num_iterations() << "\n";
        ul_sol = extract_solution(M);
    }

    double rel_err = linf_relative_error(ul_sol, tl_sol);
    std::cout << "  L∞ rel.err UL vs TL: " << std::scientific << rel_err << "\n";
    check(rel_err < 1e-8, "UL ≡ TL (same solution)");
}


// ─── UL + NH incremental (larger deformation) ─────────────────────────────

static void test_6_gmsh_ul_nh_incremental() {
    std::cout << "\n─── Test 6: Gmsh 64×hex27 — UL + Neo-Hookean incremental ───\n";

    Domain<DIM> D;
    GmshDomainBuilder builder(MESH_FILE, D);

    auto nh = continuum::CompressibleNeoHookean<3>::from_E_nu(E_modulus, nu_poisson);
    MaterialInstance<continuum::NeoHookeanRelation<3>> inst{nh};
    Material<ThreeDimensionalMaterial> mat{inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::UpdatedLagrangian, NDOF> M{D, mat};
    M.fix_z(0.0);
    M.setup();
    apply_z_traction(M, 50.0);

    NonlinearAnalysis<ThreeDimensionalMaterial, continuum::UpdatedLagrangian> nl{&M};
    nl.solve_incremental(5);

    auto reason = nl.converged_reason();
    check(reason > 0, "SNES converged (all 5 load steps)");

    double max_uz = max_z_displacement(M);
    std::cout << "  Max |uz|: " << std::scientific << max_uz << "\n";
    check(max_uz > 0.0, "Positive z-displacement");
    check(std::isfinite(max_uz), "Finite displacement");
}


// =============================================================================
//  STEP 3: Plasticity + SNES incremental
// =============================================================================

// ─── SmallStrain + J2 Von Mises: beyond yield point ───────────────────────

static void test_7_gmsh_j2_small_strain() {
    std::cout << "\n─── Test 7: Gmsh 64×hex27 — SmallStrain + J2 Plasticity ───\n";

    Domain<DIM> D;
    GmshDomainBuilder builder(MESH_FILE, D);

    J2PlasticMaterial3D j2_inst{E_modulus, nu_poisson, sigma_y0, H_hardening};
    Material<ThreeDimensionalMaterial> mat{j2_inst, InelasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF> M{D, mat};
    M.fix_z(0.0);
    M.setup();

    // Choose force large enough to cause widespread yielding:
    //   σ_y = 0.25 on a L=10 cube face, A = 100  → F_yield ≈ 0.25 * 100 = 25
    //   Apply F = 100  → ~4× yield, significant plastic flow even with triaxial
    //   constraint from clamped face (von Mises equivalent reduced)
    apply_z_traction(M, 100.0);

    NonlinearAnalysis<ThreeDimensionalMaterial, continuum::SmallStrain> nl{&M};
    nl.solve_incremental(20);

    auto reason = nl.converged_reason();
    std::cout << "  SNES converged reason: " << static_cast<int>(reason) << "\n";
    check(reason > 0, "SNES converged (20 load steps, J2)");

    double max_uz = max_z_displacement(M);
    std::cout << "  Max |uz| (plastic): " << std::scientific << max_uz << "\n";
    check(max_uz > 0.0, "Non-zero displacement");
    check(std::isfinite(max_uz), "Finite displacement");

    // Compare against actual FE elastic solution at same load.
    // For a clamped cube, the 1D bar formula FL/(EA) overestimates displacement;
    // we compute the actual elastic displacement via a linear FE solve (test 9).
    // Here just verify plastic displacement is meaningfully large.
    double elastic_1d_ref = 100.0 * 10.0 / (E_modulus * 100.0);  // = 0.05
    std::cout << "  1D bar reference: " << std::scientific << elastic_1d_ref << "\n";
    check(max_uz > elastic_1d_ref, "Plastic deformation > 1D bar elastic reference");
}


// ─── TL + J2 Von Mises incremental (geometrically + materially nonlinear) ──

static void test_8_gmsh_tl_j2_incremental() {
    std::cout << "\n─── Test 8: Gmsh 64×hex27 — TL + J2 incremental ───\n";

    Domain<DIM> D;
    GmshDomainBuilder builder(MESH_FILE, D);

    J2PlasticMaterial3D j2_inst{E_modulus, nu_poisson, sigma_y0, H_hardening};
    Material<ThreeDimensionalMaterial> mat{j2_inst, InelasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::TotalLagrangian, NDOF> M{D, mat};
    M.fix_z(0.0);
    M.setup();
    apply_z_traction(M, 100.0);

    NonlinearAnalysis<ThreeDimensionalMaterial, continuum::TotalLagrangian> nl{&M};
    nl.solve_incremental(20);

    auto reason = nl.converged_reason();
    std::cout << "  SNES converged reason: " << static_cast<int>(reason) << "\n";
    check(reason > 0, "SNES converged (TL + J2, 20 load steps)");

    double max_uz = max_z_displacement(M);
    std::cout << "  Max |uz| (TL + plastic): " << std::scientific << max_uz << "\n";
    check(max_uz > 0.0, "Positive displacement");
    check(std::isfinite(max_uz), "Finite displacement");
}


// ─── Test 9: Elastic-vs-plastic cross-validation ──────────────────────────
//
//  Same mesh + BCs + load, solved with:
//    (a) Linear elastic — reference elastic displacement
//    (b) J2 plasticity  — larger displacement (already from test 7)
//  Verify the plastic solution is measurably softer.

static void test_9_elastic_vs_plastic() {
    std::cout << "\n─── Test 9: Elastic vs Plastic cross-validation ───\n";

    // (a) Elastic reference at F = 100
    double max_uz_elastic = 0.0;
    {
        Domain<DIM> D;
        GmshDomainBuilder builder(MESH_FILE, D);

        ContinuumIsotropicElasticMaterial linear_mat{E_modulus, nu_poisson};
        Material<ThreeDimensionalMaterial> mat{linear_mat, ElasticUpdate{}};

        Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF> M{D, mat};
        M.fix_z(0.0);
        M.setup();
        apply_z_traction(M, 100.0);

        NonlinearAnalysis<ThreeDimensionalMaterial, continuum::SmallStrain> nl{&M};
        nl.solve();
        max_uz_elastic = max_z_displacement(M);
    }

    // (b) Plastic at same load (20 increments)
    double max_uz_plastic = 0.0;
    {
        Domain<DIM> D;
        GmshDomainBuilder builder(MESH_FILE, D);

        J2PlasticMaterial3D j2_inst{E_modulus, nu_poisson, sigma_y0, H_hardening};
        Material<ThreeDimensionalMaterial> mat{j2_inst, InelasticUpdate{}};

        Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF> M{D, mat};
        M.fix_z(0.0);
        M.setup();
        apply_z_traction(M, 100.0);

        NonlinearAnalysis<ThreeDimensionalMaterial, continuum::SmallStrain> nl{&M};
        nl.solve_incremental(20);
        check(nl.converged_reason() > 0, "Plastic solve converged");
        max_uz_plastic = max_z_displacement(M);
    }

    std::cout << "  Max |uz| elastic: " << std::scientific << max_uz_elastic << "\n";
    std::cout << "  Max |uz| plastic: " << std::scientific << max_uz_plastic << "\n";

    double ratio = max_uz_plastic / max_uz_elastic;
    std::cout << "  Softening ratio (plastic/elastic): " << std::fixed
              << std::setprecision(2) << ratio << "\n";

    check(max_uz_plastic > max_uz_elastic,
          "Plastic displacement > elastic displacement");
    check(ratio > 1.5,
          "Significant softening (ratio > 1.5)");
    check(std::isfinite(ratio), "Finite ratio");
}


// =============================================================================
//  Main
// =============================================================================

int main(int argc, char** args) {
    PetscInitialize(&argc, &args, nullptr, nullptr);

    // Configure SNES for direct solver (deterministic, moderate-size problem)
    PetscOptionsSetValue(nullptr, "-ksp_type",  "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type",   "lu");
    PetscOptionsSetValue(nullptr, "-snes_rtol", "1e-8");
    PetscOptionsSetValue(nullptr, "-snes_atol", "1e-10");
    PetscOptionsSetValue(nullptr, "-snes_max_it", "100");
    PetscOptionsSetValue(nullptr, "-snes_converged_reason", nullptr);
    // Full Newton step (basic line search) — critical for plasticity
    PetscOptionsSetValue(nullptr, "-snes_linesearch_type", "basic");

    {
        std::cout << "============================================================\n"
                  << " Phase 7B: Multi-element Gmsh + UL + Plasticity (SNES)\n"
                  << "============================================================\n"
                  << " Mesh: 4×4×4 hex27 (64 elements, 729 nodes)\n"
                  << " Domain: cube [0,10]³\n"
                  << " Fixed face: z = 0\n"
                  << "============================================================\n";

        // Step 1: Multi-element Gmsh + SNES (hyperelastic)
        test_1_gmsh_small_strain_linear();
        test_2_gmsh_tl_svk();
        test_3_gmsh_tl_svk_incremental();
        test_4_gmsh_tl_nh_incremental();

        // Step 2: UpdatedLagrangian → SNES
        test_5_gmsh_ul_svk();
        test_6_gmsh_ul_nh_incremental();

        // Step 3: Plasticity + SNES incremental
        test_7_gmsh_j2_small_strain();
        test_8_gmsh_tl_j2_incremental();
        test_9_elastic_vs_plastic();

        std::cout << "\n============================================================\n"
                  << " Results: " << passed << " passed, "
                  << failed << " failed  (of " << (passed + failed) << ")\n"
                  << "============================================================\n";
    }

    PetscFinalize();
    return (failed > 0) ? 1 : 0;
}
