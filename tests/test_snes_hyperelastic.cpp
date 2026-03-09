// =============================================================================
//  test_snes_hyperelastic.cpp — Phase 7: End-to-end SNES integration tests
// =============================================================================
//
//  Validates the full nonlinear pipeline:
//
//    Domain → Model → ContinuumElement → KinematicPolicy → HyperelasticRelation
//    → Material<> type-erasure → NonlinearAnalysis (SNES) → solution
//
//  Tests:
//    1. SmallStrain + linear elastic  → SNES converges in 1 iteration (linear)
//    2. SmallStrain + SVK via type-erasure → same as (1) for small load
//    3. TotalLagrangian + SVK + SNES  → matches SmallStrain for small load
//    4. TotalLagrangian + Neo-Hookean  → matches SmallStrain for small load
//    5. TotalLagrangian + SVK incremental → converges for larger deformation
//    6. Parallel assembly equivalence  → two-phase assembly matches sequential
//
//  Mesh: single hex8 element, unit cube [0,1]³.
//  BCs:  x = 0 face clamped (all DOFs), uniaxial tension on x = 1 face.
//
//  Requires PETSc runtime (PetscInitialize / PetscFinalize).
//  Compile/link: PkgConfig::PETSC  MPI::MPI_CXX  Eigen3::Eigen
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

#include <petsc.h>

// ── Project headers ───────────────────────────────────────────────────────────

#include "header_files.hh"                        // Domain, Model, NLAnalysis, etc.
#include "src/continuum/HyperelasticRelation.hh"  // SVKRelation, NeoHookeanRelation

// ── Constants ─────────────────────────────────────────────────────────────────

static constexpr std::size_t DIM  = 3;
static constexpr std::size_t NDOF = DIM;

static constexpr double E_modulus  = 200.0;
static constexpr double nu_poisson = 0.3;

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

/// Create a unit cube [0,1]³ domain with 1 hex8 element (2×2×2 GP).
static void create_unit_cube(Domain<DIM>& D) {
    D.preallocate_node_capacity(8);
    D.add_node(0, 0.0, 0.0, 0.0);  // x=0 face
    D.add_node(1, 1.0, 0.0, 0.0);  // x=1 face
    D.add_node(2, 0.0, 1.0, 0.0);  // x=0 face
    D.add_node(3, 1.0, 1.0, 0.0);  // x=1 face
    D.add_node(4, 0.0, 0.0, 1.0);  // x=0 face
    D.add_node(5, 1.0, 0.0, 1.0);  // x=1 face
    D.add_node(6, 0.0, 1.0, 1.0);  // x=0 face
    D.add_node(7, 1.0, 1.0, 1.0);  // x=1 face

    std::array<PetscInt, 8> conn{0, 1, 2, 3, 4, 5, 6, 7};
    D.make_element<LagrangeElement<3, 2, 2, 2>>(
        GaussLegendreCellIntegrator<2, 2, 2>{}, 0, conn.data());
    D.assemble_sieve();
}

/// Extract full local solution vector from a model's state_vector().
static std::vector<double> extract_solution(auto& model) {
    const PetscScalar* arr;
    PetscInt n;
    VecGetLocalSize(model.state_vector(), &n);
    VecGetArrayRead(model.state_vector(), &arr);
    std::vector<double> sol(arr, arr + n);
    VecRestoreArrayRead(model.state_vector(), &arr);
    return sol;
}

/// Extract displacement of a specific node from a model's current_state.
static std::array<double, DIM> node_displacement(auto& model, std::size_t node_idx) {
    auto& node = model.get_domain().node(node_idx);
    auto dof_idx = node.dof_index();
    std::array<double, DIM> u{};
    PetscInt idx[DIM];
    for (std::size_t i = 0; i < DIM; ++i)
        idx[i] = dof_idx[i];
    VecGetValues(model.state_vector(), static_cast<PetscInt>(DIM), idx, u.data());
    return u;
}

/// Apply distributed uniaxial tension on the x = 1 face (nodes 1,3,5,7).
template <typename ModelT>
static void apply_uniaxial_tension(ModelT& M, double total_force) {
    const double f_per_node = total_force / 4.0;
    for (std::size_t id : {1ul, 3ul, 5ul, 7ul})
        M.apply_node_force(id, f_per_node, 0.0, 0.0);
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


// =============================================================================
//  Test 1: SmallStrain + linear elastic material + SNES
// =============================================================================
//
//  A linear problem should converge in exactly 1 Newton iteration.
//  This establishes the reference solution for comparison tests.

static std::vector<double> ref_solution;  // stored for later comparisons

static void test_1_small_strain_linear_elastic() {
    std::cout << "\n--- Test 1: SmallStrain + Linear Elastic + SNES ---\n";

    Domain<DIM> D;
    create_unit_cube(D);

    ContinuumIsotropicElasticMaterial linear_mat{E_modulus, nu_poisson};
    Material<ThreeDimensionalMaterial> mat{linear_mat, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF> M{D, mat};
    M.fix_x(0.0);
    M.setup();
    apply_uniaxial_tension(M, 0.02);

    NonlinearAnalysis<ThreeDimensionalMaterial, continuum::SmallStrain> nl{&M};
    nl.solve();

    auto reason = nl.converged_reason();
    auto its    = nl.num_iterations();

    check(reason > 0, "SNES converged");
    check(its <= 2,   "Converged in ≤ 2 iterations (linear problem)");

    // Verify x-displacement at x=1 face is positive (tension)
    auto u1 = node_displacement(M, 1);
    check(u1[0] > 0.0, "Node 1 x-displacement > 0 (tension)");

    // All x=1 face nodes should have same x-displacement (uniform tension)
    auto u3 = node_displacement(M, 3);
    auto u5 = node_displacement(M, 5);
    auto u7 = node_displacement(M, 7);
    double tol = 1e-10;
    check(std::abs(u1[0] - u3[0]) < tol && std::abs(u1[0] - u5[0]) < tol
          && std::abs(u1[0] - u7[0]) < tol,
          "Uniform x-displacement on x=1 face");

    std::cout << "  x-displacement at x=1 face: " << std::scientific
              << std::setprecision(6) << u1[0] << "\n";

    // Store reference solution
    ref_solution = extract_solution(M);
}


// =============================================================================
//  Test 2: SmallStrain + SVK through type-erasure + SNES
// =============================================================================
//
//  SVK with SmallStrain reduces to linear elasticity (Green-Lagrange = ε for
//  small strains), so the result must match Test 1 exactly.

static void test_2_small_strain_svk() {
    std::cout << "\n--- Test 2: SmallStrain + SVK (type-erasure) + SNES ---\n";

    Domain<DIM> D;
    create_unit_cube(D);

    // SVK through MaterialInstance + type-erasure
    auto svk_model = continuum::SaintVenantKirchhoff<3>::from_E_nu(E_modulus, nu_poisson);
    MaterialInstance<continuum::SVKRelation<3>> svk_inst{svk_model};
    Material<ThreeDimensionalMaterial> mat{svk_inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF> M{D, mat};
    M.fix_x(0.0);
    M.setup();
    apply_uniaxial_tension(M, 0.02);

    NonlinearAnalysis<ThreeDimensionalMaterial, continuum::SmallStrain> nl{&M};
    nl.solve();

    check(nl.converged_reason() > 0, "SNES converged");

    auto sol = extract_solution(M);
    double rel_err = linf_relative_error(sol, ref_solution);
    std::cout << "  L∞ relative error vs linear elastic: " << std::scientific
              << rel_err << "\n";
    check(rel_err < 1e-8, "SVK matches linear elastic for SmallStrain");
}


// =============================================================================
//  Test 3: TotalLagrangian + SVK + SNES
// =============================================================================
//
//  For small loads, TL + SVK ≈ SmallStrain + linear elastic.
//  The difference is O(ε²) from geometric nonlinearity.

static void test_3_total_lagrangian_svk() {
    std::cout << "\n--- Test 3: TotalLagrangian + SVK + SNES ---\n";

    Domain<DIM> D;
    create_unit_cube(D);

    auto svk_model = continuum::SaintVenantKirchhoff<3>::from_E_nu(E_modulus, nu_poisson);
    MaterialInstance<continuum::SVKRelation<3>> svk_inst{svk_model};
    Material<ThreeDimensionalMaterial> mat{svk_inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::TotalLagrangian, NDOF> M{D, mat};
    M.fix_x(0.0);
    M.setup();
    apply_uniaxial_tension(M, 0.02);

    NonlinearAnalysis<ThreeDimensionalMaterial, continuum::TotalLagrangian> nl{&M};
    nl.solve();

    auto reason = nl.converged_reason();
    auto its    = nl.num_iterations();

    check(reason > 0, "SNES converged");
    std::cout << "  SNES iterations: " << its << "\n";

    // Compare with SmallStrain reference
    auto sol = extract_solution(M);
    double rel_err = linf_relative_error(sol, ref_solution);
    std::cout << "  L∞ relative error vs SmallStrain: " << std::scientific
              << rel_err << "\n";
    check(rel_err < 1e-3, "TL+SVK ≈ SmallStrain for small load (< 0.1%)");

    // Basic physics check
    auto u1 = node_displacement(M, 1);
    check(u1[0] > 0.0, "Positive x-displacement (tension)");
}


// =============================================================================
//  Test 4: TotalLagrangian + Neo-Hookean + SNES
// =============================================================================
//
//  For small loads, NH ≈ SVK ≈ SmallStrain (all hyperelastic models have
//  the same linearization at the reference configuration).

static void test_4_total_lagrangian_neohookean() {
    std::cout << "\n--- Test 4: TotalLagrangian + Neo-Hookean + SNES ---\n";

    Domain<DIM> D;
    create_unit_cube(D);

    auto nh_model = continuum::CompressibleNeoHookean<3>::from_E_nu(E_modulus, nu_poisson);
    MaterialInstance<continuum::NeoHookeanRelation<3>> nh_inst{nh_model};
    Material<ThreeDimensionalMaterial> mat{nh_inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::TotalLagrangian, NDOF> M{D, mat};
    M.fix_x(0.0);
    M.setup();
    apply_uniaxial_tension(M, 0.02);

    NonlinearAnalysis<ThreeDimensionalMaterial, continuum::TotalLagrangian> nl{&M};
    nl.solve();

    auto reason = nl.converged_reason();
    auto its    = nl.num_iterations();

    check(reason > 0, "SNES converged");
    std::cout << "  SNES iterations: " << its << "\n";

    // Compare with SmallStrain reference
    auto sol = extract_solution(M);
    double rel_err = linf_relative_error(sol, ref_solution);
    std::cout << "  L∞ relative error vs SmallStrain: " << std::scientific
              << rel_err << "\n";
    check(rel_err < 1e-3, "TL+NH ≈ SmallStrain for small load (< 0.1%)");

    // NH and SVK should also match each other closely
    auto u1 = node_displacement(M, 1);
    check(u1[0] > 0.0, "Positive x-displacement (tension)");
}


// =============================================================================
//  Test 5: TotalLagrangian + SVK — incremental loading (larger deformation)
// =============================================================================
//
//  Larger load → geometric nonlinearity matters.
//  Verifies that incremental loading converges with SNES.

static void test_5_tl_svk_incremental() {
    std::cout << "\n--- Test 5: TL + SVK — Incremental loading ---\n";

    Domain<DIM> D;
    create_unit_cube(D);

    auto svk_model = continuum::SaintVenantKirchhoff<3>::from_E_nu(E_modulus, nu_poisson);
    MaterialInstance<continuum::SVKRelation<3>> svk_inst{svk_model};
    Material<ThreeDimensionalMaterial> mat{svk_inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::TotalLagrangian, NDOF> M{D, mat};
    M.fix_x(0.0);
    M.setup();

    // Larger force → noticeable geometric nonlinearity
    apply_uniaxial_tension(M, 2.0);

    NonlinearAnalysis<ThreeDimensionalMaterial, continuum::TotalLagrangian> nl{&M};
    nl.solve_incremental(5);  // 5 load steps

    auto reason = nl.converged_reason();
    check(reason > 0, "SNES converged on final load step");

    auto u1 = node_displacement(M, 1);
    std::cout << "  Final x-displacement at node 1: " << std::scientific
              << u1[0] << "\n";
    check(u1[0] > 0.0, "Positive x-displacement under large load");
    check(std::isfinite(u1[0]), "Finite displacement value");
}


// =============================================================================
//  Test 6: TotalLagrangian + NH — incremental loading
// =============================================================================
//
//  Neo-Hookean under larger deformation with incremental SNES.

static void test_6_tl_nh_incremental() {
    std::cout << "\n--- Test 6: TL + Neo-Hookean — Incremental loading ---\n";

    Domain<DIM> D;
    create_unit_cube(D);

    auto nh_model = continuum::CompressibleNeoHookean<3>::from_E_nu(E_modulus, nu_poisson);
    MaterialInstance<continuum::NeoHookeanRelation<3>> nh_inst{nh_model};
    Material<ThreeDimensionalMaterial> mat{nh_inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::TotalLagrangian, NDOF> M{D, mat};
    M.fix_x(0.0);
    M.setup();

    apply_uniaxial_tension(M, 2.0);

    NonlinearAnalysis<ThreeDimensionalMaterial, continuum::TotalLagrangian> nl{&M};
    nl.solve_incremental(5);

    auto reason = nl.converged_reason();
    check(reason > 0, "SNES converged on final load step");

    auto u1 = node_displacement(M, 1);
    std::cout << "  Final x-displacement at node 1: " << std::scientific
              << u1[0] << "\n";
    check(u1[0] > 0.0, "Positive x-displacement under large load");
    check(std::isfinite(u1[0]), "Finite displacement value");
}


// =============================================================================
//  Test 7: Verify parallel assembly produces identical results
// =============================================================================
//
//  Runs the same problem twice with the same settings.  Since the two-phase
//  compute-then-inject assembly is used in both cases (FormJacobian /
//  FormResidual were refactored), this test verifies self-consistency.
//  With a multi-element mesh, OpenMP parallelism (if built with -fopenmp)
//  exercises the parallel code path.

static void test_7_assembly_consistency() {
    std::cout << "\n--- Test 7: Assembly consistency (two runs) ---\n";

    // Run 1
    std::vector<double> sol1;
    {
        Domain<DIM> D;
        create_unit_cube(D);
        auto svk = continuum::SaintVenantKirchhoff<3>::from_E_nu(E_modulus, nu_poisson);
        MaterialInstance<continuum::SVKRelation<3>> inst{svk};
        Material<ThreeDimensionalMaterial> mat{inst, ElasticUpdate{}};

        Model<ThreeDimensionalMaterial, continuum::TotalLagrangian, NDOF> M{D, mat};
        M.fix_x(0.0);
        M.setup();
        apply_uniaxial_tension(M, 0.5);

        NonlinearAnalysis<ThreeDimensionalMaterial, continuum::TotalLagrangian> nl{&M};
        nl.solve();
        sol1 = extract_solution(M);
    }

    // Run 2 — identical problem
    std::vector<double> sol2;
    {
        Domain<DIM> D;
        create_unit_cube(D);
        auto svk = continuum::SaintVenantKirchhoff<3>::from_E_nu(E_modulus, nu_poisson);
        MaterialInstance<continuum::SVKRelation<3>> inst{svk};
        Material<ThreeDimensionalMaterial> mat{inst, ElasticUpdate{}};

        Model<ThreeDimensionalMaterial, continuum::TotalLagrangian, NDOF> M{D, mat};
        M.fix_x(0.0);
        M.setup();
        apply_uniaxial_tension(M, 0.5);

        NonlinearAnalysis<ThreeDimensionalMaterial, continuum::TotalLagrangian> nl{&M};
        nl.solve();
        sol2 = extract_solution(M);
    }

    check(sol1.size() == sol2.size(), "Solution vectors same size");
    double rel_err = linf_relative_error(sol1, sol2);
    std::cout << "  L∞ relative error between runs: " << std::scientific
              << rel_err << "\n";
    check(rel_err < 1e-14, "Identical solutions from two runs");
}


// =============================================================================
//  Main
// =============================================================================

int main(int argc, char** args) {
    PetscInitialize(&argc, &args, nullptr, nullptr);

    // Configure SNES for direct solver (deterministic, small problem)
    PetscOptionsSetValue(nullptr, "-ksp_type",  "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type",   "lu");
    PetscOptionsSetValue(nullptr, "-snes_rtol", "1e-12");
    PetscOptionsSetValue(nullptr, "-snes_atol", "1e-14");

    {
        std::cout << "============================================\n"
                  << " Phase 7: SNES + Hyperelastic end-to-end\n"
                  << "============================================\n";

        test_1_small_strain_linear_elastic();
        test_2_small_strain_svk();
        test_3_total_lagrangian_svk();
        test_4_total_lagrangian_neohookean();
        test_5_tl_svk_incremental();
        test_6_tl_nh_incremental();
        test_7_assembly_consistency();

        std::cout << "\n============================================\n"
                  << " Results: " << passed << " passed, "
                  << failed << " failed  (of " << (passed + failed) << ")\n"
                  << "============================================\n";
    }

    PetscFinalize();

    return (failed > 0) ? 1 : 0;
}
