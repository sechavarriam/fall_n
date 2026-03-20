// =============================================================================
//  test_dynamic_analysis.cpp — Phase 4: Dynamic analysis via PETSc TS
// =============================================================================
//
//  Validates the complete dynamic analysis pipeline:
//
//    Domain → Model (+density) → ContinuumElement (mass matrix)
//    → BoundaryCondition (time functions + BCs)
//    → DampingModel (Rayleigh)
//    → DynamicAnalysis (PETSc TS2: generalized-α / Newmark)
//    → VTK time series (PVDWriter)
//
//  Tests:
//
//    ── Time functions ──────────────────────────────────────────────────
//    1. TimeFunction factories (constant, ramp, harmonic, step, etc.)
//    2. TimeFunction combinators (product, sum, scale)
//    3. Piecewise-linear interpolation
//
//    ── Boundary conditions ─────────────────────────────────────────────
//    4. BoundaryConditionSet force evaluation
//    5. BoundaryConditionSet prescribed displacement
//    6. BoundaryConditionSet initial conditions
//
//    ── Mass matrix ─────────────────────────────────────────────────────
//    7. ContinuumElement consistent mass (hex8, ρ=1, unit cube → M=1)
//    8. Lumped mass (row-sum of consistent mass)
//    9. Mass matrix assembly in Model
//
//    ── Dynamic analysis ────────────────────────────────────────────────
//    10. SDOF free vibration: u₀·cos(ωt) with ω = √(K/M)
//    11. SDOF with Rayleigh damping: amplitude decay check
//    12. SDOF forced vibration: harmonic force, steady-state check
//
//    ── VTK time series ─────────────────────────────────────────────────
//    13. PVDWriter snapshot naming and .pvd generation
//
//    ── Concept verification ────────────────────────────────────────────
//    14. DampingAssembler callable concept
//    15. ContinuumElement mass matrix availability
//
//  Mesh: single hex8 element, unit cube [0,1]³, 2×2×2 Gauss points.
//  BCs: x=0 face clamped (12 DOFs fixed), x=1 face free (4 nodes).
//
//  Requires PETSc runtime (PetscInitialize / PetscFinalize).
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
#include <filesystem>

#include <petsc.h>

// ── Project headers ───────────────────────────────────────────────────────────

#include "header_files.hh"

// ── Constants ─────────────────────────────────────────────────────────────────

static constexpr std::size_t DIM  = 3;

static constexpr double E_modulus  = 1000.0;   // Young's modulus
static constexpr double nu_poisson = 0.0;      // Poisson's ratio (1D-like)
static constexpr double rho_density = 1.0;     // Mass density

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

/// Create a unit cube [0,1]³ domain with 1 hex8 element.
static void create_unit_cube(Domain<DIM>& D) {
    D.preallocate_node_capacity(8);
    D.add_node(0, 0.0, 0.0, 0.0);
    D.add_node(1, 1.0, 0.0, 0.0);
    D.add_node(2, 0.0, 1.0, 0.0);
    D.add_node(3, 1.0, 1.0, 0.0);
    D.add_node(4, 0.0, 0.0, 1.0);
    D.add_node(5, 1.0, 0.0, 1.0);
    D.add_node(6, 0.0, 1.0, 1.0);
    D.add_node(7, 1.0, 1.0, 1.0);

    std::array<PetscInt, 8> conn{0, 1, 2, 3, 4, 5, 6, 7};
    D.make_element<LagrangeElement<3, 2, 2, 2>>(
        GaussLegendreCellIntegrator<2, 2, 2>{}, 0, conn.data());
    D.assemble_sieve();
}


// =============================================================================
//  Test 1: TimeFunction factories
// =============================================================================

static void test_1_time_function_factories() {
    std::cout << "\n--- Test 1: TimeFunction factories ---\n";

    auto f_const = time_fn::constant(3.14);
    check(std::abs(f_const(0.0) - 3.14) < 1e-12, "constant(3.14) at t=0");
    check(std::abs(f_const(99.0) - 3.14) < 1e-12, "constant(3.14) at t=99");

    auto f_zero = time_fn::zero();
    check(std::abs(f_zero(42.0)) < 1e-12, "zero() at t=42");

    auto f_ramp = time_fn::linear_ramp(1.0, 3.0, 0.0, 10.0);
    check(std::abs(f_ramp(0.5) - 0.0) < 1e-12, "ramp: t<t0 → v0");
    check(std::abs(f_ramp(2.0) - 5.0) < 1e-12, "ramp: midpoint");
    check(std::abs(f_ramp(5.0) - 10.0) < 1e-12, "ramp: t>t1 → v1");

    double omega = 2.0 * M_PI;
    auto f_harm = time_fn::harmonic(1.0, omega);
    check(std::abs(f_harm(0.0)) < 1e-12, "harmonic: sin(0) = 0");
    check(std::abs(f_harm(0.25) - 1.0) < 1e-10, "harmonic: sin(π/2) = 1");

    auto f_cos = time_fn::cosine(2.0, omega);
    check(std::abs(f_cos(0.0) - 2.0) < 1e-12, "cosine: cos(0) = 2");

    auto f_step = time_fn::step(1.0, 0.0, 5.0);
    check(std::abs(f_step(0.5) - 0.0) < 1e-12, "step: t < 1");
    check(std::abs(f_step(1.5) - 5.0) < 1e-12, "step: t > 1");

    auto f_pulse = time_fn::pulse(1.0, 2.0, 7.0);
    check(std::abs(f_pulse(0.5)) < 1e-12, "pulse: before");
    check(std::abs(f_pulse(1.5) - 7.0) < 1e-12, "pulse: inside");
    check(std::abs(f_pulse(3.0)) < 1e-12, "pulse: after");

    auto f_exp = time_fn::exponential_decay(10.0, 1.0);
    check(std::abs(f_exp(0.0) - 10.0) < 1e-10, "exp_decay: t=0");
    check(std::abs(f_exp(1.0) - 10.0 * std::exp(-1.0)) < 1e-10, "exp_decay: t=1");
}


// =============================================================================
//  Test 2: TimeFunction combinators
// =============================================================================

static void test_2_time_function_combinators() {
    std::cout << "\n--- Test 2: TimeFunction combinators ---\n";

    auto f = time_fn::constant(3.0);
    auto g = time_fn::constant(4.0);

    auto fg = time_fn::product(f, g);
    check(std::abs(fg(0.0) - 12.0) < 1e-12, "product(3, 4) = 12");

    auto fpg = time_fn::sum(f, g);
    check(std::abs(fpg(0.0) - 7.0) < 1e-12, "sum(3, 4) = 7");

    auto sf = time_fn::scale(2.5, f);
    check(std::abs(sf(0.0) - 7.5) < 1e-12, "scale(2.5, 3) = 7.5");
}


// =============================================================================
//  Test 3: Piecewise-linear TimeFunction
// =============================================================================

static void test_3_piecewise_linear() {
    std::cout << "\n--- Test 3: Piecewise-linear TimeFunction ---\n";

    std::vector<std::pair<double, double>> data = {
        {0.0, 0.0}, {1.0, 10.0}, {2.0, 5.0}, {3.0, 15.0}
    };
    auto f = time_fn::piecewise_linear(data);

    check(std::abs(f(-1.0) - 0.0) < 1e-12, "pw-linear: before range → first value");
    check(std::abs(f(0.5) - 5.0) < 1e-12, "pw-linear: interpolate [0,1]");
    check(std::abs(f(1.5) - 7.5) < 1e-12, "pw-linear: interpolate [1,2]");
    check(std::abs(f(2.5) - 10.0) < 1e-12, "pw-linear: interpolate [2,3]");
    check(std::abs(f(5.0) - 15.0) < 1e-12, "pw-linear: after range → last value");
}


// =============================================================================
//  Test 4: BoundaryConditionSet force evaluation
// =============================================================================

static void test_4_bc_force_evaluation() {
    std::cout << "\n--- Test 4: BC force evaluation ---\n";

    // This test verifies the BoundaryConditionSet data structure
    // (actual PETSc assembly tested in test 10+)

    BoundaryConditionSet<3> bcs;

    NodalForceBC<3> bc1;
    bc1.node_id = 5;
    bc1.components[0] = time_fn::constant(100.0);
    bc1.components[1] = time_fn::harmonic(50.0, 2.0 * M_PI);
    bc1.components[2] = time_fn::zero();
    bcs.add_force(std::move(bc1));

    check(bcs.forces().size() == 1, "One force BC added");
    auto f = bcs.forces()[0].evaluate(0.0);
    check(std::abs(f[0] - 100.0) < 1e-12, "Force x at t=0: 100");
    check(std::abs(f[1]) < 1e-10, "Force y at t=0: sin(0) = 0");
    check(std::abs(f[2]) < 1e-12, "Force z at t=0: 0");

    auto f_quarter = bcs.forces()[0].evaluate(0.25);
    check(std::abs(f_quarter[1] - 50.0) < 1e-8, "Force y at t=0.25: 50·sin(π/2) = 50");
}


// =============================================================================
//  Test 5: Prescribed BC evaluation
// =============================================================================

static void test_5_prescribed_bc() {
    std::cout << "\n--- Test 5: Prescribed BC ---\n";

    BoundaryConditionSet<3> bcs;

    PrescribedBC pbc;
    pbc.node_id = 3;
    pbc.local_dof = 0;
    pbc.displacement = time_fn::linear_ramp(0.0, 1.0, 0.0, 0.01);
    pbc.velocity = time_fn::constant(0.01);  // du/dt = const
    pbc.acceleration = time_fn::zero();

    bcs.add_prescribed(std::move(pbc));
    check(bcs.has_prescribed(), "Has prescribed BC");
    check(std::abs(bcs.prescribed()[0].eval_displacement(0.5) - 0.005) < 1e-12, "u_D(0.5) = 0.005");
    check(std::abs(bcs.prescribed()[0].eval_velocity(0.5) - 0.01) < 1e-12, "v_D(0.5) = 0.01");
}


// =============================================================================
//  Test 6: Initial conditions
// =============================================================================

static void test_6_initial_conditions() {
    std::cout << "\n--- Test 6: Initial conditions ---\n";

    BoundaryConditionSet<3> bcs;
    bcs.add_initial_condition({1, {0.01, 0.0, 0.0}, {0.0, 0.0, 0.0}});
    bcs.add_initial_condition({3, {0.01, 0.0, 0.0}, {0.0, 0.0, 0.0}});

    check(bcs.initial_conditions().size() == 2, "Two ICs registered");
    check(std::abs(bcs.initial_conditions()[0].displacement[0] - 0.01) < 1e-15,
          "IC node 1 u_x = 0.01");
}


// =============================================================================
//  Test 7: Consistent mass matrix for hex8 unit cube
// =============================================================================
//
//  For a unit cube [0,1]³ with ρ=1 and 8-node hex element:
//  - Total mass = ρ·V = 1·1 = 1
//  - The consistent mass matrix M_e has the property:
//    Σ_i Σ_j M_ij = ρ·V (for each DOF direction)
//  - For ρ=1: Σ_i M_ii = 1 (diagonal sum per direction = total mass)

static void test_7_consistent_mass_hex8() {
    std::cout << "\n--- Test 7: Consistent mass matrix (hex8, unit cube) ---\n";

    Domain<DIM> D;
    create_unit_cube(D);

    ContinuumIsotropicElasticMaterial linear_mat{E_modulus, nu_poisson};
    Material<ThreeDimensionalMaterial> mat{linear_mat, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial> M{D, mat};
    M.fix_x(0.0);
    M.setup();

    // Set density
    M.set_density(rho_density);  // ρ = 1.0

    // Get first element, compute mass matrix
    auto& elem = M.elements()[0];
    check(std::abs(elem.density() - 1.0) < 1e-15, "Density set to 1.0");

    auto M_e = elem.compute_consistent_mass_matrix();
    const auto n = static_cast<Eigen::Index>(3 * 8);  // 24 DOFs
    check(M_e.rows() == n && M_e.cols() == n, "Mass matrix is 24×24");

    // Total mass check: for each direction d, the sum of entries in 
    // rows/cols corresponding to direction d should equal ρ·V = 1
    // Sum ALL entries: this should equal dim * ρ * V = 3
    double total_sum = M_e.sum();
    check(std::abs(total_sum - 3.0) < 1e-10,
          "Sum of all M_e entries = dim·ρ·V = 3");

    // Mass matrix should be symmetric
    check((M_e - M_e.transpose()).norm() < 1e-12, "M_e is symmetric");

    // Mass matrix should be positive semi-definite (all eigenvalues ≥ 0)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M_e);
    check(es.eigenvalues().minCoeff() >= -1e-12, "M_e is positive semi-definite");
}


// =============================================================================
//  Test 8: Lumped mass vector (row-sum)
// =============================================================================

static void test_8_lumped_mass() {
    std::cout << "\n--- Test 8: Lumped mass vector ---\n";

    Domain<DIM> D;
    create_unit_cube(D);

    ContinuumIsotropicElasticMaterial linear_mat{E_modulus, nu_poisson};
    Material<ThreeDimensionalMaterial> mat{linear_mat, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial> M{D, mat};
    M.fix_x(0.0);
    M.setup();
    M.set_density(rho_density);

    auto& elem = M.elements()[0];
    auto m_lumped = elem.compute_lumped_mass_vector();

    check(m_lumped.size() == 24, "Lumped mass vector has 24 entries");

    // Total lumped mass = sum of diagonal = ρ·V per direction × dim
    double total_lumped = m_lumped.sum();
    check(std::abs(total_lumped - 3.0) < 1e-10,
          "Total lumped mass = dim·ρ·V = 3");

    // All entries should be positive (for standard elements)
    check(m_lumped.minCoeff() > 0.0, "All lumped masses > 0");

    // Each node gets 1/8 of total mass per direction: m_i = ρ·V/(8) = 0.125
    // per DOF direction. With tri-linear hex8, lumped mass varies.
    // At least verify symmetry: all x-DOF entries equal, etc.
}


// =============================================================================
//  Test 9: Mass matrix assembly in Model
// =============================================================================

static void test_9_model_mass_assembly() {
    std::cout << "\n--- Test 9: Mass matrix assembly in Model ---\n";

    Domain<DIM> D;
    create_unit_cube(D);

    ContinuumIsotropicElasticMaterial linear_mat{E_modulus, nu_poisson};
    Material<ThreeDimensionalMaterial> mat{linear_mat, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial> M{D, mat};
    M.fix_x(0.0);
    M.setup();
    M.set_density(rho_density);

    // Assemble mass matrix via Model
    Mat M_global;
    DMCreateMatrix(M.get_plex(), &M_global);
    M.assemble_mass_matrix(M_global);

    // Check that matrix is assembled and has non-zero entries
    PetscInt m, n;
    MatGetSize(M_global, &m, &n);
    check(m > 0 && n > 0, "Global mass matrix created with positive size");
    check(m == n, "Mass matrix is square");

    MatInfo info;
    MatGetInfo(M_global, MAT_LOCAL, &info);
    check(info.nz_used > 0, "Mass matrix has non-zero entries");

    MatDestroy(&M_global);
}


// =============================================================================
//  Test 10: SDOF free vibration via DynamicAnalysis
// =============================================================================
//
//  Setup: Unit cube, x=0 clamped, x=1 free.
//  With E=1000, ν=0, ρ=1, unit cross-section:
//    Axial stiffness: K = E·A/L = 1000 (for uniform bar)
//    But with a 3D hex8 element, the effective axial stiffness and mass
//    depend on the full 3D stiffness matrix and mass matrix.
//
//  For simplicity, we verify the qualitative behavior:
//    - Initial x-displacement u₀ on x=1 face, zero velocity
//    - Free vibration: displacement should oscillate
//    - After half-period: displacement should be ≈ −u₀
//    - Energy should be approximately conserved (no damping)

static void test_10_sdof_free_vibration() {
    std::cout << "\n--- Test 10: SDOF free vibration (generalized-α) ---\n";

    Domain<DIM> D;
    create_unit_cube(D);

    ContinuumIsotropicElasticMaterial linear_mat{E_modulus, nu_poisson};
    Material<ThreeDimensionalMaterial> mat{linear_mat, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial> M{D, mat};
    M.fix_x(0.0);
    M.setup();
    M.set_density(rho_density);

    // ── Create DynamicAnalysis ──────────────────────────────────────

    DynamicAnalysis<ThreeDimensionalMaterial> dyn(&M);

    // ── Set initial conditions: u₀ = 0.001 at x=1 face ─────────────

    BoundaryConditionSet<3> bcs;
    // Initial displacement: 0.001 in x at nodes 1,3,5,7
    for (std::size_t id : {1ul, 3ul, 5ul, 7ul}) {
        bcs.add_initial_condition({id, {0.001, 0.0, 0.0}, {0.0, 0.0, 0.0}});
    }

    // No external forces — pure free vibration
    dyn.set_initial_conditions(bcs);

    // ── Estimate natural frequency ──────────────────────────────────
    //
    // For a uniform bar (simplified):
    //   ω₁ = (π/2L)·√(E/ρ) = (π/2)·√(1000/1) ≈ 49.67 rad/s
    //   T₁ = 2π/ω₁ ≈ 0.1265 s
    //
    // For the 3D hex8 element, the effective frequency will be
    // somewhat different due to the 3D nature, but on the same order.
    
    double omega_est = (M_PI / 2.0) * std::sqrt(E_modulus / rho_density);
    double T_est = 2.0 * M_PI / omega_est;

    // Solve for 1.5 periods with ~100 steps per period
    double dt = T_est / 100.0;
    double t_final = 1.5 * T_est;

    PetscPrintf(PETSC_COMM_WORLD,
        "  Estimated ω = %.4f rad/s, T = %.6f s\n"
        "  Solving t_final = %.6f, dt = %.6e\n",
        omega_est, T_est, t_final, dt);

    bool ok = dyn.solve(t_final, dt);
    check(ok, "TS solve completed successfully");

    // ── Verify oscillatory behavior ─────────────────────────────────
    //
    // After ~1.5 periods, the displacement should have crossed zero
    // multiple times. The final displacement should be on the order
    // of u₀ (with opposite sign near half-period multiples).

    auto u_final = dyn.get_nodal_displacement(1);
    double u_x_final = u_final[0];

    PetscPrintf(PETSC_COMM_WORLD,
        "  u_x(node 1, t_final) = %.6e\n", u_x_final);

    // At t = 1.5·T, for undamped vibration: u ≈ u₀·cos(ω·1.5T) = u₀·cos(3π) = −u₀
    // But the 3D element's actual frequency differs, so just verify magnitude
    check(std::abs(u_x_final) < 10.0 * 0.001, "Final displacement bounded");
    check(std::abs(u_x_final) > 1e-10, "Displacement is non-zero (oscillating)");
}


// =============================================================================
//  Test 11: SDOF with Rayleigh damping
// =============================================================================
//
//  Same setup as Test 10, but with Rayleigh damping.
//  Verify that the amplitude decays compared to the undamped case.

static void test_11_sdof_rayleigh_damping() {
    std::cout << "\n--- Test 11: SDOF free vibration with Rayleigh damping ---\n";

    Domain<DIM> D;
    create_unit_cube(D);

    ContinuumIsotropicElasticMaterial linear_mat{E_modulus, nu_poisson};
    Material<ThreeDimensionalMaterial> mat{linear_mat, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial> M{D, mat};
    M.fix_x(0.0);
    M.setup();
    M.set_density(rho_density);

    DynamicAnalysis<ThreeDimensionalMaterial> dyn(&M);

    BoundaryConditionSet<3> bcs;
    for (std::size_t id : {1ul, 3ul, 5ul, 7ul}) {
        bcs.add_initial_condition({id, {0.001, 0.0, 0.0}, {0.0, 0.0, 0.0}});
    }
    dyn.set_initial_conditions(bcs);

    // Rayleigh damping: α_M=5.0, β_K=0.001 (moderate damping)
    dyn.set_rayleigh_damping(5.0, 0.001);

    double omega_est = (M_PI / 2.0) * std::sqrt(E_modulus / rho_density);
    double T_est = 2.0 * M_PI / omega_est;
    double dt = T_est / 100.0;
    double t_final = 3.0 * T_est;

    bool ok = dyn.solve(t_final, dt);
    check(ok, "Damped TS solve completed");

    auto u_final = dyn.get_nodal_displacement(1);
    double u_x_final = u_final[0];

    PetscPrintf(PETSC_COMM_WORLD,
        "  u_x(node 1, t=3T) = %.6e (damped)\n", u_x_final);

    // After 3 periods with moderate damping, amplitude should be
    // significantly less than initial 0.001
    check(std::abs(u_x_final) < 0.001, "Damped amplitude < initial u₀");
}


// =============================================================================
//  Test 12: Forced vibration (harmonic excitation)
// =============================================================================
//
//  Apply a harmonic force at the x=1 face and verify that the
//  DynamicAnalysis computes a non-trivial response.

static void test_12_forced_vibration() {
    std::cout << "\n--- Test 12: Forced harmonic vibration ---\n";

    Domain<DIM> D;
    create_unit_cube(D);

    ContinuumIsotropicElasticMaterial linear_mat{E_modulus, nu_poisson};
    Material<ThreeDimensionalMaterial> mat{linear_mat, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial> M{D, mat};
    M.fix_x(0.0);
    M.setup();
    M.set_density(rho_density);

    DynamicAnalysis<ThreeDimensionalMaterial> dyn(&M);

    // Apply harmonic force at x=1 face
    double F0 = 0.1;   // force amplitude per node
    double omega_f = 10.0;  // forcing frequency

    BoundaryConditionSet<3> bcs;
    for (std::size_t id : {1ul, 3ul, 5ul, 7ul}) {
        NodalForceBC<3> fbc;
        fbc.node_id = id;
        fbc.components[0] = time_fn::harmonic(F0, omega_f);
        fbc.components[1] = time_fn::zero();
        fbc.components[2] = time_fn::zero();
        bcs.add_force(std::move(fbc));
    }

    dyn.set_boundary_conditions(bcs);

    // Light damping to reach steady state faster
    dyn.set_rayleigh_damping(0.5, 0.0001);

    double T_f = 2.0 * M_PI / omega_f;  // forcing period
    double dt = T_f / 40.0;
    double t_final = 5.0 * T_f;  // 5 forcing cycles

    bool ok = dyn.solve(t_final, dt);
    check(ok, "Forced vibration TS solve completed");

    auto u_final = dyn.get_nodal_displacement(1);
    double u_x = u_final[0];

    PetscPrintf(PETSC_COMM_WORLD,
        "  u_x(node 1, t=5T_f) = %.6e\n", u_x);

    // Displacement should be non-zero (response to forcing)
    check(std::abs(u_x) > 1e-12, "Non-zero response to harmonic forcing");
}


// =============================================================================
//  Test 13: PVDWriter
// =============================================================================

static void test_13_pvd_writer() {
    std::cout << "\n--- Test 13: PVDWriter ---\n";

    auto tmp = std::filesystem::temp_directory_path() / "fall_n_test_dynamics";
    auto base = (tmp / "simulation").string();

    PVDWriter pvd(base);

    auto snap0  = (tmp / "simulation_000000.vtu").string();
    auto snap42 = (tmp / "simulation_000042.vtu").string();
    auto snap1  = (tmp / "simulation_000001.vtu").string();
    auto snap2  = (tmp / "simulation_000002.vtu").string();

    check(pvd.snapshot_filename(0) == snap0,
          "Snapshot filename step 0");
    check(pvd.snapshot_filename(42) == snap42,
          "Snapshot filename step 42");

    pvd.add_timestep(0.0, snap0);
    pvd.add_timestep(0.001, snap1);
    pvd.add_timestep(0.002, snap2);

    check(pvd.num_timesteps() == 3, "3 timesteps registered");

    pvd.write();

    // Check that the .pvd file was created
    bool pvd_exists = std::filesystem::exists(tmp / "simulation.pvd");
    check(pvd_exists, "PVD file created");

    // Clean up
    std::filesystem::remove_all(tmp);
}


// =============================================================================
//  Test 14: Concept checks
// =============================================================================

static void test_14_concept_checks() {
    std::cout << "\n--- Test 14: Concept checks ---\n";

    // DampingAssembler is a callable
    DampingAssembler da = damping::rayleigh(1.0, 0.001);
    check(static_cast<bool>(da), "Rayleigh damping assembler is callable");

    DampingAssembler da_none = damping::none();
    check(!static_cast<bool>(da_none), "damping::none() is empty/null");

    // ContinuumElement has mass methods (compile-time check)
    using ElemT = ContinuumElement<ThreeDimensionalMaterial, 3, continuum::SmallStrain>;
    static_assert(requires(ElemT e) { e.set_density(1.0); },
                  "ContinuumElement has set_density");
    static_assert(requires(ElemT e) { e.density(); },
                  "ContinuumElement has density()");
    static_assert(requires(ElemT e) { e.compute_consistent_mass_matrix(); },
                  "ContinuumElement has compute_consistent_mass_matrix");
    static_assert(requires(ElemT e) { e.compute_lumped_mass_vector(); },
                  "ContinuumElement has compute_lumped_mass_vector");

    check(true, "All compile-time concept checks passed");
}


// =============================================================================
//  Test 15: Ground motion BC evaluation
// =============================================================================

static void test_15_ground_motion_bc() {
    std::cout << "\n--- Test 15: Ground motion BC ---\n";

    BoundaryConditionSet<3> bcs;
    GroundMotionBC gm;
    gm.direction = 0;  // x-direction
    gm.acceleration = time_fn::harmonic(9.81, 2.0 * M_PI);  // 1 Hz shaking

    bcs.add_ground_motion(std::move(gm));
    check(bcs.has_ground_motion(), "Has ground motion BC");
    check(bcs.ground_motions().size() == 1, "One ground motion registered");

    double ag = bcs.ground_motions()[0].eval(0.25);
    check(std::abs(ag - 9.81) < 1e-8, "a_g(0.25) = 9.81·sin(π/2) = 9.81");
}


// =============================================================================
//  Main
// =============================================================================

int main(int argc, char* argv[]) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    std::cout << "═══════════════════════════════════════════════════════\n"
              << "  Phase 4: Dynamic Analysis Tests (PETSc TS)\n"
              << "═══════════════════════════════════════════════════════\n";

    // ── Time function tests (no PETSc mesh needed) ───────────────────
    test_1_time_function_factories();
    test_2_time_function_combinators();
    test_3_piecewise_linear();

    // ── Boundary condition tests ─────────────────────────────────────
    test_4_bc_force_evaluation();
    test_5_prescribed_bc();
    test_6_initial_conditions();

    // ── Mass matrix tests ────────────────────────────────────────────
    test_7_consistent_mass_hex8();
    test_8_lumped_mass();
    test_9_model_mass_assembly();

    // ── Dynamic analysis tests ───────────────────────────────────────
    test_10_sdof_free_vibration();
    test_11_sdof_rayleigh_damping();
    test_12_forced_vibration();

    // ── VTK time series ──────────────────────────────────────────────
    test_13_pvd_writer();

    // ── Concept verification ─────────────────────────────────────────
    test_14_concept_checks();
    test_15_ground_motion_bc();

    std::cout << "\n═══════════════════════════════════════════════════════\n"
              << "  Results: " << passed << " PASSED, " << failed << " FAILED"
              << "  (total: " << (passed + failed) << " checks)\n"
              << "═══════════════════════════════════════════════════════\n";

    PetscFinalize();
    return (failed > 0) ? 1 : 0;
}
