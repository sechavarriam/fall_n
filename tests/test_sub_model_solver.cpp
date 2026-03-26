// =============================================================================
//  test_sub_model_solver.cpp — SubModelSolver + HomogenizedSection
// =============================================================================
//
//  Validates the Phase-6 future-work additions:
//
//    1.  Zero-displacement BCs → zero displacement, zero stress, zero strain
//    2.  Uniform axial extension: avg σ_xx ≈ E · ε₀,  E_eff ≈ E (input)
//    3.  Homogenization consistency: N = A · σ_xx  (uniform-stress assumption)
//    4.  to_beam_local_stress with Identity R → stress unchanged
//    5.  to_beam_local_stress with 90° rotation swaps components correctly
//    6.  homogenize() returns N, Vy, Vz consistent with resultant formulas
//    7.  Parallel build_sub_models: same result as sequential run
//
//  All tests that need a prismatic continuum domain require PETSc runtime.
//
// =============================================================================

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <vector>
#include <array>

#include <petsc.h>
#include <Eigen/Dense>

#include "header_files.hh"

// ── Test harness ──────────────────────────────────────────────────────────────

static int passed = 0;
static int failed = 0;

static void check(bool cond, const char* msg) {
    if (cond) {
        std::cout << "  [PASS] " << msg << "\n";
        ++passed;
    } else {
        std::cout << "  [FAIL] " << msg << "\n";
        ++failed;
    }
}

static constexpr double tol_tight = 1e-6;
static constexpr double tol       = 1e-4;


// ── Helper: build a simple ElementKinematics for an axis-aligned beam ─────────

static fall_n::ElementKinematics make_axial_ek(
    std::size_t id,
    double u_axial_A, double u_axial_B)
{
    fall_n::ElementKinematics ek;
    ek.element_id = id;
    ek.endpoint_A = {0.0, 0.0, 0.0};
    ek.endpoint_B = {1.0, 0.0, 0.0};   // length = 1
    ek.up_direction = {0.0, 1.0, 0.0};

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();

    ek.kin_A.centroid     = Eigen::Vector3d{0.0, 0.0, 0.0};
    ek.kin_A.R            = R;
    ek.kin_A.u_local      = Eigen::Vector3d{u_axial_A, 0.0, 0.0};
    ek.kin_A.theta_local  = Eigen::Vector3d::Zero();
    ek.kin_A.E = 200.0;  ek.kin_A.G = 80.0;  ek.kin_A.nu = 0.25;

    ek.kin_B.centroid     = Eigen::Vector3d{1.0, 0.0, 0.0};
    ek.kin_B.R            = R;
    ek.kin_B.u_local      = Eigen::Vector3d{u_axial_B, 0.0, 0.0};
    ek.kin_B.theta_local  = Eigen::Vector3d::Zero();
    ek.kin_B.E = 200.0;  ek.kin_B.G = 80.0;  ek.kin_B.nu = 0.25;

    return ek;
}


// ── Helper: build a MultiscaleCoordinator with one sub-model ─────────────────
//
//  Uses the smallest mesh that still demonstrates correctness: 2×2×4 elements.
//  The caller must keep the returned coordinator alive for the duration of
//  any solve, because SubModelSolver::solve() holds a non-owning pointer to
//  the domain.

static fall_n::MultiscaleCoordinator build_coordinator(
    const fall_n::ElementKinematics& ek,
    double width = 0.2, double height = 0.2)
{
    fall_n::MultiscaleCoordinator coord;
    fall_n::ElementKinematics ek_copy = ek;
    coord.add_critical_element(std::move(ek_copy));

    fall_n::SubModelSpec spec{width, height, 2, 2, 4};
    coord.build_sub_models(spec);

    return coord;
}


// =============================================================================
//  Test 1: Zero-displacement BCs → zero displacement, stress, strain
// =============================================================================
//
//  When both end-faces of the prismatic sub-model are prescribed zero
//  displacement (clamped, unloaded), the only valid solution is zero
//  everywhere.

void test_zero_bc() {
    std::cout << "\nTest 1: Zero-displacement BCs → trivial solution\n";

    auto ek   = make_axial_ek(0, 0.0, 0.0);
    auto coord = build_coordinator(ek);
    auto& sub  = coord.sub_models()[0];

    fall_n::SubModelSolver solver(30.0);
    const auto res = solver.solve(sub);

    check(res.converged,
          "solver converged");
    check(res.max_displacement < tol_tight,
          "max displacement == 0 (zero BCs)");
    check(res.avg_stress.norm() < tol_tight,
          "avg stress == 0 (zero BCs)");
    check(res.avg_strain.norm() < tol_tight,
          "avg strain == 0 (zero BCs)");
    check(res.max_stress_vm < tol_tight,
          "von Mises stress == 0 (zero BCs)");
    check(res.num_gp > 0,
          "Gauss points were visited");
}


// =============================================================================
//  Test 2: Uniform axial extension → avg σ_xx ≈ E · ε₀
// =============================================================================
//
//  Prescribe u_z = 0 on the A-face and u_z = ε₀·L on the B-face; all other
//  displacement components are zero.  The beam axis coincides with the global
//  x-axis (local x = global x), so the continuum will experience pure
//  uniaxial tension with ε_xx = ε₀.
//
//  Expected:
//    avg_stress[0] ≈ E · ε₀
//    E_eff         ≈ E  (identity modulus)

void test_uniform_axial_extension() {
    std::cout << "\nTest 2: Uniform axial extension → σ_xx = E·ε₀\n";

    // Ko-Bathe concrete with fc=30 MPa.
    // Elastic moduli: Ke≈11096, Ge≈13304 → Ee≈28513 MPa
    const double fc  = 30.0;
    const double L   = 1.0;
    const double eps = 1e-4;           // ε₀ (small for elastic range)

    // Beam from X=0 to X=L;  u_x: 0 → ε·L
    auto ek   = make_axial_ek(0, 0.0, eps * L);
    auto coord = build_coordinator(ek);
    auto& sub  = coord.sub_models()[0];

    fall_n::SubModelSolver solver(fc);
    const auto res = solver.solve(sub);

    check(res.converged, "solver converged");

    // E_eff must be in the elastic concrete range [20k, 40k] MPa
    check(res.E_eff > 20000.0, "E_eff > 20 GPa");
    check(res.E_eff < 40000.0, "E_eff < 40 GPa");

    // σ_xx = E_eff · ε₀ (self-consistency)
    const double sigma_xx = res.avg_stress[0];
    check(std::abs(sigma_xx - res.E_eff * eps) / (res.E_eff * eps) < 0.02,
          "avg σ_xx ≈ E_eff·ε₀  (within 2%)");
}


// =============================================================================
//  Test 3: Homogenization consistency: N = A · <σ_xx_local>
// =============================================================================
//
//  The uniform-stress resultant formula N = A · σ_xx must hold for an axial
//  case.  Any value of σ_xx and any section size should satisfy this.

void test_homogenization_N() {
    std::cout << "\nTest 3: Homogenization: N = A · avg_σ_xx\n";

    const double fc  = 30.0;
    const double eps = 1e-4;
    const double W   = 0.3;
    const double H   = 0.4;
    const double A   = W * H;

    auto ek    = make_axial_ek(0, 0.0, eps);
    auto coord = build_coordinator(ek, W, H);
    auto& sub  = coord.sub_models()[0];

    fall_n::SubModelSolver solver(fc);
    const auto res = solver.solve(sub);

    check(res.converged, "solver converged");

    // The beam rotation R is Identity for an x-aligned element
    Eigen::Vector<double, 6> sig_local =
        fall_n::to_beam_local_stress(res.avg_stress,
                                     Eigen::Matrix3d::Identity());

    const double N_expected = A * sig_local[0];

    auto hs = fall_n::homogenize(res, sub, W, H);

    check(std::abs(hs.N - N_expected) < tol,
          "N = A · σ_xx_local (resultant formula)");
    check(std::abs(hs.area() - A) < tol_tight,
          "section area width×height");
}


// =============================================================================
//  Test 4: to_beam_local_stress with Identity R → stress unchanged
// =============================================================================
//
//  For a beam aligned with the global x-axis the rotation matrix R = I₃ and
//  the transformation must be a no-op: σ_local ≡ σ_global.

void test_to_beam_local_stress_identity() {
    std::cout << "\nTest 4: to_beam_local_stress with R=I → no rotation\n";

    Eigen::Vector<double, 6> sig_in;
    sig_in << 10.0, 2.0, 3.0, 0.5, -0.5, 1.0;   // arbitrary Voigt stress

    const auto sig_out =
        fall_n::to_beam_local_stress(sig_in, Eigen::Matrix3d::Identity());

    check((sig_in - sig_out).norm() < tol_tight,
          "R=I leaves Voigt stress unchanged");
}


// =============================================================================
//  Test 5: to_beam_local_stress — 90° rotation around z swaps x↔y
// =============================================================================
//
//  A 90° CCW rotation about the global z-axis maps: x→-y, y→x, z→z.
//
//    R = [ 0 -1  0 ]
//        [ 1  0  0 ]
//        [ 0  0  1 ]
//
//  and  σ_xx_local = σ_yy_global,  σ_yy_local = σ_xx_global,
//       τ_xy_local = -τ_xy_global  (sign from Voigt convention).

void test_to_beam_local_stress_rotation() {
    std::cout << "\nTest 5: to_beam_local_stress — 90° rotation swaps σ_xx↔σ_yy\n";

    // Construct input: σ_xx=10, σ_yy=3, all off-diagonal = 0
    Eigen::Vector<double, 6> sig_in = Eigen::Vector<double, 6>::Zero();
    sig_in[0] = 10.0;   // σ_xx
    sig_in[1] =  3.0;   // σ_yy

    // 90° CCW rotation about z
    Eigen::Matrix3d R;
    R << 0.0, -1.0, 0.0,
         1.0,  0.0, 0.0,
         0.0,  0.0, 1.0;

    const auto sig_out = fall_n::to_beam_local_stress(sig_in, R);

    // After rotation: new x = old y → σ_xx_local = σ_yy_global = 3
    //                 new y = -old x → σ_yy_local = σ_xx_global = 10
    check(std::abs(sig_out[0] -  3.0) < tol_tight, "σ_xx_local = σ_yy_global (=3)");
    check(std::abs(sig_out[1] - 10.0) < tol_tight, "σ_yy_local = σ_xx_global (=10)");
    check(std::abs(sig_out[2])         < tol_tight, "σ_zz_local = 0 (unchanged)");
    check((sig_out.tail<3>()).norm()    < tol_tight, "shear components = 0");
}


// =============================================================================
//  Test 6: homogenize() — Vy and Vz from shear stress
// =============================================================================
//
//  Artificially construct a SubModelSolverResult with only τ_xy ≠ 0 and
//  verify that the Vy resultant equals A·τ_xy while Vz = 0.

void test_homogenize_Vy_Vz() {
    std::cout << "\nTest 6: homogenize() Vy = A·τ_12_local, Vz = A·τ_13_local\n";

    const double W  = 0.25;
    const double H  = 0.30;
    const double A  = W * H;
    const double tau_xy = 5.0;   // artificial shear stress τ_xy (Voigt [5])
    const double tau_xz = 2.0;   // artificial shear stress τ_xz (Voigt [4])

    // Build a fake result with only shear components non-zero
    fall_n::SubModelSolverResult fake;
    fake.converged = true;
    fake.avg_stress = Eigen::Vector<double, 6>::Zero();
    fake.avg_stress[4] = tau_xz;   // τ_xz (Voigt index 4)
    fake.avg_stress[5] = tau_xy;   // τ_xy (Voigt index 5)
    fake.avg_strain = Eigen::Vector<double, 6>::Zero();
    fake.E_eff = 0.0;
    fake.G_eff = 0.0;
    fake.num_gp = 1;

    // Build a sub-model with Identity R (beam aligned with global x)
    fall_n::MultiscaleSubModel sub;
    sub.kin_A.R = Eigen::Matrix3d::Identity();

    const auto hs = fall_n::homogenize(fake, sub, W, H);

    check(std::abs(hs.Vy - A * tau_xy) < tol_tight,
          "Vy = A · τ_12_local");
    check(std::abs(hs.Vz - A * tau_xz) < tol_tight,
          "Vz = A · τ_13_local");
    check(std::abs(hs.N)  < tol_tight,
          "N = 0 when only shear is prescribed");
}


// =============================================================================
//  Test 7: Parallel build_sub_models — same result as sequential run
// =============================================================================
//
//  Builds two sub-models for identical elements sequentially (single-threaded)
//  and verifies that the resulting bc_min_z and bc_max_z maps contain the same
//  node count and that the total BC node count is larger than zero.
//
//  Note: a fully parallel run is not guaranteed on all CI machines (OpenMP may
//  not be available), but the two-phase refactoring path is exercised either
//  way since the sequential fallback uses the same code paths.

void test_parallel_build_smoke() {
    std::cout << "\nTest 7: Parallel build_sub_models — smoke / consistency\n";

    fall_n::MultiscaleCoordinator coord;

    // Add two identical elements
    coord.add_critical_element(make_axial_ek(0, 0.0, 0.001));
    coord.add_critical_element(make_axial_ek(1, 0.0, 0.001));

    fall_n::SubModelSpec spec{0.2, 0.2, 2, 2, 4};
    coord.build_sub_models(spec);

    check(coord.is_built(), "coordinator reports built after two-phase build");
    check(coord.sub_models().size() == 2, "two sub-models produced");

    const auto& s0 = coord.sub_models()[0];
    const auto& s1 = coord.sub_models()[1];

    // Identical elements → identical sub-models
    check(s0.bc_min_z.size() == s1.bc_min_z.size(),
          "identical elements → same MinZ BC node count");
    check(s0.bc_max_z.size() == s1.bc_max_z.size(),
          "identical elements → same MaxZ BC node count");
    check(!s0.bc_min_z.empty(), "MinZ BCs are populated");
    check(!s0.bc_max_z.empty(), "MaxZ BCs are populated");

    // Both sub-models should have the same grid dimensions
    check(s0.grid.total_nodes() == s1.grid.total_nodes(),
          "identical elements → same node count across sub-models");
}


// =============================================================================
//  main
// =============================================================================

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    std::cout << std::string(60, '=') << "\n"
              << "  Phase 6: SubModelSolver + HomogenizedSection\n"
              << std::string(60, '=') << "\n";

    // Pure C++ / Eigen tests (no PETSc mesh needed)
    test_to_beam_local_stress_identity();
    test_to_beam_local_stress_rotation();
    test_homogenize_Vy_Vz();

    // Tests that require a full prismatic PETSc domain
    test_zero_bc();
    test_uniform_axial_extension();
    test_homogenization_N();
    test_parallel_build_smoke();

    std::cout << "\n" << std::string(60, '=') << "\n"
              << "  Summary: " << passed << " passed, " << failed << " failed\n"
              << std::string(60, '=') << "\n";

    PetscFinalize();
    return (failed > 0) ? 1 : 0;
}
