// =============================================================================
//  test_field_transfer.cpp — Phase 4: Beam → Continuum Field Transfer
// =============================================================================
//
//  Validates the kinematic bridge between a Timoshenko beam element and a
//  continuum (hex) sub-model:
//
//    1.  Pure-translation beam → uniform displacement on section
//    2.  Pure-bending → linear displacement distribution
//    3.  Stress reconstruction: σ_xx linear, τ_xy constant
//    4.  displacement_at_global_point projects into local frame
//    5.  compute_boundary_displacements maps to continuum nodes
//    6.  Sub-model pipeline: align mesh + compute boundary BCs
//    7.  Rotated beam: field transfer works in non-axis-aligned frames
//
//  Mesh: single 2-node Timoshenko beam (L=2, elastic, E=200, ν=0.3).
//
//  Requires PETSc runtime.
//
// =============================================================================

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <array>

#include <petsc.h>
#include <Eigen/Dense>

#include "header_files.hh"

// ── Constants ─────────────────────────────────────────────────────────────────

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

static constexpr double tol = 1e-10;


// =============================================================================
//  Test 1: Pure-translation → uniform displacement on section
// =============================================================================
//
//  If the beam centroid displaces by (ux, 0, 0) with zero rotation,
//  every point in the section should have the same displacement.

void test_pure_translation() {
    std::cout << "\nTest 1: Pure-translation → uniform section displacement\n";

    fall_n::SectionKinematics kin;
    kin.centroid = Eigen::Vector3d{5.0, 0.0, 0.0};
    kin.R = Eigen::Matrix3d::Identity();  // beam axis = global x
    kin.u_local = Eigen::Vector3d{0.01, 0.0, 0.0};  // axial displacement
    kin.theta_local = Eigen::Vector3d::Zero();

    // At any (y,z), displacement should be the same
    auto u_00 = fall_n::section_displacement_at(kin, 0.0, 0.0);
    auto u_10 = fall_n::section_displacement_at(kin, 0.1, 0.0);
    auto u_01 = fall_n::section_displacement_at(kin, 0.0, 0.1);
    auto u_11 = fall_n::section_displacement_at(kin, 0.1, 0.1);

    check((u_00 - u_10).norm() < tol, "u(0,0) == u(0.1, 0)");
    check((u_00 - u_01).norm() < tol, "u(0,0) == u(0, 0.1)");
    check((u_00 - u_11).norm() < tol, "u(0,0) == u(0.1, 0.1)");
    check(std::abs(u_00[0] - 0.01) < tol, "u_x = 0.01 (global x)");
    check(std::abs(u_00[1]) < tol, "u_y = 0");
    check(std::abs(u_00[2]) < tol, "u_z = 0");
}


// =============================================================================
//  Test 2: Pure-bending → linear displacement distribution
// =============================================================================
//
//  If the section has a rotation θ_z about local z (bending in the x-y
//  plane), points above/below the centroid should have opposite
//  axial displacements.
//
//  θ_local = (0, 0, θz):
//  u_local(y,z) = u_c + (0,0,θz) × (0,y,z) = u_c + (-θz·y, 0, 0)  (wait, let me compute)
//
//  Actually: θ × (0,y,z) where θ = (0,0,θz):
//  = (θz·z - 0, 0 - 0, 0 - θz·0) ... no, let me do this properly:
//  (0,0,θz) × (0,y,z) = (0·z - θz·y, θz·0 - 0·z, 0·y - 0·0) = (-θz·y, 0, 0)
//
//  So u_local(y,z) = u_c + (-θz·y, 0, 0)
//
//  This means: rotation about z causes axial displacement proportional to -y.

void test_pure_bending() {
    std::cout << "\nTest 2: Pure-bending → linear displacement\n";

    fall_n::SectionKinematics kin;
    kin.centroid = Eigen::Vector3d::Zero();
    kin.R = Eigen::Matrix3d::Identity();
    kin.u_local = Eigen::Vector3d::Zero();
    kin.theta_local = Eigen::Vector3d{0.0, 0.0, 0.01};  // rotation about z

    // At centroid (0,0): no displacement
    auto u_center = fall_n::section_displacement_at(kin, 0.0, 0.0);
    check(u_center.norm() < tol, "centroid displacement = 0");

    // At y=+0.5: u_x = -θz·y = -0.005
    auto u_top = fall_n::section_displacement_at(kin, 0.5, 0.0);
    check(std::abs(u_top[0] - (-0.005)) < tol, "u_x at y=+0.5 is -θz·y");

    // At y=-0.5: u_x = +0.005
    auto u_bot = fall_n::section_displacement_at(kin, -0.5, 0.0);
    check(std::abs(u_bot[0] - 0.005) < tol, "u_x at y=-0.5 is +θz·y");

    // Symmetry: u_top + u_bot ~ 0
    check((u_top + u_bot).norm() < tol, "antisymmetric about centroid");
}


// =============================================================================
//  Test 3: Stress reconstruction
// =============================================================================
//
//  Given axial strain ε₀ and curvature κ_y:
//      ε_xx(y,z) = ε₀ − z·κ_y
//      σ_xx = E·ε_xx

void test_stress_reconstruction() {
    std::cout << "\nTest 3: Stress reconstruction\n";

    fall_n::SectionKinematics kin;
    kin.E  = 200.0;
    kin.G  = 80.0;
    kin.nu = 0.25;
    kin.eps_0   = 0.001;    // 0.1% axial strain
    kin.kappa_y = 0.002;    // curvature about y
    kin.kappa_z = 0.0;
    kin.gamma_y = 0.0005;   // shear
    kin.gamma_z = 0.0;

    // At (y=0, z=0): ε_xx = ε₀ = 0.001
    auto sig_00 = fall_n::section_stress_at(kin, 0.0, 0.0);
    check(std::abs(sig_00[0] - 200.0 * 0.001) < tol, "σ_xx at centroid = E·ε₀");
    check(std::abs(sig_00[3] - 80.0 * 0.0005) < tol, "τ_xy = G·γ_y");

    // At (y=0, z=+0.1): ε_xx = 0.001 - 0.1*0.002 = 0.0008
    auto sig_z1 = fall_n::section_stress_at(kin, 0.0, 0.1);
    double expected_eps = 0.001 - 0.1 * 0.002;
    check(std::abs(sig_z1[0] - 200.0 * expected_eps) < tol,
          "σ_xx varies linearly with z");

    // Strain reconstruction
    auto eps_00 = fall_n::section_strain_at(kin, 0.0, 0.0);
    check(std::abs(eps_00[0] - 0.001) < tol, "ε_xx at centroid = ε₀");
    check(std::abs(eps_00[3] - 0.0005) < tol, "γ_xy at centroid");

    // σ_yy, σ_zz, τ_yz = 0 (beam theory)
    check(std::abs(sig_00[1]) < tol, "σ_yy = 0");
    check(std::abs(sig_00[2]) < tol, "σ_zz = 0");
    check(std::abs(sig_00[5]) < tol, "τ_yz = 0");
}


// =============================================================================
//  Test 4: displacement_at_global_point projects correctly
// =============================================================================

void test_global_point_projection() {
    std::cout << "\nTest 4: displacement_at_global_point projection\n";

    // Beam at (1,0,0) with local frame = identity (beam axis = global X)
    // Rotation θ_z = 0.01 as before
    fall_n::SectionKinematics kin;
    kin.centroid = Eigen::Vector3d{1.0, 0.0, 0.0};
    kin.R = Eigen::Matrix3d::Identity();
    kin.u_local = Eigen::Vector3d{0.0, 0.002, 0.0};  // transverse displacement
    kin.theta_local = Eigen::Vector3d{0.0, 0.0, 0.01};

    // Point at (1.0, 0.3, 0.0) — offset in global y = local y = +0.3
    Eigen::Vector3d P{1.0, 0.3, 0.0};
    auto u_gp = fall_n::displacement_at_global_point(kin, P);

    // Should equal section_displacement_at(kin, 0.3, 0.0)
    auto u_ref = fall_n::section_displacement_at(kin, 0.3, 0.0);
    check((u_gp - u_ref).norm() < tol, "global_point matches section_displacement");
}


// =============================================================================
//  Test 5: compute_boundary_displacements maps to continuum nodes
// =============================================================================

void test_boundary_displacement_mapping() {
    std::cout << "\nTest 5: compute_boundary_displacements\n";

    // Create a small prismatic mesh: 2×2×1 (4 nodes on each z-face)
    fall_n::PrismaticSpec pspec{
        .width  = 0.4,
        .height = 0.6,
        .length = 1.0,
        .nx = 2, .ny = 2, .nz = 1,
        .origin = {0.0, 0.0, 0.5},  // centred at z=0.5 → z in [0, 1]
        .e_x = {1.0, 0.0, 0.0},
        .e_y = {0.0, 1.0, 0.0},
        .e_z = {0.0, 0.0, 1.0}
    };
    auto [domain, grid] = fall_n::make_prismatic_domain(pspec);

    // Beam section kinematics at z=0 face
    fall_n::SectionKinematics kin;
    kin.centroid = Eigen::Vector3d{0.0, 0.0, 0.0};  // at z=0
    kin.R = Eigen::Matrix3d{
        {0, 0, 1},  // local x = global z (beam axis)
        {1, 0, 0},  // local y = global x
        {0, 1, 0}   // local z = global y
    };
    kin.u_local = Eigen::Vector3d{0.001, 0.0, 0.0};  // axial (along z)
    kin.theta_local = Eigen::Vector3d::Zero();

    auto face_min = grid.nodes_on_face(fall_n::PrismFace::MinZ);
    auto bcs = fall_n::compute_boundary_displacements(kin, domain, face_min);

    check(bcs.size() == face_min.size(),
          "one BC per boundary node");

    // Pure translation → all nodes should have same displacement
    bool all_equal = true;
    for (std::size_t i = 1; i < bcs.size(); ++i) {
        if ((bcs[i].second - bcs[0].second).norm() > tol) {
            all_equal = false;
            break;
        }
    }
    check(all_equal, "pure translation → uniform BCs");

    // Displacement should be (0, 0, 0.001) in global frame
    // because u_local = (0.001, 0, 0) and R^T maps local x to global z
    const auto u_expected = kin.R.transpose() * kin.u_local;
    check((bcs[0].second - u_expected).norm() < tol,
          "BC displacement matches R^T * u_local");
}


// =============================================================================
//  Test 6: Sub-model pipeline — prismatic mesh + boundary BCs
// =============================================================================
//
//  Exercises the same composition that build_beam_submodel performs:
//    align_to_beam → make_prismatic_domain → compute_boundary_displacements
//  using manually constructed SectionKinematics (no real beam solve).
//
//  Beam: cantilever along global X, L=2, section 0.3×0.5.
//  Clamped end (A, x=0) has zero displacement.
//  Free end   (B, x=2) has transverse displacement and rotation.

void test_submodel_pipeline() {
    std::cout << "\nTest 6: Sub-model pipeline (mesh + BCs)\n";

    const double b = 0.3;   // section width  (y-direction in beam local)
    const double h = 0.5;   // section height (z-direction in beam local)
    const double L = 2.0;

    // Beam endpoints
    std::array<double, 3> A{0.0, 0.0, 0.0};
    std::array<double, 3> B{L,   0.0, 0.0};
    std::array<double, 3> up{0.0, 1.0, 0.0};  // local y = global Y

    // Create prismatic hex mesh aligned to the beam
    auto pspec = fall_n::align_to_beam(A, B, up, b, h, 2, 2, 4);
    auto [domain, grid] = fall_n::make_prismatic_domain(pspec);

    check(grid.total_nodes() == 3 * 3 * 5,
          "prismatic mesh has correct node count");
    check(grid.total_elements() == 2 * 2 * 4,
          "prismatic mesh has correct element count");

    // Beam R matrix: beam axis = global X → local x = global X
    // R maps global→local, so R = I for an axis-aligned beam
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();

    // Clamped end kinematics (ξ=-1 → z=0 face): zero displacement
    fall_n::SectionKinematics kin_A;
    kin_A.centroid = Eigen::Vector3d{0.0, 0.0, 0.0};
    kin_A.R = R;
    kin_A.u_local     = Eigen::Vector3d::Zero();
    kin_A.theta_local = Eigen::Vector3d::Zero();
    kin_A.E = 200.0;  kin_A.G = 80.0;  kin_A.nu = 0.25;

    // Free end kinematics (ξ=+1 → z=L face): transverse + rotation
    fall_n::SectionKinematics kin_B;
    kin_B.centroid = Eigen::Vector3d{L, 0.0, 0.0};
    kin_B.R = R;
    kin_B.u_local     = Eigen::Vector3d{0.0, 0.005, 0.0};  // transverse
    kin_B.theta_local = Eigen::Vector3d{0.0, 0.0, 0.002};  // rotation about z
    kin_B.E = 200.0;  kin_B.G = 80.0;  kin_B.nu = 0.25;

    // Compute boundary displacements at both faces
    auto face_min = grid.nodes_on_face(fall_n::PrismFace::MinZ);
    auto face_max = grid.nodes_on_face(fall_n::PrismFace::MaxZ);

    auto bc_A = fall_n::compute_boundary_displacements(kin_A, domain, face_min);
    auto bc_B = fall_n::compute_boundary_displacements(kin_B, domain, face_max);

    check(!bc_A.empty(), "MinZ BCs populated");
    check(!bc_B.empty(), "MaxZ BCs populated");
    check(bc_A.size() == face_min.size(), "one BC per MinZ node");
    check(bc_B.size() == face_max.size(), "one BC per MaxZ node");

    // Clamped end: all BCs should be zero
    double max_bc_A = 0.0;
    for (const auto& [id, u] : bc_A)
        max_bc_A = std::max(max_bc_A, u.norm());
    check(max_bc_A < tol, "clamped end BCs are zero");

    // Free end: BCs should be non-zero (has rotation → linear variation)
    double max_bc_B = 0.0;
    for (const auto& [id, u] : bc_B)
        max_bc_B = std::max(max_bc_B, u.norm());
    check(max_bc_B > tol, "free end BCs are non-zero");

    // Free end centroid node displacement = R^T * u_local = (0, 0.005, 0)
    // (rotation adds axial component proportional to y offset)
    // Verify that at least one BC differs from the centroid value (due to rotation)
    Eigen::Vector3d u_cent = R.transpose() * kin_B.u_local;
    bool has_variation = false;
    for (const auto& [id, u] : bc_B) {
        if ((u - u_cent).norm() > tol) {
            has_variation = true;
            break;
        }
    }
    check(has_variation, "rotation causes spatial variation in free-end BCs");

    std::cout << "    max |BC|_A = " << max_bc_A
              << ", max |BC|_B = " << max_bc_B << "\n";
}


// =============================================================================
//  Test 7: Rotated beam — field transfer in non-axis-aligned frame
// =============================================================================

void test_rotated_beam_transfer() {
    std::cout << "\nTest 7: Rotated beam field transfer\n";

    // Beam with rotation: local x = (1,1,0)/√2, local y = (-1,1,0)/√2, local z = (0,0,1)
    fall_n::SectionKinematics kin;
    kin.centroid = Eigen::Vector3d{1.0, 1.0, 0.0};

    // R maps global → local: R * v_global = v_local
    // If local axes in global are:
    //   e_x_local = (1,1,0)/√2  (beam axis)
    //   e_y_local = (-1,1,0)/√2
    //   e_z_local = (0,0,1)
    // Then R^T = [e_x_local | e_y_local | e_z_local]
    // So R = (R^T)^T
    const double s2 = 1.0 / std::sqrt(2.0);
    Eigen::Matrix3d Rt;
    Rt.col(0) = Eigen::Vector3d{ s2,  s2, 0.0};  // local x in global
    Rt.col(1) = Eigen::Vector3d{-s2,  s2, 0.0};  // local y in global
    Rt.col(2) = Eigen::Vector3d{ 0.0, 0.0, 1.0}; // local z in global
    kin.R = Rt.transpose();  // R maps global → local

    // Pure axial displacement in beam local frame
    kin.u_local = Eigen::Vector3d{0.01, 0.0, 0.0};
    kin.theta_local = Eigen::Vector3d::Zero();

    // At centroid, global displacement = R^T * (0.01, 0, 0)
    auto u_cent = fall_n::section_displacement_at(kin, 0.0, 0.0);
    Eigen::Vector3d expected = Rt * Eigen::Vector3d{0.01, 0.0, 0.0};
    check((u_cent - expected).norm() < tol,
          "centroid displacement correct in rotated frame");

    // A point offset by 0.1 in local y → global direction (-s2, s2, 0)*0.1
    Eigen::Vector3d P_global = kin.centroid + 0.1 * Rt.col(1);
    auto u_gp = fall_n::displacement_at_global_point(kin, P_global);
    auto u_ref = fall_n::section_displacement_at(kin, 0.1, 0.0);
    check((u_gp - u_ref).norm() < tol,
          "rotated global point maps to correct section coords");

    // With rotation θ about local z: u_x_local = -θz*y
    kin.theta_local = Eigen::Vector3d{0.0, 0.0, 0.005};
    auto u_off = fall_n::section_displacement_at(kin, 0.2, 0.0);
    // Expected: u_local = (0.01 - 0.005*0.2, 0, 0) = (0.009, 0, 0)
    // u_global = R^T * (0.009, 0, 0)
    Eigen::Vector3d exp2 = Rt * Eigen::Vector3d{0.009, 0.0, 0.0};
    check((u_off - exp2).norm() < tol,
          "bending contribution correct in rotated frame");
}


// =============================================================================
//  main
// =============================================================================

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    std::cout << std::string(60, '=') << "\n"
              << "  Phase 4: Beam -> Continuum Field Transfer Verification\n"
              << std::string(60, '=') << "\n";

    test_pure_translation();
    test_pure_bending();
    test_stress_reconstruction();
    test_global_point_projection();
    test_boundary_displacement_mapping();
    test_submodel_pipeline();
    test_rotated_beam_transfer();

    std::cout << "\n" << std::string(60, '=') << "\n"
              << "  Summary: " << passed << " passed, " << failed << " failed\n"
              << std::string(60, '=') << "\n";

    PetscFinalize();
    return (failed > 0) ? 1 : 0;
}
