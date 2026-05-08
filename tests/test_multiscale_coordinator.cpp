// =============================================================================
//  test_multiscale_coordinator.cpp — Phase 5: MultiscaleCoordinator
// =============================================================================
//
//  Validates the downscaling orchestrator that converts beam analysis
//  results into continuum sub-models:
//
//    1.  Single-element downscaling: one critical element → one sub-model
//    2.  Multi-element downscaling: two elements → two sub-models
//    3.  Report statistics: node/element counts, displacement summary
//    4.  Clear and rebuild: coordinator is reusable
//    5.  Zero-displacement element: clamped beam → zero BCs everywhere
//    6.  Mixed state: one deformed + one undeformed element
//    7.  Rotated element: non-axis-aligned beam → correct BCs
//
//  All tests construct SectionKinematics manually (no PETSc solve needed).
//  Requires PETSc runtime for Domain<3> construction.
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

static constexpr double tol = 1e-10;


// ── Helper: make ElementKinematics for an axis-aligned beam ──────────────────

static fall_n::ElementKinematics make_beam_kinematics(
    std::size_t id,
    double x_A, double x_B,
    const Eigen::Vector3d& u_A, const Eigen::Vector3d& theta_A,
    const Eigen::Vector3d& u_B, const Eigen::Vector3d& theta_B)
{
    fall_n::ElementKinematics ek;
    ek.element_id = id;

    ek.endpoint_A = {x_A, 0.0, 0.0};
    ek.endpoint_B = {x_B, 0.0, 0.0};
    ek.up_direction = {0.0, 1.0, 0.0};

    // Beam along global X → R = I (local x = global X)
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();

    ek.kin_A.centroid = Eigen::Vector3d(x_A, 0.0, 0.0);
    ek.kin_A.R = R;
    ek.kin_A.u_local = u_A;
    ek.kin_A.theta_local = theta_A;
    ek.kin_A.E = 200.0;  ek.kin_A.G = 80.0;  ek.kin_A.nu = 0.25;

    ek.kin_B.centroid = Eigen::Vector3d(x_B, 0.0, 0.0);
    ek.kin_B.R = R;
    ek.kin_B.u_local = u_B;
    ek.kin_B.theta_local = theta_B;
    ek.kin_B.E = 200.0;  ek.kin_B.G = 80.0;  ek.kin_B.nu = 0.25;

    return ek;
}


// =============================================================================
//  Test 1: Single-element downscaling
// =============================================================================

void test_single_element() {
    std::cout << "\nTest 1: Single-element downscaling\n";

    fall_n::MultiscaleCoordinator coord;

    auto ek = make_beam_kinematics(
        42,
        0.0, 2.0,
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        Eigen::Vector3d(0.0, 0.003, 0.0), Eigen::Vector3d(0.0, 0.0, 0.001));

    coord.add_critical_element(std::move(ek));
    check(coord.num_critical() == 1, "one critical element registered");
    check(!coord.is_built(), "not built yet");

    fall_n::SubModelSpec spec{0.3, 0.5, 2, 2, 4};
    coord.build_sub_models(spec);

    check(coord.is_built(), "built after build_sub_models");
    check(coord.sub_models().size() == 1, "one sub-model created");

    const auto& sub = coord.sub_models()[0];
    check(sub.parent_element_id == 42, "parent element ID preserved");
    check(sub.grid.total_nodes() == 3 * 3 * 5, "correct node count");
    check(sub.grid.total_elements() == 2 * 2 * 4, "correct element count");
    check(!sub.bc_min_z.empty(), "MinZ BCs populated");
    check(!sub.bc_max_z.empty(), "MaxZ BCs populated");
}


// =============================================================================
//  Test 2: Multi-element downscaling
// =============================================================================

void test_multi_element() {
    std::cout << "\nTest 2: Multi-element downscaling\n";

    fall_n::MultiscaleCoordinator coord;

    // Element 0: beam from x=0 to x=3
    coord.add_critical_element(make_beam_kinematics(
        0, 0.0, 3.0,
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        Eigen::Vector3d(0.001, 0.0, 0.0), Eigen::Vector3d::Zero()));

    // Element 1: beam from x=3 to x=6
    coord.add_critical_element(make_beam_kinematics(
        1, 3.0, 6.0,
        Eigen::Vector3d(0.001, 0.0, 0.0), Eigen::Vector3d::Zero(),
        Eigen::Vector3d(0.002, 0.001, 0.0), Eigen::Vector3d(0.0, 0.0, 0.0005)));

    check(coord.num_critical() == 2, "two critical elements");

    fall_n::SubModelSpec spec{0.4, 0.4, 3, 3, 6};
    coord.build_sub_models(spec);

    check(coord.sub_models().size() == 2, "two sub-models created");
    check(coord.sub_models()[0].parent_element_id == 0, "first sub-model ID");
    check(coord.sub_models()[1].parent_element_id == 1, "second sub-model ID");

    // Each sub-model: (3+1)*(3+1)*(6+1) = 4*4*7 = 112 nodes
    check(coord.sub_models()[0].grid.total_nodes() == 4 * 4 * 7,
          "sub-model 0 node count");
    check(coord.sub_models()[1].grid.total_nodes() == 4 * 4 * 7,
          "sub-model 1 node count");
}


// =============================================================================
//  Test 3: Report statistics
// =============================================================================

void test_report() {
    std::cout << "\nTest 3: Report statistics\n";

    fall_n::MultiscaleCoordinator coord;

    coord.add_critical_element(make_beam_kinematics(
        10, 0.0, 1.0,
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        Eigen::Vector3d(0.0, 0.005, 0.0), Eigen::Vector3d::Zero()));

    fall_n::SubModelSpec spec{0.2, 0.2, 2, 2, 2};
    coord.build_sub_models(spec);

    auto r = coord.report();
    check(r.num_elements == 1, "report: 1 element");
    check(r.num_sub_models == 1, "report: 1 sub-model");
    check(r.total_nodes == 3 * 3 * 3, "report: total nodes");
    check(r.total_elements == 2 * 2 * 2, "report: total elements");
    check(r.max_displacement > 0.0, "report: max displacement > 0");
    check(r.mean_displacement >= 0.0, "report: mean displacement >= 0");
    check(r.max_displacement >= r.mean_displacement,
          "report: max >= mean");

    std::cout << "    max_disp = " << r.max_displacement
              << ", mean_disp = " << r.mean_displacement << "\n";
}


// =============================================================================
//  Test 4: Clear and rebuild
// =============================================================================

void test_clear_rebuild() {
    std::cout << "\nTest 4: Clear and rebuild\n";

    fall_n::MultiscaleCoordinator coord;

    coord.add_critical_element(make_beam_kinematics(
        0, 0.0, 1.0,
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        Eigen::Vector3d(0.001, 0.0, 0.0), Eigen::Vector3d::Zero()));

    fall_n::SubModelSpec spec{0.2, 0.2, 2, 2, 2};
    coord.build_sub_models(spec);
    check(coord.sub_models().size() == 1, "one sub-model after first build");

    // Clear and add different elements
    coord.clear();
    check(coord.num_critical() == 0, "zero elements after clear");
    check(coord.sub_models().empty(), "no sub-models after clear");
    check(!coord.is_built(), "not built after clear");

    // Add two elements and rebuild
    coord.add_critical_element(make_beam_kinematics(
        5, 0.0, 2.0,
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        Eigen::Vector3d(0.0, 0.001, 0.0), Eigen::Vector3d::Zero()));
    coord.add_critical_element(make_beam_kinematics(
        6, 2.0, 4.0,
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        Eigen::Vector3d(0.0, 0.002, 0.0), Eigen::Vector3d::Zero()));

    coord.build_sub_models(spec);
    check(coord.sub_models().size() == 2, "two sub-models after rebuild");
    check(coord.is_built(), "built after rebuild");
}


// =============================================================================
//  Test 5: Zero-displacement element
// =============================================================================

void test_zero_displacement() {
    std::cout << "\nTest 5: Zero-displacement element\n";

    fall_n::MultiscaleCoordinator coord;

    // Both ends zero displacement / rotation
    coord.add_critical_element(make_beam_kinematics(
        0, 0.0, 1.5,
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()));

    fall_n::SubModelSpec spec{0.3, 0.3, 2, 2, 3};
    coord.build_sub_models(spec);

    const auto& sub = coord.sub_models()[0];

    // All BCs should be zero
    double max_bc = 0.0;
    for (const auto& [id, u] : sub.bc_min_z)
        max_bc = std::max(max_bc, u.norm());
    for (const auto& [id, u] : sub.bc_max_z)
        max_bc = std::max(max_bc, u.norm());

    check(max_bc < tol, "zero displacement → zero BCs everywhere");

    auto r = coord.report();
    check(r.max_displacement < tol, "report: zero max displacement");
}


// =============================================================================
//  Test 6: Mixed state — one deformed, one undeformed
// =============================================================================

void test_mixed_state() {
    std::cout << "\nTest 6: Mixed state\n";

    fall_n::MultiscaleCoordinator coord;

    // Element 0: undeformed
    coord.add_critical_element(make_beam_kinematics(
        0, 0.0, 1.0,
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()));

    // Element 1: deformed with bending
    coord.add_critical_element(make_beam_kinematics(
        1, 1.0, 2.0,
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        Eigen::Vector3d(0.0, 0.01, 0.0), Eigen::Vector3d(0.0, 0.0, 0.005)));

    fall_n::SubModelSpec spec{0.3, 0.5, 2, 2, 3};
    coord.build_sub_models(spec);

    // Sub-model 0 BCs: all zero
    double max_0 = 0.0;
    for (const auto& [id, u] : coord.sub_models()[0].bc_min_z)
        max_0 = std::max(max_0, u.norm());
    for (const auto& [id, u] : coord.sub_models()[0].bc_max_z)
        max_0 = std::max(max_0, u.norm());
    check(max_0 < tol, "undeformed element has zero BCs");

    // Sub-model 1 BCs: non-zero at end B
    double max_1B = 0.0;
    for (const auto& [id, u] : coord.sub_models()[1].bc_max_z)
        max_1B = std::max(max_1B, u.norm());
    check(max_1B > tol, "deformed element has non-zero end-B BCs");

    // Sub-model 1 end A: zero (kin_A has zero displacement)
    double max_1A = 0.0;
    for (const auto& [id, u] : coord.sub_models()[1].bc_min_z)
        max_1A = std::max(max_1A, u.norm());
    check(max_1A < tol, "deformed element end-A still zero");
}


// =============================================================================
//  Test 7: Rotated beam — non-axis-aligned
// =============================================================================

void test_rotated_element() {
    std::cout << "\nTest 7: Rotated beam element\n";

    fall_n::MultiscaleCoordinator coord;

    // Beam along global Y (from (0,0,0) to (0,3,0)), up = global Z
    fall_n::ElementKinematics ek;
    ek.element_id = 99;
    ek.endpoint_A = {0.0, 0.0, 0.0};
    ek.endpoint_B = {0.0, 3.0, 0.0};
    ek.up_direction = {0.0, 0.0, 1.0};

    // R maps global→local: beam axis (global Y) → local X
    // local x = global Y, local y = global Z, local z = -global X
    // (depends on align_to_beam cross product convention)
    // We'll use a general R consistent with up=Z, axis=Y
    Eigen::Matrix3d Rt;  // R^T: local→global
    Rt.col(0) = Eigen::Vector3d(0.0, 1.0, 0.0);   // local x → global Y
    Rt.col(1) = Eigen::Vector3d(0.0, 0.0, 1.0);   // local y → global Z
    Rt.col(2) = Eigen::Vector3d(1.0, 0.0, 0.0);   // local z → global X
    Eigen::Matrix3d R = Rt.transpose();

    // End A: zero
    ek.kin_A.centroid = Eigen::Vector3d(0.0, 0.0, 0.0);
    ek.kin_A.R = R;
    ek.kin_A.u_local = Eigen::Vector3d::Zero();
    ek.kin_A.theta_local = Eigen::Vector3d::Zero();
    ek.kin_A.E = 200.0;  ek.kin_A.G = 80.0;  ek.kin_A.nu = 0.25;

    // End B: axial displacement (u_local_x = 0.002)
    ek.kin_B.centroid = Eigen::Vector3d(0.0, 3.0, 0.0);
    ek.kin_B.R = R;
    ek.kin_B.u_local = Eigen::Vector3d(0.002, 0.0, 0.0);
    ek.kin_B.theta_local = Eigen::Vector3d::Zero();
    ek.kin_B.E = 200.0;  ek.kin_B.G = 80.0;  ek.kin_B.nu = 0.25;

    coord.add_critical_element(std::move(ek));

    fall_n::SubModelSpec spec{0.3, 0.3, 2, 2, 4};
    coord.build_sub_models(spec);

    check(coord.sub_models().size() == 1, "one sub-model");

    // End A: zero BCs
    double max_A = 0.0;
    for (const auto& [id, u] : coord.sub_models()[0].bc_min_z)
        max_A = std::max(max_A, u.norm());
    check(max_A < tol, "end A zero BCs");

    // End B: uniform axial displacement
    // u_local = (0.002, 0, 0), u_global = R^T * u_local
    Eigen::Vector3d u_global_expected = Rt * Eigen::Vector3d(0.002, 0.0, 0.0);
    bool all_match = true;
    for (const auto& [id, u] : coord.sub_models()[0].bc_max_z) {
        if ((u - u_global_expected).norm() > 1e-8) {
            all_match = false;
            break;
        }
    }
    check(all_match, "end B uniform axial displacement in global frame");

    // Expected: u_global = (0, 0.002, 0) since local x = global Y
    check(std::abs(u_global_expected[1] - 0.002) < tol,
          "axial displacement maps to global Y");
}

// =============================================================================
//  Test 8: Distinct local-site IDs for repeated parent elements
// =============================================================================

void test_repeated_parent_local_site_ids() {
    std::cout << "\nTest 8: Repeated parent local-site IDs\n";

    fall_n::MultiscaleCoordinator coord;

    auto fixed_site = make_beam_kinematics(
        42, 0.0, 2.0,
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        Eigen::Vector3d(0.0, 0.002, 0.0), Eigen::Vector3d::Zero());
    fixed_site.local_site_index = 7;
    fixed_site.site_z_over_l = 0.05;
    coord.add_critical_element(std::move(fixed_site));

    auto loaded_site = make_beam_kinematics(
        42, 0.0, 2.0,
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        Eigen::Vector3d(0.0, 0.002, 0.0), Eigen::Vector3d::Zero());
    loaded_site.local_site_index = 8;
    loaded_site.site_z_over_l = 0.95;
    coord.add_critical_element(std::move(loaded_site));

    fall_n::SubModelSpec spec{0.3, 0.3, 1, 1, 2};
    coord.build_sub_models(spec);

    check(coord.sub_models().size() == 2,
          "two sub-models created for one parent element");
    check(coord.sub_models()[0].parent_element_id == 42,
          "first parent element preserved");
    check(coord.sub_models()[1].parent_element_id == 42,
          "second parent element preserved");
    check(coord.sub_models()[0].vtk_site_id() == 7,
          "first VTK site ID preserved");
    check(coord.sub_models()[1].vtk_site_id() == 8,
          "second VTK site ID preserved");
    check(std::abs(coord.sub_models()[0].site_z_over_l - 0.05) < tol,
          "first site z/L preserved");
    check(std::abs(coord.sub_models()[1].site_z_over_l - 0.95) < tol,
          "second site z/L preserved");
}


// =============================================================================
//  main
// =============================================================================

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    std::cout << std::string(60, '=') << "\n"
              << "  Phase 5: MultiscaleCoordinator Verification\n"
              << std::string(60, '=') << "\n";

    test_single_element();
    test_multi_element();
    test_report();
    test_clear_rebuild();
    test_zero_displacement();
    test_mixed_state();
    test_rotated_element();
    test_repeated_parent_local_site_ids();

    std::cout << "\n" << std::string(60, '=') << "\n"
              << "  Summary: " << passed << " passed, " << failed << " failed\n"
              << std::string(60, '=') << "\n";

    PetscFinalize();
    return (failed > 0) ? 1 : 0;
}
