// ============================================================================
//  Tests for section::SectionGeometry, section::NodeSection,
//  and MaterialSection
// ============================================================================

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numbers>

#include <Eigen/Dense>

#include "src/elements/section/SectionGeometry.hh"
#include "src/elements/section/NodeSection.hh"
#include "src/elements/section/MaterialSection.hh"

// For MaterialSection test we need a concrete material
#include "src/materials/LinealElasticMaterial.hh"
#include "src/materials/update_strategy/IntegrationStrategy.hh"

// ── Helpers ─────────────────────────────────────────────────────────────────

static int  g_pass = 0;
static int  g_fail = 0;

#define ASSERT_NEAR(a, b, tol)                                                 \
    do {                                                                       \
        if (std::abs((a) - (b)) > (tol)) {                                     \
            std::cerr << "  FAIL  " << __FILE__ << ":" << __LINE__             \
                      << "  |" << (a) << " - " << (b) << "| = "               \
                      << std::abs((a) - (b)) << " > " << (tol) << "\n";       \
            ++g_fail;                                                          \
            return;                                                            \
        }                                                                      \
    } while (0)

#define ASSERT_TRUE(cond)                                                      \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cerr << "  FAIL  " << __FILE__ << ":" << __LINE__             \
                      << "  " #cond " is false\n";                             \
            ++g_fail;                                                          \
            return;                                                            \
        }                                                                      \
    } while (0)

#define RUN_TEST(fn)                                                           \
    do {                                                                       \
        int _before = g_fail;                                                  \
        fn();                                                                  \
        if (g_fail == _before) { ++g_pass; std::cout << "  PASS  " << #fn << "\n"; } \
        else                   { std::cout << "  FAIL  " << #fn << "\n"; }     \
    } while (0)

// ============================================================================
//  SectionGeometry tests
// ============================================================================

void test_rectangular_section_area() {
    section::RectangularSection rect(0.3, 0.5); // 30cm × 50cm
    ASSERT_NEAR(rect.area(), 0.15, 1e-12);
}

void test_rectangular_section_inertia() {
    section::RectangularSection rect(0.3, 0.5);
    // I_y = bh³/12 = 0.3·0.5³/12 = 0.3·0.125/12 = 0.003125
    ASSERT_NEAR(rect.moment_y(), 0.3 * 0.125 / 12.0, 1e-12);
    // I_z = hb³/12 = 0.5·0.3³/12 = 0.5·0.027/12 = 0.001125
    ASSERT_NEAR(rect.moment_z(), 0.5 * 0.027 / 12.0, 1e-12);
}

void test_rectangular_section_shear_factors() {
    section::RectangularSection rect(0.3, 0.5);
    ASSERT_NEAR(rect.shear_factor_y(), 5.0 / 6.0, 1e-12);
    ASSERT_NEAR(rect.shear_factor_z(), 5.0 / 6.0, 1e-12);
}

void test_circular_section_area() {
    section::CircularSection circ(0.1); // r = 10cm
    ASSERT_NEAR(circ.area(), std::numbers::pi * 0.01, 1e-12);
}

void test_circular_section_inertia_symmetry() {
    section::CircularSection circ(0.15);
    // I_y == I_z for a circle
    ASSERT_NEAR(circ.moment_y(), circ.moment_z(), 1e-15);
    // I = π·r⁴/4
    double expected = std::numbers::pi * std::pow(0.15, 4) / 4.0;
    ASSERT_NEAR(circ.moment_y(), expected, 1e-12);
}

void test_circular_section_torsion() {
    section::CircularSection circ(0.1);
    // J = π·r⁴/2 = 2·I
    ASSERT_NEAR(circ.torsion_J(), 2.0 * circ.moment_y(), 1e-15);
}

void test_generic_section() {
    section::GenericSection gen(0.05, 1e-4, 2e-4, 3e-4, 0.8, 0.7);
    ASSERT_NEAR(gen.area(), 0.05, 1e-15);
    ASSERT_NEAR(gen.moment_y(), 1e-4, 1e-15);
    ASSERT_NEAR(gen.moment_z(), 2e-4, 1e-15);
    ASSERT_NEAR(gen.torsion_J(), 3e-4, 1e-15);
    ASSERT_NEAR(gen.shear_factor_y(), 0.8, 1e-15);
    ASSERT_NEAR(gen.shear_factor_z(), 0.7, 1e-15);
}

void test_section_geometry_concept() {
    ASSERT_TRUE((section::SectionGeometryLike<section::RectangularSection>));
    ASSERT_TRUE((section::SectionGeometryLike<section::CircularSection>));
    ASSERT_TRUE((section::SectionGeometryLike<section::GenericSection>));
}

// ============================================================================
//  SectionFrame tests
// ============================================================================

void test_section_frame_identity() {
    section::SectionFrame<3> frame;
    Eigen::Vector3d v{1.0, 2.0, 3.0};
    auto v_local = frame.to_local(v);
    ASSERT_NEAR((v_local - v).norm(), 0.0, 1e-15);
}

void test_section_frame_rotation() {
    // 90° rotation about z: x→y, y→−x
    Eigen::Matrix3d R;
    R << 0, 1, 0,
        -1, 0, 0,
         0, 0, 1;

    section::SectionFrame<3> frame(R);

    Eigen::Vector3d v{1.0, 0.0, 0.0};
    auto v_local = frame.to_local(v);
    ASSERT_NEAR(v_local(0), 0.0, 1e-15);
    ASSERT_NEAR(v_local(1), -1.0, 1e-15);
    ASSERT_NEAR(v_local(2), 0.0, 1e-15);
}

void test_section_frame_round_trip() {
    // Arbitrary rotation
    Eigen::Matrix3d R;
    R << 0.6, 0.8, 0.0,
        -0.8, 0.6, 0.0,
         0.0, 0.0, 1.0;

    section::SectionFrame<3> frame(R);

    Eigen::Vector3d v{1.5, -2.3, 0.7};
    auto v_round = frame.to_global(frame.to_local(v));
    ASSERT_NEAR((v_round - v).norm(), 0.0, 1e-12);
}

// ============================================================================
//  NodeSection tests
// ============================================================================

void test_node_section_binds_node() {
    Node<3> node(0, 1.0, 2.0, 3.0);
    node.set_num_dof(6);

    section::RectangularSection geom(0.3, 0.5);
    section::NodeSection<3, section::RectangularSection> ns(node, geom);

    ASSERT_TRUE(ns.node_id() == 0);
    ASSERT_TRUE(ns.num_dof() == 6);
    ASSERT_NEAR(ns.area(), 0.15, 1e-12);
}

void test_node_section_with_frame() {
    Node<3> node(1, 0.0, 0.0, 0.0);
    node.set_num_dof(6);

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    section::SectionFrame<3> frame(R);
    section::GenericSection geom(0.1, 1e-3, 2e-3, 5e-4);

    section::NodeSection<3, section::GenericSection> ns(node, frame, geom);

    ASSERT_TRUE(&ns.node() == &node);
    ASSERT_NEAR(ns.moment_y(), 1e-3, 1e-15);
}

void test_node_section_delegates_dofs() {
    Node<3> node(5, 1.0, 2.0, 3.0);
    node.set_num_dof(6);
    node.set_dof_index(0, 10);
    node.set_dof_index(1, 11);

    section::NodeSection<3> ns(node);

    auto dofs = ns.dof_index();
    ASSERT_TRUE(dofs.size() == 6);
    ASSERT_TRUE(dofs[0] == 10);
    ASSERT_TRUE(dofs[1] == 11);
    ASSERT_TRUE(ns.dof_data() == node.dof_data());
}

void test_node_section_modifies_through_reference() {
    Node<3> node(0, 0.0, 0.0, 0.0);
    node.set_num_dof(3);

    section::NodeSection<3> ns(node);

    // Modify DOFs through node reference
    ns.node().set_dof_index(0, 42);
    ASSERT_TRUE(node.dof_index()[0] == 42);
}

void test_node_section_2d() {
    Node<2> node(0, 1.0, 2.0);
    node.set_num_dof(3);

    section::RectangularSection geom(0.2, 0.4);
    section::NodeSection<2, section::RectangularSection> ns(node, geom);

    ASSERT_TRUE(ns.dim == 2);
    ASSERT_NEAR(ns.area(), 0.08, 1e-12);
    ASSERT_TRUE(ns.data() == node.data());
}

// ============================================================================
//  MaterialSection tests (require PETSc init for Material)
// ============================================================================

void test_material_section_construction() {
    // Build a Timoshenko beam section constitutive site using the legacy alias.
    TimoshenkoBeamMaterial3D mat_instance{200.0e9, 80.0e9, 0.01, 8.33e-6, 8.33e-6, 1.41e-5};

    ConstitutiveHandle<TimoshenkoBeamConstitutiveSpace3D> mat{mat_instance, ElasticUpdate{}};

    SectionConstitutiveSite<TimoshenkoBeamConstitutiveSpace3D> ms(std::move(mat));

    // The section stiffness matrix should be 6×6
    auto C = ms.C();
    ASSERT_TRUE(C.rows() == 6);
    ASSERT_TRUE(C.cols() == 6);

    // Diagonal should be non-zero
    ASSERT_TRUE(C(0, 0) > 0);  // EA
    ASSERT_TRUE(C(1, 1) > 0);  // EI_y
}

void test_material_section_bind_integration_point() {
    TimoshenkoBeamMaterial3D mat_instance{200e9, 80e9, 0.01, 8.33e-6, 8.33e-6, 1.41e-5};

    ConstitutiveHandle<TimoshenkoBeamConstitutiveSpace3D> mat{mat_instance, ElasticUpdate{}};

    SectionConstitutiveSite<TimoshenkoBeamConstitutiveSpace3D> ms(std::move(mat));

    IntegrationPoint<3> gp;
    std::array<double, 3> coord{0.5, 0.0, 0.0};
    gp.set_coord(coord);

    ms.bind_integration_point(gp);

    auto c = ms.coord();
    ASSERT_NEAR(c[0], 0.5, 1e-15);
    ASSERT_TRUE(ms.data() == gp.data());
    ASSERT_NEAR(ms.weight(), 0.0, 1e-15);
}

// ============================================================================

int main() {
    std::cout << "=== Section Tests ===\n";

    // SectionGeometry
    RUN_TEST(test_rectangular_section_area);
    RUN_TEST(test_rectangular_section_inertia);
    RUN_TEST(test_rectangular_section_shear_factors);
    RUN_TEST(test_circular_section_area);
    RUN_TEST(test_circular_section_inertia_symmetry);
    RUN_TEST(test_circular_section_torsion);
    RUN_TEST(test_generic_section);
    RUN_TEST(test_section_geometry_concept);

    // SectionFrame
    RUN_TEST(test_section_frame_identity);
    RUN_TEST(test_section_frame_rotation);
    RUN_TEST(test_section_frame_round_trip);

    // NodeSection
    RUN_TEST(test_node_section_binds_node);
    RUN_TEST(test_node_section_with_frame);
    RUN_TEST(test_node_section_delegates_dofs);
    RUN_TEST(test_node_section_modifies_through_reference);
    RUN_TEST(test_node_section_2d);

    // MaterialSection
    RUN_TEST(test_material_section_construction);
    RUN_TEST(test_material_section_bind_integration_point);

    std::cout << "\n=== " << g_pass << " PASSED, " << g_fail << " FAILED ===\n";
    return g_fail == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
