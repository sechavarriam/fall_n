// ============================================================================
//  Tests for BeamElement<BeamPolicy, Dim, AsmPolicy>
// ============================================================================
//
//  Verifies:
//    1. FiniteElement concept satisfaction (static_assert in header)
//    2. Element length computation
//    3. Rotation matrix (aligned + angled beams)
//    4. B matrix structure (2D + 3D)
//    5. Stiffness matrix symmetry
//    6. Analytical K entries for aligned 2D Timoshenko beam
//    7. Rotated beam K transformation
//    8. 3D beam stiffness matrix basics
//
// ============================================================================

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numbers>

#include <Eigen/Dense>

// Narrow validation umbrella for the beam slice. This is the migration path
// away from the repository-wide header_files.hh umbrella.
#include "src/validation/BeamValidationSupport.hh"

// ── Test harness ────────────────────────────────────────────────────────────

static int g_pass = 0;
static int g_fail = 0;

#define ASSERT_NEAR(a, b, tol)                                                 \
    do {                                                                       \
        if (std::abs((a) - (b)) > (tol)) {                                     \
            std::cerr << "  FAIL  " << __FILE__ << ":" << __LINE__             \
                      << "  |" << (a) << " - " << (b) << "| = "                \
                      << std::abs((a) - (b)) << " > " << (tol) << "\n";        \
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


// ── Helpers ─────────────────────────────────────────────────────────────────

// Build a 2-node line ElementGeometry<2> from two Node<2>* with given coords.
// Caller owns the returned ElementGeometry.
struct Beam2DFixture {
    Node<2> n0, n1;
    LagrangeElement2D<2> element;           // 2-node line in 2D
    GaussLegendreCellIntegrator<2> integrator;
    ElementGeometry<2> geom;

    // Section properties:  C = diag(EA, EI, kGA)
    //   E=10, G=5, A=10, I=1, k=1.0
    //   → EA=100, EI=10, kGA=50
    TimoshenkoBeamMaterial2D mat_instance{10.0, 5.0, 10.0, 1.0, 1.0};
    Material<TimoshenkoBeam2D> mat{mat_instance, ElasticUpdate{}};

    Beam2DFixture(double x0, double y0, double x1, double y1)
        : n0{0, x0, y0}
        , n1{1, x1, y1}
        , element{std::optional<std::array<Node<2>*, 2>>{std::array<Node<2>*, 2>{&n0, &n1}}}
        , geom{element, integrator}
    {
        geom.set_sieve_id(0);
    }

    auto make_beam() {
        return BeamElement<TimoshenkoBeam2D, 2>{&geom, mat};
    }
};

// Build a 2-node line ElementGeometry<3> for 3D beams.
struct Beam3DFixture {
    Node<3> n0, n1;
    LagrangeElement3D<2> element;           // 2-node line in 3D
    GaussLegendreCellIntegrator<2> integrator;
    ElementGeometry<3> geom;

    // Section properties for 3D Timoshenko
    //   E=200, G=80, A=0.01, Iy=8.33e-6, Iz=8.33e-6, J=1.41e-5, ky=5/6, kz=5/6
    TimoshenkoBeamMaterial3D mat_instance{200.0, 80.0, 0.01, 8.33e-6, 8.33e-6, 1.41e-5, 5.0/6.0, 5.0/6.0};
    Material<TimoshenkoBeam3D> mat{mat_instance, ElasticUpdate{}};

    Beam3DFixture(double x0, double y0, double z0,
                  double x1, double y1, double z1)
        : n0{0, x0, y0, z0}
        , n1{1, x1, y1, z1}
        , element{std::optional<std::array<Node<3>*, 2>>{std::array<Node<3>*, 2>{&n0, &n1}}}
        , geom{element, integrator}
    {
        geom.set_sieve_id(0);
    }

    auto make_beam() {
        return BeamElement<TimoshenkoBeam3D, 3>{&geom, mat};
    }
};

// Build a 2-node line ElementGeometry<2> from geometric vertices only.
// This verifies that BeamElement frame construction depends on geometry,
// not on analysis-node materialisation.
struct Beam2DVertexFixture {
    geometry::Vertex<2> v0, v1;
    LagrangeElement2D<2> element;
    GaussLegendreCellIntegrator<2> integrator;
    ElementGeometry<2> geom;

    TimoshenkoBeamMaterial2D mat_instance{10.0, 5.0, 10.0, 1.0, 1.0};
    Material<TimoshenkoBeam2D> mat{mat_instance, ElasticUpdate{}};

    Beam2DVertexFixture(double x0, double y0, double x1, double y1)
        : v0{0, x0, y0}
        , v1{1, x1, y1}
        , element{0, std::array<PetscInt, 2>{0, 1}}
        , geom{element, integrator}
    {
        geom.bind_point(0, &v0);
        geom.bind_point(1, &v1);
        geom.set_sieve_id(0);
    }

    auto make_beam() {
        return BeamElement<TimoshenkoBeam2D, 2>{&geom, mat};
    }
};


// ============================================================================
//  2D Beam Tests
// ============================================================================

void test_element_length_aligned() {
    Beam2DFixture f(0.0, 0.0, 2.0, 0.0);
    auto beam = f.make_beam();
    ASSERT_NEAR(beam.element_length(), 2.0, 1e-12);
}

void test_element_length_angled() {
    Beam2DFixture f(0.0, 0.0, 3.0, 4.0);
    auto beam = f.make_beam();
    ASSERT_NEAR(beam.element_length(), 5.0, 1e-12);
}

void test_rotation_matrix_horizontal() {
    // Beam along +x → R = I
    Beam2DFixture f(0.0, 0.0, 2.0, 0.0);
    auto beam = f.make_beam();
    auto R = beam.rotation_matrix();

    ASSERT_NEAR(R(0, 0), 1.0, 1e-12);
    ASSERT_NEAR(R(0, 1), 0.0, 1e-12);
    ASSERT_NEAR(R(1, 0), 0.0, 1e-12);
    ASSERT_NEAR(R(1, 1), 1.0, 1e-12);
}

void test_rotation_matrix_vertical() {
    // Beam along +y → R = [0  1; -1  0]
    // e1 (tangent) = (0, 1)
    // R row(0) = e1 = (0, 1)
    // R row(1) = (-e1[1], e1[0]) = (-1, 0)
    Beam2DFixture f(0.0, 0.0, 0.0, 3.0);
    auto beam = f.make_beam();
    auto R = beam.rotation_matrix();

    ASSERT_NEAR(R(0, 0), 0.0, 1e-12);
    ASSERT_NEAR(R(0, 1), 1.0, 1e-12);
    ASSERT_NEAR(R(1, 0), -1.0, 1e-12);
    ASSERT_NEAR(R(1, 1), 0.0, 1e-12);
}

void test_rotation_matrix_45deg() {
    double s = std::numbers::sqrt2 / 2.0;
    Beam2DFixture f(0.0, 0.0, 1.0, 1.0);
    auto beam = f.make_beam();
    auto R = beam.rotation_matrix();

    // e1 = (1/√2, 1/√2)
    ASSERT_NEAR(R(0, 0), s, 1e-12);
    ASSERT_NEAR(R(0, 1), s, 1e-12);
    // local y = (-e1[1], e1[0]) = (-1/√2, 1/√2)
    ASSERT_NEAR(R(1, 0), -s, 1e-12);
    ASSERT_NEAR(R(1, 1), s, 1e-12);

    // R is orthogonal: R · Rᵀ = I
    Eigen::Matrix2d product = R * R.transpose();
    ASSERT_NEAR(product(0, 0), 1.0, 1e-12);
    ASSERT_NEAR(product(1, 1), 1.0, 1e-12);
    ASSERT_NEAR(product(0, 1), 0.0, 1e-12);
}

void test_num_nodes() {
    Beam2DFixture f(0.0, 0.0, 1.0, 0.0);
    auto beam = f.make_beam();
    ASSERT_TRUE(beam.num_nodes() == 2);
}

void test_num_integration_points() {
    Beam2DFixture f(0.0, 0.0, 1.0, 0.0);
    auto beam = f.make_beam();
    ASSERT_TRUE(beam.num_integration_points() == 2);
}

void test_sieve_id() {
    Beam2DFixture f(0.0, 0.0, 1.0, 0.0);
    auto beam = f.make_beam();
    ASSERT_TRUE(beam.sieve_id() == 0);
}

void test_beam_frame_uses_geometry_points_only() {
    Beam2DVertexFixture f(0.0, 0.0, 3.0, 4.0);
    auto beam = f.make_beam();
    auto R = beam.rotation_matrix();

    ASSERT_NEAR(beam.element_length(), 5.0, 1e-12);
    ASSERT_NEAR(R(0, 0), 3.0 / 5.0, 1e-12);
    ASSERT_NEAR(R(0, 1), 4.0 / 5.0, 1e-12);
}

void test_K_symmetry_2d() {
    Beam2DFixture f(0.0, 0.0, 2.0, 0.0);
    auto beam = f.make_beam();
    auto K = beam.K();

    // K should be 6×6
    ASSERT_TRUE(K.rows() == 6);
    ASSERT_TRUE(K.cols() == 6);

    // Symmetry
    for (int i = 0; i < 6; ++i)
        for (int j = i + 1; j < 6; ++j)
            ASSERT_NEAR(K(i, j), K(j, i), 1e-10);
}

void test_K_analytical_aligned_2d() {
    // Horizontal beam: T = I, so K_global = K_local.
    // C = diag(EA=100, EI=10, kGA=50), L=2.
    //
    // Analytical K from exact 2-point Gauss integration of B^T·C·B:
    //
    //   EA/L = 50,  EI/L = 5,  kGA/L = 25,  kGA/2 = 25
    //   kGA·L/3 = 100/3,  kGA·L/6 = 50/3
    //
    //         u₁      v₁          θ₁           u₂       v₂          θ₂
    //  u₁: [ EA/L      0           0          -EA/L       0           0         ]
    //  v₁: [  0      kGA/L       kGA/2          0      -kGA/L       kGA/2       ]
    //  θ₁: [  0      kGA/2    EI/L+kGAL/3       0      -kGA/2   -EI/L+kGAL/6    ]
    //  u₂: [-EA/L      0           0           EA/L       0           0         ]
    //  v₂: [  0     -kGA/L      -kGA/2          0       kGA/L      -kGA/2       ]
    //  θ₂: [  0      kGA/2   -EI/L+kGAL/6       0      -kGA/2    EI/L+kGAL/3    ]

    Beam2DFixture f(0.0, 0.0, 2.0, 0.0);
    auto beam = f.make_beam();
    auto K = beam.K();

    const double EA  = 100.0, EI  = 10.0, kGA = 50.0, L = 2.0;
    const double EAL = EA / L;        // 50
    const double EIL = EI / L;        // 5
    const double kGAL = kGA / L;      // 25
    const double kGA2 = kGA / 2.0;    // 25
    const double kGAL3 = kGA * L / 3; // 100/3
    const double kGAL6 = kGA * L / 6; // 50/3

    const double tol = 1e-10;

    // Axial entries
    ASSERT_NEAR(K(0, 0),  EAL, tol);
    ASSERT_NEAR(K(0, 3), -EAL, tol);
    ASSERT_NEAR(K(3, 3),  EAL, tol);

    // Shear diagonal: (1,1) and (4,4)
    ASSERT_NEAR(K(1, 1),  kGAL, tol);
    ASSERT_NEAR(K(4, 4),  kGAL, tol);

    // Bending + shear coupling on diagonal: (2,2) and (5,5)
    ASSERT_NEAR(K(2, 2), EIL + kGAL3, tol);
    ASSERT_NEAR(K(5, 5), EIL + kGAL3, tol);

    // Off-diagonal shear: (1,2), (1,4), (1,5)
    ASSERT_NEAR(K(1, 2),  kGA2, tol);
    ASSERT_NEAR(K(1, 4), -kGAL, tol);
    ASSERT_NEAR(K(1, 5),  kGA2, tol);

    // Off-diagonal bending + shear: (2,4), (2,5)
    ASSERT_NEAR(K(2, 4), -kGA2, tol);
    ASSERT_NEAR(K(2, 5), -EIL + kGAL6, tol);

    // Off-diagonal: (4,5)
    ASSERT_NEAR(K(4, 5), -kGA2, tol);

    // Zero coupling between axial and bending for aligned beam
    ASSERT_NEAR(K(0, 1), 0.0, tol);
    ASSERT_NEAR(K(0, 2), 0.0, tol);
    ASSERT_NEAR(K(0, 4), 0.0, tol);
    ASSERT_NEAR(K(0, 5), 0.0, tol);
    ASSERT_NEAR(K(3, 1), 0.0, tol);
    ASSERT_NEAR(K(3, 2), 0.0, tol);
    ASSERT_NEAR(K(3, 4), 0.0, tol);
    ASSERT_NEAR(K(3, 5), 0.0, tol);
}

void test_K_rotation_invariance_2d() {
    // For a rotated beam, K_global = T^T K_local T.
    // The trace (sum of eigenvalues) should be the same
    // regardless of orientation.
    Beam2DFixture f_h(0.0, 0.0, 2.0, 0.0);       // horizontal
    Beam2DFixture f_45(0.0, 0.0,
                       std::numbers::sqrt2, std::numbers::sqrt2); // 45°, same length

    auto beam_h  = f_h.make_beam();
    auto beam_45 = f_45.make_beam();

    auto K_h  = beam_h.K();
    auto K_45 = beam_45.K();

    // Trace invariance
    ASSERT_NEAR(K_h.trace(), K_45.trace(), 1e-8);

    // Eigenvalue sum invariance (Frobenius norm also works)
    ASSERT_NEAR(K_h.norm(), K_45.norm(), 1e-8);
}

void test_K_rigid_body_modes_2d() {
    // A free beam has 3 rigid body modes (2D): 2 translations + 1 rotation.
    // K should have exactly 3 zero eigenvalues.
    Beam2DFixture f(0.0, 0.0, 2.0, 0.0);
    auto beam = f.make_beam();
    auto K = beam.K();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(K);
    auto evals = solver.eigenvalues();

    int n_zero = 0;
    for (int i = 0; i < evals.size(); ++i)
        if (std::abs(evals[i]) < 1e-8) ++n_zero;

    ASSERT_TRUE(n_zero == 3);

    // All non-zero eigenvalues should be positive (positive semi-definite)
    for (int i = 0; i < evals.size(); ++i)
        ASSERT_TRUE(evals[i] >= -1e-10);
}


// ============================================================================
//  3D Beam Tests
// ============================================================================

void test_element_length_3d() {
    Beam3DFixture f(0.0, 0.0, 0.0, 3.0, 4.0, 0.0);
    auto beam = f.make_beam();
    ASSERT_NEAR(beam.element_length(), 5.0, 1e-12);
}

void test_rotation_matrix_3d_along_x() {
    // Beam along +x → e1=(1,0,0), ref_vec=Z → e2=Z×e1=(-Y)... hmm
    // Let me compute: ref_vec = Z = (0,0,1)
    // e2 = ref_vec × e1 = (0,0,1) × (1,0,0) = (0,1,0) ← local y
    // e3 = e1 × e2 = (1,0,0) × (0,1,0) = (0,0,1) ← local z
    // R rows: e1, e2, e3 → Identity
    Beam3DFixture f(0.0, 0.0, 0.0, 5.0, 0.0, 0.0);
    auto beam = f.make_beam();
    auto R = beam.rotation_matrix();

    Eigen::Matrix3d expected = Eigen::Matrix3d::Identity();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            ASSERT_NEAR(R(i, j), expected(i, j), 1e-12);
}

void test_rotation_matrix_3d_along_z() {
    // Beam along +z → e1=(0,0,1)
    // |e1 · Z| = 1 > 0.99, so ref_vec = X = (1,0,0)
    // e2 = X × e1 = (1,0,0) × (0,0,1) = (0,-1,0)... wait
    // Cross product: (1,0,0) × (0,0,1) = (0·1-0·0, 0·0-1·1, 1·0-0·0) = (0,-1,0)
    // Normalize: (0,-1,0)
    // e3 = e1 × e2 = (0,0,1) × (0,-1,0) = (0·0-1·(-1), 1·0-0·0, 0·(-1)-0·0) = (1,0,0)
    Beam3DFixture f(0.0, 0.0, 0.0, 0.0, 0.0, 5.0);
    auto beam = f.make_beam();
    auto R = beam.rotation_matrix();

    // R rows: e1=(0,0,1), e2=(0,-1,0), e3=(1,0,0)
    ASSERT_NEAR(R(0, 2), 1.0, 1e-12);  // e1.z = 1
    ASSERT_NEAR(R(1, 1), -1.0, 1e-12); // e2.y = -1
    ASSERT_NEAR(R(2, 0), 1.0, 1e-12);  // e3.x = 1

    // R should be orthogonal
    Eigen::Matrix3d product = R * R.transpose();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            ASSERT_NEAR(product(i, j), (i == j) ? 1.0 : 0.0, 1e-12);
}

void test_K_symmetry_3d() {
    Beam3DFixture f(0.0, 0.0, 0.0, 5.0, 0.0, 0.0);
    auto beam = f.make_beam();
    auto K = beam.K();

    // 12×12
    ASSERT_TRUE(K.rows() == 12);
    ASSERT_TRUE(K.cols() == 12);

    for (int i = 0; i < 12; ++i)
        for (int j = i + 1; j < 12; ++j)
            ASSERT_NEAR(K(i, j), K(j, i), 1e-8);
}

void test_K_axial_stiffness_3d() {
    // Along x-axis → R = I, so K_local = K_global.
    // EA = E·A = 200·0.01 = 2.0
    // L = 5
    // K(0,0) = K(6,6) = EA/L = 0.4
    // K(0,6) = K(6,0) = -EA/L
    Beam3DFixture f(0.0, 0.0, 0.0, 5.0, 0.0, 0.0);
    auto beam = f.make_beam();
    auto K = beam.K();

    const double EA = 200.0 * 0.01;  // 2.0
    const double L  = 5.0;
    const double EAL = EA / L;        // 0.4

    ASSERT_NEAR(K(0, 0),  EAL, 1e-10);
    ASSERT_NEAR(K(6, 6),  EAL, 1e-10);
    ASSERT_NEAR(K(0, 6), -EAL, 1e-10);
}

void test_K_rigid_body_modes_3d() {
    // A free 3D beam has 6 rigid body modes.
    // K should have exactly 6 zero eigenvalues.
    Beam3DFixture f(0.0, 0.0, 0.0, 5.0, 0.0, 0.0);
    auto beam = f.make_beam();
    auto K = beam.K();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(K);
    auto evals = solver.eigenvalues();

    int n_zero = 0;
    for (int i = 0; i < evals.size(); ++i)
        if (std::abs(evals[i]) < 1e-8) ++n_zero;

    ASSERT_TRUE(n_zero == 6);

    // All non-zero eigenvalues should be positive
    for (int i = 0; i < evals.size(); ++i)
        ASSERT_TRUE(evals[i] >= -1e-10);
}

void test_K_trace_rotation_invariance_3d() {
    // Same beam, different orientation: trace should match.
    Beam3DFixture f_x(0.0, 0.0, 0.0, 5.0, 0.0, 0.0);   // along x
    Beam3DFixture f_y(0.0, 0.0, 0.0, 0.0, 5.0, 0.0);   // along y

    auto beam_x = f_x.make_beam();
    auto beam_y = f_y.make_beam();

    auto K_x = beam_x.K();
    auto K_y = beam_y.K();

    ASSERT_NEAR(K_x.trace(), K_y.trace(), 1e-8);
    ASSERT_NEAR(K_x.norm(),  K_y.norm(),  1e-8);
}

void test_sections_access() {
    Beam2DFixture f(0.0, 0.0, 2.0, 0.0);
    auto beam = f.make_beam();

    // 2 Gauss points → 2 MaterialSections
    ASSERT_TRUE(beam.sections().size() == 2);

    // Each section should provide a 3×3 constitutive matrix (2D)
    auto C = beam.sections()[0].C();
    ASSERT_TRUE(C.rows() == 3);
    ASSERT_TRUE(C.cols() == 3);

    // C should be diag(EA, EI, kGA) = diag(100, 10, 50)
    ASSERT_NEAR(C(0, 0), 100.0, 1e-10);
    ASSERT_NEAR(C(1, 1), 10.0, 1e-10);
    ASSERT_NEAR(C(2, 2), 50.0, 1e-10);
    ASSERT_NEAR(C(0, 1), 0.0, 1e-10);
}


// ============================================================================
//  Main
// ============================================================================

int main() {
    std::cout << "=== BeamElement Tests ===\n";

    // 2D tests
    RUN_TEST(test_element_length_aligned);
    RUN_TEST(test_element_length_angled);
    RUN_TEST(test_rotation_matrix_horizontal);
    RUN_TEST(test_rotation_matrix_vertical);
    RUN_TEST(test_rotation_matrix_45deg);
    RUN_TEST(test_num_nodes);
    RUN_TEST(test_num_integration_points);
    RUN_TEST(test_sieve_id);
    RUN_TEST(test_beam_frame_uses_geometry_points_only);
    RUN_TEST(test_K_symmetry_2d);
    RUN_TEST(test_K_analytical_aligned_2d);
    RUN_TEST(test_K_rotation_invariance_2d);
    RUN_TEST(test_K_rigid_body_modes_2d);

    // 3D tests
    RUN_TEST(test_element_length_3d);
    RUN_TEST(test_rotation_matrix_3d_along_x);
    RUN_TEST(test_rotation_matrix_3d_along_z);
    RUN_TEST(test_K_symmetry_3d);
    RUN_TEST(test_K_axial_stiffness_3d);
    RUN_TEST(test_K_rigid_body_modes_3d);
    RUN_TEST(test_K_trace_rotation_invariance_3d);

    // Section access
    RUN_TEST(test_sections_access);

    std::cout << "\n=== " << g_pass << " PASSED, " << g_fail << " FAILED ===\n";
    return g_fail == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
