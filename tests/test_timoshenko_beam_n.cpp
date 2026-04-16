// ============================================================================
//  Tests for TimoshenkoBeamN<N, BeamPolicy, AsmPolicy>
// ============================================================================
//
//  Verifies:
//    1. FiniteElement concept satisfaction (static_assert in header)
//    2. N=2 equivalence with BeamElement<TimoshenkoBeam3D, 3>
//    3. Stiffness matrix symmetry (N=2, 3, 4)
//    4. Rigid body modes — 6 zero eigenvalues (N=2, 3, 4)
//    5. Axial stiffness patch test
//    6. Rotation invariance (K trace & norm)
//    7. Section access
//    8. Shear basis construction
//    9. Higher-order element rank check
//   10. Curved beam: circular arc (N=3)
//
// ============================================================================

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <stdexcept>

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


// ──────────────────────────────────────────────────────────────────────────────
//  Fixtures
// ──────────────────────────────────────────────────────────────────────────────

// Section properties (same as existing beam tests):
//   E=200, G=80, A=0.01, Iy=8.33e-6, Iz=8.33e-6, J=1.41e-5, ky=5/6, kz=5/6
//   → EA=2, EIy=EIz≈1.666e-3, kyGA=80·0.01·5/6≈0.6667, GJ≈1.128e-3

// 2-node fixture (for equivalence test with BeamElement)
struct BeamN2Fixture {
    Node<3> n0, n1;
    LagrangeElement3D<2> element;
    GaussLegendreCellIntegrator<1> integrator; // N-1 = 1 GP
    ElementGeometry<3> geom;

    TimoshenkoBeamMaterial3D mat_instance{200.0, 80.0, 0.01, 8.33e-6, 8.33e-6, 1.41e-5, 5.0/6.0, 5.0/6.0};
    Material<TimoshenkoBeam3D> mat{mat_instance, ElasticUpdate{}};

    BeamN2Fixture(double x0, double y0, double z0,
                  double x1, double y1, double z1)
        : n0{0, x0, y0, z0}
        , n1{1, x1, y1, z1}
        , element{std::optional<std::array<Node<3>*, 2>>{std::array<Node<3>*, 2>{&n0, &n1}}}
        , geom{element, integrator}
    {
        geom.set_sieve_id(0);
    }

    auto make_beam() {
        return TimoshenkoBeamN<2>{&geom, mat};
    }
};

// Reference 2-node BeamElement (for equivalence)
// Must use 1 GP to match TimoshenkoBeamN<2> which uses N-1=1 GP.
struct RefBeam3DFixture {
    Node<3> n0, n1;
    LagrangeElement3D<2> element;
    GaussLegendreCellIntegrator<1> integrator; // 1 GP (same as TimoshenkoBeamN<2>)
    ElementGeometry<3> geom;

    TimoshenkoBeamMaterial3D mat_instance{200.0, 80.0, 0.01, 8.33e-6, 8.33e-6, 1.41e-5, 5.0/6.0, 5.0/6.0};
    Material<TimoshenkoBeam3D> mat{mat_instance, ElasticUpdate{}};

    RefBeam3DFixture(double x0, double y0, double z0,
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

// 3-node fixture (quadratic beam)
struct BeamN3Fixture {
    Node<3> n0, n1, n2;
    LagrangeElement3D<3> element;
    GaussLegendreCellIntegrator<2> integrator; // N-1 = 2 GPs
    ElementGeometry<3> geom;

    TimoshenkoBeamMaterial3D mat_instance{200.0, 80.0, 0.01, 8.33e-6, 8.33e-6, 1.41e-5, 5.0/6.0, 5.0/6.0};
    Material<TimoshenkoBeam3D> mat{mat_instance, ElasticUpdate{}};

    // Straight 3-node beam along x: nodes at 0, L/2, L
    BeamN3Fixture(double L)
        : n0{0, 0.0, 0.0, 0.0}
        , n1{1, L/2.0, 0.0, 0.0}
        , n2{2, L, 0.0, 0.0}
        , element{std::optional<std::array<Node<3>*, 3>>{std::array<Node<3>*, 3>{&n0, &n1, &n2}}}
        , geom{element, integrator}
    {
        geom.set_sieve_id(0);
    }

    // Arbitrary 3-node beam
    BeamN3Fixture(double x0, double y0, double z0,
                  double x1, double y1, double z1,
                  double x2, double y2, double z2)
        : n0{0, x0, y0, z0}
        , n1{1, x1, y1, z1}
        , n2{2, x2, y2, z2}
        , element{std::optional<std::array<Node<3>*, 3>>{std::array<Node<3>*, 3>{&n0, &n1, &n2}}}
        , geom{element, integrator}
    {
        geom.set_sieve_id(0);
    }

    auto make_beam() {
        return TimoshenkoBeamN<3>{&geom, mat};
    }
};

// 4-node fixture (cubic beam)
struct BeamN4Fixture {
    Node<3> n0, n1, n2, n3;
    LagrangeElement3D<4> element;
    GaussLegendreCellIntegrator<3> integrator; // N-1 = 3 GPs
    ElementGeometry<3> geom;

    TimoshenkoBeamMaterial3D mat_instance{200.0, 80.0, 0.01, 8.33e-6, 8.33e-6, 1.41e-5, 5.0/6.0, 5.0/6.0};
    Material<TimoshenkoBeam3D> mat{mat_instance, ElasticUpdate{}};

    // Straight 4-node beam along x: nodes at 0, L/3, 2L/3, L
    BeamN4Fixture(double L)
        : n0{0, 0.0, 0.0, 0.0}
        , n1{1, L / 3.0, 0.0, 0.0}
        , n2{2, 2.0 * L / 3.0, 0.0, 0.0}
        , n3{3, L, 0.0, 0.0}
        , element{std::optional<std::array<Node<3>*, 4>>{std::array<Node<3>*, 4>{&n0, &n1, &n2, &n3}}}
        , geom{element, integrator}
    {
        geom.set_sieve_id(0);
    }

    auto make_beam() {
        return TimoshenkoBeamN<4>{&geom, mat};
    }
};


// ============================================================================
//  Test: N=2 stiffness equivalence with BeamElement
// ============================================================================

void test_N2_K_equivalence_with_BeamElement() {
    // Both should produce the same 12×12 stiffness matrix
    // for a straight beam along x, L=5.
    //
    // Note: BeamElement uses 2 GP (full integration for linear),
    // TimoshenkoBeamN<2> uses N-1=1 GP (reduced integration for shear).
    // For a straight 2-node beam with uniform section, the results should
    // match exactly because:
    // - Axial/bending/torsion: linear shape functions, 1 GP exact for constant integrands
    // - Shear: reduced basis of order 0 (constant) on 1 GP — same as 1-point integration
    //
    // With 1 GP, the axial stiffness integral:
    //   ∫ (dN/ds)^T EA (dN/ds) ds  with 1 GP at ξ=0, w=2, dm=L/2
    //   = 2 · (L/2) · dN1/ds · EA · dN1/ds  → EA·(-1/L)^2·L = EA/L  ✓
    //
    // This works because (dN/ds)^2 is constant for linear shape functions.

    BeamN2Fixture fn(0.0, 0.0, 0.0, 5.0, 0.0, 0.0);
    auto beamN = fn.make_beam();
    auto K_N = beamN.K();

    RefBeam3DFixture fr(0.0, 0.0, 0.0, 5.0, 0.0, 0.0);
    auto beam_ref = fr.make_beam();
    auto K_ref = beam_ref.K();

    ASSERT_TRUE(K_N.rows() == 12 && K_N.cols() == 12);
    ASSERT_TRUE(K_ref.rows() == 12 && K_ref.cols() == 12);

    for (int i = 0; i < 12; ++i)
        for (int j = 0; j < 12; ++j)
            ASSERT_NEAR(K_N(i, j), K_ref(i, j), 1e-8);
}

// ============================================================================
//  Test: K symmetry (N=2, 3, 4)
// ============================================================================

void test_K_symmetry_N2() {
    BeamN2Fixture f(0.0, 0.0, 0.0, 5.0, 0.0, 0.0);
    auto beam = f.make_beam();
    auto K = beam.K();

    ASSERT_TRUE(K.rows() == 12 && K.cols() == 12);
    for (int i = 0; i < 12; ++i)
        for (int j = i + 1; j < 12; ++j)
            ASSERT_NEAR(K(i, j), K(j, i), 1e-8);
}

void test_K_symmetry_N3() {
    BeamN3Fixture f(5.0);
    auto beam = f.make_beam();
    auto K = beam.K();

    ASSERT_TRUE(K.rows() == 18 && K.cols() == 18);
    for (int i = 0; i < 18; ++i)
        for (int j = i + 1; j < 18; ++j)
            ASSERT_NEAR(K(i, j), K(j, i), 1e-8);
}

void test_K_symmetry_N4() {
    BeamN4Fixture f(5.0);
    auto beam = f.make_beam();
    auto K = beam.K();

    ASSERT_TRUE(K.rows() == 24 && K.cols() == 24);
    for (int i = 0; i < 24; ++i)
        for (int j = i + 1; j < 24; ++j)
            ASSERT_NEAR(K(i, j), K(j, i), 1e-8);
}

// ============================================================================
//  Test: Rigid body modes — 6 zero eigenvalues for free 3D beam
// ============================================================================

void test_rigid_body_modes_N2() {
    BeamN2Fixture f(0.0, 0.0, 0.0, 5.0, 0.0, 0.0);
    auto beam = f.make_beam();
    auto K = beam.K();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(K);
    auto evals = solver.eigenvalues();

    int n_zero = 0;
    for (int i = 0; i < evals.size(); ++i)
        if (std::abs(evals[i]) < 1e-8) ++n_zero;

    ASSERT_TRUE(n_zero == 6);

    for (int i = 0; i < evals.size(); ++i)
        ASSERT_TRUE(evals[i] >= -1e-10);
}

void test_rigid_body_modes_N3() {
    BeamN3Fixture f(5.0);
    auto beam = f.make_beam();
    auto K = beam.K();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(K);
    auto evals = solver.eigenvalues();

    int n_zero = 0;
    for (int i = 0; i < evals.size(); ++i)
        if (std::abs(evals[i]) < 1e-8) ++n_zero;

    // 3-node beam in 3D: 18 DOFs, 6 rigid body modes
    ASSERT_TRUE(n_zero == 6);

    // All eigenvalues >= 0
    for (int i = 0; i < evals.size(); ++i)
        ASSERT_TRUE(evals[i] >= -1e-10);
}

void test_rigid_body_modes_N4() {
    BeamN4Fixture f(5.0);
    auto beam = f.make_beam();
    auto K = beam.K();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(K);
    auto evals = solver.eigenvalues();

    int n_zero = 0;
    for (int i = 0; i < evals.size(); ++i)
        if (std::abs(evals[i]) < 1e-8) ++n_zero;

    // 4-node beam: 24 DOFs, 6 rigid body modes
    ASSERT_TRUE(n_zero == 6);

    for (int i = 0; i < evals.size(); ++i)
        ASSERT_TRUE(evals[i] >= -1e-10);
}

// ============================================================================
//  Test: Axial stiffness — constant strain patch test
// ============================================================================

void test_axial_stiffness_N2() {
    // Along x, L=5, EA=200·0.01=2
    // K(0,0) = K(6,6) = EA/L = 0.4
    // K(0,6) = -EA/L
    BeamN2Fixture f(0.0, 0.0, 0.0, 5.0, 0.0, 0.0);
    auto beam = f.make_beam();
    auto K = beam.K();

    const double EA = 200.0 * 0.01;
    const double L  = 5.0;
    const double EAL = EA / L;

    ASSERT_NEAR(K(0, 0),  EAL, 1e-10);
    ASSERT_NEAR(K(6, 6),  EAL, 1e-10);
    ASSERT_NEAR(K(0, 6), -EAL, 1e-10);
}

void test_axial_stiffness_N3() {
    // Straight 3-node beam, L=6, EA=2.
    // For a 3-node (quadratic) beam along x with equally-spaced nodes,
    // the axial stiffness block should yield the same total stiffness
    // as a 2-element assembly of 2-node beams, each of length L/2.
    //
    // Standard result (2 GP exact for cubic):
    //    k_axial = EA/L · [7/3  -8/3   1/3]
    //                      [-8/3  16/3 -8/3]
    //                      [1/3  -8/3   7/3]
    BeamN3Fixture f(6.0);
    auto beam = f.make_beam();
    auto K = beam.K();

    const double EA  = 200.0 * 0.01; // 2.0
    const double L   = 6.0;
    const double EAL = EA / L;

    // Axial DOFs are at indices: node0→0, node1→6, node2→12
    ASSERT_NEAR(K(0, 0),   EAL * 7.0 / 3.0,  1e-8);
    ASSERT_NEAR(K(0, 6),   EAL * (-8.0 / 3.0), 1e-8);
    ASSERT_NEAR(K(0, 12),  EAL * 1.0 / 3.0,  1e-8);
    ASSERT_NEAR(K(6, 6),   EAL * 16.0 / 3.0, 1e-8);
    ASSERT_NEAR(K(12, 12), EAL * 7.0 / 3.0,  1e-8);
}

// ============================================================================
//  Test: Rotation invariance of K (trace & Frobenius norm)
// ============================================================================

void test_rotation_invariance_N3() {
    // Same beam, different orientations: trace and norm should match.
    BeamN3Fixture fx(0.0, 0.0, 0.0,  2.5, 0.0, 0.0,  5.0, 0.0, 0.0);  // along x
    BeamN3Fixture fy(0.0, 0.0, 0.0,  0.0, 2.5, 0.0,  0.0, 5.0, 0.0);  // along y

    auto beam_x = fx.make_beam();
    auto beam_y = fy.make_beam();

    auto K_x = beam_x.K();
    auto K_y = beam_y.K();

    ASSERT_NEAR(K_x.trace(), K_y.trace(), 1e-6);
    ASSERT_NEAR(K_x.norm(),  K_y.norm(),  1e-6);
}

// ============================================================================
//  Test: Sections access
// ============================================================================

void test_sections_N3() {
    BeamN3Fixture f(5.0);
    auto beam = f.make_beam();

    // N-1 = 2 Gauss points → 2 sections
    ASSERT_TRUE(beam.sections().size() == 2);

    // Each section provides a 6×6 constitutive matrix
    auto C = beam.sections()[0].C();
    ASSERT_TRUE(C.rows() == 6);
    ASSERT_TRUE(C.cols() == 6);

    // C should be diag(EA, EIy, EIz, kyGA, kzGA, GJ)
    const double EA   = 200.0 * 0.01;
    const double EIy  = 200.0 * 8.33e-6;
    const double EIz  = 200.0 * 8.33e-6;
    const double kyGA = (5.0/6.0) * 80.0 * 0.01;
    const double kzGA = (5.0/6.0) * 80.0 * 0.01;
    const double GJ   = 80.0 * 1.41e-5;

    ASSERT_NEAR(C(0, 0), EA,   1e-10);
    ASSERT_NEAR(C(1, 1), EIy,  1e-10);
    ASSERT_NEAR(C(2, 2), EIz,  1e-10);
    ASSERT_NEAR(C(3, 3), kyGA, 1e-10);
    ASSERT_NEAR(C(4, 4), kzGA, 1e-10);
    ASSERT_NEAR(C(5, 5), GJ,   1e-10);

    // Off-diagonal should be zero
    ASSERT_NEAR(C(0, 1), 0.0, 1e-10);
}

// ============================================================================
//  Test: Shear basis construction
// ============================================================================

void test_shear_basis_N3() {
    BeamN3Fixture f(5.0);
    auto beam = f.make_beam();
    const auto& basis = beam.shear_interpolation_basis();

    // N-1 = 2 Gauss points → 2-point (linear) shear basis
    ASSERT_TRUE(basis.size() == 2);

    // Gauss-Legendre 2-point nodes: ±1/√3
    const double gp = 1.0 / std::sqrt(3.0);
    ASSERT_NEAR(basis.x(0), -gp, 1e-14);
    ASSERT_NEAR(basis.x(1),  gp, 1e-14);

    // Partition of unity: sum of basis functions = 1 at any point
    for (double xi : {-1.0, -0.5, 0.0, 0.5, 1.0}) {
        double sum = 0.0;
        for (std::size_t i = 0; i < basis.size(); ++i)
            sum += basis[i](xi);
        ASSERT_NEAR(sum, 1.0, 1e-14);
    }
}

// ============================================================================
//  Test: Element topology queries
// ============================================================================

void test_topology_queries_N3() {
    BeamN3Fixture f(5.0);
    auto beam = f.make_beam();

    ASSERT_TRUE(beam.num_nodes() == 3);
    ASSERT_TRUE(beam.num_integration_points() == 2);
    ASSERT_TRUE(beam.sieve_id() == 0);
}

void test_topology_queries_N4() {
    BeamN4Fixture f(5.0);
    auto beam = f.make_beam();

    ASSERT_TRUE(beam.num_nodes() == 4);
    ASSERT_TRUE(beam.num_integration_points() == 3);
}

void test_topology_queries_N2() {
    BeamN2Fixture f(0.0, 0.0, 0.0, 5.0, 0.0, 0.0);
    auto beam = f.make_beam();

    ASSERT_TRUE(beam.num_nodes() == 2);
    ASSERT_TRUE(beam.num_integration_points() == 1);
    ASSERT_TRUE(beam.sieve_id() == 0);
}

void test_quadrature_contract_N2() {
    Node<3> n0{0, 0.0, 0.0, 0.0};
    Node<3> n1{1, 5.0, 0.0, 0.0};
    LagrangeElement3D<2> element{
        std::optional<std::array<Node<3>*, 2>>{
            std::array<Node<3>*, 2>{&n0, &n1}
        }
    };
    GaussLegendreCellIntegrator<2> integrator;
    ElementGeometry<3> geom{element, integrator};

    TimoshenkoBeamMaterial3D mat_instance{
        200.0, 80.0, 0.01, 8.33e-6, 8.33e-6, 1.41e-5, 5.0 / 6.0, 5.0 / 6.0
    };
    Material<TimoshenkoBeam3D> mat{mat_instance, ElasticUpdate{}};

    bool threw = false;
    try {
        [[maybe_unused]] auto beam = TimoshenkoBeamN<2>{&geom, mat};
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

// ============================================================================
//  Test: Rank check — K should have rank (6N - 6)
// ============================================================================

void test_K_rank_N3() {
    BeamN3Fixture f(5.0);
    auto beam = f.make_beam();
    auto K = beam.K();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(K);
    auto evals = solver.eigenvalues();

    int n_nonzero = 0;
    for (int i = 0; i < evals.size(); ++i)
        if (std::abs(evals[i]) > 1e-8) ++n_nonzero;

    // 18 DOFs - 6 rigid body modes = 12 non-zero eigenvalues
    ASSERT_TRUE(n_nonzero == 12);
}

void test_K_rank_N4() {
    BeamN4Fixture f(5.0);
    auto beam = f.make_beam();
    auto K = beam.K();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(K);
    auto evals = solver.eigenvalues();

    int n_nonzero = 0;
    for (int i = 0; i < evals.size(); ++i)
        if (std::abs(evals[i]) > 1e-8) ++n_nonzero;

    // 24 DOFs - 6 rigid body modes = 18 non-zero eigenvalues
    ASSERT_TRUE(n_nonzero == 18);
}

// ============================================================================
//  Test: Curved beam — quarter-circle arc (N=3)
// ============================================================================

void test_curved_beam_K_symmetry_N3() {
    // Quarter-circle arc in XY plane, radius R=10
    // Nodes at angles 0°, 45°, 90°
    const double R = 10.0;
    const double pi_4 = std::numbers::pi / 4.0;

    BeamN3Fixture f(
        R,                  0.0,                0.0,     // node 0: (R, 0, 0)
        R * std::cos(pi_4), R * std::sin(pi_4), 0.0,    // node 1: (R/√2, R/√2, 0)
        0.0,                R,                  0.0      // node 2: (0, R, 0)
    );
    auto beam = f.make_beam();
    auto K = beam.K();

    ASSERT_TRUE(K.rows() == 18 && K.cols() == 18);

    // Symmetry
    for (int i = 0; i < 18; ++i)
        for (int j = i + 1; j < 18; ++j)
            ASSERT_NEAR(K(i, j), K(j, i), 1e-8);
}

void test_curved_beam_rigid_body_modes_N3() {
    // Quarter-circle: should still have 6 rigid body modes
    const double R = 10.0;
    const double pi_4 = std::numbers::pi / 4.0;

    BeamN3Fixture f(
        R, 0.0, 0.0,
        R * std::cos(pi_4), R * std::sin(pi_4), 0.0,
        0.0, R, 0.0
    );
    auto beam = f.make_beam();
    auto K = beam.K();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(K);
    auto evals = solver.eigenvalues();

    int n_zero = 0;
    for (int i = 0; i < evals.size(); ++i)
        if (std::abs(evals[i]) < 1e-8) ++n_zero;

    ASSERT_TRUE(n_zero == 6);

    for (int i = 0; i < evals.size(); ++i)
        ASSERT_TRUE(evals[i] >= -1e-10);
}

void test_curved_beam_K_positive_semidefinite_N3() {
    // Half-circle arc
    const double R = 5.0;

    BeamN3Fixture f(
        R, 0.0, 0.0,
        0.0, R, 0.0,
        -R, 0.0, 0.0
    );
    auto beam = f.make_beam();
    auto K = beam.K();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(K);
    auto evals = solver.eigenvalues();

    // All eigenvalues >= 0 (positive semi-definite)
    for (int i = 0; i < evals.size(); ++i)
        ASSERT_TRUE(evals[i] >= -1e-10);
}


// ============================================================================
//  Test: Rotation matrix for curved beam
// ============================================================================

void test_rotation_matrix_N3_curved() {
    // Quarter-circle arc in XY plane
    // Tangent at center (ξ=0) should point roughly at 45° in XY plane
    const double R = 10.0;
    const double pi_4 = std::numbers::pi / 4.0;

    BeamN3Fixture f(
        R, 0.0, 0.0,
        R * std::cos(pi_4), R * std::sin(pi_4), 0.0,
        0.0, R, 0.0
    );
    auto beam = f.make_beam();
    auto R_mat = beam.rotation_matrix();

    // R should be orthogonal
    Eigen::Matrix3d product = R_mat * R_mat.transpose();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            ASSERT_NEAR(product(i, j), (i == j) ? 1.0 : 0.0, 1e-12);
}


// ============================================================================
//  Main
// ============================================================================

int main() {
    std::cout << "=== TimoshenkoBeamN Tests ===\n";

    // N=2 equivalence
    RUN_TEST(test_N2_K_equivalence_with_BeamElement);

    // Symmetry
    RUN_TEST(test_K_symmetry_N2);
    RUN_TEST(test_K_symmetry_N3);
    RUN_TEST(test_K_symmetry_N4);

    // Rigid body modes
    RUN_TEST(test_rigid_body_modes_N2);
    RUN_TEST(test_rigid_body_modes_N3);
    RUN_TEST(test_rigid_body_modes_N4);

    // Axial stiffness
    RUN_TEST(test_axial_stiffness_N2);
    RUN_TEST(test_axial_stiffness_N3);

    // Rotation invariance
    RUN_TEST(test_rotation_invariance_N3);

    // Section access
    RUN_TEST(test_sections_N3);

    // Shear basis
    RUN_TEST(test_shear_basis_N3);

    // Topology
    RUN_TEST(test_topology_queries_N2);
    RUN_TEST(test_topology_queries_N3);
    RUN_TEST(test_topology_queries_N4);
    RUN_TEST(test_quadrature_contract_N2);

    // Rank
    RUN_TEST(test_K_rank_N3);
    RUN_TEST(test_K_rank_N4);

    // Curved beam
    RUN_TEST(test_curved_beam_K_symmetry_N3);
    RUN_TEST(test_curved_beam_rigid_body_modes_N3);
    RUN_TEST(test_curved_beam_K_positive_semidefinite_N3);
    RUN_TEST(test_rotation_matrix_N3_curved);

    std::cout << "\n=== " << g_pass << " PASSED, " << g_fail << " FAILED ===\n";
    return g_fail == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
