// ============================================================================
//  Tests for beam-axis quadrature families and their use in TimoshenkoBeamN
// ============================================================================

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include <Eigen/Dense>

// Narrow validation umbrella for the beam slice. This is the migration path
// away from the repository-wide header_files.hh umbrella.
#include "src/validation/BeamValidationSupport.hh"

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

template <typename Quadrature>
double integrate_monomial(int degree) {
    Quadrature q;
    return q([&](std::span<const double> xi) {
        return std::pow(xi[0], degree);
    });
}

double exact_line_monomial_integral(int degree) {
    if (degree % 2 != 0) {
        return 0.0;
    }
    return 2.0 / static_cast<double>(degree + 1);
}

template <BeamAxisQuadratureFamily Family, std::size_t n>
void check_exactness_family() {
    using Quadrature = BeamAxisQuadratureT<Family, n>;
    constexpr int max_degree = BeamAxisQuadratureTraits<Family, n>::polynomial_exactness_degree;
    for (int degree = 0; degree <= max_degree; ++degree) {
        ASSERT_NEAR(integrate_monomial<Quadrature>(degree),
                    exact_line_monomial_integral(degree), 1e-11);
    }
}

void test_gauss_legendre_exactness() {
    check_exactness_family<BeamAxisQuadratureFamily::GaussLegendre, 1>();
    check_exactness_family<BeamAxisQuadratureFamily::GaussLegendre, 2>();
    check_exactness_family<BeamAxisQuadratureFamily::GaussLegendre, 3>();
    check_exactness_family<BeamAxisQuadratureFamily::GaussLegendre, 4>();
    check_exactness_family<BeamAxisQuadratureFamily::GaussLegendre, 5>();
}

void test_gauss_lobatto_exactness() {
    check_exactness_family<BeamAxisQuadratureFamily::GaussLobatto, 1>();
    check_exactness_family<BeamAxisQuadratureFamily::GaussLobatto, 2>();
    check_exactness_family<BeamAxisQuadratureFamily::GaussLobatto, 3>();
    check_exactness_family<BeamAxisQuadratureFamily::GaussLobatto, 4>();
    check_exactness_family<BeamAxisQuadratureFamily::GaussLobatto, 5>();
}

void test_gauss_radau_left_exactness() {
    check_exactness_family<BeamAxisQuadratureFamily::GaussRadauLeft, 1>();
    check_exactness_family<BeamAxisQuadratureFamily::GaussRadauLeft, 2>();
    check_exactness_family<BeamAxisQuadratureFamily::GaussRadauLeft, 3>();
    check_exactness_family<BeamAxisQuadratureFamily::GaussRadauLeft, 4>();
    check_exactness_family<BeamAxisQuadratureFamily::GaussRadauLeft, 5>();
}

void test_gauss_radau_right_mirrors_left_rule() {
    using Left = BeamAxisQuadratureT<BeamAxisQuadratureFamily::GaussRadauLeft, 4>;
    using Right = BeamAxisQuadratureT<BeamAxisQuadratureFamily::GaussRadauRight, 4>;

    ASSERT_TRUE(Left::num_integration_points == Right::num_integration_points);

    for (std::size_t i = 0; i < Left::num_integration_points; ++i) {
        const auto left_xi = Left::reference_integration_point(i)[0];
        const auto right_xi = Right::reference_integration_point(Left::num_integration_points - 1 - i)[0];
        ASSERT_NEAR(left_xi, -right_xi, 1e-14);
        ASSERT_NEAR(Left::weight(i), Right::weight(Left::num_integration_points - 1 - i), 1e-14);
    }
}

template <typename Integrator>
struct BeamFixtureN3 {
    Node<3> n0, n1, n2;
    LagrangeElement3D<3> element;
    Integrator integrator;
    ElementGeometry<3> geom;

    TimoshenkoBeamMaterial3D mat_instance{200.0, 80.0, 0.01, 8.33e-6, 8.33e-6, 1.41e-5, 5.0/6.0, 5.0/6.0};
    Material<TimoshenkoBeam3D> mat{mat_instance, ElasticUpdate{}};

    BeamFixtureN3(double L)
        : n0{0, 0.0, 0.0, 0.0}
        , n1{1, L/2.0, 0.0, 0.0}
        , n2{2, L, 0.0, 0.0}
        , element{std::optional<std::array<Node<3>*, 3>>{std::array<Node<3>*, 3>{&n0, &n1, &n2}}}
        , geom{element, integrator}
    {
        geom.set_sieve_id(0);
    }

    auto make_beam() { return TimoshenkoBeamN<3>{&geom, mat}; }
};

template <typename Integrator>
struct BeamFixtureN4 {
    Node<3> n0, n1, n2, n3;
    LagrangeElement3D<4> element;
    Integrator integrator;
    ElementGeometry<3> geom;

    TimoshenkoBeamMaterial3D mat_instance{200.0, 80.0, 0.01, 8.33e-6, 8.33e-6, 1.41e-5, 5.0/6.0, 5.0/6.0};
    Material<TimoshenkoBeam3D> mat{mat_instance, ElasticUpdate{}};

    BeamFixtureN4(double L)
        : n0{0, 0.0, 0.0, 0.0}
        , n1{1, L / 3.0, 0.0, 0.0}
        , n2{2, 2.0 * L / 3.0, 0.0, 0.0}
        , n3{3, L, 0.0, 0.0}
        , element{std::optional<std::array<Node<3>*, 4>>{std::array<Node<3>*, 4>{&n0, &n1, &n2, &n3}}}
        , geom{element, integrator}
    {
        geom.set_sieve_id(0);
    }

    auto make_beam() { return TimoshenkoBeamN<4>{&geom, mat}; }
};

void test_timoshenko_beam_lobatto_tracks_station_coordinates() {
    BeamFixtureN3<GaussLobattoCellIntegrator<2>> fixture(5.0);
    auto beam = fixture.make_beam();
    const auto& basis = beam.shear_interpolation_basis();

    ASSERT_TRUE(basis.size() == 2);
    ASSERT_NEAR(basis.x(0), -1.0, 1e-14);
    ASSERT_NEAR(basis.x(1),  1.0, 1e-14);

    for (std::size_t i = 0; i < basis.size(); ++i) {
        ASSERT_NEAR(basis.x(i), beam.geometry().reference_integration_point(i)[0], 1e-14);
    }
}

void test_timoshenko_beam_radau_tracks_station_coordinates() {
    BeamFixtureN4<GaussRadauCellIntegrator<3, GaussRadau::Endpoint::Left>> fixture(5.0);
    auto beam = fixture.make_beam();
    const auto& basis = beam.shear_interpolation_basis();

    ASSERT_TRUE(basis.size() == 3);
    ASSERT_NEAR(basis.x(0), -1.0, 1e-14);

    for (std::size_t i = 0; i < basis.size(); ++i) {
        ASSERT_NEAR(basis.x(i), beam.geometry().reference_integration_point(i)[0], 1e-14);
    }
}

void test_timoshenko_beam_lobatto_K_symmetry() {
    BeamFixtureN3<GaussLobattoCellIntegrator<2>> fixture(5.0);
    auto beam = fixture.make_beam();
    auto K = beam.K();

    ASSERT_TRUE(K.rows() == 18 && K.cols() == 18);
    for (int i = 0; i < 18; ++i)
        for (int j = i + 1; j < 18; ++j)
            ASSERT_NEAR(K(i, j), K(j, i), 1e-8);
}

void test_timoshenko_beam_radau_K_symmetry() {
    BeamFixtureN4<GaussRadauCellIntegrator<3, GaussRadau::Endpoint::Left>> fixture(5.0);
    auto beam = fixture.make_beam();
    auto K = beam.K();

    ASSERT_TRUE(K.rows() == 24 && K.cols() == 24);
    for (int i = 0; i < 24; ++i)
        for (int j = i + 1; j < 24; ++j)
            ASSERT_NEAR(K(i, j), K(j, i), 1e-8);
}

int main() {
    std::cout << "Running beam-axis quadrature tests...\n";

    RUN_TEST(test_gauss_legendre_exactness);
    RUN_TEST(test_gauss_lobatto_exactness);
    RUN_TEST(test_gauss_radau_left_exactness);
    RUN_TEST(test_gauss_radau_right_mirrors_left_rule);
    RUN_TEST(test_timoshenko_beam_lobatto_tracks_station_coordinates);
    RUN_TEST(test_timoshenko_beam_radau_tracks_station_coordinates);
    RUN_TEST(test_timoshenko_beam_lobatto_K_symmetry);
    RUN_TEST(test_timoshenko_beam_radau_K_symmetry);

    std::cout << "\nPassed: " << g_pass << "  Failed: " << g_fail << "\n";
    return g_fail == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
