#include "../src/materials/ConstitutiveRelation.hh"
#include "../src/materials/constitutive_models/lineal/OrthotropicBilinearConcreteProxy.hh"
#include "../src/materials/constitutive_models/lineal/OrthotropicBimodularConcreteProxy.hh"

#include <Eigen/Dense>

#include <cmath>
#include <iostream>
#include <vector>

namespace {

int total_tests = 0;
int passed_tests = 0;

void check(bool condition, const char* message)
{
    ++total_tests;
    if (condition) {
        ++passed_tests;
        std::cout << "[PASS] " << message << "\n";
    } else {
        std::cout << "[FAIL] " << message << "\n";
    }
}

Strain<6> make_strain(double exx,
                      double eyy = 0.0,
                      double ezz = 0.0,
                      double gyz = 0.0,
                      double gxz = 0.0,
                      double gxy = 0.0)
{
    Eigen::Matrix<double, 6, 1> v;
    v << exx, eyy, ezz, gyz, gxz, gxy;
    Strain<6> strain;
    strain.set_components(v);
    return strain;
}

bool close(double lhs, double rhs, double scale = 1.0, double tol = 1.0e-10)
{
    return std::abs(lhs - rhs) <= tol * std::max(scale, 1.0);
}

} // namespace

static_assert(ConstitutiveRelation<OrthotropicBilinearConcreteProxy>);

int main()
{
    // Representative C25/30-like parameters.
    const double Ec      = 30000.0;   // MPa
    const double fc      = 25.0;      // MPa  (compressive yield surrogate)
    const double hc      = 0.05;      // post-compressive-yield ratio
    const double Et_rat  = 1.0;       // identical initial slope in tension
    const double ft      = 2.5;       // MPa  (tensile yield surrogate)
    const double ht      = 0.01;      // residual post-cracking ratio
    const double nu_like = 0.20;

    OrthotropicBilinearConcreteProxy proxy{
        Ec, fc, hc, Et_rat, ft, ht, nu_like, 1.0};

    OrthotropicBimodularConcreteProxy reference{
        Ec, /*tension_ratio=*/Et_rat, nu_like, 1.0};

    const double eps_yc = fc / Ec;       // ~8.33e-4
    const double eps_yt = ft / Ec;       // ~8.33e-5

    // ----------------------------------------------------------------------
    // 1. Pre-yield tangent parity with the bimodular elastic proxy.
    // ----------------------------------------------------------------------
    {
        const auto eps = make_strain(0.5 * eps_yt);
        const double t_bil = proxy.tangent(eps)(0, 0);
        const double t_ref = reference.tangent(eps)(0, 0);
        check(close(t_bil, t_ref, std::abs(t_ref)),
              "pre-yield tension tangent matches bimodular proxy");
    }
    {
        const auto eps = make_strain(-0.5 * eps_yc);
        const double t_bil = proxy.tangent(eps)(0, 0);
        const double t_ref = reference.tangent(eps)(0, 0);
        check(close(t_bil, t_ref, std::abs(t_ref)),
              "pre-yield compression tangent matches bimodular proxy");
    }

    // ----------------------------------------------------------------------
    // 2. Tensile knee continuity in stress.
    // ----------------------------------------------------------------------
    {
        const double delta = 1.0e-12;
        const auto eps_minus = make_strain(eps_yt - delta);
        const auto eps_plus  = make_strain(eps_yt + delta);
        const double s_minus = proxy.compute_response(eps_minus)[0];
        const double s_plus  = proxy.compute_response(eps_plus)[0];
        check(close(s_minus, s_plus, ft, 1.0e-6),
              "stress is continuous across the tensile yield strain");
        check(close(s_minus, ft, ft, 1.0e-6),
              "tensile yield stress is reached at eps_yt");
    }

    // ----------------------------------------------------------------------
    // 3. Post-yield tensile slope equals h_t * E_t.
    // ----------------------------------------------------------------------
    {
        const auto eps_a = make_strain(2.0 * eps_yt);
        const auto eps_b = make_strain(3.0 * eps_yt);
        const double s_a = proxy.compute_response(eps_a)[0];
        const double s_b = proxy.compute_response(eps_b)[0];
        const double slope = (s_b - s_a) / (eps_b[0] - eps_a[0]);
        const double expected = ht * Ec;
        check(close(slope, expected, std::abs(expected), 1.0e-6),
              "post-yield tensile slope equals h_t * E_t");
    }

    // ----------------------------------------------------------------------
    // 4. Post-yield compressive slope equals h_c * E_c.
    // ----------------------------------------------------------------------
    {
        const auto eps_a = make_strain(-2.0 * eps_yc);
        const auto eps_b = make_strain(-3.0 * eps_yc);
        const double s_a = proxy.compute_response(eps_a)[0];
        const double s_b = proxy.compute_response(eps_b)[0];
        const double slope = (s_b - s_a) / (eps_b[0] - eps_a[0]);
        const double expected = hc * Ec;
        check(close(slope, expected, std::abs(expected), 1.0e-6),
              "post-yield compressive slope equals h_c * E_c");
        check(close(s_a, -fc + hc * Ec * (eps_a[0] + eps_yc),
                    std::abs(s_a), 1.0e-6),
              "compressive branch reproduces analytical bilinear law");
    }

    // ----------------------------------------------------------------------
    // 5. Tangent picks the post-yield slope on the soft branch only.
    // ----------------------------------------------------------------------
    {
        const auto eps_pre  = make_strain(0.99 * eps_yt);
        const auto eps_post = make_strain(1.01 * eps_yt);
        const double t_pre  = proxy.tangent(eps_pre)(0, 0);
        const double t_post = proxy.tangent(eps_post)(0, 0);
        check(close(t_pre, Ec, Ec),
              "tangent stays at E_t just below the tensile knee");
        check(close(t_post, ht * Ec, std::abs(ht * Ec)),
              "tangent drops to h_t * E_t just above the tensile knee");
    }

    // ----------------------------------------------------------------------
    // 6. Shear is linear elastic and decoupled from normal axes.
    // ----------------------------------------------------------------------
    {
        const double gxy = 1.5e-4;
        const auto eps = make_strain(0.0, 0.0, 0.0, 0.0, 0.0, gxy);
        const double G = proxy.shear_moduli_mpa()[2];
        const auto stress = proxy.compute_response(eps);
        check(close(stress[5], G * gxy, std::abs(G * gxy)),
              "shear xy stays linear elastic");
        check(std::abs(stress[0]) < 1.0e-12
                  && std::abs(stress[1]) < 1.0e-12
                  && std::abs(stress[2]) < 1.0e-12,
              "shear strain does not couple into normal stresses");
    }

    // ----------------------------------------------------------------------
    // 7. Path-independence: round-trip gives the same response.
    // ----------------------------------------------------------------------
    {
        const std::vector<double> history{
            0.0,
            2.5 * eps_yt,
            -3.0 * eps_yc,
            0.0,
            2.5 * eps_yt};
        double s_first = 0.0;
        double s_last = 0.0;
        for (std::size_t k = 0; k < history.size(); ++k) {
            const auto eps = make_strain(history[k]);
            const double s = proxy.compute_response(eps)[0];
            if (k == 1) s_first = s;
            if (k + 1 == history.size()) s_last = s;
        }
        check(close(s_first, s_last, std::abs(s_first), 1.0e-12),
              "proxy is path-independent across cyclic excursions");
    }

    std::cout << passed_tests << "/" << total_tests << " tests passed\n";
    return passed_tests == total_tests ? 0 : 1;
}
