#include "../src/materials/ConstitutiveRelation.hh"
#include "../src/materials/constitutive_models/non_lineal/FixedCrackBandConcrete3D.hh"

#include <Eigen/Dense>

#include <cmath>
#include <iostream>

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
                      double eyy,
                      double ezz,
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

} // namespace

static_assert(ConstitutiveRelation<FixedCrackBandConcrete3D>);
static_assert(InelasticConstitutiveRelation<FixedCrackBandConcrete3D>);
static_assert(
    ExternallyStateDrivenConstitutiveRelation<FixedCrackBandConcrete3D>);

int main()
{
    FixedCrackBandConcrete3D concrete{
        30.0, 30000.0, 0.20, 2.0, 0.06, 100.0};

    auto alpha = FixedCrackBandConcrete3DState{};
    const double eps_crack = concrete.tensile_cracking_strain();
    const double intact_shear_modulus = 30000.0 / (2.0 * (1.0 + 0.20));

    {
        const auto peak_compression = make_strain(0.0, 0.0, -2.0e-3);
        const auto stress = concrete.compute_response(peak_compression, alpha);
        concrete.commit(alpha, peak_compression);

        check(stress[2] < -29.5 && stress[2] > -30.5,
              "pure compression reaches fpc at eps0");
        check(alpha.d() == 0.0,
              "pure compression does not create tensile damage");
        check(alpha.num_cracks == 0,
              "pure compression does not create fixed crack directions");
    }

    {
        alpha = {};
        const auto below_onset = make_strain(0.5 * eps_crack, 0.0, 0.0);
        const auto stress = concrete.compute_response(below_onset, alpha);
        concrete.commit(alpha, below_onset);

        check(alpha.d() == 0.0,
              "subcritical tension remains undamaged");
        check(stress[0] > 0.95 && stress[0] < 1.05,
              "subcritical tension uses full initial tensile stiffness");
    }

    {
        const auto cracked = make_strain(1.0e-3, 0.0, 0.0);
        const auto stress = concrete.compute_response(cracked, alpha);
        concrete.commit(alpha, cracked);

        check(alpha.num_cracks == 1,
              "uniaxial tension stores one fixed crack normal");
        check(alpha.d() > 0.95,
              "post-peak tension creates strong crack-band damage");
        check(std::abs(std::abs(alpha.crack_normals[0].x()) - 1.0) < 1.0e-12,
              "first fixed crack normal aligns with the tensile direction");
        check(stress[0] > 0.0 && stress[0] < 0.25,
              "regularized tensile stress drops below tensile strength");

        const auto sheared = make_strain(1.0e-3, 0.0, 0.0, 0.0, 0.0, 1.0e-4);
        const auto tangent = concrete.tangent(sheared, alpha);
        check(tangent(5, 5) < 0.25 * intact_shear_modulus,
              "open fixed crack degrades local shear transfer");

        const auto closed = make_strain(-2.0e-3, 0.0, 0.0);
        const auto closed_stress = concrete.compute_response(closed, alpha);
        concrete.commit(alpha, closed);
        check(closed_stress[0] < -29.5 && closed_stress[0] > -30.5,
              "closed fixed crack recovers the compression envelope");
        check(alpha.crack_closed[0],
              "committed crack is marked closed under compression");
    }

    {
        const auto transverse_crack = make_strain(0.0, 1.0e-3, 0.0);
        concrete.commit(alpha, transverse_crack);
        check(alpha.num_cracks >= 2,
              "orthogonal tensile reversal may store a second crack normal");
    }

    {
        auto rotated_state = FixedCrackBandConcrete3DState{};
        const double e = 1.0e-3;
        const auto rotated = make_strain(0.5 * e, 0.5 * e, 0.0, 0.0, 0.0, e);
        const auto stress = concrete.compute_response(rotated, rotated_state);
        concrete.commit(rotated_state, rotated);
        const Eigen::Vector3d n =
            Eigen::Vector3d{1.0, 1.0, 0.0}.normalized();

        check(rotated_state.num_cracks == 1,
              "rotated uniaxial tension creates one crack normal");
        check(std::abs(rotated_state.crack_normals[0].normalized().dot(n)) >
                  0.95,
              "fixed crack normal follows principal tensile strain direction");
        check(std::abs(stress[5]) > 1.0e-2,
              "rotated fixed crack response is transformed back to global shear");
    }

    std::cout << passed_tests << "/" << total_tests << " tests passed\n";
    return passed_tests == total_tests ? 0 : 1;
}
