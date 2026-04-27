#include "../src/materials/ConstitutiveRelation.hh"
#include "../src/materials/constitutive_models/non_lineal/TensileCrackBandDamageConcreteProxy3D.hh"

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

static_assert(
    ConstitutiveRelation<TensileCrackBandDamageConcreteProxy3D>);
static_assert(
    InelasticConstitutiveRelation<TensileCrackBandDamageConcreteProxy3D>);
static_assert(
    ExternallyStateDrivenConstitutiveRelation<
        TensileCrackBandDamageConcreteProxy3D>);

int main()
{
    TensileCrackBandDamageConcreteProxy3D concrete{
        30000.0, 0.10, 0.20, 2.0, 0.06, 100.0};

    auto alpha = TensileCrackBandDamageConcreteProxy3DState{};

    {
        const auto compression = make_strain(0.0, 0.0, -1.0e-3);
        const auto stress = concrete.compute_response(compression, alpha);
        concrete.commit(alpha, compression);

        check(stress[2] < -29.0 && stress[2] > -31.0,
              "compression branch retains the concrete compression modulus");
        check(alpha.damage == 0.0,
              "pure compression does not trigger tensile damage");
        check(alpha.num_cracks == 0,
              "pure compression does not create diagnostic cracks");
    }

    {
        alpha = {};
        const auto below_onset = make_strain(2.0e-4, 0.0, 0.0);
        const auto stress = concrete.compute_response(below_onset, alpha);
        concrete.commit(alpha, below_onset);

        check(alpha.damage == 0.0,
              "subcritical tensile strain remains undamaged");
        check(stress[0] > 0.55 && stress[0] < 0.65,
              "subcritical tensile stress uses the tensile proxy modulus");
    }

    {
        const auto cracked = make_strain(3.0e-3, 0.0, 0.0);
        const auto stress = concrete.compute_response(cracked, alpha);
        concrete.commit(alpha, cracked);

        check(alpha.damage > 0.90,
              "post-peak tensile strain creates strong scalar damage");
        check(alpha.num_cracks == 1,
              "damaged state exports one diagnostic crack normal");
        check(alpha.crack_strain[0] > 0.0,
              "damaged state exports a positive crack opening strain");
        check(stress[0] < 1.0,
              "softened tensile stress is below the tensile strength scale");

        const double committed_damage = alpha.damage;
        const auto unload = make_strain(2.0e-4, 0.0, 0.0);
        const auto unload_stress = concrete.compute_response(unload, alpha);
        concrete.commit(alpha, unload);

        check(alpha.damage >= committed_damage,
              "damage is irreversible under unloading");
        check(unload_stress[0] < 0.1,
              "unloading tensile stiffness remains degraded");

        const auto closed = make_strain(-1.0e-3, 0.0, 0.0);
        const auto closed_stress = concrete.compute_response(closed, alpha);
        concrete.commit(alpha, closed);

        check(closed_stress[0] < -29.0 && closed_stress[0] > -31.0,
              "closed crack recovers normal compression stiffness");
        check(alpha.num_cracks == 1 && alpha.crack_closed[0],
              "committed crack is marked closed under compression");
    }

    {
        TensileCrackBandDamageConcreteProxy3D short_band{
            30000.0, 0.10, 0.20, 2.0, 0.06, 50.0};
        TensileCrackBandDamageConcreteProxy3D long_band{
            30000.0, 0.10, 0.20, 2.0, 0.06, 200.0};
        auto short_state = TensileCrackBandDamageConcreteProxy3DState{};
        auto long_state = TensileCrackBandDamageConcreteProxy3DState{};
        const auto strain = make_strain(1.5e-3, 0.0, 0.0);
        short_band.commit(short_state, strain);
        long_band.commit(long_state, strain);

        check(long_state.damage > short_state.damage,
              "longer crack-band length dissipates faster at fixed strain");
    }

    std::cout << passed_tests << "/" << total_tests << " tests passed\n";
    return passed_tests == total_tests ? 0 : 1;
}
