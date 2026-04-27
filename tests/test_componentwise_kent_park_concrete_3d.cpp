#include "../src/materials/ConstitutiveRelation.hh"
#include "../src/materials/constitutive_models/non_lineal/ComponentwiseKentParkConcrete3D.hh"

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

Strain<1> make_uniaxial(double eps)
{
    Strain<1> strain;
    strain.set_components(eps);
    return strain;
}

bool close(double lhs, double rhs, double scale = 1.0)
{
    return std::abs(lhs - rhs) <= 1.0e-10 * std::max(scale, 1.0);
}

} // namespace

static_assert(ConstitutiveRelation<ComponentwiseKentParkConcrete3D>);
static_assert(InelasticConstitutiveRelation<ComponentwiseKentParkConcrete3D>);
static_assert(
    ExternallyStateDrivenConstitutiveRelation<
        ComponentwiseKentParkConcrete3D>);

int main()
{
    const KentParkConcreteTensionConfig tension{
        .tensile_strength = 0.02 * 30.0,
        .softening_multiplier = 0.50,
        .residual_tangent_ratio = 1.0e-6,
        .crack_transition_multiplier = 0.50,
    };
    KentParkConcrete uniaxial{30.0, tension};
    ComponentwiseKentParkConcrete3D concrete{30.0, tension, 0.20};

    auto one_d_state = KentParkState{};
    auto three_d_state = ComponentwiseKentParkConcrete3DState{};
    const std::vector<double> history{
        0.0,
        0.5 * uniaxial.tensile_cracking_strain(),
        1.25 * uniaxial.tensile_cracking_strain(),
        0.0,
        -5.0e-4,
        -2.0e-3,
        -5.0e-4,
        2.0 * uniaxial.tensile_cracking_strain()};

    bool stress_matches = true;
    bool positive_tangent_policy_holds = true;
    bool positive_exact_tangents_match = true;
    for (const double eps : history) {
        const auto eps1 = make_uniaxial(eps);
        const auto eps3 = make_strain(eps, 0.0, 0.0);

        const double sig1 =
            uniaxial.compute_response(eps1, one_d_state).components();
        const double sig3 =
            concrete.compute_response(eps3, three_d_state)[0];
        const double et1 = uniaxial.tangent(eps1, one_d_state)(0, 0);
        const double et3 = concrete.tangent(eps3, three_d_state)(0, 0);

        stress_matches = stress_matches && close(sig1, sig3, std::abs(sig1));
        positive_tangent_policy_holds =
            positive_tangent_policy_holds && et3 > 0.0;
        if (et1 > 0.0) {
            positive_exact_tangents_match =
                positive_exact_tangents_match &&
                close(et1, et3, std::abs(et1));
        }

        uniaxial.commit(one_d_state, eps1);
        concrete.commit(three_d_state, eps3);
    }

    check(stress_matches,
          "3D componentwise normal stress matches Kent-Park uniaxial law");
    check(positive_tangent_policy_holds,
          "3D componentwise tangent remains positive for global Newton");
    check(positive_exact_tangents_match,
          "positive Kent-Park tangents are preserved exactly");
    check(three_d_state.normal_states[0].cracked,
          "3D state preserves Kent-Park crack history on the loaded axis");
    check(!three_d_state.normal_states[1].cracked &&
              !three_d_state.normal_states[2].cracked,
          "unloaded normal axes remain independent and uncracked");
    check(three_d_state.num_cracks == 1,
          "componentwise state exports one diagnostic crack");

    {
        const auto shear = make_strain(0.0, 0.0, 0.0, 1.0e-3);
        const auto stress = concrete.compute_response(shear, {});
        check(stress[3] > 0.0,
              "componentwise branch keeps an elastic shear response");
        check(std::abs(stress[0]) < 1.0e-12 &&
                  std::abs(stress[1]) < 1.0e-12 &&
                  std::abs(stress[2]) < 1.0e-12,
              "pure shear does not create artificial normal stress");
    }

    std::cout << passed_tests << "/" << total_tests << " tests passed\n";
    return passed_tests == total_tests ? 0 : 1;
}
