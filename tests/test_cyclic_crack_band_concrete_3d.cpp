#include "../src/materials/ConstitutiveRelation.hh"
#include "../src/materials/constitutive_models/non_lineal/CyclicCrackBandConcrete3D.hh"

#include <Eigen/Dense>

#include <algorithm>
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

CyclicCrackBandConcrete3D::TangentT finite_difference_tangent(
    const CyclicCrackBandConcrete3D& concrete,
    const Strain<6>& strain,
    const CyclicCrackBandConcrete3DState& alpha,
    double step)
{
    CyclicCrackBandConcrete3D::TangentT tangent =
        CyclicCrackBandConcrete3D::TangentT::Zero();
    for (Eigen::Index j = 0; j < 6; ++j) {
        Eigen::Matrix<double, 6, 1> plus_values = strain.components();
        Eigen::Matrix<double, 6, 1> minus_values = strain.components();
        plus_values(j) += step;
        minus_values(j) -= step;

        Strain<6> plus;
        Strain<6> minus;
        plus.set_components(plus_values);
        minus.set_components(minus_values);

        const auto plus_stress = concrete.compute_response(plus, alpha);
        const auto minus_stress = concrete.compute_response(minus, alpha);
        tangent.col(j) =
            (plus_stress.components() - minus_stress.components()) /
            (2.0 * step);
    }
    return tangent;
}

} // namespace

static_assert(ConstitutiveRelation<CyclicCrackBandConcrete3D>);
static_assert(InelasticConstitutiveRelation<CyclicCrackBandConcrete3D>);
static_assert(
    ExternallyStateDrivenConstitutiveRelation<CyclicCrackBandConcrete3D>);

int main()
{
    CyclicCrackBandConcrete3D concrete{
        30.0, 30000.0, 0.20, 2.0, 0.06, 100.0};

    auto alpha = CyclicCrackBandConcrete3DState{};
    const double eps_crack = concrete.tensile_cracking_strain();

    {
        const auto peak_compression = make_strain(0.0, 0.0, -2.0e-3);
        const auto stress = concrete.compute_response(peak_compression, alpha);
        concrete.commit(alpha, peak_compression);

        check(stress[2] < -29.5 && stress[2] > -30.5,
              "Kent-Park compression reaches fpc at eps0");
        check(alpha.damage == 0.0,
              "pure compression does not trigger tensile crack damage");
        check(alpha.num_cracks == 0,
              "pure compression does not create diagnostic cracks");
    }

    {
        alpha = {};
        const auto below_onset = make_strain(0.5 * eps_crack, 0.0, 0.0);
        const auto stress = concrete.compute_response(below_onset, alpha);
        concrete.commit(alpha, below_onset);

        check(alpha.damage == 0.0,
              "subcritical tension remains undamaged");
        check(stress[0] > 0.95 && stress[0] < 1.05,
              "subcritical tension uses full Kent-Park-compatible Ec");
    }

    {
        const auto cracked = make_strain(1.0e-3, 0.0, 0.0);
        const auto stress = concrete.compute_response(cracked, alpha);
        concrete.commit(alpha, cracked);

        check(alpha.damage > 0.95,
              "post-peak tension creates strong crack-band damage");
        check(alpha.num_cracks == 1,
              "damaged state exports one diagnostic crack normal");
        check(alpha.crack_strain[0] > 0.0,
              "damaged state exports a positive crack opening strain");
        check(stress[0] < 0.25,
              "regularized tension softening drops below the tensile strength");
        check(concrete.equivalent_crack_opening_mm(alpha) > 0.09,
              "crack-band material exposes an objective equivalent crack opening");
        check(concrete.retained_shear_stiffness_ratio(alpha) >=
                  concrete.residual_shear_stiffness_ratio(),
              "crack-band material exposes the active retained shear fraction");

        const double committed_damage = alpha.damage;
        const auto unload = make_strain(0.5 * eps_crack, 0.0, 0.0);
        const auto unload_stress = concrete.compute_response(unload, alpha);
        concrete.commit(alpha, unload);

        check(alpha.damage >= committed_damage,
              "tensile damage is irreversible under unloading");
        check(unload_stress[0] < 0.05,
              "unloading tensile stiffness remains degraded");

        const auto barely_closed = make_strain(-1.0e-4, 0.0, 0.0);
        const auto barely_closed_stress =
            concrete.compute_response(barely_closed, alpha);

        check(std::abs(barely_closed_stress[0]) < 0.5,
              "large tensile crack remains mostly open near zero compression");

        const auto closed = make_strain(-2.0e-3, 0.0, 0.0);
        const auto closed_stress = concrete.compute_response(closed, alpha);
        concrete.commit(alpha, closed);

        check(closed_stress[0] < -29.5 && closed_stress[0] > -30.5,
              "closed crack recovers the compression envelope");
        check(alpha.num_cracks == 1 && alpha.crack_closed[0],
              "committed crack is marked closed under compression");
    }

    {
        CyclicCrackBandConcrete3D short_band{
            30.0, 30000.0, 0.20, 2.0, 0.06, 50.0};
        CyclicCrackBandConcrete3D long_band{
            30.0, 30000.0, 0.20, 2.0, 0.06, 200.0};
        auto short_state = CyclicCrackBandConcrete3DState{};
        auto long_state = CyclicCrackBandConcrete3DState{};
        const auto strain = make_strain(5.0e-4, 0.0, 0.0);
        short_band.commit(short_state, strain);
        long_band.commit(long_state, strain);

        check(long_state.damage > short_state.damage,
              "larger crack-band length softens faster at fixed strain");
    }

    {
        const double confinement_factor = 1.21;
        const double confined_fpc = confinement_factor * 30.0;
        const double confined_eps0 = -0.002 * confinement_factor;
        const double eps50u =
            (3.0 + 0.29 * 30.0) / (145.0 * 30.0 - 1000.0);
        const double eps50h = 0.75 * 0.015 * std::sqrt(0.19 / 0.08);
        const double z_slope =
            0.5 / std::max(eps50u + eps50h + confined_eps0, 1.0e-6);
        CyclicCrackBandConcrete3D confined{
            confined_fpc,
            30000.0,
            0.20,
            2.0,
            0.06,
            100.0,
            1.0e-6,
            0.20,
            0.05,
            confined_eps0,
            0.20,
            z_slope};
        auto confined_state = CyclicCrackBandConcrete3DState{};
        const auto peak_strain = make_strain(0.0, 0.0, confined_eps0);
        const auto peak_stress =
            confined.compute_response(peak_strain, confined_state);

        check(std::abs(confined.peak_compressive_strain() - confined_eps0) <
                  1.0e-14,
              "confined crack-band concrete stores the Kent-Park peak strain");
        check(peak_stress[2] < -36.0 && peak_stress[2] > -36.6,
              "confined crack-band concrete reaches K*fpc at eps0(K)");
        check(std::abs(confined.kent_park_z_slope() - z_slope) <
                  1.0e-12,
              "confined crack-band concrete stores the Kent-Park descent slope");
    }

    {
        CyclicCrackBandConcrete3D degrading_shear{
            30.0,
            30000.0,
            0.20,
            0.6,
            0.06,
            800.0,
            1.0e-6,
            0.01,
            0.05,
            -0.002,
            0.20,
            0.0,
            0.001,
            0.015};
        const auto small_opening = make_strain(4.0e-3, 0.0, 0.0);
        const auto large_opening = make_strain(4.0e-2, 0.0, 0.0);
        const auto small_tangent =
            degrading_shear.tangent(small_opening, {});
        const auto large_tangent =
            degrading_shear.tangent(large_opening, {});

        check(degrading_shear.large_opening_residual_shear_stiffness_ratio() <
                  degrading_shear.residual_shear_stiffness_ratio(),
              "crack-band concrete stores a lower large-opening shear floor");
        check(large_tangent(3, 3) < small_tangent(3, 3),
              "open-crack shear retention decays with crack opening");
    }

    {
        CyclicCrackBandConcrete3D tangent_checked{
            30.0,
            30000.0,
            0.20,
            2.0,
            0.06,
            100.0,
            0.02,
            0.05,
            0.05};
        auto state = CyclicCrackBandConcrete3DState{};
        tangent_checked.commit(
            state,
            make_strain(7.0e-4, 1.0e-4, -2.0e-4, 1.0e-4, 0.0, 0.0));
        const auto trial =
            make_strain(6.0e-4, 1.5e-4, -2.5e-4, 1.3e-4, 0.5e-4, 0.2e-4);

        tangent_checked.set_material_tangent_mode(
            CyclicCrackBandConcrete3DTangentMode::
                AdaptiveCentralDifference);
        const auto consistent = tangent_checked.tangent(trial, state);
        const auto reference =
            finite_difference_tangent(tangent_checked, trial, state, 5.0e-8);
        const double relative_error =
            (consistent - reference).norm() /
            std::max(reference.norm(), 1.0);

        check(relative_error < 5.0e-3,
              "adaptive central tangent matches independent finite difference");
        check(consistent.allFinite(),
              "adaptive central tangent remains finite in a cracked state");

        auto loading_state = CyclicCrackBandConcrete3DState{};
        const auto active_softening =
            make_strain(3.5e-4, 0.0, 0.0, 0.0, 0.0, 0.0);
        const auto softening_tangent =
            tangent_checked.tangent(active_softening, loading_state);

        check(softening_tangent(0, 0) < 0.0,
              "consistent crack-band tangent captures active tensile softening");

        tangent_checked.set_material_tangent_mode(
            CyclicCrackBandConcrete3DTangentMode::
                AdaptiveCentralDifferenceWithSecantFallback);
        const auto fallback_tangent =
            tangent_checked.tangent(trial, state);
        check(fallback_tangent.allFinite(),
              "central tangent with secant fallback remains finite");
    }

    std::cout << passed_tests << "/" << total_tests << " tests passed\n";
    return passed_tests == total_tests ? 0 : 1;
}
