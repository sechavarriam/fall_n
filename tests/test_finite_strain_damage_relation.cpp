#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>

#include <Eigen/Dense>

#include "src/continuum/Continuum.hh"
#include "src/materials/ConstitutiveIntegrator.hh"
#include "src/materials/LinealElasticMaterial.hh"
#include "src/materials/Material.hh"
#include "src/materials/local_problem/NewtonLocalSolver.hh"
#include "src/model/MaterialPoint.hh"

using namespace continuum;

namespace {

int passed = 0;
int failed = 0;

void report(const char* name, bool ok) {
    if (ok) {
        ++passed;
        std::cout << "  PASS  " << name << "\n";
    } else {
        ++failed;
        std::cout << "  FAIL  " << name << "\n";
    }
}

template <typename VecA, typename VecB>
double max_abs_diff(const VecA& a, const VecB& b) {
    return (a - b).cwiseAbs().maxCoeff();
}

template <typename MatA, typename MatB>
double max_abs_diff_m(const MatA& a, const MatB& b) {
    return (a - b).cwiseAbs().maxCoeff();
}

[[nodiscard]] ConstitutiveKinematics<3> make_total_lagrangian_kinematics(double lambda_x) {
    Tensor2<3> F = Tensor2<3>::identity();
    F(0, 0) = lambda_x;

    GPKinematics<3> gp;
    gp.F = F;
    gp.detF = F.determinant();
    gp.strain_voigt = strain::green_lagrange(F).voigt_engineering();
    return make_constitutive_kinematics<TotalLagrangian>(gp);
}

[[nodiscard]] ConstitutiveKinematics<3> make_updated_lagrangian_kinematics(double lambda_x) {
    Tensor2<3> F = Tensor2<3>::identity();
    F(0, 0) = lambda_x;

    GPKinematics<3> gp;
    gp.F = F;
    gp.detF = F.determinant();
    gp.strain_voigt = strain::green_lagrange(F).voigt_engineering();
    return make_constitutive_kinematics<UpdatedLagrangian>(gp);
}

[[nodiscard]] ConstitutiveKinematics<3> make_green_lagrange_reference_kinematics(
    double E11) {
    ConstitutiveKinematics<3> kin;
    kin.active_strain_measure = StrainMeasureKind::green_lagrange;
    kin.conjugate_stress_measure = StressMeasureKind::second_piola_kirchhoff;
    kin.green_lagrange_strain = SymmetricTensor2<3>{E11, 0.0, 0.0, 0.0, 0.0, 0.0};
    kin.infinitesimal_strain = kin.green_lagrange_strain;
    kin.almansi_strain = kin.green_lagrange_strain;
    kin.engineering_strain = kin.green_lagrange_strain.voigt_engineering();
    kin.F = Tensor2<3>::identity();
    kin.detF = 1.0;
    return kin;
}

struct ExactDamageLocalSolver {
    template <typename Problem, typename Relation, typename ContextT>
    [[nodiscard]] auto solve(
        const Problem&,
        const Relation&,
        const ContextT& context) const
        -> LocalNewtonSolveResult<typename Problem::UnknownT>
    {
        using UnknownT = typename Problem::UnknownT;
        UnknownT target = UnknownT::Zero();
        target(0) = context.trial_state.target_kappa;
        target(1) = context.trial_state.target_damage;
        return {target, true, 1, 0.0};
    }
};

using DamageRelation = NeoHookeanDamageRelation<3>;
using DamageLocalProblem = damage::RateIndependentDamageLocalProblem;
using DamageNewtonIntegrator = ContinuumLocalProblemIntegrator<
    DamageLocalProblem,
    NewtonLocalSolver<8>>;
using DamageExactIntegrator = ContinuumLocalProblemIntegrator<
    DamageLocalProblem,
    ExactDamageLocalSolver>;

static_assert(ConstitutiveRelation<DamageRelation>);
static_assert(InelasticConstitutiveRelation<DamageRelation>);
static_assert(ExternallyStateDrivenConstitutiveRelation<DamageRelation>);
static_assert(ContinuumKinematicsAwareConstitutiveRelation<DamageRelation>);
static_assert(ExternallyStateDrivenContinuumRelation<DamageRelation>);
static_assert(ContinuumLocalProblemPolicy<DamageLocalProblem, DamageRelation>);
static_assert(LocalNonlinearSolverPolicy<
    ExactDamageLocalSolver,
    DamageLocalProblem,
    DamageRelation,
    DamageLocalProblem::ContextT<DamageRelation>>);

[[nodiscard]] DamageRelation make_relation() {
    const auto model = CompressibleNeoHookean<3>::from_E_nu(30000.0, 0.20);
    const auto evolution =
        damage::ExponentialDamageEvolution{0.020, 0.050, 1.0e-6};
    return DamageRelation{model, damage::PositiveGreenLagrangeEquivalentStrain<3>{}, evolution};
}

void test_damage_remains_inactive_below_threshold() {
    auto relation = make_relation();
    auto alpha = DamageRelation::InternalVariablesT{};
    const auto kin = make_total_lagrangian_kinematics(1.01);
    const auto undamaged = HyperelasticRelation{relation.model()};

    relation.commit(alpha, kin);
    const auto stress = relation.compute_response(kin, alpha);
    const auto tangent = relation.tangent(kin, alpha);
    const auto undamaged_stress = undamaged.compute_response(kin);

    report("fs_damage_below_threshold_damage_zero", std::abs(alpha.damage) < 1.0e-14);
    report("fs_damage_below_threshold_kappa_positive", alpha.kappa > 0.0);
    report("fs_damage_below_threshold_matches_elastic_stress",
        max_abs_diff(stress.components(), undamaged_stress.components()) < 1.0e-12);
    report("fs_damage_below_threshold_matches_elastic_tangent",
        max_abs_diff_m(tangent, undamaged.tangent(kin)) < 1.0e-12);
}

void test_damage_grows_and_unloading_does_not_heal() {
    auto relation = make_relation();
    auto alpha = DamageRelation::InternalVariablesT{};

    const auto kin_high = make_total_lagrangian_kinematics(1.08);
    relation.commit(alpha, kin_high);
    const double damage_after_loading = alpha.damage;
    const double kappa_after_loading = alpha.kappa;

    const auto damaged_stress = relation.compute_response(kin_high, alpha);
    const auto undamaged_stress =
        HyperelasticRelation{relation.model()}.compute_response(kin_high);

    const auto kin_unload = make_total_lagrangian_kinematics(1.02);
    relation.commit(alpha, kin_unload);

    report("fs_damage_grows_under_tension", damage_after_loading > 1.0e-10);
    report("fs_damage_reduces_stress",
        std::abs(damaged_stress.components()(0)) <
        std::abs(undamaged_stress.components()(0)));
    report("fs_damage_unloading_preserves_damage",
        std::abs(alpha.damage - damage_after_loading) < 1.0e-14);
    report("fs_damage_unloading_preserves_kappa",
        std::abs(alpha.kappa - kappa_after_loading) < 1.0e-14);
}

void test_damage_spatial_path_returns_degraded_cauchy_stress() {
    auto relation = make_relation();
    auto alpha = DamageRelation::InternalVariablesT{};
    const auto kin_peak = make_total_lagrangian_kinematics(1.08);
    relation.commit(alpha, kin_peak);

    const auto kin_ul = make_updated_lagrangian_kinematics(1.04);

    const double g = relation.damage_evolution().degradation(alpha.damage);
    const auto S0 = relation.model().second_piola_kirchhoff(kin_ul.green_lagrange_strain);
    const auto sigma_expected = stress::cauchy_from_2pk(S0 * g, kin_ul.F);
    const auto sigma_actual = relation.compute_response(kin_ul, alpha);
    const auto sigma_expected_voigt = tensor_to_stress<3>(sigma_expected);

    const auto c_expected =
        g * ops::push_forward_tangent(
                relation.model().material_tangent(kin_ul.green_lagrange_strain),
                kin_ul.F).voigt_matrix();
    const auto c_actual = relation.tangent(kin_ul, alpha);

    report("fs_damage_spatial_returns_cauchy",
        max_abs_diff(sigma_actual.components(),
                     sigma_expected_voigt.components()) < 1.0e-12);
    report("fs_damage_spatial_returns_degraded_tangent",
        max_abs_diff_m(c_actual, c_expected) < 1.0e-12);
}

void test_damage_active_loading_tangent_matches_integrated_finite_difference() {
    auto relation = make_relation();
    const auto alpha0 = DamageRelation::InternalVariablesT{};
    const auto kin = make_green_lagrange_reference_kinematics(0.060);

    const auto result =
        ContinuumLocalIntegrationAlgorithm<DamageLocalProblem, NewtonLocalSolver<8>>{}
            .integrate(relation, kin, alpha0);

    const double h = 1.0e-7;
    auto kin_p = make_green_lagrange_reference_kinematics(0.060 + h);
    auto kin_m = make_green_lagrange_reference_kinematics(0.060 - h);
    const auto result_p =
        ContinuumLocalIntegrationAlgorithm<DamageLocalProblem, NewtonLocalSolver<8>>{}
            .integrate(relation, kin_p, alpha0);
    const auto result_m =
        ContinuumLocalIntegrationAlgorithm<DamageLocalProblem, NewtonLocalSolver<8>>{}
            .integrate(relation, kin_m, alpha0);

    const auto fd_column =
        (result_p.response.components() - result_m.response.components()) / (2.0 * h);

    report("fs_damage_active_loading_tangent_is_consistent",
        max_abs_diff(fd_column, result.tangent.col(0)) < 5.0e-5);
    report("fs_damage_active_loading_is_marked_inelastic", result.inelastic);
}

void test_damage_unloading_tangent_reduces_to_secant() {
    auto relation = make_relation();
    const auto alpha0 = DamageRelation::InternalVariablesT{};

    const auto kin_peak = make_green_lagrange_reference_kinematics(0.080);
    const auto peak_result =
        ContinuumLocalIntegrationAlgorithm<DamageLocalProblem, NewtonLocalSolver<8>>{}
            .integrate(relation, kin_peak, alpha0);

    const auto kin_unload = make_green_lagrange_reference_kinematics(0.030);
    const auto unload_result =
        ContinuumLocalIntegrationAlgorithm<DamageLocalProblem, NewtonLocalSolver<8>>{}
            .integrate(relation, kin_unload, peak_result.algorithmic_state);

    const double g =
        relation.damage_evolution().degradation(peak_result.algorithmic_state.damage);
    const auto C_secant =
        g * relation.model().material_tangent(kin_unload.green_lagrange_strain).voigt_matrix();

    report("fs_damage_unloading_tangent_is_secant",
        max_abs_diff_m(unload_result.tangent, C_secant) < 1.0e-12);
    report("fs_damage_unloading_does_not_reactivate",
        !unload_result.inelastic);
}

void test_damage_local_problem_with_injected_solvers() {
    auto relation = make_relation();
    const auto kin = make_total_lagrangian_kinematics(1.07);
    const auto alpha0 = DamageRelation::InternalVariablesT{};

    using NewtonAlgorithm = ContinuumLocalIntegrationAlgorithm<
        DamageLocalProblem,
        NewtonLocalSolver<8>>;
    using ExactAlgorithm = ContinuumLocalIntegrationAlgorithm<
        DamageLocalProblem,
        ExactDamageLocalSolver>;

    const auto result_newton = NewtonAlgorithm{}.integrate(relation, kin, alpha0);
    const auto result_exact = ExactAlgorithm{}.integrate(relation, kin, alpha0);

    report("fs_damage_local_problem_marks_inelastic", result_newton.inelastic);
    report("fs_damage_local_problem_state_matches_exact",
        std::abs(result_newton.algorithmic_state.damage -
                 result_exact.algorithmic_state.damage) < 1.0e-12 &&
        std::abs(result_newton.algorithmic_state.kappa -
                 result_exact.algorithmic_state.kappa) < 1.0e-12);
    report("fs_damage_local_problem_response_matches_exact",
        max_abs_diff(result_newton.response.components(),
                     result_exact.response.components()) < 1.0e-12);
    report("fs_damage_local_problem_tangent_matches_exact",
        max_abs_diff_m(result_newton.tangent, result_exact.tangent) < 1.0e-12);
}

void test_damage_integrates_through_material_handle_and_snapshot() {
    NeoHookeanDamageMaterial3D site{make_relation()};
    Material<ThreeDimensionalMaterial> material{site, DamageNewtonIntegrator{}};
    const auto kin = make_total_lagrangian_kinematics(1.075);

    const auto state_before = material.current_state().components();
    const auto stress_before = material.compute_response(kin);
    const auto state_after_prediction = material.current_state().components();
    material.commit(kin);
    const auto stress_after = material.compute_response(kin);
    const auto snap = material.internal_field_snapshot();

    report("fs_damage_material_prediction_is_side_effect_free",
        max_abs_diff(state_before, state_after_prediction) < 1.0e-14);
    report("fs_damage_material_commit_updates_current_state",
        max_abs_diff(material.current_state().components(),
                     kin.active_strain_voigt()) < 1.0e-12);
    report("fs_damage_material_snapshot_contains_damage",
        snap.has_damage() && snap.damage.value() > 1.0e-10);
    report("fs_damage_material_prediction_matches_committed_response",
        max_abs_diff(stress_before.components(), stress_after.components()) < 1.0e-12);
}

void test_damage_integrates_through_material_point() {
    NeoHookeanDamageMaterial3D site{make_relation()};
    Material<ThreeDimensionalMaterial> material{site, DamageExactIntegrator{}};
    MaterialPoint<ThreeDimensionalMaterial> point{material};

    IntegrationPoint<3> gp;
    gp.set_coord(std::array<double, 3>{0.2, 0.3, 0.4});
    gp.set_weight(1.5);
    point.bind_integration_point(gp);

    const auto kin = make_total_lagrangian_kinematics(1.09);
    point.commit(kin);
    const auto snap = point.internal_field_snapshot();
    const auto stress = point.compute_response(kin);

    report("fs_damage_material_point_binds_gauss_site",
        point.integration_point() != nullptr &&
        std::abs(point.integration_point()->weight() - 1.5) < 1.0e-14);
    report("fs_damage_material_point_snapshot_contains_damage",
        snap.has_damage() && snap.damage.value() > 1.0e-10);
    report("fs_damage_material_point_returns_nonzero_stress",
        stress.components().norm() > 1.0e-12);
}

} // namespace

int main() {
    std::cout << "=== Finite-Strain Damage Relation Tests ===\n";

    test_damage_remains_inactive_below_threshold();
    test_damage_grows_and_unloading_does_not_heal();
    test_damage_spatial_path_returns_degraded_cauchy_stress();
    test_damage_active_loading_tangent_matches_integrated_finite_difference();
    test_damage_unloading_tangent_reduces_to_secant();
    test_damage_local_problem_with_injected_solvers();
    test_damage_integrates_through_material_handle_and_snapshot();
    test_damage_integrates_through_material_point();

    std::cout << "\nPassed: " << passed << "  Failed: " << failed << "\n";
    return failed == 0 ? 0 : 1;
}
