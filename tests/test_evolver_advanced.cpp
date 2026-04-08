#include "header_files.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include <Eigen/Dense>
#include <petsc.h>

using namespace fall_n;

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool cond, const char* msg)
{
    if (cond) {
        ++g_pass;
        std::cout << "  [PASS] " << msg << "\n";
    } else {
        ++g_fail;
        std::cout << "  [FAIL] " << msg << "\n";
    }
}

[[nodiscard]] double relative_norm(const Eigen::Vector<double, 6>& a,
                                   const Eigen::Vector<double, 6>& b)
{
    const double denom = std::max({1.0, a.norm(), b.norm()});
    return (a - b).norm() / denom;
}

[[nodiscard]] ElementKinematics make_ek(
    std::size_t id,
    const Eigen::Vector3d& u_B,
    const Eigen::Vector3d& theta_B)
{
    ElementKinematics ek;
    ek.element_id = id;
    ek.endpoint_A = {0.0, 0.0, 0.0};
    ek.endpoint_B = {1.0, 0.0, 0.0};
    ek.up_direction = {0.0, 1.0, 0.0};

    const Eigen::Matrix3d R = Eigen::Matrix3d::Identity();

    ek.kin_A.centroid = Eigen::Vector3d{0.0, 0.0, 0.0};
    ek.kin_A.R = R;
    ek.kin_A.u_local = Eigen::Vector3d::Zero();
    ek.kin_A.theta_local = Eigen::Vector3d::Zero();
    ek.kin_A.E = 200.0;
    ek.kin_A.G = 80.0;
    ek.kin_A.nu = 0.25;

    ek.kin_B.centroid = Eigen::Vector3d{1.0, 0.0, 0.0};
    ek.kin_B.R = R;
    ek.kin_B.u_local = u_B;
    ek.kin_B.theta_local = theta_B;
    ek.kin_B.E = 200.0;
    ek.kin_B.G = 80.0;
    ek.kin_B.nu = 0.25;

    return ek;
}

struct SolvedCase {
    SubModelSolverResult result{};
    SectionHomogenizedResponse response{};
    Eigen::Vector<double, 6> volume_average{
        Eigen::Vector<double, 6>::Zero()};
};

[[nodiscard]] SolvedCase solve_case(const Eigen::Vector3d& u_B,
                                    const Eigen::Vector3d& theta_B,
                                    double width,
                                    double height,
                                    double h_pert = 1.0e-6)
{
    MultiscaleCoordinator coord;
    coord.add_critical_element(make_ek(0, u_B, theta_B));
    coord.build_sub_models(SubModelSpec{width, height, 2, 2, 4});

    NonlinearSubModelEvolver ev(coord.sub_models()[0], 30.0, ".", 9999);
    ev.set_regularization_policy(RegularizationPolicyKind::None, 0.0);

    SolvedCase solved;
    solved.result = ev.solve_step(0.0);
    solved.response = ev.section_response(width, height, h_pert);
    solved.volume_average = ev.compute_volume_average_forces(width, height);
    return solved;
}

void test_reinforced_evolver()
{
    std::cout << "\n== Reinforced evolver ==\n";

    const double fc = 30.0;
    const double W = 0.30;
    const double H = 0.40;
    const double eps = 1.0e-4;

    auto ek = make_ek(0, Eigen::Vector3d{eps, 0.0, 0.0}, Eigen::Vector3d::Zero());

    MultiscaleCoordinator coord_plain;
    coord_plain.add_critical_element(ElementKinematics{ek});
    coord_plain.build_sub_models(SubModelSpec{W, H, 2, 2, 4});

    NonlinearSubModelEvolver plain_ev(coord_plain.sub_models()[0], fc, ".", 9999);
    plain_ev.set_regularization_policy(RegularizationPolicyKind::None, 0.0);
    const auto r_plain = plain_ev.solve_step(0.0);
    check(r_plain.converged, "plain evolver converges");

    MultiscaleCoordinator coord_rebar;
    coord_rebar.add_critical_element(ElementKinematics{ek});

    SubModelSpec spec{W, H, 2, 2, 4};
    spec.rebar_bars = {
        {-W / 2 + 0.04, -H / 2 + 0.04, 3.14e-4},
        { W / 2 - 0.04, -H / 2 + 0.04, 3.14e-4},
        {-W / 2 + 0.04,  H / 2 - 0.04, 3.14e-4},
        { W / 2 - 0.04,  H / 2 - 0.04, 3.14e-4},
    };
    coord_rebar.build_sub_models(spec);

    NonlinearSubModelEvolver rebar_ev(coord_rebar.sub_models()[0], fc, ".", 9999);
    rebar_ev.set_rebar_material(200000.0, 420.0, 0.01);
    rebar_ev.set_regularization_policy(RegularizationPolicyKind::None, 0.0);
    const auto r_rebar = rebar_ev.solve_step(0.0);

    check(r_rebar.converged, "reinforced evolver converges");
    check(r_rebar.E_eff > r_plain.E_eff,
          "reinforcement increases the apparent axial stiffness");
    check(r_rebar.E_eff > 0.0, "reinforced axial stiffness stays positive");
}

void test_adaptive_substepping()
{
    std::cout << "\n== Adaptive sub-stepping ==\n";

    const double W = 0.20;
    const double H = 0.20;

    auto ek = make_ek(
        0, Eigen::Vector3d{1.0e-4, 0.0, 0.0}, Eigen::Vector3d::Zero());

    MultiscaleCoordinator coord;
    coord.add_critical_element(ElementKinematics{ek});
    coord.build_sub_models(SubModelSpec{W, H, 2, 2, 4});

    NonlinearSubModelEvolver ev(coord.sub_models()[0], 30.0, ".", 9999);
    ev.set_regularization_policy(RegularizationPolicyKind::None, 0.0);
    const auto r0 = ev.solve_step(0.0);
    check(r0.converged, "initial solve converges");

    ev.enable_arc_length(true);
    check(ev.arc_length_active(), "arc-length mode can be enabled");

    SectionKinematics kin_B = coord.sub_models()[0].kin_B;
    kin_B.u_local[0] = 5.0e-4;
    ev.update_kinematics(coord.sub_models()[0].kin_A, kin_B);

    const auto r1 = ev.solve_step(0.02);
    check(r1.converged, "adaptive sub-stepping converges for a larger increment");
    check(r1.max_displacement > r0.max_displacement,
          "displacement grows after the larger increment");

    kin_B.u_local[0] = 8.0e-4;
    ev.update_kinematics(coord.sub_models()[0].kin_A, kin_B);

    const auto r2 = ev.solve_step(0.04);
    check(r2.converged, "a second adaptive increment also converges");
    check(r2.max_displacement > r1.max_displacement,
          "displacement continues to grow monotonically");
}

void test_adaptive_substepping_accepts_exact_budget_completion()
{
    std::cout << "\n== Adaptive sub-stepping budget completion ==\n";

    const double W = 0.20;
    const double H = 0.20;

    auto ek = make_ek(
        0, Eigen::Vector3d{1.0e-4, 0.0, 0.0}, Eigen::Vector3d::Zero());

    MultiscaleCoordinator coord;
    coord.add_critical_element(ElementKinematics{ek});
    coord.build_sub_models(SubModelSpec{W, H, 2, 2, 4});

    NonlinearSubModelEvolver ev(coord.sub_models()[0], 30.0, ".", 9999);
    ev.set_regularization_policy(RegularizationPolicyKind::None, 0.0);
    const auto r0 = ev.solve_step(0.0);
    check(r0.converged, "initial solve converges");

    ev.enable_arc_length(true);
    ev.set_adaptive_substepping_limits(2, 10);

    SectionKinematics kin_B = coord.sub_models()[0].kin_B;
    kin_B.u_local[0] = 5.0e-4;
    ev.update_kinematics(coord.sub_models()[0].kin_A, kin_B);

    const auto r1 = ev.solve_step(0.02);
    check(r1.converged,
          "adaptive solve that reaches the target at the substep budget stays converged");
    check(r1.used_arc_length,
          "budget-completion regression exercises the adaptive path");
    check(r1.adaptive_substeps == 2,
          "the target is reached exactly at the configured adaptive substep budget");
    check(std::abs(r1.achieved_fraction - 1.0) < 1.0e-12,
          "achieved fraction reports full completion");
}

void test_mode_specific_homogenized_responses()
{
    std::cout << "\n== Mode-specific homogenized responses ==\n";

    struct LoadCase {
        const char* name;
        Eigen::Vector3d u_B;
        Eigen::Vector3d theta_B;
        int dominant_force;
        int tangent_diag;
        double dominance_ratio;
    };

    const double W = 0.20;
    const double H = 0.20;
    const std::array<LoadCase, 6> cases{{
        {"axial",   Eigen::Vector3d{1.0e-4, 0.0, 0.0}, Eigen::Vector3d::Zero(), 0, 0, 2.0},
        {"bend_y",  Eigen::Vector3d::Zero(),            Eigen::Vector3d{0.0, 1.0e-4, 0.0}, 1, 1, 1.05},
        {"bend_z",  Eigen::Vector3d::Zero(),            Eigen::Vector3d{0.0, 0.0, 1.0e-4}, 2, 2, 1.05},
        {"shear_y", Eigen::Vector3d{0.0, 1.0e-4, 0.0},  Eigen::Vector3d::Zero(), 3, 3, 2.0},
        {"shear_z", Eigen::Vector3d{0.0, 0.0, 1.0e-4},  Eigen::Vector3d::Zero(), 4, 4, 2.0},
        {"torsion", Eigen::Vector3d::Zero(),            Eigen::Vector3d{1.0e-4, 0.0, 0.0}, 5, 5, 2.0},
    }};

    for (const auto& load_case : cases) {
        std::cout << "  case: " << load_case.name << "\n";
        const auto solved = solve_case(
            load_case.u_B, load_case.theta_B, W, H, 1.0e-6);

        check(solved.result.converged, "sub-model converges for the pure load case");
        check(solved.response.status != ResponseStatus::SolveFailed,
              "homogenized response is available");
        check(solved.response.tangent_scheme
                  == TangentLinearizationScheme::LinearizedCondensation,
              "elastic reference cases use the condensed linearized tangent");
        check(std::all_of(solved.response.perturbation_sizes.begin(),
                          solved.response.perturbation_sizes.end(),
                          [](double h) { return h == 0.0; }),
              "condensed linearisation leaves perturbation sizes at zero");
        check(std::all_of(solved.response.tangent_column_valid.begin(),
                          solved.response.tangent_column_valid.end(),
                          [](bool valid) { return valid; }),
              "all tangent columns are valid in the elastic reference cases");
        check(std::all_of(solved.response.tangent_column_central.begin(),
                          solved.response.tangent_column_central.end(),
                          [](bool central) { return central; }),
              "condensed tangent marks every generalized direction as directly resolved");
        check(solved.response.failed_perturbations == 0,
              "condensed linearisation requires no perturbation solves");
        check(solved.response.forces_consistent_with_tangent,
              "elastic reference cases expose a force/tangent pair flagged as consistent");
        check(solved.response.tangent(load_case.tangent_diag, load_case.tangent_diag) > 0.0,
              "the tangent diagonal of the excited mode stays positive");

        const double dominant = std::abs(
            solved.response.forces[load_case.dominant_force]);
        double runner_up = 0.0;
        for (int i = 0; i < 6; ++i) {
            if (i == load_case.dominant_force) {
                continue;
            }
            runner_up = std::max(runner_up, std::abs(solved.response.forces[i]));
        }

        check(dominant > 0.0, "the expected resultant is non-zero");
        if (load_case.dominant_force == 1 || load_case.dominant_force == 2) {
            const int cross_bending =
                (load_case.dominant_force == 1) ? 2 : 1;
            check(dominant > std::abs(solved.response.forces[cross_bending]),
                  "the expected bending moment exceeds the cross-bending coupling");
        } else {
            check(dominant > load_case.dominance_ratio * runner_up,
                  "the expected resultant dominates the force vector");
        }
    }
}

void test_linearized_consistency_energy_and_operator_comparison()
{
    std::cout << "\n== Linearized consistency and energy ==\n";

    const double W = 0.20;
    const double H = 0.20;
    const double eps0 = 1.0e-4;
    const double eps1 = 1.2e-4;

    const auto base = solve_case(
        Eigen::Vector3d{eps0, 0.0, 0.0}, Eigen::Vector3d::Zero(), W, H, 1.0e-6);
    const auto perturbed = solve_case(
        Eigen::Vector3d{eps1, 0.0, 0.0}, Eigen::Vector3d::Zero(), W, H, 1.0e-6);

    check(base.result.converged && perturbed.result.converged,
          "base and perturbed axial reference cases converge");

    Eigen::Vector<double, 6> delta_e = Eigen::Vector<double, 6>::Zero();
    delta_e[0] = eps1 - eps0;

    const Eigen::Vector<double, 6> delta_s_actual =
        perturbed.response.forces - base.response.forces;
    const Eigen::Vector<double, 6> delta_s_pred =
        base.response.tangent * delta_e;

    const double linearization_error =
        relative_norm(delta_s_actual, delta_s_pred);
    check(linearization_error < 0.20,
          "boundary-reaction tangent predicts the axial force increment");

    const double operator_gap =
        relative_norm(base.response.forces, base.volume_average);
    check(operator_gap < 0.10,
          "boundary reactions and volume averaging stay close in the elastic axial case");

    const double macro_power = base.response.forces[0] * eps0;
    const double micro_power =
        base.result.avg_stress[0] * base.result.avg_strain[0] * W * H;
    const double power_gap = std::abs(macro_power - micro_power)
        / std::max({1.0, std::abs(macro_power), std::abs(micro_power)});

    std::cout << std::setprecision(6)
              << "  linearization_error = " << linearization_error << "\n"
              << "  operator_gap        = " << operator_gap << "\n"
              << "  power_gap           = " << power_gap << "\n";

    check(power_gap < 0.15,
          "macro axial power and micro average power remain consistent");
}

void test_crack_summary_keeps_missing_damage_scalar_honest()
{
    std::cout << "\n== Crack summary semantics ==\n";

    const double W = 0.20;
    const double H = 0.20;
    auto ek = make_ek(
        0, Eigen::Vector3d{8.0e-4, 0.0, 0.0}, Eigen::Vector3d::Zero());

    MultiscaleCoordinator coord;
    coord.add_critical_element(ElementKinematics{ek});
    coord.build_sub_models(SubModelSpec{W, H, 2, 2, 4});

    NonlinearSubModelEvolver ev(coord.sub_models()[0], 30.0, ".", 9999);
    ev.set_regularization_policy(RegularizationPolicyKind::None, 0.0);
    ev.set_min_crack_opening(0.0);

    const auto solve = ev.solve_step(0.0);
    ev.end_of_step(0.0);
    const auto summary = ev.crack_summary();

    check(solve.converged, "cracking reference case converges");
    check(!summary.damage_scalar_available,
          "crack summary does not invent a scalar damage metric for Ko-Bathe 3D");
    check(std::isnan(summary.max_damage_scalar),
          "missing scalar damage is encoded as NaN rather than zero");
    if (summary.total_cracks > 0) {
        check(summary.fracture_history_available,
              "when cracks are recorded, crack summary exposes fracture-history invariants");
        check(summary.max_tau_o_max >= 0.0,
              "fracture-history shear invariant remains non-negative");
    } else {
        check(!summary.fracture_history_available,
              "without recorded cracks, fracture-history aggregates stay absent");
    }
}

void test_condensed_operator_matches_forced_fd()
{
    std::cout << "\n== Condensed operator versus forced FD ==\n";

    const double W = 0.20;
    const double H = 0.20;
    const auto ek = make_ek(
        0, Eigen::Vector3d{1.0e-4, 2.5e-5, -1.5e-5},
        Eigen::Vector3d{2.0e-5, 1.0e-5, -1.5e-5});

    MultiscaleCoordinator coord;
    coord.add_critical_element(ElementKinematics{ek});
    coord.build_sub_models(SubModelSpec{W, H, 2, 2, 4});

    NonlinearSubModelEvolver ev(coord.sub_models()[0], 30.0, ".", 9999);
    ev.set_regularization_policy(RegularizationPolicyKind::None, 0.0);

    const auto solve = ev.solve_step(0.0);
    check(solve.converged,
          "mixed elastic reference case converges before operator comparison");

    const auto condensed = ev.section_response(W, H, 1.0e-6);
    check(condensed.tangent_scheme
              == TangentLinearizationScheme::LinearizedCondensation,
          "production path uses linearized condensation in the comparison case");
    check(condensed.tangent_mode_requested
              == TangentComputationMode::PreferLinearizedCondensation,
          "production path reports the preferred tangent computation mode");
    check(condensed.condensed_tangent_status
              == CondensedTangentStatus::Success,
          "production path reports a successful condensed operator");

    ev.set_tangent_computation_mode(
        TangentComputationMode::ForceAdaptiveFiniteDifference);
    const auto forced_fd = ev.section_response(W, H, 1.0e-6);

    check(forced_fd.tangent_scheme
              == TangentLinearizationScheme::AdaptiveFiniteDifference,
          "validation path can force the adaptive finite-difference tangent");
    check(forced_fd.tangent_mode_requested
              == TangentComputationMode::ForceAdaptiveFiniteDifference,
          "forced validation path reports the requested tangent mode");
    check(forced_fd.condensed_tangent_status
              == CondensedTangentStatus::ForcedAdaptiveFiniteDifference,
          "forced validation path records that condensation was intentionally skipped");
    check(forced_fd.failed_perturbations == 0,
          "forced FD comparison case resolves all perturbations");

    const double tangent_gap =
        (condensed.tangent - forced_fd.tangent).norm()
        / std::max({1.0, condensed.tangent.norm(), forced_fd.tangent.norm()});
    const double force_gap =
        relative_norm(condensed.forces, forced_fd.forces);

    std::cout << std::setprecision(6)
              << "  tangent_gap        = " << tangent_gap << "\n"
              << "  force_gap          = " << force_gap << "\n";

    check(tangent_gap < 6.0e-2,
          "condensed tangent stays close to forced adaptive FD in the elastic case");
    check(force_gap < 1.0e-12,
          "condensed and forced FD paths read the same boundary forces");
}

void test_condensed_operator_reuses_symbolic_pattern()
{
    std::cout << "\n== Condensed operator pattern reuse ==\n";

    const double W = 0.20;
    const double H = 0.20;
    const auto ek = make_ek(
        0, Eigen::Vector3d{8.0e-5, -1.5e-5, 1.0e-5},
        Eigen::Vector3d{1.0e-5, -2.0e-5, 1.5e-5});

    MultiscaleCoordinator coord;
    coord.add_critical_element(ElementKinematics{ek});
    coord.build_sub_models(SubModelSpec{W, H, 2, 2, 4});

    NonlinearSubModelEvolver ev(coord.sub_models()[0], 30.0, ".", 9999);
    ev.set_regularization_policy(RegularizationPolicyKind::None, 0.0);

    const auto solve = ev.solve_step(0.0);
    check(solve.converged,
          "mixed elastic reference case converges before pattern reuse audit");

    const auto first = ev.section_response(W, H, 1.0e-6);
    const auto second = ev.section_response(W, H, 1.0e-6);

    check(first.condensed_tangent_status == CondensedTangentStatus::Success,
          "first condensed response is available");
    check(second.condensed_tangent_status == CondensedTangentStatus::Success,
          "second condensed response is available");
    check(!first.condensed_pattern_reused,
          "first condensed call performs the initial symbolic analysis");
    check(second.condensed_pattern_reused,
          "second condensed call reuses the symbolic factorization pattern");
    check(first.condensed_symbolic_factorizations == 1,
          "first condensed call reports one symbolic factorization");
    check(second.condensed_symbolic_factorizations == 1,
          "second condensed call does not trigger a second symbolic factorization");
    check((first.tangent - second.tangent).norm()
              / std::max({1.0, first.tangent.norm(), second.tangent.norm()})
          < 1.0e-14,
          "pattern reuse leaves the condensed tangent unchanged");
}

}  // namespace

int main(int argc, char** argv)
{
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    PetscOptionsSetValue(nullptr, "-snes_monitor_cancel", "");
    PetscOptionsSetValue(nullptr, "-ksp_monitor_cancel", "");

    std::cout << std::string(72, '=') << "\n"
              << "  Advanced multiscale evolver verification\n"
              << std::string(72, '=') << "\n";

    test_reinforced_evolver();
    test_adaptive_substepping();
    test_adaptive_substepping_accepts_exact_budget_completion();
    test_mode_specific_homogenized_responses();
    test_linearized_consistency_energy_and_operator_comparison();
    test_crack_summary_keeps_missing_damage_scalar_honest();
    test_condensed_operator_matches_forced_fd();
    test_condensed_operator_reuses_symbolic_pattern();

    std::cout << "\n" << std::string(72, '=') << "\n"
              << "  Summary: " << g_pass << " passed, " << g_fail
              << " failed\n"
              << std::string(72, '=') << "\n";

    PetscFinalize();
    return (g_fail == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
