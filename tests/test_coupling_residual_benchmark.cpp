#include "header_files.hh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

#include "src/reconstruction/SectionOperatorValidationNorm.hh"

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

    ek.kin_A.centroid = Eigen::Vector3d{0.0, 0.0, 0.0};
    ek.kin_A.R = Eigen::Matrix3d::Identity();
    ek.kin_A.u_local = Eigen::Vector3d::Zero();
    ek.kin_A.theta_local = Eigen::Vector3d::Zero();
    ek.kin_A.E = 200.0;
    ek.kin_A.G = 80.0;
    ek.kin_A.nu = 0.25;

    ek.kin_B = ek.kin_A;
    ek.kin_B.centroid = Eigen::Vector3d{1.0, 0.0, 0.0};
    ek.kin_B.u_local = u_B;
    ek.kin_B.theta_local = theta_B;
    return ek;
}

struct BenchmarkCase {
    std::string_view name;
    Eigen::Vector3d u_B;
    Eigen::Vector3d theta_B;
};

struct BenchmarkResult {
    std::string_view name;
    double force_gap_f{0.0};
    double force_gap_sw{0.0};
    double force_gap_de{0.0};
    double tangent_gap_f{0.0};
    double tangent_gap_sw{0.0};
    double tangent_gap_de{0.0};
    bool accepted_f{false};
    bool accepted_sw{false};
    bool accepted_de{false};
    double iteration_cost_s{0.0};
    double projected_f_time_s{0.0};
    double projected_sw_time_s{0.0};
    double projected_de_time_s{0.0};
};

template <typename Fn>
[[nodiscard]] double best_of_seconds(int repeats, Fn&& fn)
{
    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i < repeats; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const auto t1 = std::chrono::steady_clock::now();
        const std::chrono::duration<double> dt = t1 - t0;
        best = std::min(best, dt.count());
    }
    return best;
}

[[nodiscard]] Eigen::Vector<double, 6>
reference_generalized_state(const BenchmarkCase& load_case)
{
    Eigen::Vector<double, 6> e_ref = Eigen::Vector<double, 6>::Zero();
    e_ref << load_case.u_B[0],
        load_case.theta_B[1],
        load_case.theta_B[2],
        load_case.u_B[1],
        load_case.u_B[2],
        load_case.theta_B[0];
    return e_ref;
}

[[nodiscard]] BenchmarkResult run_case(const BenchmarkCase& load_case,
                                       double width,
                                       double height,
                                       double force_tol,
                                       double tangent_tol)
{
    MultiscaleCoordinator coord;
    coord.add_critical_element(make_ek(
        0, load_case.u_B, load_case.theta_B));
    coord.build_sub_models(SubModelSpec{width, height, 2, 2, 4});

    NonlinearSubModelEvolver ev(coord.sub_models()[0], 30.0, ".", 9999);
    ev.set_regularization_policy(RegularizationPolicyKind::None, 0.0);

    const auto solve = ev.solve_step(0.0);
    if (!solve.converged) {
        return BenchmarkResult{load_case.name};
    }

    ev.set_tangent_computation_mode(
        TangentComputationMode::PreferLinearizedCondensation);
    const auto current = ev.section_response(width, height, 1.0e-6);

    ev.set_tangent_computation_mode(
        TangentComputationMode::ForceAdaptiveFiniteDifference);
    const auto previous = ev.section_response(width, height, 1.0e-6);

    const auto e_ref = reference_generalized_state(load_case);
    const auto scales_f = make_section_operator_validation_scales(
        TangentValidationNormKind::RelativeFrobenius,
        width,
        height,
        e_ref);
    const auto scales_sw = make_section_operator_validation_scales(
        TangentValidationNormKind::StateWeightedFrobenius,
        width,
        height,
        e_ref,
        current.forces);
    const auto scales_de = make_section_operator_validation_scales(
        TangentValidationNormKind::DualEnergyScaled,
        width,
        height,
        e_ref,
        current.forces);

    const auto force_metrics_f = compute_section_vector_validation_metrics(
        current.forces, previous.forces, scales_f);
    const auto force_metrics_sw = compute_section_vector_validation_metrics(
        current.forces, previous.forces, scales_sw);
    const auto force_metrics_de = compute_section_vector_validation_metrics(
        current.forces, previous.forces, scales_de);
    const auto tangent_metrics_f = compute_section_operator_validation_metrics(
        current.tangent, previous.tangent, scales_f);
    const auto tangent_metrics_sw = compute_section_operator_validation_metrics(
        current.tangent, previous.tangent, scales_sw);
    const auto tangent_metrics_de = compute_section_operator_validation_metrics(
        current.tangent, previous.tangent, scales_de);

    ev.set_tangent_computation_mode(
        TangentComputationMode::PreferLinearizedCondensation);
    const double iteration_cost = best_of_seconds(3, [&]() {
        const auto response = ev.section_response(width, height, 1.0e-6);
        (void)response;
    });

    BenchmarkResult result;
    result.name = load_case.name;
    result.force_gap_f = force_metrics_f.relative_gap;
    result.force_gap_sw = force_metrics_sw.relative_gap;
    result.force_gap_de = force_metrics_de.relative_gap;
    result.tangent_gap_f = tangent_metrics_f.relative_gap;
    result.tangent_gap_sw = tangent_metrics_sw.relative_gap;
    result.tangent_gap_de = tangent_metrics_de.relative_gap;
    result.accepted_f =
        result.force_gap_f <= force_tol && result.tangent_gap_f <= tangent_tol;
    result.accepted_sw =
        result.force_gap_sw <= force_tol
        && result.tangent_gap_sw <= tangent_tol;
    result.accepted_de =
        result.force_gap_de <= force_tol
        && result.tangent_gap_de <= tangent_tol;
    result.iteration_cost_s = iteration_cost;
    result.projected_f_time_s =
        result.accepted_f ? iteration_cost : 2.0 * iteration_cost;
    result.projected_sw_time_s =
        result.accepted_sw ? iteration_cost : 2.0 * iteration_cost;
    result.projected_de_time_s =
        result.accepted_de ? iteration_cost : 2.0 * iteration_cost;
    return result;
}

void test_coupling_residual_benchmark()
{
    std::cout << "\n== Coupling residual norm benchmark ==\n";

    const double W = 0.20;
    const double H = 0.20;
    const double force_tol = 1.0e-10;
    const double tangent_tol = 4.5e-2;
    const std::array<BenchmarkCase, 4> cases{{
        {"mixed_a",
         Eigen::Vector3d{1.0e-4, 2.5e-5, -1.5e-5},
         Eigen::Vector3d{2.0e-5, 1.0e-5, -1.5e-5}},
        {"mixed_b",
         Eigen::Vector3d{8.0e-5, -4.0e-5, 2.0e-5},
         Eigen::Vector3d{2.5e-5, 2.0e-5, -1.0e-5}},
        {"mixed_c",
         Eigen::Vector3d{6.0e-5, 3.5e-5, -3.0e-5},
         Eigen::Vector3d{-2.5e-5, 1.5e-5, 2.0e-5}},
        {"mixed_d",
         Eigen::Vector3d{9.0e-5, -2.5e-5, 4.0e-5},
         Eigen::Vector3d{1.0e-5, -2.0e-5, 2.5e-5}},
    }};

    int accepts_f = 0;
    int accepts_sw = 0;
    int accepts_de = 0;
    double projected_f_time = 0.0;
    double projected_sw_time = 0.0;
    double projected_de_time = 0.0;

    std::cout << std::left
              << std::setw(10) << "case"
              << std::setw(14) << "rf_F"
              << std::setw(14) << "rf_SW"
              << std::setw(14) << "rf_DE"
              << std::setw(14) << "rD_F"
              << std::setw(14) << "rD_SW"
              << std::setw(14) << "rD_DE"
              << std::setw(10) << "acc_F"
              << std::setw(10) << "acc_SW"
              << std::setw(10) << "acc_DE"
              << std::setw(14) << "t_iter[s]"
              << std::setw(16) << "proj_F[s]"
              << std::setw(16) << "proj_SW[s]"
              << std::setw(16) << "proj_DE[s]"
              << "\n";

    for (const auto& load_case : cases) {
        const auto result = run_case(
            load_case, W, H, force_tol, tangent_tol);
        accepts_f += result.accepted_f ? 1 : 0;
        accepts_sw += result.accepted_sw ? 1 : 0;
        accepts_de += result.accepted_de ? 1 : 0;
        projected_f_time += result.projected_f_time_s;
        projected_sw_time += result.projected_sw_time_s;
        projected_de_time += result.projected_de_time_s;

        std::cout << std::left
                  << std::setw(10) << result.name
                  << std::setw(14) << std::setprecision(6)
                  << result.force_gap_f
                  << std::setw(14) << result.force_gap_sw
                  << std::setw(14) << result.force_gap_de
                  << std::setw(14) << result.tangent_gap_f
                  << std::setw(14) << result.tangent_gap_sw
                  << std::setw(14) << result.tangent_gap_de
                  << std::setw(10) << (result.accepted_f ? "yes" : "no")
                  << std::setw(10) << (result.accepted_sw ? "yes" : "no")
                  << std::setw(10) << (result.accepted_de ? "yes" : "no")
                  << std::setw(14) << result.iteration_cost_s
                  << std::setw(16) << result.projected_f_time_s
                  << std::setw(16) << result.projected_sw_time_s
                  << std::setw(16) << result.projected_de_time_s
                  << "\n";
    }

    const double projected_speedup =
        projected_sw_time > 0.0 ? projected_f_time / projected_sw_time : 1.0;

    std::cout << "  force_tol = " << force_tol << "\n"
              << "  tangent_tol = " << tangent_tol << "\n"
              << "  accepts_f = " << accepts_f << "\n"
              << "  accepts_sw = " << accepts_sw << "\n"
              << "  accepts_de = " << accepts_de << "\n"
              << "  projected_f_time = " << projected_f_time << " s\n"
              << "  projected_sw_time = " << projected_sw_time << " s\n"
              << "  projected_de_time = " << projected_de_time << " s\n"
              << "  projected_speedup = " << projected_speedup << "x\n";

    check(accepts_sw >= accepts_f,
          "state-weighted residual norm does not reduce accepted FE2 surrogate states");
    check(projected_sw_time <= projected_f_time + 1.0e-12,
          "state-weighted residual norm does not worsen projected outer-iteration cost");
    check(accepts_de >= accepts_f,
          "dual-energy residual norm does not reduce accepted FE2 surrogate states relative to legacy Frobenius");
    check(projected_de_time <= projected_f_time + 1.0e-12,
          "dual-energy residual norm does not worsen projected outer-iteration cost relative to legacy Frobenius");
    check(projected_speedup >= 1.0,
          "residual benchmark reports a non-degrading projected speedup");
}

}  // namespace

int main(int argc, char** argv)
{
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    PetscOptionsSetValue(nullptr, "-snes_monitor_cancel", "");
    PetscOptionsSetValue(nullptr, "-ksp_monitor_cancel", "");

    std::cout << std::string(72, '=') << "\n"
              << "  Coupling residual benchmark\n"
              << std::string(72, '=') << "\n";

    test_coupling_residual_benchmark();

    std::cout << "\n" << std::string(72, '=') << "\n"
              << "  Summary: " << g_pass << " passed, " << g_fail
              << " failed\n"
              << std::string(72, '=') << "\n";

    PetscFinalize();
    return (g_fail == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
