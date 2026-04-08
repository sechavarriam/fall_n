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
    double condensed_time_s{0.0};
    double fd_time_s{0.0};
    double frobenius_gap{0.0};
    double state_weighted_gap{0.0};
    bool frobenius_accepted{false};
    bool state_weighted_accepted{false};
    double projected_frobenius_time_s{0.0};
    double projected_state_weighted_time_s{0.0};
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

[[nodiscard]] bool accepted_condensed_response(
    const SectionHomogenizedResponse& response)
{
    return response.tangent_scheme
               == TangentLinearizationScheme::LinearizedCondensation
        && response.condensed_tangent_status
               == CondensedTangentStatus::Success
        && response.tangent_validation_status
               == TangentValidationStatus::Accepted;
}

[[nodiscard]] BenchmarkResult run_case(const BenchmarkCase& load_case,
                                       double width,
                                       double height,
                                       double tolerance)
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
    const double condensed_time = best_of_seconds(3, [&]() {
        const auto response = ev.section_response(width, height, 1.0e-6);
        (void)response;
    });

    ev.set_tangent_computation_mode(
        TangentComputationMode::ForceAdaptiveFiniteDifference);
    const double fd_time = best_of_seconds(2, [&]() {
        const auto response = ev.section_response(width, height, 1.0e-6);
        (void)response;
    });

    ev.set_tangent_computation_mode(
        TangentComputationMode::
            ValidateCondensationAgainstAdaptiveFiniteDifference);
    ev.set_tangent_validation_relative_tolerance(tolerance);

    ev.set_tangent_validation_norm(
        TangentValidationNormKind::RelativeFrobenius);
    const auto frobenius = ev.section_response(width, height, 1.0e-6);

    ev.set_tangent_validation_norm(
        TangentValidationNormKind::StateWeightedFrobenius);
    const auto state_weighted = ev.section_response(width, height, 1.0e-6);

    BenchmarkResult result;
    result.name = load_case.name;
    result.condensed_time_s = condensed_time;
    result.fd_time_s = fd_time;
    result.frobenius_gap = frobenius.tangent_validation_relative_gap;
    result.state_weighted_gap = state_weighted.tangent_validation_relative_gap;
    result.frobenius_accepted = accepted_condensed_response(frobenius);
    result.state_weighted_accepted = accepted_condensed_response(state_weighted);
    result.projected_frobenius_time_s =
        result.frobenius_accepted ? condensed_time : fd_time;
    result.projected_state_weighted_time_s =
        result.state_weighted_accepted ? condensed_time : fd_time;
    return result;
}

void test_validation_norm_benchmark()
{
    std::cout << "\n== Tangent validation norm benchmark ==\n";

    const double W = 0.20;
    const double H = 0.20;
    const double tolerance = 4.5e-2;
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

    std::vector<BenchmarkResult> results;
    results.reserve(cases.size());

    int frobenius_accepts = 0;
    int state_weighted_accepts = 0;
    double projected_frobenius_time = 0.0;
    double projected_state_weighted_time = 0.0;

    std::cout << std::left
              << std::setw(10) << "case"
              << std::setw(14) << "gap_F"
              << std::setw(14) << "gap_SW"
              << std::setw(10) << "acc_F"
              << std::setw(10) << "acc_SW"
              << std::setw(14) << "t_cond[s]"
              << std::setw(14) << "t_fd[s]"
              << std::setw(16) << "proj_F[s]"
              << std::setw(16) << "proj_SW[s]"
              << "\n";

    for (const auto& load_case : cases) {
        const auto result = run_case(load_case, W, H, tolerance);
        results.push_back(result);
        frobenius_accepts += result.frobenius_accepted ? 1 : 0;
        state_weighted_accepts += result.state_weighted_accepted ? 1 : 0;
        projected_frobenius_time += result.projected_frobenius_time_s;
        projected_state_weighted_time += result.projected_state_weighted_time_s;

        std::cout << std::left
                  << std::setw(10) << result.name
                  << std::setw(14) << std::setprecision(6)
                  << result.frobenius_gap
                  << std::setw(14) << result.state_weighted_gap
                  << std::setw(10) << (result.frobenius_accepted ? "yes" : "no")
                  << std::setw(10) << (result.state_weighted_accepted ? "yes" : "no")
                  << std::setw(14) << result.condensed_time_s
                  << std::setw(14) << result.fd_time_s
                  << std::setw(16) << result.projected_frobenius_time_s
                  << std::setw(16) << result.projected_state_weighted_time_s
                  << "\n";
    }

    const double projected_speedup =
        projected_state_weighted_time > 0.0
            ? projected_frobenius_time / projected_state_weighted_time
            : 1.0;

    std::cout << "  tolerance = " << tolerance << "\n"
              << "  frobenius_accepts = " << frobenius_accepts << "\n"
              << "  state_weighted_accepts = " << state_weighted_accepts << "\n"
              << "  projected_frobenius_time = "
              << projected_frobenius_time << " s\n"
              << "  projected_state_weighted_time = "
              << projected_state_weighted_time << " s\n"
              << "  projected_speedup = "
              << projected_speedup << "x\n";

    check(state_weighted_accepts >= frobenius_accepts,
          "state-weighted norm does not reduce the number of accepted condensed operators");
    check(projected_state_weighted_time
              <= projected_frobenius_time + 1.0e-12,
          "state-weighted norm does not worsen the projected production tangent cost");
    check(projected_speedup >= 1.0,
          "state-weighted norm benchmark reports a non-degrading projected speedup");
}

}  // namespace

int main(int argc, char** argv)
{
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    PetscOptionsSetValue(nullptr, "-snes_monitor_cancel", "");
    PetscOptionsSetValue(nullptr, "-ksp_monitor_cancel", "");

    std::cout << std::string(72, '=') << "\n"
              << "  Tangent validation benchmark\n"
              << std::string(72, '=') << "\n";

    test_validation_norm_benchmark();

    std::cout << "\n" << std::string(72, '=') << "\n"
              << "  Summary: " << g_pass << " passed, " << g_fail
              << " failed\n"
              << std::string(72, '=') << "\n";

    PetscFinalize();
    return (g_fail == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
