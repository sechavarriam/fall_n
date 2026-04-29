#include <cmath>
#include <iostream>

#include <Eigen/Dense>

#include "src/analysis/BorderedMixedControlNewton.hh"

namespace {

using namespace fall_n;

int g_pass = 0;
int g_fail = 0;

#define CHECK_TRUE(cond, msg)                                                   \
    do {                                                                        \
        if (cond) {                                                             \
            std::cout << "  [PASS] " << msg << "\n";                           \
            ++g_pass;                                                           \
        } else {                                                                \
            std::cout << "  [FAIL] " << msg << "\n";                           \
            ++g_fail;                                                           \
        }                                                                       \
    } while (0)

void test_linear_displacement_control_equilibrium()
{
    auto evaluator = [](const BorderedMixedControlState& state) {
        const double u = state.unknowns[0];
        const double lambda = state.load_parameter;

        BorderedMixedControlEvaluation e;
        e.residual = Eigen::VectorXd::Constant(1, 10.0 * u - lambda);
        e.tangent = Eigen::MatrixXd::Constant(1, 1, 10.0);
        e.load_column = Eigen::VectorXd::Constant(1, -1.0);
        e.constraint = u - 2.0;
        e.constraint_gradient = Eigen::VectorXd::Constant(1, 1.0);
        e.constraint_load_derivative = 0.0;
        return e;
    };

    BorderedMixedControlState state;
    state.unknowns = Eigen::VectorXd::Zero(1);
    state.load_parameter = 0.0;

    const auto result = solve_bordered_mixed_control_newton(
        std::move(state),
        evaluator,
        BorderedMixedControlNewtonSettings{
            .line_search_enabled = false});

    CHECK_TRUE(result.converged(),
               "bordered Newton converges for linear displacement control");
    CHECK_TRUE(std::abs(result.state.unknowns[0] - 2.0) < 1.0e-12,
               "augmented solve enforces the displacement constraint");
    CHECK_TRUE(std::abs(result.state.load_parameter - 20.0) < 1.0e-12,
               "augmented solve recovers the consistent load parameter");
}

void test_nonlinear_mixed_arc_constraint()
{
    constexpr double radius = 0.25;
    auto evaluator = [](const BorderedMixedControlState& state) {
        const double u = state.unknowns[0];
        const double lambda = state.load_parameter;

        BorderedMixedControlEvaluation e;
        e.residual = Eigen::VectorXd::Constant(1, u * u - lambda);
        e.tangent = Eigen::MatrixXd::Constant(1, 1, 2.0 * u);
        e.load_column = Eigen::VectorXd::Constant(1, -1.0);
        e.constraint =
            (u - 1.0) * (u - 1.0) +
            (lambda - 1.0) * (lambda - 1.0) -
            radius * radius;
        e.constraint_gradient =
            Eigen::VectorXd::Constant(1, 2.0 * (u - 1.0));
        e.constraint_load_derivative = 2.0 * (lambda - 1.0);
        return e;
    };

    BorderedMixedControlState state;
    state.unknowns = Eigen::VectorXd::Constant(1, 1.15);
    state.load_parameter = 1.15;

    const auto result = solve_bordered_mixed_control_newton(
        std::move(state),
        evaluator,
        BorderedMixedControlNewtonSettings{
            .residual_tolerance = 1.0e-12,
            .constraint_tolerance = 1.0e-12,
            .line_search_enabled = true});

    CHECK_TRUE(result.converged(),
               "bordered Newton converges for a nonlinear mixed arc constraint");
    CHECK_TRUE(std::abs(
                   result.state.unknowns[0] * result.state.unknowns[0] -
                   result.state.load_parameter) < 1.0e-10,
               "nonlinear equilibrium residual is closed");
    CHECK_TRUE(std::abs(
                   (result.state.unknowns[0] - 1.0) *
                       (result.state.unknowns[0] - 1.0) +
                   (result.state.load_parameter - 1.0) *
                       (result.state.load_parameter - 1.0) -
                   radius * radius) < 1.0e-10,
               "mixed arc constraint is closed");
}

void test_invalid_dimension_is_rejected()
{
    auto evaluator = [](const BorderedMixedControlState&) {
        BorderedMixedControlEvaluation e;
        e.residual = Eigen::VectorXd::Zero(2);
        e.tangent = Eigen::MatrixXd::Identity(1, 1);
        e.load_column = Eigen::VectorXd::Ones(1);
        e.constraint_gradient = Eigen::VectorXd::Ones(1);
        return e;
    };

    BorderedMixedControlState state;
    state.unknowns = Eigen::VectorXd::Zero(1);

    const auto result = solve_bordered_mixed_control_newton(
        std::move(state),
        evaluator);

    CHECK_TRUE(
        result.status ==
            BorderedMixedControlNewtonStatus::invalid_evaluation,
        "bordered Newton rejects inconsistent callback dimensions");
}

} // namespace

int main()
{
    test_linear_displacement_control_equilibrium();
    test_nonlinear_mixed_arc_constraint();
    test_invalid_dimension_is_rejected();

    std::cout << "\nPassed: " << g_pass << "  Failed: " << g_fail << "\n";
    return g_fail == 0 ? 0 : 1;
}
