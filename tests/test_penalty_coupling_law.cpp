#include <cmath>
#include <cstdlib>
#include <iostream>

#include "src/analysis/PenaltyCoupling.hh"

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool condition, const char* message)
{
    if (condition) {
        ++g_pass;
        std::cout << "  [PASS] " << message << "\n";
    } else {
        ++g_fail;
        std::cout << "  [FAIL] " << message << "\n";
    }
}

bool near(double a, double b, double tol)
{
    return std::abs(a - b) <= tol;
}

void test_linear_default()
{
    fall_n::PenaltyCouplingLaw law;
    const double alpha = 280000.0;
    const double gap = 1.2e-4;
    const auto response = law.evaluate(gap, alpha);

    check(near(response.force, alpha * gap, 1.0e-12),
          "default penalty force remains linear");
    check(near(response.tangent, alpha, 1.0e-12),
          "default penalty tangent remains linear");
}

void test_bond_slip_regularization()
{
    fall_n::PenaltyCouplingLaw law;
    law.bond_slip_regularization = true;
    law.slip_reference_m = 5.0e-4;
    law.residual_stiffness_ratio = 0.2;

    const double alpha = 280000.0;
    const double small_gap = 1.0e-8;
    const double large_gap = 4.0e-3;

    const auto origin = law.evaluate(0.0, alpha);
    const auto positive = law.evaluate(large_gap, alpha);
    const auto negative = law.evaluate(-large_gap, alpha);
    const auto small = law.evaluate(small_gap, alpha);

    check(near(origin.force, 0.0, 1.0e-14),
          "bond-slip force is zero at zero slip");
    check(near(origin.tangent, alpha, 1.0e-9),
          "bond-slip tangent starts at the initial penalty");
    check(positive.tangent < alpha && positive.tangent > 0.0,
          "bond-slip tangent softens but stays positive");
    check(near(positive.force, -negative.force, 1.0e-9),
          "bond-slip force is odd in the slip gap");
    check(std::abs(small.force / small_gap - alpha) < 5.0,
          "small-slip secant stiffness stays close to initial penalty");

    const double h = 1.0e-7;
    const auto plus = law.evaluate(large_gap + h, alpha);
    const auto minus = law.evaluate(large_gap - h, alpha);
    const double fd_tangent = (plus.force - minus.force) / (2.0 * h);

    check(std::abs(fd_tangent - positive.tangent) < 1.0e-1,
          "bond-slip analytic tangent matches finite difference");
}

void test_adaptive_bond_slip_regularization()
{
    fall_n::PenaltyCouplingLaw law;
    law.bond_slip_regularization = true;
    law.slip_reference_m = 5.0e-4;
    law.residual_stiffness_ratio = 0.2;
    law.adaptive_slip_regularization = true;
    law.adaptive_slip_reference_max_factor = 4.0;
    law.adaptive_residual_stiffness_ratio_floor = 0.05;

    const double alpha = 280000.0;
    const double small_gap = 1.0e-8;
    const double large_gap = 4.0e-3;

    const auto small_state = law.effective_state(small_gap);
    const auto large_state = law.effective_state(large_gap);
    const auto large = law.evaluate(large_gap, alpha);

    check(near(small_state.slip_reference_m, law.slip_reference_m, 1.0e-8),
          "adaptive bond-slip keeps the initial reference at tiny slip");
    check(large_state.slip_reference_m > law.slip_reference_m,
          "adaptive bond-slip increases the effective slip reference");
    check(large_state.residual_stiffness_ratio <
              law.residual_stiffness_ratio,
          "adaptive bond-slip degrades residual bond stiffness");
    check(large.tangent > 0.0 && large.tangent < alpha,
          "adaptive bond-slip tangent remains positive and softened");

    const double h = 1.0e-7;
    const auto plus = law.evaluate(large_gap + h, alpha);
    const auto minus = law.evaluate(large_gap - h, alpha);
    const double fd_tangent = (plus.force - minus.force) / (2.0 * h);

    check(std::abs(fd_tangent - large.tangent) < 1.0e-1,
          "adaptive bond-slip analytic tangent matches finite difference");
}

}  // namespace

int main()
{
    std::cout << "Penalty coupling law tests\n";

    test_linear_default();
    test_bond_slip_regularization();
    test_adaptive_bond_slip_regularization();

    std::cout << "Summary: " << g_pass << " passed, " << g_fail
              << " failed\n";
    return (g_fail == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
