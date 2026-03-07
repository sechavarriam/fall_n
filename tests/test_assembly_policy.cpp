// ============================================================================
//  Tests for assembly::DirectAssembly and assembly::CondensedAssembly
// ============================================================================

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include <Eigen/Dense>

#include "src/elements/assembly/AssemblyPolicy.hh"

// ── Helpers ─────────────────────────────────────────────────────────────────

static int  g_pass = 0;
static int  g_fail = 0;

#define ASSERT_NEAR(a, b, tol)                                                 \
    do {                                                                       \
        if (std::abs((a) - (b)) > (tol)) {                                     \
            std::cerr << "  FAIL  " << __FILE__ << ":" << __LINE__             \
                      << "  |" << (a) << " - " << (b) << "| = "               \
                      << std::abs((a) - (b)) << " > " << (tol) << "\n";       \
            ++g_fail;                                                          \
            return;                                                            \
        }                                                                      \
    } while (0)

#define ASSERT_MATRIX_NEAR(A, B, tol)                                          \
    do {                                                                       \
        if ((A).rows() != (B).rows() || (A).cols() != (B).cols()) {            \
            std::cerr << "  FAIL  " << __FILE__ << ":" << __LINE__             \
                      << "  dimension mismatch\n";                             \
            ++g_fail;                                                          \
            return;                                                            \
        }                                                                      \
        for (Eigen::Index _i = 0; _i < (A).rows(); ++_i)                      \
            for (Eigen::Index _j = 0; _j < (A).cols(); ++_j)                  \
                if (std::abs((A)(_i, _j) - (B)(_i, _j)) > (tol)) {            \
                    std::cerr << "  FAIL  " << __FILE__ << ":" << __LINE__     \
                              << "  at (" << _i << "," << _j << "): "          \
                              << (A)(_i, _j) << " vs " << (B)(_i, _j) << "\n";\
                    ++g_fail;                                                  \
                    return;                                                    \
                }                                                              \
    } while (0)

#define ASSERT_VECTOR_NEAR(A, B, tol)                                          \
    do {                                                                       \
        if ((A).size() != (B).size()) {                                        \
            std::cerr << "  FAIL  " << __FILE__ << ":" << __LINE__             \
                      << "  size mismatch: " << (A).size()                     \
                      << " vs " << (B).size() << "\n";                         \
            ++g_fail;                                                          \
            return;                                                            \
        }                                                                      \
        for (Eigen::Index _i = 0; _i < (A).size(); ++_i)                      \
            if (std::abs((A)(_i) - (B)(_i)) > (tol)) {                        \
                std::cerr << "  FAIL  " << __FILE__ << ":" << __LINE__         \
                          << "  at [" << _i << "]: "                           \
                          << (A)(_i) << " vs " << (B)(_i) << "\n";            \
                ++g_fail;                                                      \
                return;                                                        \
            }                                                                  \
    } while (0)

#define ASSERT_TRUE(cond)                                                      \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cerr << "  FAIL  " << __FILE__ << ":" << __LINE__             \
                      << "  " #cond " is false\n";                             \
            ++g_fail;                                                          \
            return;                                                            \
        }                                                                      \
    } while (0)

#define ASSERT_FALSE(cond) ASSERT_TRUE(!(cond))

#define RUN_TEST(fn)                                                           \
    do {                                                                       \
        int _before = g_fail;                                                  \
        fn();                                                                  \
        if (g_fail == _before) { ++g_pass; std::cout << "  PASS  " << #fn << "\n"; } \
        else                   { std::cout << "  FAIL  " << #fn << "\n"; }     \
    } while (0)

// ── Shared test fixture: a 4×4 SPD system with 2 ext + 2 int DOFs ──────────

static auto make_test_system() {
    Eigen::MatrixXd K(4, 4);
    K << 10,  2,  1,  0,
          2,  8,  0,  1,
          1,  0,  5,  1,
          0,  1,  1,  6;

    Eigen::VectorXd f(4);
    f << 5, 3, 1, 2;

    return std::make_pair(K, f);
}

// ============================================================================
//  DirectAssembly tests
// ============================================================================

void test_direct_assembly_compile_time_flag() {
    ASSERT_TRUE(assembly::DirectAssembly::exposes_internal_dofs);
}

void test_direct_assembly_passthrough() {
    auto [K, f] = make_test_system();
    assembly::DirectAssembly policy;

    auto ps = policy.prepare(K, f, 2);

    ASSERT_MATRIX_NEAR(ps.K, K, 1e-15);
    ASSERT_VECTOR_NEAR(ps.f, f, 1e-15);
}

void test_direct_assembly_recover_is_empty() {
    assembly::DirectAssembly policy;
    Eigen::VectorXd u_ext(2);
    u_ext << 1.0, 2.0;

    auto u_int = policy.recover_internal(u_ext);
    ASSERT_TRUE(u_int.size() == 0);
}

// ============================================================================
//  CondensedAssembly tests
// ============================================================================

void test_condensed_assembly_compile_time_flag() {
    ASSERT_FALSE(assembly::CondensedAssembly::exposes_internal_dofs);
}

void test_condensed_assembly_produces_reduced_system() {
    auto [K, f] = make_test_system();
    assembly::CondensedAssembly policy;

    auto ps = policy.prepare(K, f, 2);

    // K̂ should be 2×2 (external DOFs only)
    ASSERT_TRUE(ps.K.rows() == 2);
    ASSERT_TRUE(ps.K.cols() == 2);
    ASSERT_TRUE(ps.f.size() == 2);
}

void test_condensed_assembly_roundtrip() {
    auto [K, f] = make_test_system();
    assembly::CondensedAssembly policy;

    auto ps = policy.prepare(K, f, 2);

    // Solve reduced system
    Eigen::VectorXd u_ext = ps.K.ldlt().solve(ps.f);

    // Recover internal DOFs
    auto u_int = policy.recover_internal(u_ext);

    // Verify against full direct solve
    Eigen::VectorXd u_full = K.ldlt().solve(f);

    constexpr double tol = 1e-12;
    ASSERT_VECTOR_NEAR(u_ext, u_full.head(2), tol);
    ASSERT_VECTOR_NEAR(u_int, u_full.tail(2), tol);

    // Verify residual
    Eigen::VectorXd u_assembled(4);
    u_assembled << u_ext, u_int;
    double residual = (K * u_assembled - f).norm();
    ASSERT_NEAR(residual, 0.0, tol);
}

void test_condensed_assembly_cache_cleared_after_recover() {
    auto [K, f] = make_test_system();
    assembly::CondensedAssembly policy;

    policy.prepare(K, f, 2);
    ASSERT_TRUE(policy.has_cache());

    Eigen::VectorXd u_ext = Eigen::VectorXd::Zero(2);
    policy.recover_internal(u_ext);
    ASSERT_FALSE(policy.has_cache());
}

void test_condensed_vs_direct_solutions_agree() {
    // 6×6 SPD, 4 ext + 2 int
    Eigen::MatrixXd A(6, 6);
    A << 3, 1, 0, 1, 0, 1,
         1, 4, 1, 0, 1, 0,
         0, 1, 5, 1, 0, 1,
         1, 0, 1, 6, 1, 0,
         0, 1, 0, 1, 7, 1,
         1, 0, 1, 0, 1, 8;
    Eigen::MatrixXd K = A.transpose() * A + 10.0 * Eigen::MatrixXd::Identity(6, 6);
    Eigen::VectorXd f(6);
    f << 1, 2, 3, 4, 5, 6;

    constexpr std::size_t n_ext = 4;

    // Direct: solve full system
    assembly::DirectAssembly direct;
    auto ps_d = direct.prepare(K, f, n_ext);
    Eigen::VectorXd u_direct = ps_d.K.ldlt().solve(ps_d.f);

    // Condensed: solve reduced, then recover
    assembly::CondensedAssembly condensed;
    auto ps_c = condensed.prepare(K, f, n_ext);
    Eigen::VectorXd u_ext = ps_c.K.ldlt().solve(ps_c.f);
    Eigen::VectorXd u_int = condensed.recover_internal(u_ext);

    Eigen::VectorXd u_condensed(6);
    u_condensed << u_ext, u_int;

    constexpr double tol = 1e-10;
    ASSERT_VECTOR_NEAR(u_condensed, u_direct, tol);
}

void test_concept_satisfaction() {
    // Compile-time check via static_assert is in the header.
    // This test simply verifies the concept at runtime.
    ASSERT_TRUE((assembly::AssemblyPolicyLike<assembly::DirectAssembly>));
    ASSERT_TRUE((assembly::AssemblyPolicyLike<assembly::CondensedAssembly>));
}

// ============================================================================

int main() {
    std::cout << "=== Assembly Policy Tests ===\n";

    RUN_TEST(test_direct_assembly_compile_time_flag);
    RUN_TEST(test_direct_assembly_passthrough);
    RUN_TEST(test_direct_assembly_recover_is_empty);
    RUN_TEST(test_condensed_assembly_compile_time_flag);
    RUN_TEST(test_condensed_assembly_produces_reduced_system);
    RUN_TEST(test_condensed_assembly_roundtrip);
    RUN_TEST(test_condensed_assembly_cache_cleared_after_recover);
    RUN_TEST(test_condensed_vs_direct_solutions_agree);
    RUN_TEST(test_concept_satisfaction);

    std::cout << "\n=== " << g_pass << " PASSED, " << g_fail << " FAILED ===\n";
    return g_fail == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
