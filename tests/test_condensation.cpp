// ============================================================================
//  Tests for condensation::condense / condensation::recover_internal
// ============================================================================
//
//  All test cases use analytically verifiable block systems so the expected
//  Schur complement and recovered DOFs can be computed by hand.
// ============================================================================

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include <Eigen/Dense>

#include "src/numerics/StaticCondensation.hh"

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
                      << "  size mismatch\n";                                  \
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

#define ASSERT_THROWS(expr)                                                    \
    do {                                                                       \
        bool _caught = false;                                                  \
        try { expr; } catch (...) { _caught = true; }                          \
        if (!_caught) {                                                        \
            std::cerr << "  FAIL  " << __FILE__ << ":" << __LINE__             \
                      << "  expected exception not thrown\n";                   \
            ++g_fail;                                                          \
            return;                                                            \
        }                                                                      \
    } while (0)

#define RUN_TEST(fn)                                                           \
    do {                                                                       \
        int _before = g_fail;                                                  \
        fn();                                                                  \
        if (g_fail == _before) { ++g_pass; std::cout << "  PASS  " << #fn << "\n"; } \
        else                   { std::cout << "  FAIL  " << #fn << "\n"; }     \
    } while (0)

// ============================================================================
//  Test 1 — 3×3 system: 2 external + 1 internal
// ============================================================================
//
//  K = [ 4  1 | 2 ]     f = [ 10 ]
//      [ 1  3 | 1 ]         [  5 ]
//      [------+---]         [----]
//      [ 2  1 | 5 ]         [ 15 ]
//
//  K_ii = [5],  K_ii⁻¹ = [0.2]
//  K_ei·K_ii⁻¹·K_ie = [2 1]·0.2·[2] = [2 1]·[0.4] = [0.8  0.4]
//                                [1]         [0.2]   [0.4  0.2]
//
//  K̂ = [4 1] - [0.8 0.4] = [3.2  0.6]
//      [1 3]   [0.4 0.2]   [0.6  2.8]
//
//  K_ei·K_ii⁻¹·f_i = [2 1]·0.2·15 = [2 1]·3 = [6]
//                                               [3]
//
//  f̂ = [10 5]ᵀ − [6 3]ᵀ = [4 2]ᵀ

void test_3x3_condense_and_recover() {
    Eigen::MatrixXd K(3, 3);
    K << 4, 1, 2,
         1, 3, 1,
         2, 1, 5;

    Eigen::VectorXd f(3);
    f << 10, 5, 15;

    auto cs = condensation::condense(K, f, 2);

    // Expected K̂
    Eigen::MatrixXd K_hat_expected(2, 2);
    K_hat_expected << 3.2, 0.6,
                      0.6, 2.8;

    // Expected f̂
    Eigen::VectorXd f_hat_expected(2);
    f_hat_expected << 4.0, 2.0;

    constexpr double tol = 1e-12;

    ASSERT_MATRIX_NEAR(cs.K_hat, K_hat_expected, tol);
    ASSERT_VECTOR_NEAR(cs.f_hat, f_hat_expected, tol);

    // ── Full solve and recovery ──
    // Solve K̂ · u_e = f̂
    Eigen::VectorXd u_ext = cs.K_hat.ldlt().solve(cs.f_hat);

    // Recover internal DOFs
    Eigen::VectorXd u_int = condensation::recover_internal(cs, u_ext);

    // Verify by solving the full system directly
    Eigen::VectorXd u_full = K.ldlt().solve(f);

    ASSERT_VECTOR_NEAR(u_ext, u_full.head(2), tol);
    ASSERT_VECTOR_NEAR(u_int, u_full.tail(1), tol);
}

// ============================================================================
//  Test 2 — 4×4 system: 2 external + 2 internal (identity K_ii)
// ============================================================================
//
//  Symmetric K with K_ii = I₂.  K_ii⁻¹ = I₂, so the Schur complement
//  reduces to  K̂ = K_ee − K_ei · K_ie.

void test_4x4_identity_Kii() {
    Eigen::MatrixXd K(4, 4);
    K << 10,  2,   1,  0,
          2,  8,   0,  1,
          1,  0,   1,  0,
          0,  1,   0,  1;

    Eigen::VectorXd f(4);
    f << 5, 3, 1, 2;

    auto cs = condensation::condense(K, f, 2);

    // K_ei·K_ie = [1 0; 0 1]·[1 0; 0 1] = I₂  (since K_ii⁻¹ = I)
    // Actually K_ei = [1 0; 0 1],  K_ie = [1 0; 0 1], K_ii⁻¹ = I
    // K̂ = [10 2; 2 8] - [1 0; 0 1]*I*[1 0; 0 1] = [10 2; 2 8] - [1 0; 0 1] = [9 2; 2 7]
    Eigen::MatrixXd K_hat_expected(2, 2);
    K_hat_expected << 9, 2,
                      2, 7;

    // f̂ = [5 3]ᵀ - [1 0; 0 1]*I*[1 2]ᵀ = [5 3]ᵀ - [1 2]ᵀ = [4 1]ᵀ
    Eigen::VectorXd f_hat_expected(2);
    f_hat_expected << 4, 1;

    constexpr double tol = 1e-12;

    ASSERT_MATRIX_NEAR(cs.K_hat, K_hat_expected, tol);
    ASSERT_VECTOR_NEAR(cs.f_hat, f_hat_expected, tol);

    // Full-system consistency check
    Eigen::VectorXd u_ext = cs.K_hat.ldlt().solve(cs.f_hat);
    Eigen::VectorXd u_int = condensation::recover_internal(cs, u_ext);
    Eigen::VectorXd u_full = K.ldlt().solve(f);

    ASSERT_VECTOR_NEAR(u_ext, u_full.head(2), tol);
    ASSERT_VECTOR_NEAR(u_int, u_full.tail(2), tol);
}

// ============================================================================
//  Test 3 — 6×6 system: 4 external + 2 internal (larger block)
// ============================================================================
//
//  Randomly-structured but SPD matrix.  Verifies the condensation
//  algebraically reproduces the full solve.

void test_6x6_larger_block() {
    // Build a 6×6 SPD matrix: K = AᵀA + 5I  (guaranteed SPD)
    Eigen::MatrixXd A(6, 6);
    A << 2, 1, 0, 1, 0, 1,
         1, 3, 1, 0, 1, 0,
         0, 1, 4, 1, 0, 1,
         1, 0, 1, 5, 1, 0,
         0, 1, 0, 1, 6, 1,
         1, 0, 1, 0, 1, 7;

    Eigen::MatrixXd K = A.transpose() * A + 5.0 * Eigen::MatrixXd::Identity(6, 6);

    Eigen::VectorXd f(6);
    f << 1, 2, 3, 4, 5, 6;

    constexpr std::size_t n_ext = 4;

    auto cs = condensation::condense(K, f, n_ext);

    // Solve condensed system
    Eigen::VectorXd u_ext = cs.K_hat.ldlt().solve(cs.f_hat);
    Eigen::VectorXd u_int = condensation::recover_internal(cs, u_ext);

    // Solve full system
    Eigen::VectorXd u_full = K.ldlt().solve(f);

    constexpr double tol = 1e-10;

    ASSERT_VECTOR_NEAR(u_ext, u_full.head(n_ext), tol);
    ASSERT_VECTOR_NEAR(u_int, u_full.tail(6 - n_ext), tol);

    // Also verify that reassembled solution satisfies K·u ≈ f
    Eigen::VectorXd u_assembled(6);
    u_assembled << u_ext, u_int;
    Eigen::VectorXd residual = K * u_assembled - f;

    ASSERT_NEAR(residual.norm(), 0.0, tol);
}

// ============================================================================
//  Test 4 — Stiffness-only overload (no load vector)
// ============================================================================

void test_stiffness_only_condense() {
    Eigen::MatrixXd K(3, 3);
    K << 4, 1, 2,
         1, 3, 1,
         2, 1, 5;

    auto cs = condensation::condense(K, std::size_t{2});

    // Same expected K̂ as test 1
    Eigen::MatrixXd K_hat_expected(2, 2);
    K_hat_expected << 3.2, 0.6,
                      0.6, 2.8;

    ASSERT_MATRIX_NEAR(cs.K_hat, K_hat_expected, 1e-12);
}

// ============================================================================
//  Test 5 — Symmetry preservation of K̂
// ============================================================================
//
//  If K is symmetric, K̂ must also be symmetric.

void test_symmetry_preservation() {
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(8, 8);
    Eigen::MatrixXd K = A.transpose() * A + 10.0 * Eigen::MatrixXd::Identity(8, 8);

    Eigen::VectorXd f = Eigen::VectorXd::Random(8);

    auto cs = condensation::condense(K, f, 5);

    // Check symmetry of K̂
    const double sym_error = (cs.K_hat - cs.K_hat.transpose()).norm();
    ASSERT_NEAR(sym_error, 0.0, 1e-12);
}

// ============================================================================
//  Test 6 — Round-trip: condense → solve → recover → verify K·u = f
// ============================================================================
//
//  Uses an 8×8 SPD system with 5 external and 3 internal DOFs.

void test_round_trip_8x8() {
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(8, 8);
    Eigen::MatrixXd K = A.transpose() * A + 20.0 * Eigen::MatrixXd::Identity(8, 8);

    Eigen::VectorXd f = Eigen::VectorXd::Random(8);

    constexpr std::size_t n_ext = 5;

    auto cs = condensation::condense(K, f, n_ext);
    Eigen::VectorXd u_ext = cs.K_hat.ldlt().solve(cs.f_hat);
    Eigen::VectorXd u_int = condensation::recover_internal(cs, u_ext);

    Eigen::VectorXd u(8);
    u << u_ext, u_int;

    Eigen::VectorXd residual = K * u - f;
    ASSERT_NEAR(residual.norm(), 0.0, 1e-9);
}

// ============================================================================
//  Test 7 — Error: non-square K
// ============================================================================

void test_error_non_square() {
    Eigen::MatrixXd K(3, 4);
    K.setZero();
    Eigen::VectorXd f(3);
    f.setZero();

    ASSERT_THROWS(condensation::condense(K, f, 2));
}

// ============================================================================
//  Test 8 — Error: dimension mismatch f vs K
// ============================================================================

void test_error_f_size_mismatch() {
    Eigen::MatrixXd K = Eigen::MatrixXd::Identity(4, 4);
    Eigen::VectorXd f(3);
    f.setZero();

    ASSERT_THROWS(condensation::condense(K, f, 2));
}

// ============================================================================
//  Test 9 — Error: n_ext == 0
// ============================================================================

void test_error_n_ext_zero() {
    Eigen::MatrixXd K = Eigen::MatrixXd::Identity(4, 4);
    Eigen::VectorXd f = Eigen::VectorXd::Zero(4);

    ASSERT_THROWS(condensation::condense(K, f, 0));
}

// ============================================================================
//  Test 10 — Error: n_ext >= n (no internal DOFs left)
// ============================================================================

void test_error_n_ext_too_large() {
    Eigen::MatrixXd K = Eigen::MatrixXd::Identity(4, 4);
    Eigen::VectorXd f = Eigen::VectorXd::Zero(4);

    ASSERT_THROWS(condensation::condense(K, f, 4));
    ASSERT_THROWS(condensation::condense(K, f, 5));
}

// ============================================================================
//  Test 11 — Error: singular K_ii
// ============================================================================

void test_error_singular_Kii() {
    // K_ii will be the bottom-right 2×2: zero → singular
    Eigen::MatrixXd K(4, 4);
    K << 5, 1, 0, 0,
         1, 5, 0, 0,
         0, 0, 0, 0,
         0, 0, 0, 0;

    Eigen::VectorXd f = Eigen::VectorXd::Ones(4);

    ASSERT_THROWS(condensation::condense(K, f, 2));
}

// ============================================================================
//  Test 12 — Error: recover_internal with wrong u_ext size
// ============================================================================

void test_error_recover_wrong_size() {
    Eigen::MatrixXd K(3, 3);
    K << 4, 1, 2,
         1, 3, 1,
         2, 1, 5;

    Eigen::VectorXd f(3);
    f << 1, 2, 3;

    auto cs = condensation::condense(K, f, 2);

    Eigen::VectorXd u_wrong(5); // wrong size
    u_wrong.setZero();

    ASSERT_THROWS(condensation::recover_internal(cs, u_wrong));
}

// ============================================================================
//  Test 13 — Single internal DOF: 2×2 → 1×1
// ============================================================================
//
//  Simplest possible condensation: one external, one internal DOF.
//  K = [a  b]   f = [f_e]
//      [b  c]       [f_i]
//
//  K̂ = a - b²/c,   f̂ = f_e - b·f_i/c

void test_2x2_minimal() {
    const double a = 10.0, b = 3.0, c = 5.0;
    const double fe = 7.0, fi = 2.0;

    Eigen::MatrixXd K(2, 2);
    K << a, b,
         b, c;

    Eigen::VectorXd f(2);
    f << fe, fi;

    auto cs = condensation::condense(K, f, 1);

    const double K_hat_expected = a - b * b / c;       // 10 - 9/5 = 8.2
    const double f_hat_expected = fe - b * fi / c;     // 7 - 6/5 = 5.8

    ASSERT_NEAR(cs.K_hat(0, 0), K_hat_expected, 1e-12);
    ASSERT_NEAR(cs.f_hat(0),    f_hat_expected, 1e-12);

    // Solve and recover
    double u_e = cs.f_hat(0) / cs.K_hat(0, 0);
    Eigen::VectorXd u_ext(1);
    u_ext << u_e;

    Eigen::VectorXd u_int = condensation::recover_internal(cs, u_ext);

    Eigen::VectorXd u_full = K.ldlt().solve(f);
    ASSERT_NEAR(u_ext(0), u_full(0), 1e-12);
    ASSERT_NEAR(u_int(0), u_full(1), 1e-12);
}

// ============================================================================

int main() {
    std::cout << "=== Static Condensation Tests ===\n";

    RUN_TEST(test_3x3_condense_and_recover);
    RUN_TEST(test_4x4_identity_Kii);
    RUN_TEST(test_6x6_larger_block);
    RUN_TEST(test_stiffness_only_condense);
    RUN_TEST(test_symmetry_preservation);
    RUN_TEST(test_round_trip_8x8);
    RUN_TEST(test_error_non_square);
    RUN_TEST(test_error_f_size_mismatch);
    RUN_TEST(test_error_n_ext_zero);
    RUN_TEST(test_error_n_ext_too_large);
    RUN_TEST(test_error_singular_Kii);
    RUN_TEST(test_error_recover_wrong_size);
    RUN_TEST(test_2x2_minimal);

    std::cout << "\n=== " << g_pass << " PASSED, " << g_fail << " FAILED ===\n";
    return g_fail == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
