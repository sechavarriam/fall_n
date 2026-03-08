#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numbers>

// ── Include the full continuum module ────────────────────────────────────────
#include "src/continuum/Continuum.hh"

using namespace continuum;

// ── Helpers ──────────────────────────────────────────────────────────────────

namespace {

constexpr bool approx(double a, double b, double tol = 1e-10) {
    return std::abs(a - b) <= tol;
}

int passed = 0, failed = 0;

void report(const char* name, bool ok) {
    if (ok) { ++passed; std::cout << "  PASS  " << name << "\n"; }
    else    { ++failed; std::cout << "  FAIL  " << name << "\n"; }
}

} // anonymous namespace

// =============================================================================
//  Tensor2 tests
// =============================================================================

void test_tensor2_identity_trace() {
    auto I = Tensor2<3>::identity();
    report("T2 identity trace = 3",   approx(I.trace(), 3.0));
    report("T2 identity det = 1",     approx(I.determinant(), 1.0));
}

void test_tensor2_inverse() {
    // F = simple shear: F = I + γ e1⊗e2 with γ = 0.3
    auto F = Tensor2<3>::identity();
    F(0, 1) = 0.3;

    auto Finv = F.inverse();
    auto product = F.dot(Finv);  // should be I

    report("T2 F*F^-1 ≈ I", product.approx_equal(Tensor2<3>::identity(), 1e-12));
}

void test_tensor2_transpose() {
    Tensor2<3> A;
    A(0, 1) = 1.0; A(1, 0) = 2.0;
    A(0, 2) = 3.0; A(2, 0) = 4.0;
    auto At = A.transpose();
    report("T2 transpose (0,1)↔(1,0)", approx(At(0, 1), 2.0) && approx(At(1, 0), 1.0));
    report("T2 transpose (0,2)↔(2,0)", approx(At(0, 2), 4.0) && approx(At(2, 0), 3.0));
}

void test_tensor2_double_contract() {
    auto I = Tensor2<3>::identity();
    // I : I = δ_ij δ_ij = dim = 3
    report("T2 I:I = 3", approx(I.double_contract(I), 3.0));

    Tensor2<3> A;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            A(i, j) = static_cast<double>(i * 3 + j + 1);

    // A:A = Σ a_ij^2 = 1+4+9+16+25+36+49+64+81 = 285
    report("T2 A:A = 285", approx(A.double_contract(A), 285.0));
}

void test_tensor2_symmetric_skew_decomposition() {
    Tensor2<3> A;
    A(0, 1) = 3.0; A(1, 0) = 1.0;
    A(0, 2) = 5.0; A(2, 0) = 2.0;
    A(1, 2) = 4.0; A(2, 1) = 6.0;

    auto S = A.symmetric_part();
    auto W = A.skew_part();
    auto sum = S + W;
    report("T2 sym + skw = A", sum.approx_equal(A, 1e-14));

    // sym is symmetric
    report("T2 sym(A) is symmetric", S.approx_equal(S.transpose(), 1e-14));
    // skw is skew-symmetric
    report("T2 skw(A) is skew", W.approx_equal(W.transpose() * (-1.0), 1e-14));
}

void test_tensor2_invariants() {
    // Matrix with known eigenvalues: diag(2, 3, 5)
    Tensor2<3> A;
    A(0, 0) = 2.0; A(1, 1) = 3.0; A(2, 2) = 5.0;

    // I1 = 2+3+5 = 10
    report("T2 I1 = 10", approx(A.I1(), 10.0));
    // I2 = 2*3 + 3*5 + 2*5 = 6+15+10 = 31
    report("T2 I2 = 31", approx(A.I2(), 31.0));
    // I3 = 2*3*5 = 30
    report("T2 I3 = 30", approx(A.I3(), 30.0));
}

void test_tensor2_deviatoric() {
    auto A = Tensor2<3>::identity() * 6.0;  // 6I → tr = 18
    auto devA = A.deviatoric();
    // dev(αI) = αI - (α·dim/dim)I = 0
    report("T2 dev(αI) = 0", devA.approx_equal(Tensor2<3>::zero(), 1e-14));

    Tensor2<3> B;
    B(0, 0) = 1.0; B(1, 1) = 2.0; B(2, 2) = 3.0;
    auto devB = B.deviatoric();
    // tr(dev(B)) = 0
    report("T2 tr(dev(B)) = 0", approx(devB.trace(), 0.0));
}

void test_tensor2_dyadic() {
    Eigen::Vector3d a{1.0, 0.0, 0.0};
    Eigen::Vector3d b{0.0, 1.0, 0.0};
    auto ab = Tensor2<3>::dyadic(a, b);
    // Only (0,1) should be 1
    report("T2 dyadic e1⊗e2 (0,1)=1", approx(ab(0, 1), 1.0));
    report("T2 dyadic e1⊗e2 (1,0)=0", approx(ab(1, 0), 0.0));
    report("T2 dyadic e1⊗e2 trace=0", approx(ab.trace(), 0.0));
}

// =============================================================================
//  SymmetricTensor2 tests
// =============================================================================

void test_sym_tensor2_voigt_mapping() {
    // 3D: verify that the Voigt index map is consistent
    using ST = SymmetricTensor2<3>;

    // Diagonal: (0,0)→0, (1,1)→1, (2,2)→2
    report("ST2 voigt (0,0)→0", ST::voigt_index(0, 0) == 0);
    report("ST2 voigt (1,1)→1", ST::voigt_index(1, 1) == 1);
    report("ST2 voigt (2,2)→2", ST::voigt_index(2, 2) == 2);

    // Off-diagonal: (1,2)→3, (0,2)→4, (0,1)→5
    report("ST2 voigt (1,2)→3", ST::voigt_index(1, 2) == 3);
    report("ST2 voigt (0,2)→4", ST::voigt_index(0, 2) == 4);
    report("ST2 voigt (0,1)→5", ST::voigt_index(0, 1) == 5);

    // Symmetry: (i,j) == (j,i)
    report("ST2 voigt (2,1)→3", ST::voigt_index(2, 1) == 3);
    report("ST2 voigt (2,0)→4", ST::voigt_index(2, 0) == 4);
    report("ST2 voigt (1,0)→5", ST::voigt_index(1, 0) == 5);
}

void test_sym_tensor2_multidim_subscript() {
    // C++23 T[i,j] — zero-cost matrix view over Voigt storage
    SymmetricTensor2<3> T(1.0, 2.0, 3.0, 0.4, 0.5, 0.6);

    // Read: diagonal entries match Voigt components 0,1,2
    report("ST2 T[0,0] == voigt[0]", approx(T[0,0], 1.0));
    report("ST2 T[1,1] == voigt[1]", approx(T[1,1], 2.0));
    report("ST2 T[2,2] == voigt[2]", approx(T[2,2], 3.0));

    // Read: off-diagonal (symmetric access)
    report("ST2 T[1,2] == T[2,1]", approx(T[1,2], T[2,1]));
    report("ST2 T[0,2] == 0.5",    approx(T[0,2], 0.5));
    report("ST2 T[0,1] == 0.6",    approx(T[0,1], 0.6));

    // Write through T[i,j] — modifies the underlying Voigt component
    T[0,1] = 9.9;
    report("ST2 T[0,1] write",     approx(T[0,1], 9.9));
    report("ST2 T[1,0] after write",approx(T[1,0], 9.9));  // same component
    report("ST2 voigt[5] after write",approx(T[5], 9.9));   // Voigt index 5

    // 2D
    SymmetricTensor2<2> S(10.0, 20.0, 5.0);
    report("ST2 2D S[0,0]", approx(S[0,0], 10.0));
    report("ST2 2D S[1,1]", approx(S[1,1], 20.0));
    report("ST2 2D S[0,1]", approx(S[0,1], 5.0));
    report("ST2 2D S[1,0]", approx(S[1,0], 5.0));
}

void test_subscript_vs_matrix_consistency() {
    // T[i,j] must agree with matrix()(i,j) for ALL entries of a general tensor
    SymmetricTensor2<3> T(6.0, 3.0, 5.0, 0.4, 0.7, 1.2);
    auto M = T.matrix();

    bool all_ok = true;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            if (!approx(T[i,j], M(i,j))) all_ok = false;

    report("[i,j] vs matrix() all 9 entries", all_ok);
}

void test_subscript_build_tensor_from_scratch() {
    // Build a tensor entirely through T[i,j] writes
    SymmetricTensor2<3> sigma;  // zero

    // Uniaxial stress σ₁₁ = 250 MPa
    sigma[0,0] = 250.0;

    report("[i,j] build: σ₁₁ = 250",  approx(sigma[0,0], 250.0));
    report("[i,j] build: σ₂₂ = 0",    approx(sigma[1,1], 0.0));
    report("[i,j] build: σ₃₃ = 0",    approx(sigma[2,2], 0.0));
    report("[i,j] build: trace = 250", approx(sigma.trace(), 250.0));
    report("[i,j] build: von Mises",   approx(stress::von_mises(sigma), 250.0, 1e-8));

    // Now add shear: σ₁₂ = 50  (through [1,0] → same as [0,1])
    sigma[1,0] = 50.0;
    report("[i,j] build: σ₁₂ via [1,0]", approx(sigma[0,1], 50.0));
    report("[i,j] build: voigt[5]=50",    approx(sigma[5], 50.0));

    // von Mises with shear: σ_eq = √(σ₁₁² + 3·σ₁₂²)  for this state
    double vm_expected = std::sqrt(250.0*250.0 + 3.0*50.0*50.0);
    report("[i,j] build: vM with shear", approx(stress::von_mises(sigma), vm_expected, 1e-8));
}

void test_subscript_double_contract_manual() {
    // Compute σ:ε manually via T[i,j] loops and compare to double_contract()
    SymmetricTensor2<3> sigma(100.0, 200.0, 300.0, 10.0, 20.0, 30.0);
    SymmetricTensor2<3> eps(0.001, 0.002, 0.003, 0.0001, 0.0002, 0.0003);

    // Manual: σ:ε = Σ_ij σ_ij ε_ij  (full double sum, using symmetry)
    double manual = 0.0;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            manual += sigma[i,j] * eps[i,j];

    double voigt_result = sigma.double_contract(eps);
    report("[i,j] manual σ:ε == double_contract", approx(manual, voigt_result));
}

void test_subscript_trace_manual() {
    SymmetricTensor2<3> T(7.0, 11.0, 13.0, 0.5, 0.3, 0.1);

    double manual_trace = T[0,0] + T[1,1] + T[2,2];
    report("[i,j] manual trace", approx(manual_trace, T.trace()));
}

void test_subscript_determinant_manual() {
    // det(T) via Sarrus rule using T[i,j]
    SymmetricTensor2<3> T(4.0, 5.0, 6.0, 0.3, 0.2, 0.1);

    double manual_det = T[0,0] * (T[1,1]*T[2,2] - T[1,2]*T[2,1])
                      - T[0,1] * (T[1,0]*T[2,2] - T[1,2]*T[2,0])
                      + T[0,2] * (T[1,0]*T[2,1] - T[1,1]*T[2,0]);

    report("[i,j] manual det == determinant()", approx(manual_det, T.determinant()));
}

void test_subscript_inverse_verify() {
    // Build T via [i,j], invert, verify T·T⁻¹ = I through [i,j]
    SymmetricTensor2<3> T;
    T[0,0] = 10.0;  T[1,1] = 8.0;  T[2,2] = 6.0;
    T[0,1] = 1.0;   T[0,2] = 0.5;  T[1,2] = 0.3;

    auto Tinv = T.inverse();

    // Multiply via [i,j] loops:  (T·T⁻¹)_ij = Σ_k T[i,k] * T⁻¹[k,j]
    bool ok = true;
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < 3; ++k)
                sum += T[i,k] * Tinv[k,j];
            double expected = (i == j) ? 1.0 : 0.0;
            if (!approx(sum, expected, 1e-10)) ok = false;
        }
    }
    report("[i,j] T·T⁻¹ = I via loops", ok);
}

void test_subscript_construct_strain_from_F() {
    // Manually compute Green-Lagrange E = ½(FᵀF − I) using [i,j]
    // and compare to strain::green_lagrange()
    auto F = Tensor2<3>::identity();
    F(0,0) = 1.3; F(0,1) = 0.1; F(1,1) = 0.95; F(2,2) = 1.05;

    // Manual: E_ij = ½(Σ_k F_ki F_kj − δ_ij)
    SymmetricTensor2<3> E_manual;
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = i; j < 3; ++j) {
            double FtF_ij = 0.0;
            for (std::size_t k = 0; k < 3; ++k)
                FtF_ij += F(k,i) * F(k,j);
            double delta = (i == j) ? 1.0 : 0.0;
            E_manual[i,j] = 0.5 * (FtF_ij - delta);
        }
    }

    auto E_func = strain::green_lagrange(F);
    report("[i,j] manual E_GL vs function", E_manual.approx_equal(E_func, 1e-12));
}

void test_subscript_symmetry_write_enforcement() {
    // Writing through [1,2] must be readable through [2,1]
    // (they share the same Voigt slot)
    SymmetricTensor2<3> T;

    // Write every off-diagonal through the upper-triangle index
    T[0,1] = 1.1;
    T[0,2] = 2.2;
    T[1,2] = 3.3;

    // Read through the transposed index
    report("[i,j] sym: [1,0]=[0,1]", approx(T[1,0], 1.1));
    report("[i,j] sym: [2,0]=[0,2]", approx(T[2,0], 2.2));
    report("[i,j] sym: [2,1]=[1,2]", approx(T[2,1], 3.3));

    // Now overwrite through lower-triangle and verify via upper
    T[2,0] = 7.7;
    report("[i,j] sym: overwrite [2,0]→[0,2]", approx(T[0,2], 7.7));
}

void test_subscript_2d_strain_workflow() {
    // 2D plane-strain workflow entirely through [i,j]
    Tensor2<2> F = Tensor2<2>::identity();
    F(0,0) = 1.1; F(0,1) = 0.05; F(1,1) = 0.98;

    // Manual E_GL in 2D
    SymmetricTensor2<2> E;
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = i; j < 2; ++j) {
            double FtF = 0.0;
            for (std::size_t k = 0; k < 2; ++k)
                FtF += F(k,i) * F(k,j);
            E[i,j] = 0.5 * (FtF - ((i == j) ? 1.0 : 0.0));
        }
    }

    auto E_ref = strain::green_lagrange(F);
    report("2D [i,j] E_GL workflow", E.approx_equal(E_ref, 1e-12));

    // Apply isotropic tangent via [i,j] loop: σ_IJ = Σ_KL C_Voigt(I,K) · ε_Voigt(K)
    // Instead: just verify trace of E through [i,j]
    double tr = E[0,0] + E[1,1];
    report("2D [i,j] trace of E", approx(tr, E.trace()));
}

void test_subscript_1d() {
    // 1D: only T[0,0]
    SymmetricTensor2<1> T(5.0);
    report("1D [0,0] read",  approx(T[0,0], 5.0));
    T[0,0] = 42.0;
    report("1D [0,0] write", approx(T[0,0], 42.0));
    report("1D voigt[0] after write", approx(T[0], 42.0));
}

void test_subscript_cauchy_stress_transformation() {
    // Build S (2nd PK) through [i,j], push-forward to σ, verify σ[i,j] symmetry
    SymmetricTensor2<3> S;
    S[0,0] = 100.0;  S[1,1] = 200.0;  S[2,2] = 150.0;
    S[0,1] = 30.0;   S[0,2] = 20.0;   S[1,2] = 10.0;

    auto F = Tensor2<3>::identity();
    F(0,0) = 1.2; F(0,1) = 0.15; F(1,1) = 0.9; F(2,2) = 1.1;

    auto sigma = stress::cauchy_from_2pk(S, F);

    // Verify Cauchy symmetry through [i,j]
    bool sym_ok = true;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = i+1; j < 3; ++j)
            if (!approx(sigma[i,j], sigma[j,i])) sym_ok = false;

    report("[i,j] Cauchy symmetry", sym_ok);

    // Verify trace via [i,j] matches .trace()
    double tr = sigma[0,0] + sigma[1,1] + sigma[2,2];
    report("[i,j] Cauchy trace", approx(tr, sigma.trace()));
}

void test_sym_tensor2_identity() {
    auto I = SymmetricTensor2<3>::identity();
    report("ST2 identity trace = 3",   approx(I.trace(), 3.0));
    report("ST2 identity det = 1",     approx(I.determinant(), 1.0));
    report("ST2 identity I1 = 3",      approx(I.I1(), 3.0));
    report("ST2 identity I2 = 3",      approx(I.I2(), 3.0));
    report("ST2 identity I3 = 1",      approx(I.I3(), 1.0));
}

void test_sym_tensor2_matrix_roundtrip() {
    // Create a SymmetricTensor2 from components
    SymmetricTensor2<3> S(1.0, 2.0, 3.0, 0.4, 0.5, 0.6);

    // Convert to full matrix
    auto M = S.matrix();

    // Verify symmetry
    report("ST2 matrix symmetric (0,1)=(1,0)", approx(M(0,1), M(1,0)));
    report("ST2 matrix symmetric (0,2)=(2,0)", approx(M(0,2), M(2,0)));
    report("ST2 matrix symmetric (1,2)=(2,1)", approx(M(1,2), M(2,1)));

    // Roundtrip: matrix → SymmetricTensor2 → matrix
    SymmetricTensor2<3> S2{Tensor2<3>{M}};
    report("ST2 roundtrip", S.approx_equal(S2, 1e-14));
}

void test_sym_tensor2_double_contract() {
    auto I = SymmetricTensor2<3>::identity();
    // I : I = δ_ij δ_ij = 3
    report("ST2 I:I = 3", approx(I.double_contract(I), 3.0));

    // Stress:strain product
    SymmetricTensor2<3> sigma(100.0, 200.0, 300.0, 10.0, 20.0, 30.0);
    SymmetricTensor2<3> eps(0.001, 0.002, 0.003, 0.0001, 0.0002, 0.0003);

    // σ:ε = Σ_diag σ_k ε_k + 2·Σ_offdiag σ_k ε_k
    double expected = 100*0.001 + 200*0.002 + 300*0.003
                    + 2*(10*0.0001 + 20*0.0002 + 30*0.0003);
    report("ST2 σ:ε correct", approx(sigma.double_contract(eps), expected));
}

void test_sym_tensor2_engineering_voigt() {
    SymmetricTensor2<3> eps(0.001, 0.002, 0.003, 0.004, 0.005, 0.006);

    auto eng = eps.voigt_engineering();
    // Diagonal unchanged
    report("ST2 eng voigt diag unchanged", approx(eng[0], 0.001) &&
                                            approx(eng[1], 0.002) &&
                                            approx(eng[2], 0.003));
    // Off-diagonal doubled
    report("ST2 eng voigt shear doubled",  approx(eng[3], 0.008) &&
                                            approx(eng[4], 0.010) &&
                                            approx(eng[5], 0.012));

    // Set from engineering → back to tensor
    SymmetricTensor2<3> eps2;
    eps2.set_from_engineering_voigt(eng);
    report("ST2 eng voigt roundtrip", eps.approx_equal(eps2, 1e-15));
}

void test_sym_tensor2_invariants() {
    // T = diag(2, 3, 5)  → known eigenvalues
    SymmetricTensor2<3> T(2.0, 3.0, 5.0, 0.0, 0.0, 0.0);

    report("ST2 I1 = 10", approx(T.I1(), 10.0));
    report("ST2 I2 = 31", approx(T.I2(), 31.0));
    report("ST2 I3 = 30", approx(T.I3(), 30.0));
}

void test_sym_tensor2_deviatoric() {
    SymmetricTensor2<3> T(4.0, 5.0, 6.0, 1.0, 2.0, 3.0);
    auto devT = T.deviatoric();
    // tr(dev(T)) = 0
    report("ST2 tr(dev(T)) = 0", approx(devT.trace(), 0.0, 1e-14));

    // dev(T) + sph(T) = T
    auto sphT = T.spherical();
    auto sum = devT + sphT;
    report("ST2 dev + sph = T", sum.approx_equal(T, 1e-14));
}

void test_sym_tensor2_spectral_decomposition() {
    // Diagonal tensor: eigenvalues are the diagonal entries
    SymmetricTensor2<3> D(2.0, 3.0, 5.0, 0.0, 0.0, 0.0);

    auto evals = D.eigenvalues();
    // Eigenvalues in ascending order: 2, 3, 5
    report("ST2 evals[0] = 2", approx(evals[0], 2.0));
    report("ST2 evals[1] = 3", approx(evals[1], 3.0));
    report("ST2 evals[2] = 5", approx(evals[2], 5.0));
}

void test_sym_tensor2_tensor_functions() {
    // T = diag(4, 9, 16)  → known spectral form
    SymmetricTensor2<3> T(4.0, 9.0, 16.0, 0.0, 0.0, 0.0);

    // sqrt(T) = diag(2, 3, 4)
    auto sqrtT = T.sqrt();
    report("ST2 sqrt diag(4,9,16) → diag(2,3,4)",
           approx(sqrtT(0,0), 2.0) &&
           approx(sqrtT(1,1), 3.0) &&
           approx(sqrtT(2,2), 4.0) &&
           approx(sqrtT(0,1), 0.0, 1e-12));

    // log(T) = diag(ln4, ln9, ln16)
    auto logT = T.log();
    report("ST2 log(diag(4,9,16))",
           approx(logT(0,0), std::log(4.0)) &&
           approx(logT(1,1), std::log(9.0)) &&
           approx(logT(2,2), std::log(16.0)));

    // exp(log(T)) ≈ T
    auto expLogT = logT.exp();
    report("ST2 exp(log(T)) ≈ T", expLogT.approx_equal(T, 1e-10));

    // sqrt(T)² ≈ T
    SymmetricTensor2<3> sqrtT_sq{Tensor2<3>{sqrtT.matrix() * sqrtT.matrix()}};
    report("ST2 sqrt(T)² ≈ T", sqrtT_sq.approx_equal(T, 1e-10));

    // pow(T, 0.5) ≈ sqrt(T)
    auto powHalf = T.pow(0.5);
    report("ST2 pow(T,0.5) ≈ sqrt(T)", powHalf.approx_equal(sqrtT, 1e-10));
}

void test_sym_tensor2_inverse_closed_form() {
    // ── Diagonal tensor: easy to verify ──
    SymmetricTensor2<3> D(2.0, 4.0, 5.0, 0.0, 0.0, 0.0);
    auto Dinv = D.inverse();
    report("ST2 inv diag (0,0)", approx(Dinv(0,0), 0.5));
    report("ST2 inv diag (1,1)", approx(Dinv(1,1), 0.25));
    report("ST2 inv diag (2,2)", approx(Dinv(2,2), 0.2));
    report("ST2 inv diag off-diag=0", approx(Dinv(0,1), 0.0, 1e-14));

    // ── Full tensor with shear: T · T⁻¹ = I ──
    SymmetricTensor2<3> S(6.0, 3.0, 5.0, 0.4, 0.7, 1.2);
    auto Sinv = S.inverse();
    auto product_mat = S.matrix() * Sinv.matrix();
    auto I3 = Eigen::Matrix3d::Identity();
    report("ST2 S·S⁻¹ ≈ I (full)", (product_mat - I3).norm() < 1e-10);

    // ── 2D: T·T⁻¹ = I ──
    SymmetricTensor2<2> T2(3.0, 7.0, 1.5);
    auto T2inv = T2.inverse();
    auto prod2 = T2.matrix() * T2inv.matrix();
    auto I2 = Eigen::Matrix2d::Identity();
    report("ST2 2D T·T⁻¹ ≈ I", (prod2 - I2).norm() < 1e-10);

    // ── 1D ──
    SymmetricTensor2<1> T1(4.0);
    auto T1inv = T1.inverse();
    report("ST2 1D inv", approx(T1inv[0], 0.25));
}

// =============================================================================
//  Tensor4 tests
// =============================================================================

void test_tensor4_isotropic_elasticity() {
    // Steel-like: E = 200 GPa, ν = 0.3
    double E = 200.0e3;
    double nu = 0.3;
    auto C = Tensor4<3>::isotropic_E_nu(E, nu);

    double lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
    double mu     = E / (2 * (1 + nu));

    // Check diagonal
    report("T4 C(0,0) = λ+2μ", approx(C(0,0), lambda + 2*mu));
    report("T4 C(1,1) = λ+2μ", approx(C(1,1), lambda + 2*mu));
    report("T4 C(2,2) = λ+2μ", approx(C(2,2), lambda + 2*mu));

    // Check off-diagonal (normal-normal)
    report("T4 C(0,1) = λ", approx(C(0,1), lambda));
    report("T4 C(0,2) = λ", approx(C(0,2), lambda));

    // Check shear
    report("T4 C(3,3) = μ", approx(C(3,3), mu));
    report("T4 C(4,4) = μ", approx(C(4,4), mu));
    report("T4 C(5,5) = μ", approx(C(5,5), mu));

    // Major symmetry
    report("T4 isotropic has major symmetry", C.has_major_symmetry());
}

void test_tensor4_contraction() {
    // C : ε  (uniaxial strain in x-direction)
    double E = 200.0e3;
    double nu = 0.3;
    auto C = Tensor4<3>::isotropic_E_nu(E, nu);

    // Pure uniaxial strain: ε₁₁ = 0.001, rest = 0
    SymmetricTensor2<3> eps(0.001, 0.0, 0.0, 0.0, 0.0, 0.0);
    auto sigma = C.contract(eps);

    double lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
    double mu     = E / (2 * (1 + nu));

    // σ₁₁ = (λ + 2μ) ε₁₁
    report("T4 contraction σ₁₁", approx(sigma(0,0), (lambda + 2*mu) * 0.001));
    // σ₂₂ = λ ε₁₁
    report("T4 contraction σ₂₂ = λε₁₁", approx(sigma(1,1), lambda * 0.001));
    // σ₃₃ = λ ε₁₁
    report("T4 contraction σ₃₃ = λε₁₁", approx(sigma(2,2), lambda * 0.001));
    // σ₂₃ = 0
    report("T4 contraction σ₂₃ = 0", approx(sigma(1,2), 0.0));
}

void test_tensor4_identity_tensors() {
    auto Isym = Tensor4<3>::symmetric_identity();
    auto Ivol = Tensor4<3>::volumetric_projector();
    auto Idev = Tensor4<3>::deviatoric_projector();

    // Isym acting on identity: Isym : I = I
    auto I = SymmetricTensor2<3>::identity();
    auto IonI = Isym.contract(I);
    report("T4 Isym : I = I", IonI.approx_equal(I, 1e-14));

    // Ivol : T = (1/3)tr(T) I
    SymmetricTensor2<3> T(1.0, 2.0, 3.0, 0.0, 0.0, 0.0);
    auto volT = Ivol.contract(T);
    auto expected_sph = T.spherical();
    report("T4 Ivol : T = sph(T)", volT.approx_equal(expected_sph, 1e-14));

    // Idev : T = dev(T)
    auto devT_from_proj = Idev.contract(T);
    auto devT = T.deviatoric();
    report("T4 Idev : T = dev(T)", devT_from_proj.approx_equal(devT, 1e-14));
}

void test_tensor4_outer_product() {
    auto I = SymmetricTensor2<3>::identity();
    auto IoI = ops::outer_product(I, I);

    // I⊗I acting on T gives tr(T)·I
    SymmetricTensor2<3> T(1.0, 2.0, 3.0, 0.5, 0.3, 0.1);
    auto result = IoI.contract(T);
    auto expected = I * T.trace();  // tr(T) * I
    report("T4 (I⊗I):T = tr(T)I", result.approx_equal(expected, 1e-12));
}

// =============================================================================
//  TensorOperations tests
// =============================================================================

void test_polar_decomposition_pure_rotation() {
    // Pure rotation by 45° around z-axis
    double theta = std::numbers::pi / 4.0;
    Tensor2<3> R_expected = Tensor2<3>::identity();
    R_expected(0,0) =  std::cos(theta); R_expected(0,1) = -std::sin(theta);
    R_expected(1,0) =  std::sin(theta); R_expected(1,1) =  std::cos(theta);

    // For pure rotation: F = R, U = I, V = I
    auto [R, U, V] = ops::polar_decomposition(R_expected);

    report("Polar: R ≈ F (pure rot)", R.approx_equal(R_expected, 1e-10));
    report("Polar: U ≈ I (pure rot)", U.approx_equal(SymmetricTensor2<3>::identity(), 1e-10));
    report("Polar: V ≈ I (pure rot)", V.approx_equal(SymmetricTensor2<3>::identity(), 1e-10));
}

void test_polar_decomposition_stretch() {
    // Pure stretch: F = diag(2, 3, 1)  → U = F, R = I
    Tensor2<3> F = Tensor2<3>::identity();
    F(0,0) = 2.0; F(1,1) = 3.0;

    auto [R, U, V] = ops::polar_decomposition(F);

    report("Polar: R ≈ I (pure stretch)", R.approx_equal(Tensor2<3>::identity(), 1e-10));

    SymmetricTensor2<3> U_expected(2.0, 3.0, 1.0, 0.0, 0.0, 0.0);
    report("Polar: U ≈ diag(2,3,1)", U.approx_equal(U_expected, 1e-10));
}

void test_polar_decomposition_general() {
    // General F: rotation + stretch
    double theta = 0.5;  // rad
    Tensor2<3> R_pure = Tensor2<3>::identity();
    R_pure(0,0) =  std::cos(theta); R_pure(0,1) = -std::sin(theta);
    R_pure(1,0) =  std::sin(theta); R_pure(1,1) =  std::cos(theta);

    Tensor2<3> U_pure = Tensor2<3>::identity();
    U_pure(0,0) = 1.5; U_pure(1,1) = 2.0;

    Tensor2<3> F{(R_pure.matrix() * U_pure.matrix()).eval()};

    auto [R, U, V] = ops::polar_decomposition(F);

    // Verify F = R·U
    auto F_reconstructed = Tensor2<3>{(R.matrix() * U.matrix()).eval()};
    report("Polar: F ≈ R·U (general)", F_reconstructed.approx_equal(F, 1e-10));

    // R should be orthogonal: RᵀR = I
    auto RtR = Tensor2<3>{(R.matrix().transpose() * R.matrix()).eval()};
    report("Polar: RᵀR ≈ I", RtR.approx_equal(Tensor2<3>::identity(), 1e-10));

    // det(R) ≈ 1
    report("Polar: det(R) ≈ 1", approx(R.determinant(), 1.0, 1e-10));
}

void test_polar_decomposition_2d() {
    // 2D: rotation by 30° + stretch
    double theta = std::numbers::pi / 6.0;
    Tensor2<2> R2 = Tensor2<2>::identity();
    R2(0,0) =  std::cos(theta); R2(0,1) = -std::sin(theta);
    R2(1,0) =  std::sin(theta); R2(1,1) =  std::cos(theta);

    Tensor2<2> U2 = Tensor2<2>::identity();
    U2(0,0) = 1.3; U2(1,1) = 0.8;

    Tensor2<2> F{(R2.matrix() * U2.matrix()).eval()};
    auto [R, U, V] = ops::polar_decomposition(F);

    auto F_rec = Tensor2<2>{(R.matrix() * U.matrix()).eval()};
    report("Polar 2D: F ≈ R·U", F_rec.approx_equal(F, 1e-10));
    report("Polar 2D: det(R) ≈ 1", approx(R.determinant(), 1.0, 1e-10));
}

void test_push_forward_pull_back_roundtrip() {
    // Define a deformation gradient (simple shear + stretch)
    auto F = Tensor2<3>::identity();
    F(0,0) = 1.2; F(0,1) = 0.3; F(1,1) = 0.9; F(2,2) = 1.1;

    // Arbitrary S (2nd PK)
    SymmetricTensor2<3> S(100.0, 200.0, 150.0, 10.0, 20.0, 30.0);

    // Push forward: S → σ → pull back → S'
    auto sigma = ops::push_forward(S, F);
    auto S_back = ops::pull_back(sigma, F);

    report("Push/pull roundtrip S ≈ S'", S.approx_equal(S_back, 1e-8));
}

// =============================================================================
//  Strain measure tests
// =============================================================================

void test_green_lagrange_identity() {
    // F = I → E_GL = 0
    auto F = Tensor2<3>::identity();
    auto E = strain::green_lagrange(F);
    report("E_GL(I) = 0", E.approx_equal(SymmetricTensor2<3>::zero(), 1e-14));
}

void test_green_lagrange_uniaxial() {
    // Uniaxial stretch: F = diag(λ, 1, 1)  → E₁₁ = ½(λ²-1)
    double lam = 1.3;
    Tensor2<3> F = Tensor2<3>::identity();
    F(0,0) = lam;

    auto E = strain::green_lagrange(F);
    report("E_GL uniaxial E₁₁", approx(E(0,0), 0.5*(lam*lam - 1.0)));
    report("E_GL uniaxial E₂₂", approx(E(1,1), 0.0));
    report("E_GL uniaxial E₃₃", approx(E(2,2), 0.0));
}

void test_hencky_uniaxial() {
    // Uniaxial stretch: E_H = ½ ln(C) with C = diag(λ², 1, 1)
    double lam = 1.5;
    Tensor2<3> F = Tensor2<3>::identity();
    F(0,0) = lam;

    auto E_H = strain::hencky(F);
    report("Hencky uniaxial E₁₁", approx(E_H(0,0), std::log(lam), 1e-10));
    report("Hencky uniaxial E₂₂", approx(E_H(1,1), 0.0, 1e-10));
}

void test_hencky_identity() {
    auto F = Tensor2<3>::identity();
    auto E_H = strain::hencky(F);
    report("Hencky(I) = 0", E_H.approx_equal(SymmetricTensor2<3>::zero(), 1e-12));
}

void test_seth_hill_recovers_green_lagrange() {
    // m=1 → Green-Lagrange
    double lam = 1.4;
    Tensor2<3> F = Tensor2<3>::identity();
    F(0,0) = lam; F(1,1) = 0.9; F(2,2) = 1.1;

    auto E_GL  = strain::green_lagrange(F);
    auto E_SH1 = strain::seth_hill(F, 1.0);

    report("Seth-Hill m=1 ≈ GL", E_SH1.approx_equal(E_GL, 1e-10));
}

void test_seth_hill_recovers_hencky() {
    // m → 0 → Hencky
    double lam = 1.3;
    Tensor2<3> F = Tensor2<3>::identity();
    F(0,0) = lam; F(1,1) = 0.8; F(2,2) = 1.2;

    auto E_H   = strain::hencky(F);
    auto E_SH0 = strain::seth_hill(F, 0.0);  // triggers logarithmic branch

    report("Seth-Hill m=0 ≈ Hencky", E_SH0.approx_equal(E_H, 1e-10));
}

void test_biot_strain() {
    // Biot: E_B = U - I.  For F = diag(λ, 1, 1), U = diag(λ, 1, 1)
    double lam = 1.4;
    Tensor2<3> F = Tensor2<3>::identity();
    F(0,0) = lam;

    auto E_B = strain::biot(F);
    report("Biot E₁₁ = λ-1", approx(E_B(0,0), lam - 1.0, 1e-10));
    report("Biot E₂₂ = 0",   approx(E_B(1,1), 0.0, 1e-10));
}

void test_almansi_identity() {
    auto F = Tensor2<3>::identity();
    auto e = strain::almansi(F);
    report("Almansi(I) = 0", e.approx_equal(SymmetricTensor2<3>::zero(), 1e-14));
}

void test_infinitesimal_small_deformation() {
    // Small deformation: F = I + ε∇u  with ∇u = [[0, 0.001], [-0.001, 0]]   (2D)
    Tensor2<2> F = Tensor2<2>::identity();
    F(0,1) = 0.001;
    F(1,0) = -0.001;

    auto eps = strain::infinitesimal(F);
    // ε = sym(∇u) → ε₁₂ = 0 (pure rotation)
    report("Inf strain: pure rotation → ε=0", eps.approx_equal(SymmetricTensor2<2>::zero(), 1e-14));

    // Now add actual strain
    Tensor2<2> F2 = Tensor2<2>::identity();
    F2(0,0) = 1.001; F2(1,1) = 0.999;

    auto eps2 = strain::infinitesimal(F2);
    report("Inf strain: ε₁₁ = 0.001", approx(eps2(0,0), 0.001));
    report("Inf strain: ε₂₂ = -0.001", approx(eps2(1,1), -0.001));
}

void test_all_strains_converge_for_small_F() {
    // For F ≈ I + ε∇u with small ε, all strain measures should agree
    double eps_val = 1e-6;
    Tensor2<3> F = Tensor2<3>::identity();
    F(0,0) = 1.0 + eps_val;
    F(1,1) = 1.0 - 0.5 * eps_val;
    F(2,2) = 1.0 - 0.5 * eps_val;

    auto inf  = strain::infinitesimal(F);
    auto gl   = strain::green_lagrange(F);
    auto hn   = strain::hencky(F);
    auto bt   = strain::biot(F);
    auto alm  = strain::almansi(F);

    double tol = 1e-4;  // Relative to eps_val², differences ~ eps_val²
    report("Small F: GL ≈ inf",      gl.approx_equal(inf, tol));
    report("Small F: Hencky ≈ inf",  hn.approx_equal(inf, tol));
    report("Small F: Biot ≈ inf",    bt.approx_equal(inf, tol));
    report("Small F: Almansi ≈ inf", alm.approx_equal(inf, tol));
}

// =============================================================================
//  Stress measure tests
// =============================================================================

void test_cauchy_2pk_roundtrip() {
    auto F = Tensor2<3>::identity();
    F(0,0) = 1.2; F(0,1) = 0.1; F(1,1) = 0.95; F(2,2) = 1.05;

    SymmetricTensor2<3> sigma(100.0, 50.0, 80.0, 5.0, 3.0, 7.0);

    auto S = stress::second_pk_from_cauchy(sigma, F);
    auto sigma_back = stress::cauchy_from_2pk(S, F);

    report("Cauchy → S → Cauchy roundtrip", sigma.approx_equal(sigma_back, 1e-8));
}

void test_kirchhoff_relations() {
    auto F = Tensor2<3>::identity();
    F(0,0) = 1.3;
    double J = F.determinant();

    SymmetricTensor2<3> sigma(100.0, 200.0, 150.0, 0.0, 0.0, 0.0);

    auto tau = stress::kirchhoff_from_cauchy(sigma, F);
    // τ = J σ
    auto tau_expected = sigma * J;
    report("τ = Jσ", tau.approx_equal(tau_expected, 1e-10));

    auto sigma_back = stress::cauchy_from_kirchhoff(tau, F);
    report("σ = τ/J", sigma.approx_equal(sigma_back, 1e-10));
}

void test_first_piola_kirchhoff() {
    auto F = Tensor2<3>::identity();
    F(0,0) = 1.2; F(0,1) = 0.15;

    SymmetricTensor2<3> S(100.0, 200.0, 150.0, 10.0, 20.0, 30.0);

    auto P = stress::first_pk_from_2pk(S, F);
    // P = F · S
    auto P_expected = Tensor2<3>{(F.matrix() * S.matrix()).eval()};
    report("P = F·S", P.approx_equal(P_expected, 1e-10));

    // Roundtrip: P → S
    auto S_back = stress::second_pk_from_first(P, F);
    report("S ← F⁻¹·P", S.approx_equal(S_back, 1e-8));
}

void test_von_mises() {
    // Uniaxial stress: σ = diag(σ₁₁, 0, 0)
    SymmetricTensor2<3> sigma(100.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    double vm = stress::von_mises(sigma);
    // σ_eq = σ₁₁ for uniaxial
    report("von Mises uniaxial", approx(vm, 100.0, 1e-8));

    // Pure hydrostatic: σ = p I  → von Mises = 0
    SymmetricTensor2<3> hydro(100.0, 100.0, 100.0, 0.0, 0.0, 0.0);
    report("von Mises hydrostatic = 0", approx(stress::von_mises(hydro), 0.0, 1e-10));
}

void test_hydrostatic_stress() {
    SymmetricTensor2<3> sigma(60.0, 90.0, 150.0, 0.0, 0.0, 0.0);
    double p = stress::hydrostatic(sigma);
    report("hydrostatic = tr/3", approx(p, (60.0 + 90.0 + 150.0) / 3.0));
}

void test_mandel_stress_isotropic() {
    // For isotropic materials: Mandel = C·S
    // Simple case: F = diag(2, 1, 1) → C = diag(4, 1, 1)
    Tensor2<3> F = Tensor2<3>::identity();
    F(0,0) = 2.0;

    auto C = strain::right_cauchy_green(F);
    SymmetricTensor2<3> S(100.0, 50.0, 50.0, 0.0, 0.0, 0.0);

    auto Tm = stress::mandel(S, C);
    // Tm₁₁ = C₁₁·S₁₁ = 4·100 = 400
    report("Mandel Tm₁₁ = C₁₁·S₁₁", approx(Tm(0,0), 400.0));
    // Tm₂₂ = C₂₂·S₂₂ = 1·50 = 50
    report("Mandel Tm₂₂ = C₂₂·S₂₂", approx(Tm(1,1), 50.0));
}

// =============================================================================
//  2D specialisation tests
// =============================================================================

void test_2d_green_lagrange() {
    Tensor2<2> F = Tensor2<2>::identity();
    F(0,0) = 1.5; F(1,1) = 0.8;

    auto E = strain::green_lagrange(F);
    report("2D GL E₁₁", approx(E(0,0), 0.5 * (1.5*1.5 - 1.0)));
    report("2D GL E₂₂", approx(E(1,1), 0.5 * (0.8*0.8 - 1.0)));
    report("2D GL E₁₂", approx(E(0,1), 0.0));
}

void test_2d_voigt_mapping() {
    using ST = SymmetricTensor2<2>;
    report("2D voigt (0,0)→0", ST::voigt_index(0,0) == 0);
    report("2D voigt (1,1)→1", ST::voigt_index(1,1) == 1);
    report("2D voigt (0,1)→2", ST::voigt_index(0,1) == 2);
    report("2D voigt (1,0)→2", ST::voigt_index(1,0) == 2);
}

void test_2d_tensor4_isotropic() {
    // Plane strain-like: E=100, ν=0.25
    auto C = Tensor4<2>::isotropic_E_nu(100.0, 0.25);
    double lambda = 100.0 * 0.25 / ((1 + 0.25) * (1 - 0.5));
    double mu     = 100.0 / (2 * 1.25);

    report("2D T4 C(0,0)", approx(C(0,0), lambda + 2*mu));
    report("2D T4 C(0,1)", approx(C(0,1), lambda));
    report("2D T4 C(2,2)", approx(C(2,2), mu));
}

// =============================================================================
//  main
// =============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << " Continuum Tensor Module Tests\n";
    std::cout << "========================================\n\n";

    std::cout << "── Tensor2 ──\n";
    test_tensor2_identity_trace();
    test_tensor2_inverse();
    test_tensor2_transpose();
    test_tensor2_double_contract();
    test_tensor2_symmetric_skew_decomposition();
    test_tensor2_invariants();
    test_tensor2_deviatoric();
    test_tensor2_dyadic();

    std::cout << "\n── SymmetricTensor2 ──\n";
    test_sym_tensor2_voigt_mapping();
    test_sym_tensor2_multidim_subscript();
    test_subscript_vs_matrix_consistency();
    test_subscript_build_tensor_from_scratch();
    test_subscript_double_contract_manual();
    test_subscript_trace_manual();
    test_subscript_determinant_manual();
    test_subscript_inverse_verify();
    test_subscript_construct_strain_from_F();
    test_subscript_symmetry_write_enforcement();
    test_subscript_2d_strain_workflow();
    test_subscript_1d();
    test_subscript_cauchy_stress_transformation();
    test_sym_tensor2_identity();
    test_sym_tensor2_matrix_roundtrip();
    test_sym_tensor2_double_contract();
    test_sym_tensor2_engineering_voigt();
    test_sym_tensor2_invariants();
    test_sym_tensor2_deviatoric();
    test_sym_tensor2_spectral_decomposition();
    test_sym_tensor2_tensor_functions();
    test_sym_tensor2_inverse_closed_form();

    std::cout << "\n── Tensor4 ──\n";
    test_tensor4_isotropic_elasticity();
    test_tensor4_contraction();
    test_tensor4_identity_tensors();
    test_tensor4_outer_product();

    std::cout << "\n── Polar Decomposition ──\n";
    test_polar_decomposition_pure_rotation();
    test_polar_decomposition_stretch();
    test_polar_decomposition_general();
    test_polar_decomposition_2d();

    std::cout << "\n── Push-forward / Pull-back ──\n";
    test_push_forward_pull_back_roundtrip();

    std::cout << "\n── Strain Measures ──\n";
    test_green_lagrange_identity();
    test_green_lagrange_uniaxial();
    test_hencky_uniaxial();
    test_hencky_identity();
    test_seth_hill_recovers_green_lagrange();
    test_seth_hill_recovers_hencky();
    test_biot_strain();
    test_almansi_identity();
    test_infinitesimal_small_deformation();
    test_all_strains_converge_for_small_F();

    std::cout << "\n── Stress Measures ──\n";
    test_cauchy_2pk_roundtrip();
    test_kirchhoff_relations();
    test_first_piola_kirchhoff();
    test_von_mises();
    test_hydrostatic_stress();
    test_mandel_stress_isotropic();

    std::cout << "\n── 2D specialisations ──\n";
    test_2d_green_lagrange();
    test_2d_voigt_mapping();
    test_2d_tensor4_isotropic();

    std::cout << "\n========================================\n";
    std::cout << " Results: " << passed << " passed, " << failed << " failed\n";
    std::cout << "========================================\n";

    return failed > 0 ? 1 : 0;
}
