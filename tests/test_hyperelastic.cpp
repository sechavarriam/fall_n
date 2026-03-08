// =============================================================================
//  test_hyperelastic.cpp — Tests for hyperelastic models & relation (Phase 3)
// =============================================================================
//
//  All tests use SymmetricTensor2 / Tensor4 directly (Eigen-only, no PETSc).
//
//  Testing strategy:
//    1. Compile-time concept verification
//    2. SVK energy, stress, tangent (analytical)
//    3. SVK tangent matches isotropic Lamé tensor directly
//    4. Neo-Hookean stress at zero strain → S = 0
//    5. Neo-Hookean tangent at zero strain → matches SVK (isotropic Lamé)
//    6. Neo-Hookean stress = ∂W/∂E (numerical derivative)
//    7. Neo-Hookean tangent = ∂S/∂E (numerical derivative)
//    8. Neo-Hookean tangent major symmetry (hyperelastic)
//    9. Voigt tensor ↔ engineering round-trips (SymmetricTensor2)
//   10. Manufactured deformation states (uniaxial, shear, hydrostatic)
//   11. Small strain limit (Neo-Hookean ≈ SVK)
//   12. Model-level integration (direct SymmetricTensor2 API, no PETSc)
//   13. Energy non-negativity and compression resistance (Neo-Hookean)
//   14. Parameter access (from_E_nu, young_modulus, poisson_ratio)
//
//  Note: HyperelasticRelation.hh (the ConstitutiveRelation adapter) is NOT
//  tested here because it depends on Stress.hh → Vector.hh → PETSc.
//  The relation will be exercised in the PETSc-linked integration tests
//  (Phase 4).  Here we verify the models directly via SymmetricTensor2.
//
// =============================================================================

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>

#include "src/continuum/Continuum.hh"

using namespace continuum;

namespace {

constexpr bool approx(double a, double b, double tol = 1e-10) {
    return std::abs(a - b) <= tol;
}

int passed = 0, failed = 0;

void report(const char* name, bool ok) {
    if (ok) { ++passed; std::cout << "  PASS  " << name << "\n"; }
    else    { ++failed; std::cout << "  FAIL  " << name << "\n"; }
}

// ── Random small-deformation helpers ─────────────────────────────────────────

std::mt19937 rng{42};
std::uniform_real_distribution<double> dist{-0.03, 0.03};
double rand_small() { return dist(rng); }

/// Build a random SymmetricTensor2 for testing (small magnitude).
template <std::size_t dim>
SymmetricTensor2<dim> random_E() {
    SymmetricTensor2<dim> E;
    constexpr std::size_t N = voigt_size<dim>();
    for (std::size_t k = 0; k < N; ++k)
        E[k] = rand_small();
    return E;
}

// Reference Lamé parameters for tests:  E = 200, ν = 0.3
constexpr double E_ref  = 200.0;
constexpr double nu_ref = 0.3;
constexpr double lam_ref = E_ref * nu_ref / ((1.0 + nu_ref) * (1.0 - 2.0 * nu_ref));
constexpr double mu_ref  = E_ref / (2.0 * (1.0 + nu_ref));

} // anonymous namespace


// =============================================================================
//  1. Compile-time concept verification
// =============================================================================

void test_concepts() {
    // Models
    static_assert(HyperelasticModelConcept<SaintVenantKirchhoff<1>>);
    static_assert(HyperelasticModelConcept<SaintVenantKirchhoff<2>>);
    static_assert(HyperelasticModelConcept<SaintVenantKirchhoff<3>>);
    static_assert(HyperelasticModelConcept<CompressibleNeoHookean<1>>);
    static_assert(HyperelasticModelConcept<CompressibleNeoHookean<2>>);
    static_assert(HyperelasticModelConcept<CompressibleNeoHookean<3>>);

    // SVK has constant tangent, Neo-Hookean does not
    static_assert(SaintVenantKirchhoff<3>::constant_tangent);
    static_assert(!CompressibleNeoHookean<3>::constant_tangent);

    report("concepts_all_pass", true);
}


// =============================================================================
//  2. SVK energy, stress, tangent (analytical verification)
// =============================================================================

void test_svk_energy() {
    SaintVenantKirchhoff<3> svk{lam_ref, mu_ref};

    // Zero strain → zero energy
    auto E0 = SymmetricTensor2<3>::zero();
    report("SVK_energy_zero", approx(svk.energy(E0), 0.0));

    // Pure uniaxial: E = diag(0.1, 0, 0)
    SymmetricTensor2<3> E1{0.1, 0.0, 0.0, 0.0, 0.0, 0.0};
    // W = λ/2 (0.1)² + μ (0.1)² = (λ/2 + μ) · 0.01
    double expected = (0.5 * lam_ref + mu_ref) * 0.01;
    report("SVK_energy_uniaxial", approx(svk.energy(E1), expected, 1e-12));

    // Pure shear: E = (0, 0, 0, 0, 0, 0.05)  i.e. E₁₂ = 0.05
    SymmetricTensor2<3> E2{0.0, 0.0, 0.0, 0.0, 0.0, 0.05};
    // tr(E) = 0, E:E = 2·(0.05)² = 0.005 (factor 2 for off-diagonal)
    double exp2 = mu_ref * 0.005;
    report("SVK_energy_shear", approx(svk.energy(E2), exp2, 1e-12));
}

void test_svk_stress() {
    SaintVenantKirchhoff<3> svk{lam_ref, mu_ref};

    // Zero strain → zero stress
    auto E0 = SymmetricTensor2<3>::zero();
    auto S0 = svk.second_piola_kirchhoff(E0);
    double max_S0 = 0.0;
    for (std::size_t k = 0; k < 6; ++k)
        max_S0 = std::max(max_S0, std::abs(S0[k]));
    report("SVK_stress_zero", max_S0 < 1e-15);

    // Uniaxial: E = diag(0.1, 0, 0)
    SymmetricTensor2<3> E1{0.1, 0.0, 0.0, 0.0, 0.0, 0.0};
    auto S1 = svk.second_piola_kirchhoff(E1);
    // S₁₁ = λ·0.1 + 2μ·0.1 = (λ+2μ)·0.1
    // S₂₂ = λ·0.1, S₃₃ = λ·0.1, shears = 0
    report("SVK_stress_uniaxial_11", approx(S1[0], (lam_ref + 2.0 * mu_ref) * 0.1, 1e-12));
    report("SVK_stress_uniaxial_22", approx(S1[1], lam_ref * 0.1, 1e-12));
    report("SVK_stress_uniaxial_33", approx(S1[2], lam_ref * 0.1, 1e-12));
    report("SVK_stress_uniaxial_shear",
        approx(S1[3], 0.0) && approx(S1[4], 0.0) && approx(S1[5], 0.0));

    // Pure shear: E₁₂ = 0.05 → S₁₂ = 2μ·0.05
    SymmetricTensor2<3> E2{0.0, 0.0, 0.0, 0.0, 0.0, 0.05};
    auto S2 = svk.second_piola_kirchhoff(E2);
    report("SVK_stress_shear_12", approx(S2[5], 2.0 * mu_ref * 0.05, 1e-12));
    report("SVK_stress_shear_normals",
        approx(S2[0], 0.0, 1e-12) && approx(S2[1], 0.0, 1e-12) && approx(S2[2], 0.0, 1e-12));
}

void test_svk_tangent() {
    SaintVenantKirchhoff<3> svk{lam_ref, mu_ref};

    auto E = random_E<3>();
    auto C1 = svk.material_tangent(E);
    auto C2 = svk.material_tangent();  // constant: should be identical

    auto C_ref = Tensor4<3>::isotropic_lame(lam_ref, mu_ref);

    report("SVK_tangent_eq_isotropic",
        C1.approx_equal(C_ref, 1e-14) && C2.approx_equal(C_ref, 1e-14));

    // Major symmetry
    report("SVK_tangent_major_sym", C1.has_major_symmetry(1e-14));
}

void test_svk_all_dims() {
    // Verify SVK works for 1D and 2D as well
    {
        SaintVenantKirchhoff<1> svk1{lam_ref, mu_ref};
        SymmetricTensor2<1> E1{0.05};
        // W = (λ/2 + μ)·E² = (λ/2+μ)·0.0025
        double W = svk1.energy(E1);
        report("SVK_1D_energy", approx(W, (0.5 * lam_ref + mu_ref) * 0.0025, 1e-14));

        auto S = svk1.second_piola_kirchhoff(E1);
        report("SVK_1D_stress", approx(S[0], (lam_ref + 2.0 * mu_ref) * 0.05, 1e-14));
    }
    {
        SaintVenantKirchhoff<2> svk2{lam_ref, mu_ref};
        SymmetricTensor2<2> E2{0.03, 0.02, 0.01};  // {E11, E22, E12}
        auto S = svk2.second_piola_kirchhoff(E2);
        // S11 = λ(0.03+0.02) + 2μ·0.03
        double S11_exp = lam_ref * 0.05 + 2.0 * mu_ref * 0.03;
        report("SVK_2D_stress_11", approx(S[0], S11_exp, 1e-12));
    }
}


// =============================================================================
//  3. SVK tangent matches isotropic Lamé tensor
// =============================================================================

void test_svk_matches_linear_elastic() {
    SaintVenantKirchhoff<3> svk{lam_ref, mu_ref};

    // SVK Voigt matrix must equal the isotropic Lamé tangent
    auto C_svk = svk.material_tangent();
    auto C_ref = Tensor4<3>::isotropic_lame(lam_ref, mu_ref);

    report("SVK_tangent_eq_lame_tensor", C_svk.approx_equal(C_ref, 1e-14));

    // Verify Voigt matrices match as well
    double max_diff = (C_svk.voigt_matrix() - C_ref.voigt_matrix()).cwiseAbs().maxCoeff();
    report("SVK_voigt_matrix_eq_lame", max_diff < 1e-14);

    // Verify stress = C_mixed * engineering_strain for arbitrary E
    auto E = random_E<3>();
    auto S = svk.second_piola_kirchhoff(E);

    // The mixed tangent maps engineering Voigt ε_eng to tensor Voigt S.
    // For SVK: S = λtr(E)I + 2μE, and the mixed tangent is C_ref.
    // So S_voigt = C_ref.voigt_matrix() * ε_eng, where ε_eng = E.voigt_engineering().
    Eigen::Vector<double, 6> eps_eng = E.voigt_engineering();
    Eigen::Vector<double, 6> S_expected = C_ref.voigt_matrix() * eps_eng;

    double stress_err = 0.0;
    for (std::size_t k = 0; k < 6; ++k)
        stress_err = std::max(stress_err, std::abs(S[k] - S_expected(static_cast<Eigen::Index>(k))));
    report("SVK_stress_eq_Ceng_times_eps", stress_err < 1e-12);
}


// =============================================================================
//  4. Neo-Hookean: stress at zero strain → S = 0
// =============================================================================

void test_nh_zero_strain() {
    CompressibleNeoHookean<3> nh{lam_ref, mu_ref};
    auto E0 = SymmetricTensor2<3>::zero();

    // Energy at E=0: W = μ/2(3-3) - μ·0 + λ/2·0 = 0
    report("NH_energy_zero", approx(nh.energy(E0), 0.0, 1e-14));

    // Stress at E=0: S = μ(I-I) + λ·0·I = 0
    auto S0 = nh.second_piola_kirchhoff(E0);
    double max_S = 0.0;
    for (std::size_t k = 0; k < 6; ++k)
        max_S = std::max(max_S, std::abs(S0[k]));
    report("NH_stress_zero", max_S < 1e-14);
}


// =============================================================================
//  5. Neo-Hookean tangent at zero strain → isotropic Lamé
// =============================================================================

void test_nh_tangent_at_zero() {
    CompressibleNeoHookean<3> nh{lam_ref, mu_ref};
    auto E0 = SymmetricTensor2<3>::zero();

    auto C_nh  = nh.material_tangent(E0);
    auto C_ref = Tensor4<3>::isotropic_lame(lam_ref, mu_ref);

    report("NH_tangent_zero_eq_isotropic", C_nh.approx_equal(C_ref, 1e-12));

    // Also check 1D and 2D
    {
        CompressibleNeoHookean<1> nh1{lam_ref, mu_ref};
        auto C1 = nh1.material_tangent(SymmetricTensor2<1>::zero());
        report("NH_1D_tangent_zero", approx(C1(0, 0), lam_ref + 2.0 * mu_ref, 1e-12));
    }
    {
        CompressibleNeoHookean<2> nh2{lam_ref, mu_ref};
        auto C2 = nh2.material_tangent(SymmetricTensor2<2>::zero());
        auto Cr = Tensor4<2>::isotropic_lame(lam_ref, mu_ref);
        report("NH_2D_tangent_zero", C2.approx_equal(Cr, 1e-12));
    }
}


// =============================================================================
//  6. Neo-Hookean stress = ∂W/∂E (numerical derivative)
// =============================================================================

template <std::size_t dim>
void test_nh_stress_numerical_impl(const char* label) {
    CompressibleNeoHookean<dim> nh{lam_ref, mu_ref};
    auto E = random_E<dim>();
    auto S = nh.second_piola_kirchhoff(E);

    // Numerical derivative: S_k = ∂W/∂E_k
    // where E_k is the k-th tensor Voigt component.
    // For diagonal (k < dim): ∂W/∂E_kk directly.
    // For off-diagonal (k ≥ dim): ∂W/∂E_ij, but E_ij appears twice in E:E,
    //   so the derivative accounts for the symmetry factor.
    constexpr std::size_t N = voigt_size<dim>();
    const double h = 1e-7;
    double max_err = 0.0;

    for (std::size_t k = 0; k < N; ++k) {
        SymmetricTensor2<dim> E_p = E, E_m = E;
        E_p[k] += h;
        E_m[k] -= h;

        double W_p = nh.energy(E_p);
        double W_m = nh.energy(E_m);

        // ∂W/∂(voigt_k) via central differences
        double dW_num = (W_p - W_m) / (2.0 * h);

        // For off-diagonal components: the Voigt component E[k] = E_ij,
        // and the actual derivative ∂W/∂E_ij differs from ∂W/∂(voigt_k)
        // by a factor of 2 because E_ij = E_ji.
        // Specifically: dW/d(voigt_k) = S[k] for diagonal,
        //               dW/d(voigt_k) = 2·S[k] for off-diagonal.
        double S_expected = (k >= dim) ? dW_num / 2.0 : dW_num;

        max_err = std::max(max_err, std::abs(S[k] - S_expected));
    }
    report(label, max_err < 1e-6);
}

void test_nh_stress_numerical() {
    test_nh_stress_numerical_impl<1>("NH_stress_num_deriv_1D");
    test_nh_stress_numerical_impl<2>("NH_stress_num_deriv_2D");
    test_nh_stress_numerical_impl<3>("NH_stress_num_deriv_3D");
}


// =============================================================================
//  7. Neo-Hookean tangent = ∂S/∂E (numerical derivative)
// =============================================================================

template <std::size_t dim>
void test_nh_tangent_numerical_impl(const char* label) {
    CompressibleNeoHookean<dim> nh{lam_ref, mu_ref};
    auto E = random_E<dim>();
    auto CC = nh.material_tangent(E);

    // The material_tangent returns the mixed Voigt tangent:
    //   CC_mixed(A,B) = ∂S_A / ∂ε_eng_B
    //
    // For numerical verification, compute finite differences on the
    // compute_response of the HyperelasticRelation (which takes engineering
    // Voigt and returns tensor Voigt stress).
    //
    // However, it's simpler to verify at the model level using tensor Voigt:
    //   CC_raw(A,B) = ℂ_{i(A)j(A)k(B)l(B)}
    //
    // The relationship to the numerical derivative ∂S_A/∂E_B (tensor Voigt)
    // is:
    //   CC_raw(A,B) = ∂S_A / ∂E_B                  for B < dim (normal)
    //   CC_raw(A,B) = ½ · ∂S_A / ∂E_B              for B ≥ dim (shear)
    //
    // because perturbing voigt_[B] (off-diagonal) changes E_kl AND E_lk.

    constexpr std::size_t N = voigt_size<dim>();
    const double h = 1e-7;
    double max_err = 0.0;

    for (std::size_t B = 0; B < N; ++B) {
        SymmetricTensor2<dim> E_p = E, E_m = E;
        E_p[B] += h;
        E_m[B] -= h;

        auto S_p = nh.second_piola_kirchhoff(E_p);
        auto S_m = nh.second_piola_kirchhoff(E_m);

        for (std::size_t A = 0; A < N; ++A) {
            double dS_num = (S_p[A] - S_m[A]) / (2.0 * h);

            // The raw component ℂ_{ijkl} vs numerical ∂S_A/∂(voigt_B):
            //   For B normal: they're identical.
            //   For B shear:  ∂S_A/∂(voigt_B) = 2·ℂ_{..}  (because the tensor
            //     perturbation hits both E_kl and E_lk in E:E and tr(E),
            //     but voigt_B stores E_kl only once).
            double CC_analytical = CC(A, B);
            double CC_expected = (B >= dim) ? dS_num * 0.5 : dS_num;

            max_err = std::max(max_err, std::abs(CC_analytical - CC_expected));
        }
    }
    report(label, max_err < 1e-5);
}

void test_nh_tangent_numerical() {
    test_nh_tangent_numerical_impl<1>("NH_tangent_num_deriv_1D");
    test_nh_tangent_numerical_impl<2>("NH_tangent_num_deriv_2D");
    test_nh_tangent_numerical_impl<3>("NH_tangent_num_deriv_3D");
}


// =============================================================================
//  8. Neo-Hookean tangent major symmetry
// =============================================================================

void test_nh_tangent_symmetry() {
    CompressibleNeoHookean<3> nh{lam_ref, mu_ref};

    auto E = random_E<3>();
    auto CC = nh.material_tangent(E);
    report("NH_tangent_major_sym_3D", CC.has_major_symmetry(1e-12));

    CompressibleNeoHookean<2> nh2{lam_ref, mu_ref};
    auto E2 = random_E<2>();
    auto CC2 = nh2.material_tangent(E2);
    report("NH_tangent_major_sym_2D", CC2.has_major_symmetry(1e-12));

    CompressibleNeoHookean<1> nh1{lam_ref, mu_ref};
    auto E1 = random_E<1>();
    auto CC1 = nh1.material_tangent(E1);
    report("NH_tangent_major_sym_1D", CC1.has_major_symmetry(1e-12));
}


// =============================================================================
//  9. Voigt conversion round-trips
// =============================================================================

void test_voigt_conversions() {
    // SymmetricTensor2 tensor Voigt ↔ engineering Voigt round-trip (3D)
    SymmetricTensor2<3> E{0.03, 0.02, 0.01, 0.005, -0.004, 0.007};
    auto eng = E.voigt_engineering();

    // Engineering: normals same, shears doubled
    bool ok = true;
    for (std::size_t k = 0; k < 3; ++k)
        if (std::abs(eng(static_cast<Eigen::Index>(k)) - E[k]) > 1e-15) ok = false;
    for (std::size_t k = 3; k < 6; ++k)
        if (std::abs(eng(static_cast<Eigen::Index>(k)) - 2.0 * E[k]) > 1e-15) ok = false;
    report("voigt_tensor_to_eng_3D", ok);

    // Reconstruct tensor Voigt from engineering
    SymmetricTensor2<3> E_back;
    for (std::size_t k = 0; k < 3; ++k) E_back[k] = eng(static_cast<Eigen::Index>(k));
    for (std::size_t k = 3; k < 6; ++k) E_back[k] = eng(static_cast<Eigen::Index>(k)) * 0.5;
    ok = true;
    for (std::size_t k = 0; k < 6; ++k)
        if (std::abs(E_back[k] - E[k]) > 1e-15) ok = false;
    report("voigt_eng_round_trip_3D", ok);

    // 2D
    {
        SymmetricTensor2<2> E2{0.03, 0.02, 0.01};
        auto eng2 = E2.voigt_engineering();
        report("voigt_tensor_to_eng_2D",
            approx(eng2(0), 0.03) && approx(eng2(1), 0.02)
         && approx(eng2(2), 0.02));  // 2*0.01
    }

    // 1D
    {
        SymmetricTensor2<1> E1{0.05};
        auto eng1 = E1.voigt_engineering();
        report("voigt_tensor_to_eng_1D", approx(eng1(0), 0.05));
    }
}


// =============================================================================
//  10. Manufactured deformation states
// =============================================================================

void test_nh_uniaxial_extension() {
    // Uniaxial extension: F = diag(λ, 1, 1), λ = 1.1
    // C = diag(λ², 1, 1), E = diag((λ²-1)/2, 0, 0)
    const double stretch = 1.1;
    const double E11 = 0.5 * (stretch * stretch - 1.0);

    CompressibleNeoHookean<3> nh{lam_ref, mu_ref};
    SymmetricTensor2<3> E{E11, 0.0, 0.0, 0.0, 0.0, 0.0};

    auto S = nh.second_piola_kirchhoff(E);

    // Analytical: C = diag(1.21, 1, 1), C⁻¹ = diag(1/1.21, 1, 1)
    // J = 1.1, lnJ = ln(1.1)
    double C11 = stretch * stretch;
    double C11_inv = 1.0 / C11;
    double J = stretch;
    double lnJ = std::log(J);

    double S11_exp = mu_ref * (1.0 - C11_inv) + lam_ref * lnJ * C11_inv;
    double S22_exp = mu_ref * (1.0 - 1.0) + lam_ref * lnJ * 1.0;  // = λ·lnJ

    report("NH_uniaxial_S11", approx(S[0], S11_exp, 1e-10));
    report("NH_uniaxial_S22", approx(S[1], S22_exp, 1e-10));
    report("NH_uniaxial_S33", approx(S[2], S22_exp, 1e-10));
    report("NH_uniaxial_shears_zero",
        approx(S[3], 0.0, 1e-12) && approx(S[4], 0.0, 1e-12) && approx(S[5], 0.0, 1e-12));
}

void test_nh_hydrostatic() {
    // Hydrostatic expansion: F = λI, C = λ²I, E = (λ²-1)/2 I
    const double stretch = 1.05;
    const double Ekk = 0.5 * (stretch * stretch - 1.0);

    CompressibleNeoHookean<3> nh{lam_ref, mu_ref};
    SymmetricTensor2<3> E{Ekk, Ekk, Ekk, 0.0, 0.0, 0.0};

    auto S = nh.second_piola_kirchhoff(E);

    // C = λ²·I, C⁻¹ = (1/λ²)·I, J = λ³, lnJ = 3·ln(λ)
    double J = stretch * stretch * stretch;
    double lnJ = std::log(J);
    double C_inv_kk = 1.0 / (stretch * stretch);

    // S is isotropic: S = (μ(1 - 1/λ²) + λ·lnJ/λ²)·I
    double S_exp = mu_ref * (1.0 - C_inv_kk) + lam_ref * lnJ * C_inv_kk;

    report("NH_hydrostatic_S11", approx(S[0], S_exp, 1e-10));
    report("NH_hydrostatic_S22", approx(S[1], S_exp, 1e-10));
    report("NH_hydrostatic_S33", approx(S[2], S_exp, 1e-10));
    report("NH_hydrostatic_shears",
        approx(S[3], 0.0, 1e-12) && approx(S[4], 0.0, 1e-12) && approx(S[5], 0.0, 1e-12));
}

void test_nh_simple_shear() {
    // Simple shear: F = [[1, γ, 0],[0, 1, 0],[0, 0, 1]], γ = 0.1
    // C = [[1, γ, 0],[γ, 1+γ², 0],[0, 0, 1]]
    // E = [[0, γ/2, 0],[γ/2, γ²/2, 0],[0, 0, 0]]
    const double gamma = 0.1;

    CompressibleNeoHookean<3> nh{lam_ref, mu_ref};
    SymmetricTensor2<3> E{0.0, 0.5 * gamma * gamma, 0.0, 0.0, 0.0, 0.5 * gamma};
    // Voigt: {E11=0, E22=γ²/2, E33=0, E23=0, E13=0, E12=γ/2}

    auto S = nh.second_piola_kirchhoff(E);

    // Verify det(F) = 1 for simple shear → J = 1, lnJ = 0
    // C = 2E + I → det(C) = 1 (for simple shear)
    auto C_check = E * 2.0 + SymmetricTensor2<3>::identity();
    report("NH_shear_detC_eq_1", approx(C_check.determinant(), 1.0, 1e-10));

    // S should have S₁₂ ≠ 0 (shear response)
    report("NH_shear_S12_nonzero", std::abs(S[5]) > 1e-6);

    // Since J=1 → lnJ=0, S = μ(I - C⁻¹) + 0 = μ(I - C⁻¹)
    auto C = E * 2.0 + SymmetricTensor2<3>::identity();
    auto C_inv = C.inverse();
    auto I = SymmetricTensor2<3>::identity();
    auto S_expected = (I - C_inv) * mu_ref;

    double max_err = 0.0;
    for (std::size_t k = 0; k < 6; ++k)
        max_err = std::max(max_err, std::abs(S[k] - S_expected[k]));
    report("NH_shear_S_analytical", max_err < 1e-10);
}


// =============================================================================
//  11. Small strain limit: Neo-Hookean ≈ SVK
// =============================================================================

template <std::size_t dim>
void test_small_strain_limit_impl(const char* label) {
    SaintVenantKirchhoff<dim> svk{lam_ref, mu_ref};
    CompressibleNeoHookean<dim> nh{lam_ref, mu_ref};

    // Very small strain (1e-8 scale)
    constexpr std::size_t N = voigt_size<dim>();
    SymmetricTensor2<dim> E;
    for (std::size_t k = 0; k < N; ++k)
        E[k] = 1e-8 * (static_cast<double>(k) + 1.0);

    auto S_svk = svk.second_piola_kirchhoff(E);
    auto S_nh  = nh.second_piola_kirchhoff(E);

    double max_diff = 0.0;
    for (std::size_t k = 0; k < N; ++k)
        max_diff = std::max(max_diff, std::abs(S_svk[k] - S_nh[k]));

    // Difference should be O(E²) ≈ 1e-16 (amplified by Lamé params)
    report(label, max_diff < 1e-10);
}

void test_small_strain_limit() {
    test_small_strain_limit_impl<1>("small_strain_SVK_eq_NH_1D");
    test_small_strain_limit_impl<2>("small_strain_SVK_eq_NH_2D");
    test_small_strain_limit_impl<3>("small_strain_SVK_eq_NH_3D");
}


// =============================================================================
//  12. HyperelasticRelation integration test
// =============================================================================

void test_relation_integration() {
    // Test model-level compute at the SymmetricTensor2 API level,
    // simulating what the relation adapter would do (eng → tensor → S).

    CompressibleNeoHookean<3> nh{lam_ref, mu_ref};

    // Build a tensor Voigt strain from uniaxial extension
    SymmetricTensor2<3> E{0.105, 0.0, 0.0, 0.0, 0.0, 0.0};

    auto S = nh.second_piola_kirchhoff(E);
    auto C = nh.material_tangent(E);

    // Verify S is non-zero and physically reasonable
    report("model_NH_S11_positive", S[0] > 0.0);

    // Verify tangent has major symmetry
    report("model_NH_tangent_symmetric", C.has_major_symmetry(1e-12));

    // Verify tangent (numerical) matches tangent (analytical)
    // Using tensor Voigt perturbations and accounting for the
    // off-diagonal factor (same as test 7 above)
    constexpr std::size_t N = 6;
    const double h = 1e-7;
    double max_err = 0.0;
    for (std::size_t B = 0; B < N; ++B) {
        SymmetricTensor2<3> E_p = E, E_m = E;
        E_p[B] += h;
        E_m[B] -= h;

        auto S_p = nh.second_piola_kirchhoff(E_p);
        auto S_m = nh.second_piola_kirchhoff(E_m);

        for (std::size_t A = 0; A < N; ++A) {
            double dS_num = (S_p[A] - S_m[A]) / (2.0 * h);
            double CC_expected = (B >= 3) ? dS_num * 0.5 : dS_num;
            max_err = std::max(max_err, std::abs(C(A, B) - CC_expected));
        }
    }
    report("model_NH_tangent_num_verify", max_err < 1e-5);

    // Also test SVK energy
    SaintVenantKirchhoff<3> svk{lam_ref, mu_ref};
    double W = svk.energy(E);
    double W_exp = (0.5 * lam_ref + mu_ref) * 0.105 * 0.105;
    report("model_SVK_energy", approx(W, W_exp, 1e-12));
}

void test_relation_1D_2D() {
    // 1D: SVK model directly
    {
        SaintVenantKirchhoff<1> svk1{lam_ref, mu_ref};
        SymmetricTensor2<1> E1{0.05};
        auto S1 = svk1.second_piola_kirchhoff(E1);
        report("model_SVK_1D_stress", approx(S1[0], (lam_ref + 2.0 * mu_ref) * 0.05, 1e-12));
    }

    // 2D: NH model directly
    {
        CompressibleNeoHookean<2> nh2{lam_ref, mu_ref};
        SymmetricTensor2<2> E2{0.02, 0.01, 0.0025};  // tensor Voigt: E12 = γ/2

        auto S2 = nh2.second_piola_kirchhoff(E2);
        (void)S2;  // verify it compiles; value checked via tangent symmetry
        auto C2 = nh2.material_tangent(E2);

        report("model_NH_2D_tangent_sym", C2.has_major_symmetry(1e-12));
    }
}


// =============================================================================
//  13. Energy: non-negativity and compression resistance (Neo-Hookean)
// =============================================================================

void test_nh_energy_properties() {
    CompressibleNeoHookean<3> nh{lam_ref, mu_ref};

    // Energy at zero deformation = 0
    report("NH_energy_zero_deformation",
        approx(nh.energy(SymmetricTensor2<3>::zero()), 0.0, 1e-14));

    // Energy under moderate extension should be positive
    SymmetricTensor2<3> E_ext{0.1, 0.0, 0.0, 0.0, 0.0, 0.0};
    report("NH_energy_positive_extension", nh.energy(E_ext) > 0.0);

    // Energy under moderate compression (but still J > 0)
    // E11 = -0.05 → C11 = 0.9, J = √0.9 > 0
    SymmetricTensor2<3> E_comp{-0.05, 0.0, 0.0, 0.0, 0.0, 0.0};
    report("NH_energy_positive_compression", nh.energy(E_comp) > 0.0);

    // Energy should increase with deformation magnitude
    double W1 = nh.energy(SymmetricTensor2<3>{0.01, 0.0, 0.0, 0.0, 0.0, 0.0});
    double W2 = nh.energy(SymmetricTensor2<3>{0.1, 0.0, 0.0, 0.0, 0.0, 0.0});
    report("NH_energy_increases_with_strain", W2 > W1);
}


// =============================================================================
//  14. Parameter access
// =============================================================================

void test_parameter_access() {
    auto svk = SaintVenantKirchhoff<3>::from_E_nu(E_ref, nu_ref);
    report("SVK_from_E_nu_lambda", approx(svk.lambda(), lam_ref, 1e-10));
    report("SVK_from_E_nu_mu", approx(svk.mu(), mu_ref, 1e-10));
    report("SVK_young_modulus", approx(svk.young_modulus(), E_ref, 1e-10));
    report("SVK_poisson_ratio", approx(svk.poisson_ratio(), nu_ref, 1e-10));

    auto nh = CompressibleNeoHookean<3>::from_E_nu(E_ref, nu_ref);
    report("NH_from_E_nu_lambda", approx(nh.lambda(), lam_ref, 1e-10));
    report("NH_from_E_nu_mu", approx(nh.mu(), mu_ref, 1e-10));
    report("NH_young_modulus", approx(nh.young_modulus(), E_ref, 1e-10));
    report("NH_poisson_ratio", approx(nh.poisson_ratio(), nu_ref, 1e-10));
}


// =============================================================================
//  main
// =============================================================================

int main() {
    std::cout << "\n══════════════════════════════════════════════════════\n"
              << "  Hyperelastic Model Tests (Phase 3)\n"
              << "══════════════════════════════════════════════════════\n\n";

    // 1. Concepts
    test_concepts();

    // 2. SVK
    std::cout << "\n── SVK energy ──\n";
    test_svk_energy();

    std::cout << "\n── SVK stress ──\n";
    test_svk_stress();

    std::cout << "\n── SVK tangent ──\n";
    test_svk_tangent();
    test_svk_all_dims();

    // 3. SVK matches isotropic Lamé
    std::cout << "\n── SVK ≡ isotropic Lamé ──\n";
    test_svk_matches_linear_elastic();

    // 4-5. Neo-Hookean at E=0
    std::cout << "\n── Neo-Hookean at zero strain ──\n";
    test_nh_zero_strain();
    test_nh_tangent_at_zero();

    // 6. NH stress numerical
    std::cout << "\n── NH stress = ∂W/∂E ──\n";
    test_nh_stress_numerical();

    // 7. NH tangent numerical
    std::cout << "\n── NH tangent = ∂S/∂E ──\n";
    test_nh_tangent_numerical();

    // 8. NH tangent symmetry
    std::cout << "\n── NH tangent symmetry ──\n";
    test_nh_tangent_symmetry();

    // 9. Voigt conversions
    std::cout << "\n── Voigt conversions ──\n";
    test_voigt_conversions();

    // 10. Manufactured
    std::cout << "\n── Manufactured deformations ──\n";
    test_nh_uniaxial_extension();
    test_nh_hydrostatic();
    test_nh_simple_shear();

    // 11. Small strain limit
    std::cout << "\n── Small strain limit ──\n";
    test_small_strain_limit();

    // 12. Model-level integration
    std::cout << "\n── Model integration ──\n";
    test_relation_integration();
    test_relation_1D_2D();

    // 13. Energy properties
    std::cout << "\n── Energy properties ──\n";
    test_nh_energy_properties();

    // 14. Parameter access
    std::cout << "\n── Parameter access ──\n";
    test_parameter_access();

    // Summary
    std::cout << "\n══════════════════════════════════════════════════════\n"
              << "  Results: " << passed << " passed, " << failed << " failed"
              << " (total " << (passed + failed) << ")\n"
              << "══════════════════════════════════════════════════════\n\n";

    return failed > 0 ? 1 : 0;
}
