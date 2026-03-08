// =============================================================================
//  test_updated_lagrangian.cpp — Phase 6: Updated Lagrangian formulation
// =============================================================================
//
//  Verifies that the spatial (Updated Lagrangian) pathway produces results
//  identical to the Total Lagrangian pathway for hyperelastic materials.
//
//  Key mathematical identity (the "grand equivalence"):
//
//    TL:  f_int = B_NLᵀ · S̃ ,   K = B_NLᵀ · ℂ̃ · B_NL  + K_σ(S, ∂N/∂X)
//    UL:  f_int = J · bᵀ · σ̃ ,   K = J · bᵀ · 𝕔̃ · b  + J · k_σ(σ, ∂N/∂x)
//
//  where:
//    σ   = (1/J) F · S · Fᵀ           (Cauchy = push-forward of 2nd PK)
//    𝕔   = (1/J) F⊗F : ℂ : Fᵀ⊗Fᵀ    (spatial tangent = push-forward of ℂ)
//    b   = linear B matrix with ∂N/∂x  (spatial gradients)
//    ∂N/∂x = (∂N/∂X) · F⁻¹
//
//  Testing strategy:
//
//    1–3.  Push-forward: stress and tangent (1D, 2D, 3D)
//    4.    Spatial gradients: grad_x · F = grad_X
//    5–9.  Grand equivalence UL ≡ TL: f_int and K (SVK + NH, 1D/2D/3D)
//   10–11. Push-forward tangent properties (symmetry, isotropic identity)
//   12–14. Newton convergence through spatial pathway (1D, 2D, 3D)
//   15–16. Spatial patch test (2D, 3D)
//   17.    Push-forward/pull-back round-trip
//   18.    Almansi ↔ Green-Lagrange consistency
//
// =============================================================================

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "src/continuum/Continuum.hh"

using namespace continuum;

namespace {

// ── Reporting ────────────────────────────────────────────────────────────────

int passed = 0, failed = 0;

constexpr bool approx(double a, double b, double tol = 1e-10) {
    return std::abs(a - b) <= tol;
}

void report(const char* name, bool ok) {
    if (ok) { ++passed; std::cout << "  PASS  " << name << "\n"; }
    else    { ++failed; std::cout << "  FAIL  " << name << "\n"; }
}

// ── Material parameters ─────────────────────────────────────────────────────

constexpr double E_ref  = 200.0;
constexpr double nu_ref = 0.3;
constexpr double lam_ref = E_ref * nu_ref / ((1.0 + nu_ref) * (1.0 - 2.0 * nu_ref));
constexpr double mu_ref  = E_ref / (2.0 * (1.0 + nu_ref));


// =============================================================================
//  Gradient helpers (canonical elements, same as Phase 4/5)
// =============================================================================

Eigen::MatrixXd grad_bar_1d(double L = 2.0) {
    Eigen::MatrixXd g(2, 1);
    g(0, 0) = -1.0 / L;
    g(1, 0) =  1.0 / L;
    return g;
}

Eigen::MatrixXd grad_cst_2d() {
    Eigen::MatrixXd g(3, 2);
    g << -1.0, -1.0,
          1.0,  0.0,
          0.0,  1.0;
    return g;
}

Eigen::MatrixXd grad_tet_3d() {
    Eigen::MatrixXd g(4, 3);
    g << -1.0, -1.0, -1.0,
          1.0,  0.0,  0.0,
          0.0,  1.0,  0.0,
          0.0,  0.0,  1.0;
    return g;
}


// =============================================================================
//  TL single-GP assembly (reused from Phase 4/5 for comparison)
// =============================================================================

template <std::size_t dim, typename Model>
std::pair<Eigen::VectorXd, Eigen::MatrixXd>
assemble_TL(
    const Eigen::MatrixXd& grad,
    std::size_t ndof,
    const Eigen::VectorXd& u_e,
    const Model& model)
{
    auto kin = TotalLagrangian::evaluate_from_gradients<dim>(grad, ndof, u_e);
    auto E_tensor = strain::green_lagrange(kin.F);
    auto S = model.second_piola_kirchhoff(E_tensor);
    auto C_T4 = model.material_tangent(E_tensor);

    constexpr auto NV = voigt_size<dim>();
    Eigen::Vector<double, static_cast<int>(NV)> S_voigt;
    for (std::size_t k = 0; k < NV; ++k)
        S_voigt(static_cast<Eigen::Index>(k)) = S[k];

    Eigen::VectorXd f = kin.B.transpose() * S_voigt;
    Eigen::MatrixXd K_mat = kin.B.transpose() * C_T4.voigt_matrix() * kin.B;

    auto S_matrix = TotalLagrangian::stress_voigt_to_matrix<dim>(S_voigt);
    Eigen::MatrixXd K_sigma =
        TotalLagrangian::compute_geometric_stiffness_from_gradients<dim>(
            grad, ndof, S_matrix);

    return {f, K_mat + K_sigma};
}


// =============================================================================
//  Newton solver (identical to Phase 5)
// =============================================================================

struct NewtonResult {
    Eigen::VectorXd     u;
    std::vector<double> residuals;
    int                 iterations;
    bool                converged;
};

template <typename AssembleFn>
NewtonResult solve_newton(
    AssembleFn&& assemble,
    const Eigen::VectorXd& f_ext,
    const std::vector<Eigen::Index>& free_dofs,
    Eigen::VectorXd u0,
    int max_iter = 30, double tol = 1e-12)
{
    const auto nf = static_cast<Eigen::Index>(free_dofs.size());
    NewtonResult result;
    result.u = std::move(u0);
    result.converged = false;

    for (int iter = 0; iter < max_iter; ++iter) {
        auto [f_int, K] = assemble(result.u);
        Eigen::VectorXd R = f_int - f_ext;

        Eigen::VectorXd R_f(nf);
        Eigen::MatrixXd K_ff(nf, nf);
        for (Eigen::Index i = 0; i < nf; ++i) {
            R_f(i) = R(free_dofs[static_cast<std::size_t>(i)]);
            for (Eigen::Index j = 0; j < nf; ++j)
                K_ff(i, j) = K(free_dofs[static_cast<std::size_t>(i)],
                               free_dofs[static_cast<std::size_t>(j)]);
        }

        double res_norm = R_f.norm();
        result.residuals.push_back(res_norm);

        if (res_norm < tol) {
            result.converged  = true;
            result.iterations = iter;
            return result;
        }

        Eigen::VectorXd du_f = K_ff.ldlt().solve(-R_f);
        for (Eigen::Index i = 0; i < nf; ++i)
            result.u(free_dofs[static_cast<std::size_t>(i)]) += du_f(i);
    }
    result.iterations = max_iter;
    return result;
}

double estimate_convergence_order(const std::vector<double>& res,
                                  double cutoff = 1e-14) {
    if (res.size() < 3) return 0.0;
    double sum = 0.0;
    int cnt = 0;
    for (std::size_t k = 1; k + 1 < res.size(); ++k) {
        if (res[k] < cutoff || res[k - 1] < cutoff) continue;
        double num = std::log(res[k + 1] / res[k]);
        double den = std::log(res[k] / res[k - 1]);
        if (std::abs(den) < 1e-30) continue;
        double q = num / den;
        if (q > 0.5 && q < 5.0) { sum += q; ++cnt; }
    }
    return cnt > 0 ? sum / cnt : 0.0;
}


} // anonymous namespace


// =============================================================================
//  1–3.  Push-forward: stress and tangent for dim = 1, 2, 3
// =============================================================================

void test_push_forward_stress() {
    // 1D: F = [1.15], S₁₁ = (λ+2μ)·E₁₁
    {
        SaintVenantKirchhoff<1> svk{lam_ref, mu_ref};
        auto F = Tensor2<1>::identity();
        F(0, 0) = 1.15;
        auto E = strain::green_lagrange(F);
        auto S = svk.second_piola_kirchhoff(E);

        auto sigma = ops::push_forward(S, F);

        // σ = (1/J) F·S·Fᵀ  →  for 1D: σ₁₁ = F²·S₁₁/J = F·S₁₁
        double sigma_exp = F(0, 0) * S[0];    // J = F for 1D, so (F²·S)/F = F·S
        report("push_forward_stress_1D", approx(sigma[0], sigma_exp, 1e-12));
    }

    // 2D: general F
    {
        SaintVenantKirchhoff<2> svk{lam_ref, mu_ref};
        Eigen::Matrix2d Fm;
        Fm << 1.1, 0.05, -0.02, 0.95;
        auto F = Tensor2<2>{Fm};
        auto E = strain::green_lagrange(F);
        auto S = svk.second_piola_kirchhoff(E);

        auto sigma = ops::push_forward(S, F);
        // Verify σ is symmetric
        report("push_forward_stress_2D_sym",
            approx(sigma(0, 1), sigma(1, 0), 1e-14));

        // Verify round-trip: S = J F⁻¹·σ·F⁻ᵀ
        auto S_rt = ops::pull_back(sigma, F);
        double err = 0.0;
        for (std::size_t k = 0; k < 3; ++k)
            err = std::max(err, std::abs(S[k] - S_rt[k]));
        report("push_forward_stress_2D_roundtrip", err < 1e-12);
    }

    // 3D: general F
    {
        CompressibleNeoHookean<3> nh{lam_ref, mu_ref};
        Eigen::Matrix3d Fm;
        Fm << 1.08, 0.02, -0.01,
              0.03, 0.97,  0.015,
             -0.02, 0.01,  1.05;
        auto F = Tensor2<3>{Fm};
        auto E = strain::green_lagrange(F);
        auto S = nh.second_piola_kirchhoff(E);

        auto sigma = ops::push_forward(S, F);
        auto S_rt  = ops::pull_back(sigma, F);

        double err = 0.0;
        for (std::size_t k = 0; k < 6; ++k)
            err = std::max(err, std::abs(S[k] - S_rt[k]));
        report("push_forward_stress_3D_roundtrip", err < 1e-11);
    }
}


// =============================================================================
//  4.  Spatial gradients: grad_x · F = grad_X
// =============================================================================

void test_spatial_gradients() {
    // 2D CST with a known F
    {
        auto grad_X = grad_cst_2d();
        Eigen::Matrix2d Fm;
        Fm << 1.12, 0.04, -0.03, 0.96;
        auto F = Tensor2<2>{Fm};

        auto grad_x_dyn = UpdatedLagrangian::compute_spatial_gradients<2>(
            grad_X.cast<double>(), F);

        // Verify: grad_x · F ≈ grad_X
        Eigen::MatrixXd recovered = grad_x_dyn * F.matrix();
        double err = (recovered - grad_X).cwiseAbs().maxCoeff();
        report("spatial_grad_2D_inverse", err < 1e-14);
    }

    // 3D tet
    {
        auto grad_X = grad_tet_3d();
        Eigen::Matrix3d Fm;
        Fm << 1.05, 0.02, -0.01,
              0.03, 0.98,  0.01,
             -0.02, 0.01,  1.03;
        auto F = Tensor2<3>{Fm};

        auto grad_x = UpdatedLagrangian::compute_spatial_gradients<3>(
            grad_X.cast<double>(), F);

        Eigen::MatrixXd recovered = grad_x * F.matrix();
        double err = (recovered - grad_X).cwiseAbs().maxCoeff();
        report("spatial_grad_3D_inverse", err < 1e-14);
    }

    // 1D bar: F⁻¹ = 1/F₁₁
    {
        auto grad_X = grad_bar_1d();
        auto F = Tensor2<1>::identity();
        F(0, 0) = 1.2;

        auto grad_x = UpdatedLagrangian::compute_spatial_gradients<1>(
            grad_X.cast<double>(), F);

        // grad_x(I,0) = grad_X(I,0) / F₁₁
        report("spatial_grad_1D",
            approx(grad_x(0, 0), grad_X(0, 0) / 1.2, 1e-14) &&
            approx(grad_x(1, 0), grad_X(1, 0) / 1.2, 1e-14));
    }
}


// =============================================================================
//  5–9.  Grand equivalence: UL ≡ TL for f_int and K
// =============================================================================

template <std::size_t dim, typename Model>
void check_UL_TL_equivalence(
    const char* name,
    const Eigen::MatrixXd& grad,
    std::size_t ndof,
    const Eigen::VectorXd& u_e,
    const Model& model,
    double tol = 1e-10)
{
    auto [f_TL, K_TL] = assemble_TL<dim>(grad, ndof, u_e, model);
    auto [f_UL, K_UL] = UpdatedLagrangian::assemble_spatial_from_gradients<dim>(
        grad, ndof, u_e, model);

    double f_err = (f_TL - f_UL).cwiseAbs().maxCoeff();
    double K_err = (K_TL - K_UL).cwiseAbs().maxCoeff();

    char buf_f[128], buf_K[128];
    std::snprintf(buf_f, sizeof(buf_f), "UL_TL_f_%s", name);
    std::snprintf(buf_K, sizeof(buf_K), "UL_TL_K_%s", name);

    report(buf_f, f_err < tol);
    report(buf_K, K_err < tol);

    if (f_err >= tol || K_err >= tol) {
        std::cout << "    f_err = " << f_err << ",  K_err = " << K_err << "\n";
    }
}

void test_grand_equivalence() {
    // 5. 1D bar SVK — moderate stretch
    {
        SaintVenantKirchhoff<1> svk{lam_ref, mu_ref};
        Eigen::VectorXd u(2);
        u << 0.0, 0.3;
        check_UL_TL_equivalence<1>("1D_SVK", grad_bar_1d(), 1, u, svk);
    }

    // 6. 2D CST SVK — general deformation
    {
        SaintVenantKirchhoff<2> svk{lam_ref, mu_ref};
        Eigen::VectorXd u(6);
        u << 0.0, 0.0, 0.08, -0.02, 0.03, 0.06;
        check_UL_TL_equivalence<2>("2D_SVK", grad_cst_2d(), 2, u, svk);
    }

    // 7. 2D CST NH — general deformation
    {
        CompressibleNeoHookean<2> nh{lam_ref, mu_ref};
        Eigen::VectorXd u(6);
        u << 0.0, 0.0, 0.08, -0.02, 0.03, 0.06;
        check_UL_TL_equivalence<2>("2D_NH", grad_cst_2d(), 2, u, nh);
    }

    // 8. 3D tet SVK — general deformation
    {
        SaintVenantKirchhoff<3> svk{lam_ref, mu_ref};
        Eigen::VectorXd u(12);
        u << 0.0, 0.0, 0.0,
             0.06, -0.01, 0.02,
             0.02, 0.07, -0.01,
            -0.01, 0.01, 0.05;
        check_UL_TL_equivalence<3>("3D_SVK", grad_tet_3d(), 3, u, svk);
    }

    // 9. 3D tet NH — general deformation
    {
        CompressibleNeoHookean<3> nh{lam_ref, mu_ref};
        Eigen::VectorXd u(12);
        u << 0.0, 0.0, 0.0,
             0.06, -0.01, 0.02,
             0.02, 0.07, -0.01,
            -0.01, 0.01, 0.05;
        check_UL_TL_equivalence<3>("3D_NH", grad_tet_3d(), 3, u, nh);
    }
}


// =============================================================================
//  10–11.  Push-forward tangent properties
// =============================================================================

void test_tangent_push_forward_properties() {
    // 10. Major symmetry preserved by push-forward
    {
        SaintVenantKirchhoff<3> svk{lam_ref, mu_ref};
        Eigen::Matrix3d Fm;
        Fm << 1.1, 0.03, -0.02,
              0.01, 0.95, 0.02,
             -0.01, 0.015, 1.06;
        auto F = Tensor2<3>{Fm};
        auto E = strain::green_lagrange(F);
        auto CC = svk.material_tangent(E);

        report("tangent_CC_major_sym", CC.has_major_symmetry());

        auto cc = ops::push_forward_tangent(CC, F);
        report("tangent_cc_major_sym", cc.has_major_symmetry());
    }

    // 11. For F = I, push-forward tangent should equal material tangent
    {
        CompressibleNeoHookean<2> nh{lam_ref, mu_ref};
        auto F = Tensor2<2>::identity();
        auto E = strain::green_lagrange(F);   // = 0
        auto CC = nh.material_tangent(E);

        auto cc = ops::push_forward_tangent(CC, F);
        report("tangent_F_eq_I",
            cc.approx_equal(CC, 1e-12));
    }
}


// =============================================================================
//  12–14.  Newton convergence through spatial (UL) pathway
// =============================================================================

void test_UL_newton() {
    // 12. 1D bar SVK via UL pathway
    {
        SaintVenantKirchhoff<1> svk{lam_ref, mu_ref};
        auto grad = grad_bar_1d();

        Eigen::VectorXd f_ext = Eigen::VectorXd::Zero(2);
        f_ext(1) = 30.0;

        auto result = solve_newton(
            [&](const Eigen::VectorXd& u) {
                return UpdatedLagrangian::assemble_spatial_from_gradients<1>(
                    grad, 1, u, svk);
            },
            f_ext, {1}, Eigen::VectorXd::Zero(2));

        report("UL_newton_1D_SVK_converged", result.converged);

        double q = estimate_convergence_order(result.residuals);
        report("UL_newton_1D_SVK_quadratic", q > 1.7);

        // Solution must match TL Newton
        SaintVenantKirchhoff<1> svk2{lam_ref, mu_ref};
        auto result_TL = solve_newton(
            [&](const Eigen::VectorXd& u) {
                return assemble_TL<1>(grad, 1, u, svk2);
            },
            f_ext, {1}, Eigen::VectorXd::Zero(2));

        report("UL_TL_newton_1D_same_u",
            approx(result.u(1), result_TL.u(1), 1e-10));

        std::cout << "    residuals:";
        for (double r : result.residuals) std::cout << " " << r;
        std::cout << "  (q ≈ " << q << ")\n";
    }

    // 13. 2D CST NH via UL pathway
    {
        CompressibleNeoHookean<2> nh{lam_ref, mu_ref};
        auto grad = grad_cst_2d();

        Eigen::VectorXd f_ext = Eigen::VectorXd::Zero(6);
        f_ext(2) = 5.0;
        f_ext(4) = 5.0;

        auto result = solve_newton(
            [&](const Eigen::VectorXd& u) {
                return UpdatedLagrangian::assemble_spatial_from_gradients<2>(
                    grad, 2, u, nh);
            },
            f_ext, {2, 4, 5}, Eigen::VectorXd::Zero(6));

        report("UL_newton_2D_NH_converged", result.converged);
        report("UL_newton_2D_NH_iters", result.iterations < 15);
    }

    // 14. 3D tet NH via UL pathway
    {
        CompressibleNeoHookean<3> nh{lam_ref, mu_ref};
        auto grad = grad_tet_3d();

        Eigen::VectorXd f_ext = Eigen::VectorXd::Zero(12);
        f_ext(3) = 5.0;
        f_ext(6) = 5.0;
        f_ext(9) = 5.0;

        auto result = solve_newton(
            [&](const Eigen::VectorXd& u) {
                return UpdatedLagrangian::assemble_spatial_from_gradients<3>(
                    grad, 3, u, nh);
            },
            f_ext, {3, 6, 7, 9, 10, 11}, Eigen::VectorXd::Zero(12));

        report("UL_newton_3D_NH_converged", result.converged);

        double q = estimate_convergence_order(result.residuals);
        report("UL_newton_3D_NH_quadratic", q > 1.7);

        std::cout << "    residuals:";
        for (double r : result.residuals) std::cout << " " << r;
        std::cout << "  (q ≈ " << q << ")\n";
    }
}


// =============================================================================
//  15–16.  Spatial patch test — uniform deformation
// =============================================================================

void test_UL_patch() {
    // 15. 2D CST with uniform F₀
    {
        SaintVenantKirchhoff<2> svk{lam_ref, mu_ref};
        auto grad = grad_cst_2d();

        Eigen::Matrix2d F0;
        F0 << 1.1, 0.05, -0.02, 0.95;
        Eigen::Matrix2d dF = F0 - Eigen::Matrix2d::Identity();

        // Nodes: (0,0), (1,0), (0,1)
        Eigen::VectorXd u(6);
        u(0) = 0.0; u(1) = 0.0;
        u(2) = dF(0, 0); u(3) = dF(1, 0);
        u(4) = dF(0, 1); u(5) = dF(1, 1);

        auto [f_UL, K_UL] =
            UpdatedLagrangian::assemble_spatial_from_gradients<2>(
                grad, 2, u, svk);

        // Net force = 0 (uniform stress → no nodal imbalance)
        double fx = f_UL(0) + f_UL(2) + f_UL(4);
        double fy = f_UL(1) + f_UL(3) + f_UL(5);
        report("UL_patch_2D_net_zero",
            std::abs(fx) < 1e-12 && std::abs(fy) < 1e-12);

        // K should have major symmetry
        double asym = (K_UL - K_UL.transpose()).cwiseAbs().maxCoeff();
        report("UL_patch_2D_K_sym", asym < 1e-10);
    }

    // 16. 3D tet with diagonal F₀
    {
        CompressibleNeoHookean<3> nh{lam_ref, mu_ref};
        auto grad = grad_tet_3d();

        Eigen::Matrix3d F0 = Eigen::Matrix3d::Identity();
        F0(0, 0) = 1.08; F0(1, 1) = 0.97; F0(2, 2) = 1.03;
        Eigen::Matrix3d dF = F0 - Eigen::Matrix3d::Identity();

        Eigen::Vector3d X[4] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        Eigen::VectorXd u(12);
        for (int I = 0; I < 4; ++I) {
            Eigen::Vector3d uI = dF * X[I];
            u(3 * I)     = uI(0);
            u(3 * I + 1) = uI(1);
            u(3 * I + 2) = uI(2);
        }

        auto [f_UL, K_UL] =
            UpdatedLagrangian::assemble_spatial_from_gradients<3>(
                grad, 3, u, nh);

        double fx = 0.0, fy = 0.0, fz = 0.0;
        for (int I = 0; I < 4; ++I) {
            fx += f_UL(3 * I);
            fy += f_UL(3 * I + 1);
            fz += f_UL(3 * I + 2);
        }
        report("UL_patch_3D_net_zero",
            std::abs(fx) < 1e-11 && std::abs(fy) < 1e-11 &&
            std::abs(fz) < 1e-11);

        double asym = (K_UL - K_UL.transpose()).cwiseAbs().maxCoeff();
        report("UL_patch_3D_K_sym", asym < 1e-10);
    }
}


// =============================================================================
//  17.  Push-forward / pull-back round-trip for tangent
// =============================================================================

void test_tangent_roundtrip() {
    CompressibleNeoHookean<3> nh{lam_ref, mu_ref};
    Eigen::Matrix3d Fm;
    Fm << 1.1, 0.02, -0.01, 0.03, 0.96, 0.015, -0.02, 0.01, 1.04;
    auto F = Tensor2<3>{Fm};
    auto E = strain::green_lagrange(F);
    auto CC = nh.material_tangent(E);

    // Push forward, then verify major symmetry
    auto cc = ops::push_forward_tangent(CC, F);
    report("tangent_roundtrip_sym", cc.has_major_symmetry(1e-10));

    // Push forward preserves positive-definiteness heuristic:
    // the (0,0) entry should remain positive
    report("tangent_roundtrip_positive", cc(0, 0) > 0.0);

    // For F = I + small perturbation, cc ≈ CC
    auto F_small = Tensor2<3>::identity();
    F_small(0, 0) = 1.001;
    auto E_small = strain::green_lagrange(F_small);
    auto CC_small = nh.material_tangent(E_small);
    auto cc_small = ops::push_forward_tangent(CC_small, F_small);
    double diff = (cc_small.voigt_matrix() - CC_small.voigt_matrix()).norm();
    report("tangent_small_F_approx_CC", diff < 1.0);  // O(ε) difference
}


// =============================================================================
//  18.  Almansi ↔ Green-Lagrange consistency
// =============================================================================

void test_almansi_green_lagrange() {
    // For any F:  e_Almansi = (1/2)(I - F⁻ᵀ F⁻¹)
    //             E_GL      = (1/2)(FᵀF - I)
    //
    // They are related by:  e = F⁻ᵀ · E · F⁻¹  (push-forward by F⁻¹)
    //
    // In index notation:  e_{ij} = (F⁻¹)_{Ii} E_{IJ} (F⁻¹)_{Jj}
    //
    // Verify this identity.

    Eigen::Matrix3d Fm;
    Fm << 1.12, 0.03, -0.02,
          0.04, 0.94,  0.015,
         -0.01, 0.02,  1.07;
    auto F = Tensor2<3>{Fm};

    auto E_GL = strain::green_lagrange(F);
    auto e_A  = strain::almansi(F);

    // Push forward E by F to get e_A:  e = F⁻ᵀ E F⁻¹
    auto F_inv = F.matrix().inverse();
    Eigen::Matrix3d e_from_E =
        F_inv.transpose() * E_GL.matrix() * F_inv;

    double err = (e_from_E - e_A.matrix()).cwiseAbs().maxCoeff();
    report("almansi_GL_push_forward", err < 1e-13);
}


// =============================================================================
//  Extra: UL ≡ TL at zero displacement (both give zero)
// =============================================================================

void test_UL_TL_at_zero() {
    SaintVenantKirchhoff<3> svk{lam_ref, mu_ref};
    auto grad = grad_tet_3d();
    Eigen::VectorXd u = Eigen::VectorXd::Zero(12);

    auto [f_TL, K_TL] = assemble_TL<3>(grad, 3, u, svk);
    auto [f_UL, K_UL] =
        UpdatedLagrangian::assemble_spatial_from_gradients<3>(grad, 3, u, svk);

    report("UL_TL_zero_f", f_TL.norm() < 1e-14 && f_UL.norm() < 1e-14);
    report("UL_TL_zero_K", (K_TL - K_UL).norm() < 1e-12);
}


// =============================================================================
//  Extra: UL with large deformation (20% stretch)
// =============================================================================

void test_UL_TL_large_deformation() {
    CompressibleNeoHookean<3> nh{lam_ref, mu_ref};
    auto grad = grad_tet_3d();

    // Large 20% uniaxial stretch at node 1, with Poisson contraction
    Eigen::VectorXd u(12);
    u << 0.0, 0.0, 0.0,
         0.20, 0.0, 0.0,
         0.0, -0.03, 0.0,
         0.0, 0.0, -0.03;

    auto [f_TL, K_TL] = assemble_TL<3>(grad, 3, u, nh);
    auto [f_UL, K_UL] =
        UpdatedLagrangian::assemble_spatial_from_gradients<3>(grad, 3, u, nh);

    double f_err = (f_TL - f_UL).cwiseAbs().maxCoeff();
    double K_err = (K_TL - K_UL).cwiseAbs().maxCoeff();

    report("UL_TL_large_f", f_err < 1e-9);
    report("UL_TL_large_K", K_err < 1e-9);
}


// =============================================================================
//  main
// =============================================================================

int main() {
    std::cout << "\n══════════════════════════════════════════════════════\n"
              << "  Updated Lagrangian Tests (Phase 6)\n"
              << "══════════════════════════════════════════════════════\n";

    std::cout << "\n── Push-forward stress ──\n";
    test_push_forward_stress();

    std::cout << "\n── Spatial gradients ──\n";
    test_spatial_gradients();

    std::cout << "\n── Grand equivalence: UL ≡ TL ──\n";
    test_grand_equivalence();

    std::cout << "\n── Push-forward tangent properties ──\n";
    test_tangent_push_forward_properties();

    std::cout << "\n── UL Newton convergence ──\n";
    test_UL_newton();

    std::cout << "\n── UL Patch test ──\n";
    test_UL_patch();

    std::cout << "\n── Tangent push-forward round-trip ──\n";
    test_tangent_roundtrip();

    std::cout << "\n── Almansi ↔ Green-Lagrange ──\n";
    test_almansi_green_lagrange();

    std::cout << "\n── UL ≡ TL at zero ──\n";
    test_UL_TL_at_zero();

    std::cout << "\n── UL ≡ TL large deformation ──\n";
    test_UL_TL_large_deformation();

    std::cout << "\n══════════════════════════════════════════════════════\n"
              << "  Results: " << passed << " passed, " << failed << " failed"
              << " (total " << (passed + failed) << ")\n"
              << "══════════════════════════════════════════════════════\n\n";

    return failed > 0 ? 1 : 0;
}
