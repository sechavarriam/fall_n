// =============================================================================
//  test_tl_integration.cpp — Phase 4: TotalLagrangian + Hyperelastic pipeline
// =============================================================================
//
//  Integration tests that verify the complete element-level finite-strain
//  computational chain using gradient-based APIs (Eigen-only, no PETSc).
//
//  Pipeline under test:
//
//    ∇N + u  →  F = I + ∂u/∂X
//            →  E = ½(FᵀF − I)            (Green-Lagrange)
//            →  S = ∂W/∂E                  (hyperelastic model)
//            →  B_NL(F) · δu = δε_eng      (nonlinear B matrix)
//            →  f_int = Bᵀ S               (internal forces)
//            →  K_mat = Bᵀ C B             (material stiffness)
//            →  K_σ   = geometric stiffness (initial stress)
//            →  K_t   = K_mat + K_σ         (total tangent)
//
//  Testing strategy:
//
//    1. 1D bar: analytical SVK pipeline verification
//    2. Reduction: TL at u=0 → SmallStrain B (all dims)
//    3. Reduction: TL at u=0 → SmallStrain K (1D, 2D, 3D with SVK)
//    4. Rigid body rotation → zero E, S, f_int (2D, 3D)
//    5. Numerical tangent: d(f_int)/du ≈ K_total (SVK  1D, 2D, 3D)
//    6. Numerical tangent: d(f_int)/du ≈ K_total (NH   2D, 3D)
//    7. Energy-force consistency: dW/du ≈ f_int (1D SVK, 3D NH)
//    8. K_σ symmetry (2D, 3D)
//    9. Multi-GP Q4 assembly (2D SVK)
//   10. Large deformation stability (3D NH)
//
//  Note: All tests use the *_from_gradients APIs of KinematicPolicy,
//  avoiding ElementGeometry and PETSc dependencies entirely.
//
// =============================================================================

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <tuple>

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

// ── Reference material parameters (E = 200, ν = 0.3) ────────────────────────

constexpr double E_ref  = 200.0;
constexpr double nu_ref = 0.3;
constexpr double lam_ref = E_ref * nu_ref / ((1.0 + nu_ref) * (1.0 - 2.0 * nu_ref));
constexpr double mu_ref  = E_ref / (2.0 * (1.0 + nu_ref));


// =============================================================================
//  Gradient helpers — manufacture dN/dx for canonical elements
// =============================================================================

/// 2-node bar of length L.  grad = [-1/L; 1/L].
Eigen::MatrixXd grad_bar_1d(double L = 2.0) {
    Eigen::MatrixXd g(2, 1);
    g(0, 0) = -1.0 / L;
    g(1, 0) =  1.0 / L;
    return g;
}

/// 3-node constant strain triangle (CST): nodes (0,0), (1,0), (0,1).
/// Constant gradients: grad = [[-1,-1],[1,0],[0,1]].
Eigen::MatrixXd grad_cst_2d() {
    Eigen::MatrixXd g(3, 2);
    g << -1.0, -1.0,
          1.0,  0.0,
          0.0,  1.0;
    return g;
}

/// 4-node linear tetrahedron: nodes (0,0,0), (1,0,0), (0,1,0), (0,0,1).
/// Constant gradients: grad = [[-1,-1,-1],[1,0,0],[0,1,0],[0,0,1]].
Eigen::MatrixXd grad_tet_3d() {
    Eigen::MatrixXd g(4, 3);
    g << -1.0, -1.0, -1.0,
          1.0,  0.0,  0.0,
          0.0,  1.0,  0.0,
          0.0,  0.0,  1.0;
    return g;
}

/// Q4 unit-square [0,1]²: derivatives dN/dx_phys at reference point (ξ, η).
/// Nodes: 0=(0,0), 1=(1,0), 2=(0,1), 3=(1,1)  (tensor-product order).
Eigen::MatrixXd grad_q4_unit(double xi, double eta) {
    Eigen::MatrixXd g(4, 2);
    // J = [[1/2,0],[0,1/2]] → J⁻¹ = [[2,0],[0,2]]
    // dN/dx_phys = J⁻¹ᵀ dN/dξ  but since J⁻¹ is diagonal: dN/dx = 2·dN/dξ
    g(0, 0) = -(1.0 - eta) / 2.0;  g(0, 1) = -(1.0 - xi) / 2.0;
    g(1, 0) =  (1.0 - eta) / 2.0;  g(1, 1) = -(1.0 + xi) / 2.0;
    g(2, 0) = -(1.0 + eta) / 2.0;  g(2, 1) =  (1.0 - xi) / 2.0;
    g(3, 0) =  (1.0 + eta) / 2.0;  g(3, 1) =  (1.0 + xi) / 2.0;
    return g;
}


// =============================================================================
//  Element-level computation helpers
// =============================================================================
//
//  Simulates the single-GP computation in ContinuumElement:
//    f_int = w · |J| · Bᵀ · S
//    K     = w · |J| · (Bᵀ C B + K_σ)
//
//  For simplicity, all helpers set w = |J| = 1 (single-GP with unit weight
//  and unit Jacobian determinant).

template <std::size_t dim>
using GradT = Eigen::Matrix<double, Eigen::Dynamic, static_cast<int>(dim)>;

/// Compute f_int and K_total at a single Gauss point.
///
/// Returns {f_int, K_total, W, E, S, F}.
template <std::size_t dim, typename Model>
auto compute_gp(
    const GradT<dim>& grad,
    std::size_t ndof,
    const Eigen::VectorXd& u_e,
    const Model& model)
{
    // 1. Kinematics: F, E, B_NL, strain_voigt (engineering)
    auto kin = TotalLagrangian::evaluate_from_gradients<dim>(grad, ndof, u_e);

    // 2. Green-Lagrange strain in tensor Voigt (via Tensor2 → SymmetricTensor2)
    auto E_tensor = strain::green_lagrange(kin.F);

    // 3. Model evaluation
    auto S = model.second_piola_kirchhoff(E_tensor);
    auto C_T4 = model.material_tangent(E_tensor);
    double W = model.energy(E_tensor);

    // 4. f_int = B_NLᵀ · S_voigt
    constexpr std::size_t NV = voigt_size<dim>();
    Eigen::Vector<double, static_cast<int>(NV)> S_voigt;
    for (std::size_t k = 0; k < NV; ++k)
        S_voigt(static_cast<Eigen::Index>(k)) = S[k];
    Eigen::VectorXd f_int = kin.B.transpose() * S_voigt;

    // 5. K_mat = B_NLᵀ · C_mixed · B_NL
    Eigen::MatrixXd K_mat = kin.B.transpose() * C_T4.voigt_matrix() * kin.B;

    // 6. K_σ
    auto S_matrix = TotalLagrangian::stress_voigt_to_matrix<dim>(S_voigt);
    Eigen::MatrixXd K_sigma =
        TotalLagrangian::compute_geometric_stiffness_from_gradients<dim>(
            grad, ndof, S_matrix);

    Eigen::MatrixXd K_total = K_mat + K_sigma;

    return std::tuple{f_int, K_total, W, E_tensor, S, kin.F};
}

/// Compute f_int only (for finite-difference tangent).
template <std::size_t dim, typename Model>
Eigen::VectorXd compute_fint(
    const GradT<dim>& grad,
    std::size_t ndof,
    const Eigen::VectorXd& u_e,
    const Model& model)
{
    auto [f, K, W, E, S, F] = compute_gp<dim>(grad, ndof, u_e, model);
    return f;
}

/// Compute W only (for energy tests).
template <std::size_t dim, typename Model>
double compute_energy(
    const GradT<dim>& grad,
    std::size_t ndof,
    const Eigen::VectorXd& u_e,
    const Model& model)
{
    auto kin = TotalLagrangian::evaluate_from_gradients<dim>(grad, ndof, u_e);
    auto E_tensor = strain::green_lagrange(kin.F);
    return model.energy(E_tensor);
}

/// Numerical tangent via central finite differences on f_int.
template <std::size_t dim, typename Model>
Eigen::MatrixXd numerical_tangent(
    const GradT<dim>& grad,
    std::size_t ndof,
    const Eigen::VectorXd& u_e,
    const Model& model,
    double h = 1e-7)
{
    const auto n = u_e.size();
    Eigen::MatrixXd K_num(n, n);

    for (Eigen::Index j = 0; j < n; ++j) {
        Eigen::VectorXd u_p = u_e, u_m = u_e;
        u_p(j) += h;
        u_m(j) -= h;

        Eigen::VectorXd f_p = compute_fint<dim>(grad, ndof, u_p, model);
        Eigen::VectorXd f_m = compute_fint<dim>(grad, ndof, u_m, model);

        K_num.col(j) = (f_p - f_m) / (2.0 * h);
    }
    return K_num;
}

/// Numerical dW/du via central differences (energy gradient).
template <std::size_t dim, typename Model>
Eigen::VectorXd numerical_energy_gradient(
    const GradT<dim>& grad,
    std::size_t ndof,
    const Eigen::VectorXd& u_e,
    const Model& model,
    double h = 1e-7)
{
    const auto n = u_e.size();
    Eigen::VectorXd dW(n);

    for (Eigen::Index j = 0; j < n; ++j) {
        Eigen::VectorXd u_p = u_e, u_m = u_e;
        u_p(j) += h;
        u_m(j) -= h;

        double W_p = compute_energy<dim>(grad, ndof, u_p, model);
        double W_m = compute_energy<dim>(grad, ndof, u_m, model);

        dW(j) = (W_p - W_m) / (2.0 * h);
    }
    return dW;
}


} // anonymous namespace


// =============================================================================
//  1. 1D bar: analytical SVK pipeline verification
// =============================================================================

void test_1d_bar_analytical() {
    SaintVenantKirchhoff<1> svk{lam_ref, mu_ref};
    auto grad = grad_bar_1d(2.0);  // L = 2, grad = [-0.5, 0.5]

    // Apply uniaxial displacement: u₁=0, u₂=0.2  →  F = 1 + 0.2/2 = 1.1
    Eigen::VectorXd u(2);
    u << 0.0, 0.2;

    auto [f, K, W, E, S, F] = compute_gp<1>(grad, 1, u, svk);

    // F = 1 + (u₂−u₁)/L = 1 + 0.1
    report("1D_bar_F", approx(F(0, 0), 1.1, 1e-14));

    // E = ½(F²−1) = ½(1.21−1) = 0.105
    report("1D_bar_E", approx(E[0], 0.105, 1e-14));

    // S = (λ + 2μ) · E  for 1D
    double S_exp = (lam_ref + 2.0 * mu_ref) * 0.105;
    report("1D_bar_S", approx(S[0], S_exp, 1e-10));

    // W = ½(λ + 2μ) · E²
    double W_exp = 0.5 * (lam_ref + 2.0 * mu_ref) * 0.105 * 0.105;
    report("1D_bar_W", approx(W, W_exp, 1e-10));

    // B_NL = F · grad = 1.1 · [-0.5, 0.5]
    // f_int = B_NLᵀ · S = 1.1 · [-0.5·S, 0.5·S]
    report("1D_bar_f0", approx(f(0), -1.1 * 0.5 * S_exp, 1e-10));
    report("1D_bar_f1", approx(f(1),  1.1 * 0.5 * S_exp, 1e-10));
    report("1D_bar_equilibrium", approx(f(0) + f(1), 0.0, 1e-12));
}


// =============================================================================
//  2. Reduction: TL at u = 0 → SmallStrain B (all dims)
// =============================================================================

void test_reduction_B_at_zero() {
    // At u = 0, F = I, and B_NL must equal the linear B from SmallStrain.
    {
        auto grad = grad_bar_1d();
        Eigen::VectorXd u = Eigen::VectorXd::Zero(2);
        auto kin_tl = TotalLagrangian::evaluate_from_gradients<1>(grad, 1, u);
        auto B_ss = SmallStrain::compute_B_from_gradients<1>(grad, 1);
        double err = (kin_tl.B - B_ss).cwiseAbs().maxCoeff();
        report("B_reduction_1D", err < 1e-15);
    }
    {
        auto grad = grad_cst_2d();
        Eigen::VectorXd u = Eigen::VectorXd::Zero(6);
        auto kin_tl = TotalLagrangian::evaluate_from_gradients<2>(grad, 2, u);
        auto B_ss = SmallStrain::compute_B_from_gradients<2>(grad, 2);
        double err = (kin_tl.B - B_ss).cwiseAbs().maxCoeff();
        report("B_reduction_2D", err < 1e-15);
    }
    {
        auto grad = grad_tet_3d();
        Eigen::VectorXd u = Eigen::VectorXd::Zero(12);
        auto kin_tl = TotalLagrangian::evaluate_from_gradients<3>(grad, 3, u);
        auto B_ss = SmallStrain::compute_B_from_gradients<3>(grad, 3);
        double err = (kin_tl.B - B_ss).cwiseAbs().maxCoeff();
        report("B_reduction_3D", err < 1e-15);
    }
}


// =============================================================================
//  3. Reduction: TL at u = 0 → SmallStrain K (SVK, all dims)
// =============================================================================

void test_reduction_K_at_zero() {
    // At u=0: K_TL = K_mat + K_σ.  But S(E=0) = 0, so K_σ = 0.
    // Therefore K_TL = Bᵀ C B, which must equal the SmallStrain K.

    // 1D
    {
        SaintVenantKirchhoff<1> svk{lam_ref, mu_ref};
        auto grad = grad_bar_1d();
        Eigen::VectorXd u = Eigen::VectorXd::Zero(2);
        auto [f, K_tl, W, E, S, F] = compute_gp<1>(grad, 1, u, svk);

        auto B_ss = SmallStrain::compute_B_from_gradients<1>(grad, 1);
        auto C = svk.material_tangent().voigt_matrix();
        Eigen::MatrixXd K_ss = B_ss.transpose() * C * B_ss;

        double err = (K_tl - K_ss).cwiseAbs().maxCoeff();
        report("K_reduction_1D", err < 1e-14);
    }
    // 2D
    {
        SaintVenantKirchhoff<2> svk{lam_ref, mu_ref};
        auto grad = grad_cst_2d();
        Eigen::VectorXd u = Eigen::VectorXd::Zero(6);
        auto [f, K_tl, W, E, S, F] = compute_gp<2>(grad, 2, u, svk);

        auto B_ss = SmallStrain::compute_B_from_gradients<2>(grad, 2);
        auto C = svk.material_tangent().voigt_matrix();
        Eigen::MatrixXd K_ss = B_ss.transpose() * C * B_ss;

        double err = (K_tl - K_ss).cwiseAbs().maxCoeff();
        report("K_reduction_2D", err < 1e-12);
    }
    // 3D
    {
        SaintVenantKirchhoff<3> svk{lam_ref, mu_ref};
        auto grad = grad_tet_3d();
        Eigen::VectorXd u = Eigen::VectorXd::Zero(12);
        auto [f, K_tl, W, E, S, F] = compute_gp<3>(grad, 3, u, svk);

        auto B_ss = SmallStrain::compute_B_from_gradients<3>(grad, 3);
        auto C = svk.material_tangent().voigt_matrix();
        Eigen::MatrixXd K_ss = B_ss.transpose() * C * B_ss;

        double err = (K_tl - K_ss).cwiseAbs().maxCoeff();
        report("K_reduction_3D", err < 1e-12);
    }
}


// =============================================================================
//  4. Rigid body rotation → zero E, S, f_int
// =============================================================================

void test_rigid_body_rotation() {
    const double theta = 0.35;  // ~20°
    const double ct = std::cos(theta), st = std::sin(theta);

    // ── 2D CST: nodes at (0,0), (1,0), (0,1) ────────────────────────────
    {
        SaintVenantKirchhoff<2> svk{lam_ref, mu_ref};
        auto grad = grad_cst_2d();

        // u_I = (R − I) · X_I
        // Node 0 (0,0): u = (0,0)
        // Node 1 (1,0): u = (ct−1, st)
        // Node 2 (0,1): u = (−st, ct−1)
        Eigen::VectorXd u(6);
        u << 0.0, 0.0,
             ct - 1.0, st,
             -st, ct - 1.0;

        auto [f, K, W, E, S, F] = compute_gp<2>(grad, 2, u, svk);

        // F should be the rotation matrix
        report("rigid_2D_F00", approx(F(0, 0), ct, 1e-14));
        report("rigid_2D_F01", approx(F(0, 1), -st, 1e-14));
        report("rigid_2D_F10", approx(F(1, 0), st, 1e-14));

        // E = ½(FᵀF − I) = 0
        double max_E = 0.0;
        for (std::size_t k = 0; k < 3; ++k)
            max_E = std::max(max_E, std::abs(E[k]));
        report("rigid_2D_E_zero", max_E < 1e-14);

        // S = 0
        double max_S = 0.0;
        for (std::size_t k = 0; k < 3; ++k)
            max_S = std::max(max_S, std::abs(S[k]));
        report("rigid_2D_S_zero", max_S < 1e-12);

        // f_int = 0
        report("rigid_2D_f_zero", f.cwiseAbs().maxCoeff() < 1e-12);

        // W = 0
        report("rigid_2D_W_zero", approx(W, 0.0, 1e-14));
    }

    // ── 3D tet: nodes at (0,0,0), (1,0,0), (0,1,0), (0,0,1) ────────────
    // Rotation about z-axis by θ
    {
        SaintVenantKirchhoff<3> svk{lam_ref, mu_ref};
        auto grad = grad_tet_3d();

        Eigen::VectorXd u(12);
        // Node 0 (0,0,0): u = (0,0,0)
        // Node 1 (1,0,0): u = (ct−1, st, 0)
        // Node 2 (0,1,0): u = (−st, ct−1, 0)
        // Node 3 (0,0,1): u = (0, 0, 0)
        u << 0.0, 0.0, 0.0,
             ct - 1.0, st, 0.0,
             -st, ct - 1.0, 0.0,
             0.0, 0.0, 0.0;

        auto [f, K, W, E, S, F] = compute_gp<3>(grad, 3, u, svk);

        double max_E = 0.0;
        for (std::size_t k = 0; k < 6; ++k)
            max_E = std::max(max_E, std::abs(E[k]));
        report("rigid_3D_E_zero", max_E < 1e-14);

        double max_S = 0.0;
        for (std::size_t k = 0; k < 6; ++k)
            max_S = std::max(max_S, std::abs(S[k]));
        report("rigid_3D_S_zero", max_S < 1e-12);

        report("rigid_3D_f_zero", f.cwiseAbs().maxCoeff() < 1e-12);
        report("rigid_3D_W_zero", approx(W, 0.0, 1e-14));
    }
}


// =============================================================================
//  5. Numerical tangent: SVK  (1D, 2D, 3D)
// =============================================================================

void test_numerical_tangent_svk() {
    // 1D
    {
        SaintVenantKirchhoff<1> svk{lam_ref, mu_ref};
        auto grad = grad_bar_1d();
        Eigen::VectorXd u(2);
        u << 0.0, 0.15;   // moderate stretch

        auto [f, K_a, W, E, S, F] = compute_gp<1>(grad, 1, u, svk);
        auto K_n = numerical_tangent<1>(grad, 1, u, svk);

        double err = (K_a - K_n).cwiseAbs().maxCoeff();
        report("num_tangent_1D_SVK", err < 1e-5);
    }
    // 2D CST
    {
        SaintVenantKirchhoff<2> svk{lam_ref, mu_ref};
        auto grad = grad_cst_2d();
        Eigen::VectorXd u(6);
        u << 0.0, 0.0,  0.05, 0.01,  -0.01, 0.04;

        auto [f, K_a, W, E, S, F] = compute_gp<2>(grad, 2, u, svk);
        auto K_n = numerical_tangent<2>(grad, 2, u, svk);

        double err = (K_a - K_n).cwiseAbs().maxCoeff();
        report("num_tangent_2D_SVK", err < 1e-4);
    }
    // 3D tet
    {
        SaintVenantKirchhoff<3> svk{lam_ref, mu_ref};
        auto grad = grad_tet_3d();
        Eigen::VectorXd u(12);
        u << 0.0, 0.0, 0.0,
             0.03, 0.01, -0.01,
             -0.01, 0.04, 0.005,
             0.02, -0.01, 0.03;

        auto [f, K_a, W, E, S, F] = compute_gp<3>(grad, 3, u, svk);
        auto K_n = numerical_tangent<3>(grad, 3, u, svk);

        double err = (K_a - K_n).cwiseAbs().maxCoeff();
        report("num_tangent_3D_SVK", err < 1e-4);
    }
}


// =============================================================================
//  6. Numerical tangent: Neo-Hookean  (2D, 3D)
// =============================================================================

void test_numerical_tangent_nh() {
    // 2D CST
    {
        CompressibleNeoHookean<2> nh{lam_ref, mu_ref};
        auto grad = grad_cst_2d();
        Eigen::VectorXd u(6);
        u << 0.0, 0.0,  0.04, 0.01,  -0.005, 0.03;

        auto [f, K_a, W, E, S, F] = compute_gp<2>(grad, 2, u, nh);
        auto K_n = numerical_tangent<2>(grad, 2, u, nh);

        double err = (K_a - K_n).cwiseAbs().maxCoeff();
        report("num_tangent_2D_NH", err < 1e-4);
    }
    // 3D tet
    {
        CompressibleNeoHookean<3> nh{lam_ref, mu_ref};
        auto grad = grad_tet_3d();
        Eigen::VectorXd u(12);
        u << 0.0, 0.0, 0.0,
             0.03, 0.01, -0.01,
             -0.01, 0.03, 0.005,
             0.01, -0.005, 0.02;

        auto [f, K_a, W, E, S, F] = compute_gp<3>(grad, 3, u, nh);
        auto K_n = numerical_tangent<3>(grad, 3, u, nh);

        double err = (K_a - K_n).cwiseAbs().maxCoeff();
        report("num_tangent_3D_NH", err < 1e-4);
    }
}


// =============================================================================
//  7. Energy-force consistency: dW/du ≈ f_int
// =============================================================================
//
//  For hyperelastic materials:  f_int_j = ∂W_int / ∂u_j
//  where W_int := Σ_gp  w · |J| · W(E(u))  is the total stored energy.
//
//  With a single GP (w = |J| = 1): f_int = dW/du.

void test_energy_force_consistency() {
    // 1D SVK
    {
        SaintVenantKirchhoff<1> svk{lam_ref, mu_ref};
        auto grad = grad_bar_1d();
        Eigen::VectorXd u(2);
        u << 0.0, 0.15;

        auto [f_a, K, W, E, S, F] = compute_gp<1>(grad, 1, u, svk);
        auto dW_num = numerical_energy_gradient<1>(grad, 1, u, svk);

        double err = (f_a - dW_num).cwiseAbs().maxCoeff();
        report("energy_force_1D_SVK", err < 1e-6);
    }
    // 3D NH
    {
        CompressibleNeoHookean<3> nh{lam_ref, mu_ref};
        auto grad = grad_tet_3d();
        Eigen::VectorXd u(12);
        u << 0.0, 0.0, 0.0,
             0.03, 0.01, -0.005,
             -0.01, 0.02, 0.005,
             0.005, -0.01, 0.015;

        auto [f_a, K, W, E, S, F] = compute_gp<3>(grad, 3, u, nh);
        auto dW_num = numerical_energy_gradient<3>(grad, 3, u, nh);

        double err = (f_a - dW_num).cwiseAbs().maxCoeff();
        report("energy_force_3D_NH", err < 1e-5);
    }
}


// =============================================================================
//  8. K_σ symmetry
// =============================================================================

void test_K_sigma_symmetry() {
    // 2D with arbitrary stress
    {
        auto grad = grad_cst_2d();
        Eigen::Vector3d S_voigt(10.0, 5.0, 3.0);
        auto S_matrix = TotalLagrangian::stress_voigt_to_matrix<2>(S_voigt);
        auto K_sigma =
            TotalLagrangian::compute_geometric_stiffness_from_gradients<2>(
                grad, 2, S_matrix);

        double asym = (K_sigma - K_sigma.transpose()).cwiseAbs().maxCoeff();
        report("K_sigma_sym_2D", asym < 1e-14);
    }
    // 3D with arbitrary stress
    {
        auto grad = grad_tet_3d();
        Eigen::Vector<double, 6> S_voigt;
        S_voigt << 10.0, 20.0, 30.0, 5.0, -3.0, 7.0;
        auto S_matrix = TotalLagrangian::stress_voigt_to_matrix<3>(S_voigt);
        auto K_sigma =
            TotalLagrangian::compute_geometric_stiffness_from_gradients<3>(
                grad, 3, S_matrix);

        double asym = (K_sigma - K_sigma.transpose()).cwiseAbs().maxCoeff();
        report("K_sigma_sym_3D", asym < 1e-14);
    }
    // K_total symmetry at a deformed state (NH, 3D) — hyperelastic ⟹ symmetric K
    {
        CompressibleNeoHookean<3> nh{lam_ref, mu_ref};
        auto grad = grad_tet_3d();
        Eigen::VectorXd u(12);
        u << 0.0, 0.0, 0.0,
             0.03, 0.01, -0.005,
             -0.01, 0.02, 0.005,
             0.005, -0.01, 0.015;

        auto [f, K, W, E, S, F] = compute_gp<3>(grad, 3, u, nh);
        double asym = (K - K.transpose()).cwiseAbs().maxCoeff();
        report("K_total_sym_NH_3D", asym < 1e-10);
    }
}


// =============================================================================
//  9. Multi-GP Q4 assembly (2D SVK)
// =============================================================================
//
//  Assembles K and f_int for a 4-node Q4 quad (unit square [0,1]²) using
//  2×2 Gauss quadrature.  Verifies:
//    a) K is symmetric
//    b) f_int at u=0 is zero
//    c) K at u=0 equals SmallStrain K
//    d) Numerical tangent matches analytical K at a deformed state

void test_multi_gp_q4() {
    constexpr std::size_t dim  = 2;
    constexpr std::size_t ndof = 2;
    constexpr std::size_t n_nodes = 4;
    constexpr std::size_t total_dof = ndof * n_nodes;

    SaintVenantKirchhoff<2> svk{lam_ref, mu_ref};

    // 2×2 Gauss points on [-1,1]²
    const double g = 1.0 / std::sqrt(3.0);
    const double gp_coords[4][2] = {
        {-g, -g}, {g, -g}, {-g, g}, {g, g}
    };
    const double gp_weight = 1.0;   // each GP weight = 1
    const double J_det = 0.25;      // |J| = (1/2)² for unit square

    // ── (a) K at u=0: assemble and check symmetry ────────────────────────
    {
        Eigen::VectorXd u = Eigen::VectorXd::Zero(total_dof);
        Eigen::MatrixXd K_e = Eigen::MatrixXd::Zero(total_dof, total_dof);
        Eigen::VectorXd f_e = Eigen::VectorXd::Zero(total_dof);

        for (int g_idx = 0; g_idx < 4; ++g_idx) {
            auto grad = grad_q4_unit(gp_coords[g_idx][0], gp_coords[g_idx][1]);
            auto [f, K, W, E, S, F] = compute_gp<dim>(grad, ndof, u, svk);
            K_e += gp_weight * J_det * K;
            f_e += gp_weight * J_det * f;
        }

        report("Q4_K_symmetric", (K_e - K_e.transpose()).cwiseAbs().maxCoeff() < 1e-12);
        report("Q4_f_zero_at_u0", f_e.cwiseAbs().maxCoeff() < 1e-12);
    }

    // ── (b) K at u=0 equals SmallStrain K ────────────────────────────────
    {
        Eigen::MatrixXd K_tl = Eigen::MatrixXd::Zero(total_dof, total_dof);
        Eigen::MatrixXd K_ss = Eigen::MatrixXd::Zero(total_dof, total_dof);
        Eigen::VectorXd u = Eigen::VectorXd::Zero(total_dof);
        auto C = svk.material_tangent().voigt_matrix();

        for (int g_idx = 0; g_idx < 4; ++g_idx) {
            auto grad = grad_q4_unit(gp_coords[g_idx][0], gp_coords[g_idx][1]);

            auto [f, K, W, E, S, F] = compute_gp<dim>(grad, ndof, u, svk);
            K_tl += gp_weight * J_det * K;

            auto B = SmallStrain::compute_B_from_gradients<dim>(grad, ndof);
            K_ss += gp_weight * J_det * (B.transpose() * C * B);
        }

        double err = (K_tl - K_ss).cwiseAbs().maxCoeff();
        report("Q4_K_reduction_eq_SS", err < 1e-12);
    }

    // ── (c) Numerical tangent at deformed state ──────────────────────────
    {
        Eigen::VectorXd u(total_dof);
        u << 0.0, 0.0,  0.03, 0.005,  -0.004, 0.02,  0.025, 0.025;

        // Assemble analytical K and f
        Eigen::MatrixXd K_a = Eigen::MatrixXd::Zero(total_dof, total_dof);
        Eigen::VectorXd f_a = Eigen::VectorXd::Zero(total_dof);

        for (int g_idx = 0; g_idx < 4; ++g_idx) {
            auto grad = grad_q4_unit(gp_coords[g_idx][0], gp_coords[g_idx][1]);
            auto [f, K, W, E, S, F] = compute_gp<dim>(grad, ndof, u, svk);
            K_a += gp_weight * J_det * K;
            f_a += gp_weight * J_det * f;
        }

        // Numerical tangent via central differences on full element f_int
        const double h = 1e-7;
        Eigen::MatrixXd K_n = Eigen::MatrixXd::Zero(total_dof, total_dof);

        for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(total_dof); ++j) {
            Eigen::VectorXd u_p = u, u_m = u;
            u_p(j) += h;
            u_m(j) -= h;

            Eigen::VectorXd f_p = Eigen::VectorXd::Zero(total_dof);
            Eigen::VectorXd f_m = Eigen::VectorXd::Zero(total_dof);

            for (int g_idx = 0; g_idx < 4; ++g_idx) {
                auto grad = grad_q4_unit(gp_coords[g_idx][0],
                                          gp_coords[g_idx][1]);
                f_p += gp_weight * J_det *
                       compute_fint<dim>(grad, ndof, u_p, svk);
                f_m += gp_weight * J_det *
                       compute_fint<dim>(grad, ndof, u_m, svk);
            }
            K_n.col(j) = (f_p - f_m) / (2.0 * h);
        }

        double err = (K_a - K_n).cwiseAbs().maxCoeff();
        report("Q4_num_tangent", err < 1e-3);

        // Also verify K is symmetric at the deformed state
        report("Q4_K_sym_deformed",
            (K_a - K_a.transpose()).cwiseAbs().maxCoeff() < 1e-10);
    }
}


// =============================================================================
//  10. Large deformation stability (3D NH)
// =============================================================================

void test_large_deformation_nh() {
    CompressibleNeoHookean<3> nh{lam_ref, mu_ref};
    auto grad = grad_tet_3d();

    // Pure uniaxial stretch λ = 2: F = diag(2, 1, 1)
    // Nodes: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
    // u_I = (F − I) · X_I
    // u₀ = (0,0,0),  u₁ = (1,0,0),  u₂ = (0,0,0),  u₃ = (0,0,0)
    Eigen::VectorXd u(12);
    u << 0.0, 0.0, 0.0,
         1.0, 0.0, 0.0,    // node 1: stretch in x
         0.0, 0.0, 0.0,
         0.0, 0.0, 0.0;

    auto [f, K, W, E, S, F] = compute_gp<3>(grad, 3, u, nh);

    // F should be diag(2, 1, 1)
    report("large_F11", approx(F(0, 0), 2.0, 1e-14));
    report("large_F22", approx(F(1, 1), 1.0, 1e-14));
    report("large_F33", approx(F(2, 2), 1.0, 1e-14));

    // E₁₁ = ½(4−1) = 1.5
    report("large_E11", approx(E[0], 1.5, 1e-14));
    // E₂₂ = E₃₃ = 0
    report("large_E22_E33", approx(E[1], 0.0, 1e-14) &&
                             approx(E[2], 0.0, 1e-14));

    // Energy should be positive and large
    report("large_W_positive", W > 0.0);

    // S₁₁ should be positive (tension)
    report("large_S11_positive", S[0] > 0.0);

    // Tangent should be symmetric (hyperelastic) even at large deformation
    report("large_K_sym", (K - K.transpose()).cwiseAbs().maxCoeff() < 1e-8);

    // Numerical tangent should still match
    auto K_n = numerical_tangent<3>(grad, 3, u, nh);
    double err = (K - K_n).cwiseAbs().maxCoeff();
    report("large_num_tangent_NH", err < 1e-2);  // relaxed tol for large def
}


// =============================================================================
//  main
// =============================================================================

int main() {
    std::cout << "\n══════════════════════════════════════════════════════\n"
              << "  TotalLagrangian Integration Tests (Phase 4)\n"
              << "══════════════════════════════════════════════════════\n";

    // 1. Analytical 1D bar
    std::cout << "\n── 1D bar analytical ──\n";
    test_1d_bar_analytical();

    // 2. B reduction
    std::cout << "\n── B_NL(u=0) = B_linear ──\n";
    test_reduction_B_at_zero();

    // 3. K reduction
    std::cout << "\n── K_TL(u=0) = K_SmallStrain ──\n";
    test_reduction_K_at_zero();

    // 4. Rigid body
    std::cout << "\n── Rigid body → zero E, S, f ──\n";
    test_rigid_body_rotation();

    // 5. Numerical tangent (SVK)
    std::cout << "\n── Numerical tangent (SVK) ──\n";
    test_numerical_tangent_svk();

    // 6. Numerical tangent (NH)
    std::cout << "\n── Numerical tangent (NH) ──\n";
    test_numerical_tangent_nh();

    // 7. Energy-force consistency
    std::cout << "\n── Energy-force consistency ──\n";
    test_energy_force_consistency();

    // 8. K_σ symmetry
    std::cout << "\n── K_σ and K_total symmetry ──\n";
    test_K_sigma_symmetry();

    // 9. Multi-GP Q4
    std::cout << "\n── Multi-GP Q4 assembly ──\n";
    test_multi_gp_q4();

    // 10. Large deformation
    std::cout << "\n── Large deformation (NH) ──\n";
    test_large_deformation_nh();

    // Summary
    std::cout << "\n══════════════════════════════════════════════════════\n"
              << "  Results: " << passed << " passed, " << failed << " failed"
              << " (total " << (passed + failed) << ")\n"
              << "══════════════════════════════════════════════════════\n\n";

    return failed > 0 ? 1 : 0;
}
