// =============================================================================
//  test_newton_element.cpp — Phase 5: Newton-Raphson element-level convergence
// =============================================================================
//
//  Capstone tests proving the entire finite-strain pipeline converges using
//  Newton-Raphson with consistent tangents (Phases 1-4 combined).
//
//  The solver operates on element-level DOFs using gradient-based APIs
//  (Eigen-only, no PETSc, no ElementGeometry).
//
//  Pipeline validated:
//
//    KinematicPolicy → TotalLagrangian::evaluate_from_gradients → F, E, B_NL
//    HyperelasticModel → S(E), ℂ(E)
//    f_int = Bᵀ S,   K_mat = Bᵀ ℂ B,   K_σ = geometric stiffness
//    Newton:  K_ff · Δu = −R_f   (DOF-partitioned)
//
//  Testing strategy:
//
//    1.  1D bar SVK:       force-controlled Newton, convergence
//    2.  1D bar SVK:       quadratic convergence rate verification
//    3.  1D bar SVK:       displacement-controlled reaction force check
//    4.  2D CST SVK:       force-controlled Newton, convergence
//    5.  2D CST NH:        force-controlled Newton, convergence
//    6.  3D tet NH:        force-controlled Newton, convergence
//    7.  3D tet NH:        quadratic convergence rate verification
//    8.  Q4 multi-GP SVK:  force-controlled Newton, convergence
//    9.  Load stepping:    incremental 3D NH loading
//   10.  Patch test:       uniform deformation → exact F, zero net force
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

// ── Reference material parameters  (E = 200 GPa, ν = 0.3) ──────────────────

constexpr double E_ref  = 200.0;
constexpr double nu_ref = 0.3;
constexpr double lam_ref = E_ref * nu_ref / ((1.0 + nu_ref) * (1.0 - 2.0 * nu_ref));
constexpr double mu_ref  = E_ref / (2.0 * (1.0 + nu_ref));


// =============================================================================
//  Gradient helpers — canonical element shape-function gradients ∂N/∂X
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

/// Q4 unit square [0,1]²: physical gradients at (ξ, η) ∈ [-1,1]².
Eigen::MatrixXd grad_q4_unit(double xi, double eta) {
    Eigen::MatrixXd g(4, 2);
    g(0, 0) = -(1.0 - eta) / 2.0;  g(0, 1) = -(1.0 - xi) / 2.0;
    g(1, 0) =  (1.0 - eta) / 2.0;  g(1, 1) = -(1.0 + xi) / 2.0;
    g(2, 0) = -(1.0 + eta) / 2.0;  g(2, 1) =  (1.0 - xi) / 2.0;
    g(3, 0) =  (1.0 + eta) / 2.0;  g(3, 1) =  (1.0 + xi) / 2.0;
    return g;
}


// =============================================================================
//  Single-GP assembly: returns { f_int, K_total }
// =============================================================================

template <std::size_t dim, typename Model>
std::pair<Eigen::VectorXd, Eigen::MatrixXd>
assemble_gp(
    const Eigen::MatrixXd& grad,
    std::size_t ndof,
    const Eigen::VectorXd& u_e,
    const Model& model)
{
    auto kin      = TotalLagrangian::evaluate_from_gradients<dim>(grad, ndof, u_e);
    auto E_tensor = strain::green_lagrange(kin.F);
    auto S        = model.second_piola_kirchhoff(E_tensor);
    auto C_T4     = model.material_tangent(E_tensor);

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
//  Multi-GP Q4 assembly (2×2 Gauss)
// =============================================================================

template <typename Model>
std::pair<Eigen::VectorXd, Eigen::MatrixXd>
assemble_q4(const Eigen::VectorXd& u_e, const Model& model)
{
    constexpr std::size_t dim = 2, ndof = 2, total = 8;
    Eigen::MatrixXd K_e = Eigen::MatrixXd::Zero(total, total);
    Eigen::VectorXd f_e = Eigen::VectorXd::Zero(total);

    const double g = 1.0 / std::sqrt(3.0);
    const double gp_coords[4][2] = {{-g, -g}, {g, -g}, {-g, g}, {g, g}};
    const double wJ = 0.25;   // w = 1, |J| = 1/4  for unit square

    for (int gp = 0; gp < 4; ++gp) {
        auto grad = grad_q4_unit(gp_coords[gp][0], gp_coords[gp][1]);
        auto [f, K] = assemble_gp<dim>(grad, ndof, u_e, model);
        K_e += wJ * K;
        f_e += wJ * f;
    }
    return {f_e, K_e};
}


// =============================================================================
//  Newton-Raphson solver
// =============================================================================

struct NewtonResult {
    Eigen::VectorXd        u;
    std::vector<double>    residuals;   ///< ||R_free|| at each iteration
    int                    iterations;
    bool                   converged;
};

/// Solve R(u) = f_int(u) − f_ext = 0 on the free DOFs.
///
///   assemble(u) → {f_int, K}          (full element vectors/matrices)
///   free_dofs   — indices of unconstrained DOFs
///   u0          — initial guess (constrained DOFs already set)
///
template <typename AssembleFn>
NewtonResult solve_newton(
    AssembleFn&& assemble,
    const Eigen::VectorXd& f_ext,
    const std::vector<Eigen::Index>& free_dofs,
    Eigen::VectorXd u0,
    int    max_iter = 30,
    double tol      = 1e-12)
{
    const auto nf = static_cast<Eigen::Index>(free_dofs.size());

    NewtonResult result;
    result.u         = std::move(u0);
    result.converged = false;

    for (int iter = 0; iter < max_iter; ++iter) {
        auto [f_int, K] = assemble(result.u);
        Eigen::VectorXd R = f_int - f_ext;

        // Extract free-DOF subset
        Eigen::VectorXd R_f(nf);
        Eigen::MatrixXd K_ff(nf, nf);
        for (Eigen::Index i = 0; i < nf; ++i) {
            R_f(i) = R(free_dofs[static_cast<std::size_t>(i)]);
            for (Eigen::Index j = 0; j < nf; ++j)
                K_ff(i, j) = K(free_dofs[static_cast<std::size_t>(i)],
                               free_dofs[static_cast<std::size_t>(j)]);
        }

        const double res_norm = R_f.norm();
        result.residuals.push_back(res_norm);

        if (res_norm < tol) {
            result.converged  = true;
            result.iterations = iter;
            return result;
        }

        // Solve  K_ff · Δu = −R_f
        Eigen::VectorXd du_f = K_ff.ldlt().solve(-R_f);

        for (Eigen::Index i = 0; i < nf; ++i)
            result.u(free_dofs[static_cast<std::size_t>(i)]) += du_f(i);
    }

    result.iterations = max_iter;
    return result;
}


// =============================================================================
//  Convergence-order estimator
// =============================================================================
//
//  For a sequence of residual norms {r_k}, estimate the order q
//  from consecutive triplets:
//
//      q ≈ log(r_{k+1}/r_k) / log(r_k/r_{k-1})
//
//  Returns the average over valid triplets (r_k > tol to avoid log(0)).

double estimate_convergence_order(const std::vector<double>& res,
                                  double cutoff = 1e-14)
{
    if (res.size() < 3) return 0.0;

    double sum = 0.0;
    int    cnt = 0;
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
//  1.  1D bar SVK — force-controlled Newton
// =============================================================================

void test_1d_svk_newton() {
    SaintVenantKirchhoff<1> svk{lam_ref, mu_ref};
    auto grad = grad_bar_1d();

    // Fix u₀ = 0, apply P at node 1.   DOF: [u₀, u₁].   Free: {1}.
    const double P = 30.0;
    Eigen::VectorXd f_ext = Eigen::VectorXd::Zero(2);
    f_ext(1) = P;

    auto result = solve_newton(
        [&](const Eigen::VectorXd& u) {
            return assemble_gp<1>(grad, 1, u, svk);
        },
        f_ext, {1}, Eigen::VectorXd::Zero(2));

    report("newton_1D_SVK_converged", result.converged);
    report("newton_1D_SVK_iters_reasonable", result.iterations < 15);

    // Verify equilibrium: f_int ≈ f_ext at node 1
    auto [f, K] = assemble_gp<1>(grad, 1, result.u, svk);
    report("newton_1D_SVK_equilibrium", std::abs(f(1) - P) < 1e-10);
}


// =============================================================================
//  2.  1D bar SVK — quadratic convergence rate
// =============================================================================

void test_1d_svk_convergence_rate() {
    SaintVenantKirchhoff<1> svk{lam_ref, mu_ref};
    auto grad = grad_bar_1d();

    Eigen::VectorXd f_ext = Eigen::VectorXd::Zero(2);
    f_ext(1) = 30.0;

    auto result = solve_newton(
        [&](const Eigen::VectorXd& u) {
            return assemble_gp<1>(grad, 1, u, svk);
        },
        f_ext, {1}, Eigen::VectorXd::Zero(2));

    double q = estimate_convergence_order(result.residuals);
    report("newton_1D_SVK_quadratic", q > 1.7);

    // Print convergence history for diagnostics
    std::cout << "    residuals:";
    for (double r : result.residuals) std::cout << " " << r;
    std::cout << "  (q ≈ " << q << ")\n";
}


// =============================================================================
//  3.  1D bar SVK — displacement-controlled reaction force
// =============================================================================

void test_1d_svk_displacement_controlled() {
    SaintVenantKirchhoff<1> svk{lam_ref, mu_ref};
    auto grad = grad_bar_1d();  // L = 2

    // Prescribe u₁ = 0.2 → F = 1.1, E = 0.105
    Eigen::VectorXd u(2);
    u << 0.0, 0.2;

    auto [f, K] = assemble_gp<1>(grad, 1, u, svk);

    // Analytical: S = (λ+2μ)·E,  f_int₁ = B_NL₁·S = F·grad₁·S
    // With grad₁ = 1/L = 0.5:
    const double L = 2.0;
    const double F_val = 1.0 + u(1) / L;                          // = 1.1
    const double E_val = 0.5 * (F_val * F_val - 1.0);             // = 0.105
    const double S_exp = (lam_ref + 2.0 * mu_ref) * E_val;
    const double f1_exp = F_val * (1.0 / L) * S_exp;              // B_NL₁ · S

    report("disp_ctrl_1D_reaction", approx(f(1), f1_exp, 1e-10));
    // Equilibrium: f₀ + f₁ = 0
    report("disp_ctrl_1D_equilibrium", approx(f(0) + f(1), 0.0, 1e-12));
}


// =============================================================================
//  4.  2D CST SVK — force-controlled Newton
// =============================================================================

void test_2d_cst_svk_newton() {
    SaintVenantKirchhoff<2> svk{lam_ref, mu_ref};
    auto grad = grad_cst_2d();

    // Fix node 0: DOFs {0,1}.  Fix v₁ = 0: DOF {3}.
    // Free: {2, 4, 5} = {u₁ₓ, u₂ₓ, u₂ᵧ}.
    // Tension:  P = 5 in x at nodes 1 and 2.
    Eigen::VectorXd f_ext = Eigen::VectorXd::Zero(6);
    f_ext(2) = 5.0;  // node 1 x
    f_ext(4) = 5.0;  // node 2 x

    auto result = solve_newton(
        [&](const Eigen::VectorXd& u) {
            return assemble_gp<2>(grad, 2, u, svk);
        },
        f_ext, {2, 4, 5}, Eigen::VectorXd::Zero(6));

    report("newton_2D_CST_SVK_converged", result.converged);
    report("newton_2D_CST_SVK_iters", result.iterations < 15);

    // Verify residual
    auto [f, K] = assemble_gp<2>(grad, 2, result.u, svk);
    Eigen::VectorXd R = f - f_ext;
    report("newton_2D_CST_SVK_equilibrium",
        std::abs(R(2)) < 1e-10 && std::abs(R(4)) < 1e-10 &&
        std::abs(R(5)) < 1e-10);
}


// =============================================================================
//  5.  2D CST NH — force-controlled Newton
// =============================================================================

void test_2d_cst_nh_newton() {
    CompressibleNeoHookean<2> nh{lam_ref, mu_ref};
    auto grad = grad_cst_2d();

    Eigen::VectorXd f_ext = Eigen::VectorXd::Zero(6);
    f_ext(2) = 5.0;
    f_ext(4) = 5.0;

    auto result = solve_newton(
        [&](const Eigen::VectorXd& u) {
            return assemble_gp<2>(grad, 2, u, nh);
        },
        f_ext, {2, 4, 5}, Eigen::VectorXd::Zero(6));

    report("newton_2D_CST_NH_converged", result.converged);
    report("newton_2D_CST_NH_iters", result.iterations < 15);
}


// =============================================================================
//  6.  3D tet NH — force-controlled Newton
// =============================================================================

void test_3d_tet_nh_newton() {
    CompressibleNeoHookean<3> nh{lam_ref, mu_ref};
    auto grad = grad_tet_3d();

    // Fix node 0: DOFs {0,1,2}.  Fix u₁ᵧ,u₁_z: {4,5}.  Fix u₂_z: {8}.
    // Fixed = {0,1,2,4,5,8},   Free = {3,6,7,9,10,11}.
    // Apply P = 5 in x at nodes 1,2,3.
    Eigen::VectorXd f_ext = Eigen::VectorXd::Zero(12);
    f_ext(3) = 5.0;   // node 1 x
    f_ext(6) = 5.0;   // node 2 x
    f_ext(9) = 5.0;   // node 3 x

    auto result = solve_newton(
        [&](const Eigen::VectorXd& u) {
            return assemble_gp<3>(grad, 3, u, nh);
        },
        f_ext, {3, 6, 7, 9, 10, 11}, Eigen::VectorXd::Zero(12));

    report("newton_3D_tet_NH_converged", result.converged);
    report("newton_3D_tet_NH_iters", result.iterations < 15);

    // Verify final equilibrium on free DOFs
    auto [f, K] = assemble_gp<3>(grad, 3, result.u, nh);
    Eigen::VectorXd R = f - f_ext;
    double max_R_free = 0.0;
    for (auto idx : {3, 6, 7, 9, 10, 11})
        max_R_free = std::max(max_R_free, std::abs(R(idx)));
    report("newton_3D_tet_NH_equilibrium", max_R_free < 1e-10);
}


// =============================================================================
//  7.  3D tet NH — quadratic convergence rate
// =============================================================================

void test_3d_tet_nh_convergence_rate() {
    CompressibleNeoHookean<3> nh{lam_ref, mu_ref};
    auto grad = grad_tet_3d();

    Eigen::VectorXd f_ext = Eigen::VectorXd::Zero(12);
    f_ext(3) = 5.0;
    f_ext(6) = 5.0;
    f_ext(9) = 5.0;

    auto result = solve_newton(
        [&](const Eigen::VectorXd& u) {
            return assemble_gp<3>(grad, 3, u, nh);
        },
        f_ext, {3, 6, 7, 9, 10, 11}, Eigen::VectorXd::Zero(12));

    double q = estimate_convergence_order(result.residuals);
    report("newton_3D_tet_NH_quadratic", q > 1.7);

    std::cout << "    residuals:";
    for (double r : result.residuals) std::cout << " " << r;
    std::cout << "  (q ≈ " << q << ")\n";
}


// =============================================================================
//  8.  Q4 multi-GP SVK — force-controlled Newton
// =============================================================================

void test_q4_svk_newton() {
    SaintVenantKirchhoff<2> svk{lam_ref, mu_ref};

    // Fix node 0: DOFs {0,1}.  Fix v₁ = 0: DOF {3}.
    // Free: {2, 4, 5, 6, 7}.
    // Tension P at right-side nodes 1 and 3:
    Eigen::VectorXd f_ext = Eigen::VectorXd::Zero(8);
    f_ext(2) = 3.0;   // node 1 x
    f_ext(6) = 3.0;   // node 3 x

    auto result = solve_newton(
        [&](const Eigen::VectorXd& u) {
            return assemble_q4(u, svk);
        },
        f_ext, {2, 4, 5, 6, 7}, Eigen::VectorXd::Zero(8));

    report("newton_Q4_SVK_converged", result.converged);
    report("newton_Q4_SVK_iters", result.iterations < 15);

    // K should be symmetric at equilibrium
    auto [f, K] = assemble_q4(result.u, svk);
    double asym = (K - K.transpose()).cwiseAbs().maxCoeff();
    report("newton_Q4_SVK_K_sym", asym < 1e-10);

    // Equilibrium on free DOFs
    Eigen::VectorXd R = f - f_ext;
    double max_R = 0.0;
    for (auto idx : {2, 4, 5, 6, 7})
        max_R = std::max(max_R, std::abs(R(idx)));
    report("newton_Q4_SVK_equilibrium", max_R < 1e-10);

    // Convergence rate
    double q = estimate_convergence_order(result.residuals);
    report("newton_Q4_SVK_quadratic", q > 1.7);

    std::cout << "    residuals:";
    for (double r : result.residuals) std::cout << " " << r;
    std::cout << "  (q ≈ " << q << ")\n";
}


// =============================================================================
//  9.  Load stepping — incremental 3D NH
// =============================================================================

void test_load_stepping() {
    CompressibleNeoHookean<3> nh{lam_ref, mu_ref};
    auto grad = grad_tet_3d();

    const double P_total = 40.0;       // total force per node in x
    constexpr int n_steps = 5;
    const double dP = P_total / n_steps;

    std::vector<Eigen::Index> free = {3, 6, 7, 9, 10, 11};

    Eigen::VectorXd u = Eigen::VectorXd::Zero(12);
    bool all_converged = true;
    int  total_iters   = 0;

    for (int step = 1; step <= n_steps; ++step) {
        const double P = dP * step;
        Eigen::VectorXd f_ext = Eigen::VectorXd::Zero(12);
        f_ext(3) = P;
        f_ext(6) = P;
        f_ext(9) = P;

        auto result = solve_newton(
            [&](const Eigen::VectorXd& u_) {
                return assemble_gp<3>(grad, 3, u_, nh);
            },
            f_ext, free, u);

        if (!result.converged) { all_converged = false; break; }
        total_iters += result.iterations;
        u = result.u;
    }

    report("load_stepping_all_converged", all_converged);
    report("load_stepping_total_iters", total_iters < n_steps * 15);

    // Verify final equilibrium at full load
    Eigen::VectorXd f_ext_final = Eigen::VectorXd::Zero(12);
    f_ext_final(3) = P_total;
    f_ext_final(6) = P_total;
    f_ext_final(9) = P_total;

    auto [f, K] = assemble_gp<3>(grad, 3, u, nh);
    double max_R_free = 0.0;
    for (auto idx : free)
        max_R_free = std::max(max_R_free, std::abs(f(idx) - f_ext_final(idx)));
    report("load_stepping_final_equilibrium", max_R_free < 1e-10);

    // Positive energy at final state
    auto kin = TotalLagrangian::evaluate_from_gradients<3>(grad_tet_3d(), 3, u);
    auto E_tensor = strain::green_lagrange(kin.F);
    double W = nh.energy(E_tensor);
    report("load_stepping_positive_energy", W > 0.0);

    // detF > 0 (no element inversion)
    report("load_stepping_detF_positive", kin.detF > 0.0);

    std::cout << "    u(node 1 x) = " << u(3)
              << ",  detF = " << kin.detF
              << ",  W = " << W << "\n";
}


// =============================================================================
//  10.  Patch test — uniform deformation → exact F, zero net force
// =============================================================================

void test_patch_test() {
    // 2D CST: Apply uniform F₀ = [[1.1, 0.05], [−0.02, 0.95]]
    // u_I = (F₀ − I) · X_I
    {
        SaintVenantKirchhoff<2> svk{lam_ref, mu_ref};
        auto grad = grad_cst_2d();

        Eigen::Matrix2d F0;
        F0 << 1.1, 0.05,
             -0.02, 0.95;

        Eigen::Matrix2d I2 = Eigen::Matrix2d::Identity();
        Eigen::Matrix2d dF = F0 - I2;

        // Nodes: (0,0), (1,0), (0,1)
        Eigen::VectorXd u(6);
        // Node 0 at (0,0): u = dF · [0;0] = 0
        u(0) = 0.0;  u(1) = 0.0;
        // Node 1 at (1,0): u = dF · [1;0]
        u(2) = dF(0, 0);  u(3) = dF(1, 0);
        // Node 2 at (0,1): u = dF · [0;1]
        u(4) = dF(0, 1);  u(5) = dF(1, 1);

        auto kin = TotalLagrangian::evaluate_from_gradients<2>(grad, 2, u);

        // F at GP should be F₀
        double F_err = 0.0;
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                F_err = std::max(F_err, std::abs(kin.F(i, j) - F0(i, j)));
        report("patch_2D_F_exact", F_err < 1e-14);

        // Net force = Σ f_int should be zero
        auto [f, K] = assemble_gp<2>(grad, 2, u, svk);
        // Sum forces per direction: x-sum, y-sum
        double fx = f(0) + f(2) + f(4);
        double fy = f(1) + f(3) + f(5);
        report("patch_2D_net_force_zero",
            std::abs(fx) < 1e-12 && std::abs(fy) < 1e-12);
    }

    // 3D tet: Apply uniform F₀ = diag(1.08, 0.97, 1.03)
    {
        CompressibleNeoHookean<3> nh{lam_ref, mu_ref};
        auto grad = grad_tet_3d();

        Eigen::Matrix3d F0 = Eigen::Matrix3d::Identity();
        F0(0, 0) = 1.08;  F0(1, 1) = 0.97;  F0(2, 2) = 1.03;
        Eigen::Matrix3d dF = F0 - Eigen::Matrix3d::Identity();

        // Nodes: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
        Eigen::VectorXd u(12);
        Eigen::Vector3d X[4] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        for (int I = 0; I < 4; ++I) {
            Eigen::Vector3d uI = dF * X[I];
            u(3 * I)     = uI(0);
            u(3 * I + 1) = uI(1);
            u(3 * I + 2) = uI(2);
        }

        auto kin = TotalLagrangian::evaluate_from_gradients<3>(grad, 3, u);

        // F at GP should be F₀
        double F_err = 0.0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                F_err = std::max(F_err, std::abs(kin.F(i, j) - F0(i, j)));
        report("patch_3D_F_exact", F_err < 1e-14);

        // Net force = 0
        auto [f, K] = assemble_gp<3>(grad, 3, u, nh);
        double fx = 0.0, fy = 0.0, fz = 0.0;
        for (int I = 0; I < 4; ++I) {
            fx += f(3 * I);
            fy += f(3 * I + 1);
            fz += f(3 * I + 2);
        }
        report("patch_3D_net_force_zero",
            std::abs(fx) < 1e-11 && std::abs(fy) < 1e-11 &&
            std::abs(fz) < 1e-11);

        // E should match analytical: E = ½(F₀ᵀF₀ − I)
        auto E_tensor = strain::green_lagrange(kin.F);
        auto E_exact_mat = 0.5 * (F0.transpose() * F0 - Eigen::Matrix3d::Identity());

        // Compare diagonal tensor-Voigt components
        report("patch_3D_E11", approx(E_tensor[0], E_exact_mat(0, 0), 1e-14));
        report("patch_3D_E22", approx(E_tensor[1], E_exact_mat(1, 1), 1e-14));
        report("patch_3D_E33", approx(E_tensor[2], E_exact_mat(2, 2), 1e-14));
    }
}


// =============================================================================
//  main
// =============================================================================

int main() {
    std::cout << "\n══════════════════════════════════════════════════════\n"
              << "  Newton-Raphson Element Tests (Phase 5)\n"
              << "══════════════════════════════════════════════════════\n";

    // 1. 1D SVK Newton
    std::cout << "\n── 1D bar SVK Newton ──\n";
    test_1d_svk_newton();

    // 2. 1D SVK convergence rate
    std::cout << "\n── 1D SVK convergence rate ──\n";
    test_1d_svk_convergence_rate();

    // 3. 1D displacement-controlled
    std::cout << "\n── 1D displacement-controlled ──\n";
    test_1d_svk_displacement_controlled();

    // 4. 2D CST SVK
    std::cout << "\n── 2D CST SVK Newton ──\n";
    test_2d_cst_svk_newton();

    // 5. 2D CST NH
    std::cout << "\n── 2D CST NH Newton ──\n";
    test_2d_cst_nh_newton();

    // 6. 3D tet NH
    std::cout << "\n── 3D tet NH Newton ──\n";
    test_3d_tet_nh_newton();

    // 7. 3D tet NH convergence rate
    std::cout << "\n── 3D tet NH convergence rate ──\n";
    test_3d_tet_nh_convergence_rate();

    // 8. Q4 multi-GP SVK
    std::cout << "\n── Q4 multi-GP SVK Newton ──\n";
    test_q4_svk_newton();

    // 9. Load stepping
    std::cout << "\n── Load stepping 3D NH ──\n";
    test_load_stepping();

    // 10. Patch test
    std::cout << "\n── Patch test ──\n";
    test_patch_test();

    // Summary
    std::cout << "\n══════════════════════════════════════════════════════\n"
              << "  Results: " << passed << " passed, " << failed << " failed"
              << " (total " << (passed + failed) << ")\n"
              << "══════════════════════════════════════════════════════\n\n";

    return failed > 0 ? 1 : 0;
}
