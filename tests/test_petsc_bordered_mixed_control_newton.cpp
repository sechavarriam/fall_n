#include "src/analysis/PetscBorderedMixedControlNewton.hh"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

petsc::OwnedVec make_seq_vec(std::vector<double> values)
{
    petsc::OwnedVec v;
    VecCreateSeq(
        PETSC_COMM_SELF,
        static_cast<PetscInt>(values.size()),
        v.ptr());
    for (PetscInt i = 0; i < static_cast<PetscInt>(values.size()); ++i) {
        VecSetValue(v.get(), i, values[static_cast<std::size_t>(i)], INSERT_VALUES);
    }
    VecAssemblyBegin(v.get());
    VecAssemblyEnd(v.get());
    return v;
}

petsc::OwnedMat make_seq_mat(PetscInt rows, PetscInt cols)
{
    petsc::OwnedMat m;
    MatCreateSeqAIJ(PETSC_COMM_SELF, rows, cols, cols, nullptr, m.ptr());
    MatSetOption(m.get(), MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    return m;
}

double get_value(Vec v, PetscInt i)
{
    PetscScalar value = 0.0;
    VecGetValues(v, 1, &i, &value);
    return static_cast<double>(value);
}

void assert_close(double a, double b, double tol, const char* message)
{
    if (std::abs(a - b) > tol) {
        std::cerr << message << ": expected " << b << ", got " << a << '\n';
        std::abort();
    }
}

int g_failed = 0;

void check(bool condition, const char* message)
{
    if (condition) {
        std::cout << "  PASS  " << message << "\n";
    } else {
        ++g_failed;
        std::cout << "  FAIL  " << message << "\n";
    }
}

fall_n::PetscBorderedMixedControlEvaluation make_linear_eval(
    const fall_n::PetscBorderedMixedControlState& state)
{
    const double u = get_value(state.unknowns, 0);
    const double lambda = state.load_parameter;
    fall_n::PetscBorderedMixedControlEvaluation e;
    e.residual = make_seq_vec({10.0 * u - lambda});
    e.tangent = make_seq_mat(1, 1);
    MatSetValue(e.tangent.get(), 0, 0, 10.0, INSERT_VALUES);
    MatAssemblyBegin(e.tangent.get(), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(e.tangent.get(), MAT_FINAL_ASSEMBLY);
    e.load_column = make_seq_vec({-1.0});
    e.constraint_gradient = make_seq_vec({1.0});
    e.constraint = u - 2.0;
    e.constraint_load_derivative = 0.0;
    return e;
}

fall_n::PetscBorderedMixedControlEvaluation make_nonlinear_eval(
    const fall_n::PetscBorderedMixedControlState& state)
{
    const double u = get_value(state.unknowns, 0);
    const double lambda = state.load_parameter;
    fall_n::PetscBorderedMixedControlEvaluation e;
    e.residual = make_seq_vec({u * u - lambda});
    e.tangent = make_seq_mat(1, 1);
    MatSetValue(e.tangent.get(), 0, 0, 2.0 * u, INSERT_VALUES);
    MatAssemblyBegin(e.tangent.get(), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(e.tangent.get(), MAT_FINAL_ASSEMBLY);
    e.load_column = make_seq_vec({-1.0});
    e.constraint_gradient = make_seq_vec({1.0});
    e.constraint = u - 3.0;
    e.constraint_load_derivative = 0.0;
    return e;
}

fall_n::PetscBorderedMixedControlEvaluation make_fixed_lambda_eval(
    const fall_n::PetscBorderedMixedControlState& state)
{
    const double u = get_value(state.unknowns, 0);
    const double lambda = state.load_parameter;
    fall_n::PetscBorderedMixedControlEvaluation e;
    e.residual = make_seq_vec({u * u - lambda});
    e.tangent = make_seq_mat(1, 1);
    MatSetValue(e.tangent.get(), 0, 0, 2.0 * u, INSERT_VALUES);
    MatAssemblyBegin(e.tangent.get(), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(e.tangent.get(), MAT_FINAL_ASSEMBLY);
    e.load_column = make_seq_vec({-1.0});
    e.constraint_gradient = make_seq_vec({0.0});
    e.constraint = lambda - 1.0;
    e.constraint_load_derivative = 1.0;
    return e;
}

void test_linear_displacement_constraint()
{
    auto u0 = make_seq_vec({0.0});
    const auto result = fall_n::solve_petsc_bordered_mixed_control_newton(
        fall_n::PetscBorderedMixedControlState{
            .unknowns = u0.get(),
            .load_parameter = 0.0},
        make_linear_eval,
        fall_n::PetscBorderedMixedControlNewtonSettings{
            .residual_tolerance = 1.0e-12,
            .constraint_tolerance = 1.0e-12});

    assert(result.converged());
    assert_close(get_value(result.unknowns.get(), 0), 2.0, 1.0e-12,
                 "PETSc bordered Newton enforces displacement constraint");
    assert_close(result.load_parameter, 20.0, 1.0e-12,
                 "PETSc bordered Newton recovers load parameter");
    assert(result.augmented_system_allocations == 4);
    assert(result.evaluator_calls >= result.iterations);
}

void test_fixed_control_residual_line_search_damps_bad_newton_step()
{
    auto u0 = make_seq_vec({0.1});
    const auto result = fall_n::solve_petsc_bordered_mixed_control_newton(
        fall_n::PetscBorderedMixedControlState{
            .unknowns = u0.get(),
            .load_parameter = 1.0},
        make_fixed_lambda_eval,
        fall_n::PetscBorderedMixedControlNewtonSettings{
            .max_iterations = 20,
            .residual_tolerance = 1.0e-12,
            .constraint_tolerance = 1.0e-12,
            .line_search_enabled = true,
            .max_line_search_cutbacks = 16,
            .line_search_merit =
                fall_n::PetscBorderedLineSearchMeritKind::residual_only});

    assert(result.converged());
    assert_close(get_value(result.unknowns.get(), 0), 1.0, 1.0e-10,
                 "PETSc bordered fixed-control line search closes residual");
    assert_close(result.load_parameter, 1.0, 1.0e-12,
                 "PETSc bordered fixed-control line search keeps lambda fixed");
    assert(result.augmented_system_allocations == 4);
    assert(result.evaluator_calls > result.iterations);
}

void test_nonlinear_constraint_problem()
{
    auto u0 = make_seq_vec({1.0});
    const auto result = fall_n::solve_petsc_bordered_mixed_control_newton(
        fall_n::PetscBorderedMixedControlState{
            .unknowns = u0.get(),
            .load_parameter = 1.0},
        make_nonlinear_eval,
        fall_n::PetscBorderedMixedControlNewtonSettings{
            .residual_tolerance = 1.0e-12,
            .constraint_tolerance = 1.0e-12,
            .line_search_enabled = true});

    assert(result.converged());
    assert_close(get_value(result.unknowns.get(), 0), 3.0, 1.0e-12,
                 "PETSc bordered Newton closes nonlinear unknown");
    assert_close(result.load_parameter, 9.0, 1.0e-10,
                 "PETSc bordered Newton closes nonlinear load");
}

// ── Arc-length continuation THROUGH a limit point ────────────────────────────
//  Toy equilibrium R(u,lambda) = u^2 - lambda (a parabola lambda = u^2).  Under
//  LOAD control the tangent K = dR/du = 2u vanishes at the fold (u=0, lambda=0)
//  and Newton on lambda cannot cross it.  A spherical (Crisfield) arc-length
//  continuation must trace the whole parabola from u=+1 through u=0 to u=-1.
//  This exercises exactly what the FEM driver relies on (bordered algebra +
//  arc-length constraint + secant predictor across a singular tangent), fully
//  isolated from the FEM assembly, the finite-difference load column and the
//  penalty scaling.  If this fails, the bug is in the kernel/formulation; if it
//  passes, a downstream FEM failure is scaling/plumbing, not the arc-length math.
fall_n::PetscBorderedMixedControlEvaluation make_parabola_arc_eval(
    const fall_n::PetscBorderedMixedControlState& state,
    double u_ref,
    double lambda_ref,
    double ds,
    double psi)
{
    const double u = get_value(state.unknowns, 0);
    const double lambda = state.load_parameter;
    fall_n::PetscBorderedMixedControlEvaluation e;
    e.residual = make_seq_vec({u * u - lambda});
    e.tangent = make_seq_mat(1, 1);
    MatSetValue(e.tangent.get(), 0, 0, 2.0 * u, INSERT_VALUES);  // K = 2u -> 0
    MatAssemblyBegin(e.tangent.get(), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(e.tangent.get(), MAT_FINAL_ASSEMBLY);
    e.load_column = make_seq_vec({-1.0});  // dR/dlambda = -1
    const double du = u - u_ref;
    const double dl = lambda - lambda_ref;
    e.constraint_gradient = make_seq_vec({2.0 * du});            // dc/du = 2 du
    e.constraint = du * du + psi * psi * dl * dl - ds * ds;      // spherical
    e.constraint_load_derivative = 2.0 * psi * psi * dl;         // dc/dlambda
    return e;
}

void test_arc_length_traces_through_limit_point(bool with_regularization)
{
    const double ds = 0.1;
    const double psi = 1.0;  // spherical
    double u_cur = 1.0, lam_cur = 1.0;    // on the path: 1^2 = 1
    double u_prev = 1.0, lam_prev = 1.0;
    bool have_prev = false;

    double u_min = u_cur, u_max = u_cur, max_residual = 0.0;
    bool crossed_fold = false;  // saw a converged point with |K|=|2u| < 0.2
    int accepted = 0;

    for (int step = 0; step < 200 && u_cur > -1.0; ++step) {
        double u_pred = 0.0, lam_pred = 0.0;
        if (have_prev) {  // secant predictor
            const double du = u_cur - u_prev;
            const double dl = lam_cur - lam_prev;
            const double nrm = std::sqrt(du * du + psi * psi * dl * dl);
            const double sc = nrm > 1.0e-14 ? ds / nrm : 0.0;
            u_pred = u_cur + sc * du;
            lam_pred = lam_cur + sc * dl;
        } else {          // first step: nudge toward DECREASING u (toward fold)
            u_pred = u_cur - ds;
            lam_pred = lam_cur;
        }
        const double u_ref = u_cur, lam_ref = lam_cur;
        auto u_pred_vec = make_seq_vec({u_pred});

        const auto res = fall_n::solve_petsc_bordered_mixed_control_newton(
            fall_n::PetscBorderedMixedControlState{
                .unknowns = u_pred_vec.get(),
                .load_parameter = lam_pred},
            [&](const fall_n::PetscBorderedMixedControlState& st) {
                return make_parabola_arc_eval(st, u_ref, lam_ref, ds, psi);
            },
            [&] {
                fall_n::PetscBorderedMixedControlNewtonSettings s;
                s.max_iterations = 40;
                s.residual_tolerance = 1.0e-10;
                s.constraint_tolerance = 1.0e-10;
                s.line_search_enabled = true;
                s.max_line_search_cutbacks = 20;
                if (with_regularization) {
                    s.regularization_mu_frac = 1.0e-3;  // ejercita el LM del kernel
                    s.regularization_mu_max_frac = 1.0e-1;
                }
                return s;
            }());

        if (!res.converged()) {
            std::cerr << "  arc step " << step << " did NOT converge near u="
                      << u_cur << " (status " << static_cast<int>(res.status)
                      << ")\n";
            break;
        }
        u_prev = u_cur;
        lam_prev = lam_cur;
        u_cur = get_value(res.unknowns.get(), 0);
        lam_cur = res.load_parameter;
        // residual of the accepted point on the true path (should be ~0)
        max_residual = std::max(max_residual, std::abs(u_cur * u_cur - lam_cur));
        u_min = std::min(u_min, u_cur);
        u_max = std::max(u_max, u_cur);
        if (std::abs(2.0 * u_cur) < 0.2) crossed_fold = true;
        ++accepted;
    }

    check(accepted >= 15,
          "arc-length continuation advances many steps along the parabola");
    check(crossed_fold,
          "arc-length lands a converged point in the singular-tangent zone (u~0)");
    check(u_min < -0.5,
          "arc-length CROSSES the load limit point (u goes from +1 past 0 to <-0.5)");
    check(max_residual < 1.0e-6,
          "every accepted arc-length point stays on the true equilibrium path");
}

void test_invalid_evaluation_is_rejected()
{
    auto u0 = make_seq_vec({0.0, 0.0});
    auto invalid = [](const fall_n::PetscBorderedMixedControlState&) {
        fall_n::PetscBorderedMixedControlEvaluation e;
        e.residual = make_seq_vec({0.0});
        e.tangent = make_seq_mat(1, 1);
        MatSetValue(e.tangent.get(), 0, 0, 1.0, INSERT_VALUES);
        MatAssemblyBegin(e.tangent.get(), MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(e.tangent.get(), MAT_FINAL_ASSEMBLY);
        e.load_column = make_seq_vec({1.0});
        e.constraint_gradient = make_seq_vec({1.0});
        return e;
    };

    const auto result = fall_n::solve_petsc_bordered_mixed_control_newton(
        fall_n::PetscBorderedMixedControlState{
            .unknowns = u0.get(),
            .load_parameter = 0.0},
        invalid);
    assert(result.status ==
           fall_n::BorderedMixedControlNewtonStatus::invalid_evaluation);
}

} // namespace

int main(int argc, char** argv)
{
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    test_linear_displacement_constraint();
    test_fixed_control_residual_line_search_damps_bad_newton_step();
    test_nonlinear_constraint_problem();
    test_invalid_evaluation_is_rejected();
    std::cout << "=== arc-length through a limit point (mu=0) ===\n";
    test_arc_length_traces_through_limit_point(false);
    std::cout << "=== arc-length through a limit point (kernel LM mu>0) ===\n";
    test_arc_length_traces_through_limit_point(true);
    PetscFinalize();
    return g_failed == 0 ? 0 : 1;
}
