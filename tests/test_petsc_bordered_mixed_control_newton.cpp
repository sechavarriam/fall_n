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
    PetscFinalize();
    return 0;
}
