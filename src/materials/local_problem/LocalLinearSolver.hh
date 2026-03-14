#ifndef FALL_N_LOCAL_LINEAR_SOLVER_HH
#define FALL_N_LOCAL_LINEAR_SOLVER_HH

#include <concepts>
#include <type_traits>

// =============================================================================
//  LocalLinearSolver.hh
// =============================================================================
//
//  Compile-time customization point for the linearized solve that appears
//  inside Newton-type local constitutive algorithms:
//
//      K(x^i) Δx^i = -R(x^i).
//
//  Keeping this as an explicit policy matters because different local problems
//  may prefer different linear solvers:
//    - direct scalar division for one-dimensional updates,
//    - dense LU for small fixed-size vector problems,
//    - custom factorizations,
//    - future PETSc/KSP-based solves for larger local systems or research
//      experiments.
//
// =============================================================================

template <typename Solver, typename UnknownT, typename JacobianT, typename ResidualT>
concept LocalLinearSolvePolicy = requires(
    const Solver& solver,
    const JacobianT& jacobian,
    const ResidualT& residual)
{
    { solver.solve(jacobian, residual) } -> std::same_as<UnknownT>;
};

struct DefaultLocalLinearSolver {
    template <typename JacobianT, typename ResidualT>
    [[nodiscard]] auto solve(
        const JacobianT& jacobian,
        const ResidualT& residual) const
    {
        if constexpr (std::is_arithmetic_v<std::remove_cvref_t<ResidualT>>) {
            return -residual / jacobian;
        } else {
            return jacobian.fullPivLu().solve(-residual).eval();
        }
    }
};

#endif // FALL_N_LOCAL_LINEAR_SOLVER_HH
