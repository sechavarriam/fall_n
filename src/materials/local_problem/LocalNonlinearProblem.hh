#ifndef FALL_N_LOCAL_NONLINEAR_PROBLEM_HH
#define FALL_N_LOCAL_NONLINEAR_PROBLEM_HH

#include <cmath>
#include <concepts>
#include <type_traits>
#include <utility>

// =============================================================================
//  LocalNonlinearProblem.hh
// =============================================================================
//
//  General compile-time contract for constitutive local problems of the form
//
//      R(x) = 0,
//
//  where:
//    - x is a local unknown (scalar or fixed-size vector),
//    - R is the local residual,
//    - K = dR/dx is the local Jacobian / consistent linearization.
//
//  The important point is architectural, not algorithmic: finite-strain
//  plasticity, damage, viscoplasticity, and capped models all differ mainly in
//  the definition of x, R, and K.  The nonlinear solver should therefore act on
//  an abstract local problem, not directly on a specific J2 scalar equation.
//
//  The current small-strain radial return is then just one instantiation:
//
//      x = Δγ \in R
//      R(x) = scalar consistency equation
//      K(x) = scalar derivative
//
//  Future finite-strain models can promote x to a fixed-size vector containing,
//  for example, a plastic multiplier plus internal tensorial variables, while
//  reusing the same solver and type-erased constitutive boundary.
//
// =============================================================================

template <typename Problem, typename Relation, typename ContextT>
concept LocalNonlinearProblem = requires(
    const Problem& problem,
    const Relation& relation,
    const ContextT& context,
    const typename Problem::UnknownT& unknown,
    const typename Problem::ResidualT& residual)
{
    typename Problem::UnknownT;
    typename Problem::ResidualT;
    typename Problem::JacobianT;

    { problem.initial_guess(relation, context) }
        -> std::same_as<typename Problem::UnknownT>;
    { problem.residual(relation, context, unknown) }
        -> std::same_as<typename Problem::ResidualT>;
    { problem.jacobian(relation, context, unknown) }
        -> std::same_as<typename Problem::JacobianT>;
    { problem.residual_norm(residual) } -> std::convertible_to<double>;
};

namespace local_nonlinear_problem {

template <typename T>
concept ScalarLike = std::is_arithmetic_v<std::remove_cvref_t<T>>;

template <typename ResidualT>
[[nodiscard]] inline double residual_norm(const ResidualT& residual)
{
    if constexpr (ScalarLike<ResidualT>) {
        return std::abs(residual);
    } else {
        return residual.norm();
    }
}

template <typename UnknownT, typename JacobianT, typename ResidualT>
[[nodiscard]] inline UnknownT solve_linearized_update(
    const JacobianT& jacobian,
    const ResidualT& residual)
{
    if constexpr (ScalarLike<UnknownT>) {
        return -residual / jacobian;
    } else {
        return jacobian.fullPivLu().solve(-residual);
    }
}

template <typename UnknownT>
inline void add_update(UnknownT& unknown, const UnknownT& delta_unknown)
{
    unknown += delta_unknown;
}

template <typename Problem, typename UnknownT>
inline void project_iterate(const Problem& problem, UnknownT& unknown)
{
    if constexpr (requires { problem.project_iterate(unknown); }) {
        problem.project_iterate(unknown);
    }
}

} // namespace local_nonlinear_problem

#endif // FALL_N_LOCAL_NONLINEAR_PROBLEM_HH
