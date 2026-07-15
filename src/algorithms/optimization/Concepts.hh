#ifndef FALL_N_ALGORITHMS_OPTIMIZATION_CONCEPTS_HH
#define FALL_N_ALGORITHMS_OPTIMIZATION_CONCEPTS_HH
// =============================================================================
//  Concepts.hh  --  src/algorithms/optimization
//
//  Compile-time contracts for the metaheuristic optimisation layer.
//
//  Convention: objectives are always MAXIMISED. A higher return value means a
//  fitter individual. Minimisation problems are expressed by negating the cost
//  at the objective boundary (see the Cultural Algorithm tests).
//
//  The layer is std-only: no PETSc, Eigen or FEM dependencies. A genome is a
//  contiguous block of doubles viewed through std::span; storage is the
//  caller's std::vector<double>.
// =============================================================================

#include <concepts>
#include <cstddef>
#include <random>
#include <span>
#include <vector>

namespace fall_n::algorithms {

/// An objective function maps a genome to a scalar fitness (to be maximised).
template <class F>
concept ObjectiveFunction = requires(F f, std::span<const double> genome) {
    { f(genome) } -> std::convertible_to<double>;
};

/// A bounded, samplable search space over a fixed-dimension real genome.
///  - dimension()  number of genes.
///  - lower(i)/upper(i)  per-gene box bounds.
///  - clamp(g)     projects a genome back into the box, in place.
///  - sample(rng)  draws a uniformly-random admissible genome.
template <class S>
concept SearchSpace =
    requires(const S s, std::size_t i, std::span<double> g, std::mt19937_64 rng) {
        { s.dimension() } -> std::convertible_to<std::size_t>;
        { s.lower(i) }    -> std::convertible_to<double>;
        { s.upper(i) }    -> std::convertible_to<double>;
        { s.clamp(g) }    -> std::same_as<void>;
        { s.sample(rng) } -> std::convertible_to<std::vector<double>>;
    };

}  // namespace fall_n::algorithms

#endif  // FALL_N_ALGORITHMS_OPTIMIZATION_CONCEPTS_HH
