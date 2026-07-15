#ifndef FALL_N_ALGORITHMS_CULTURAL_POPULATION_SPACE_HH
#define FALL_N_ALGORITHMS_CULTURAL_POPULATION_SPACE_HH
// =============================================================================
//  PopulationSpace.hh  --  src/algorithms/cultural
//
//  Population-level operators: binary-tournament selection and per-gene
//  Gaussian mutation. Kept as free functions so the generational loop in
//  CulturalAlgorithm.hh reads as plain data flow.
// =============================================================================

#include "src/algorithms/cultural/Individual.hh"

#include <cstddef>
#include <random>
#include <span>

namespace fall_n::algorithms::cultural {

/// Sample two individuals with replacement; return the fitter (ties -> first).
template <class Rng>
[[nodiscard]] const Individual& binary_tournament(std::span<const Individual> pop,
                                                  Rng& rng) {
    std::uniform_int_distribution<std::size_t> pick(0, pop.size() - 1);
    const Individual& a = pop[pick(rng)];
    const Individual& b = pop[pick(rng)];
    return (a.fitness >= b.fitness) ? a : b;
}

/// Add a zero-mean Gaussian step to every gene, sigma scaled by the box extent.
template <class Rng, class Space>
void gaussian_mutation(std::span<double> g, double sigma_fraction, Rng& rng,
                       const Space& space) {
    for (std::size_t i = 0; i < g.size(); ++i) {
        const double reach = space.upper(i) - space.lower(i);
        const double sigma = sigma_fraction * ((reach > 0.0) ? reach : 1.0);
        std::normal_distribution<double> step(0.0, sigma);
        g[i] += step(rng);
    }
}

}  // namespace fall_n::algorithms::cultural

#endif  // FALL_N_ALGORITHMS_CULTURAL_POPULATION_SPACE_HH
