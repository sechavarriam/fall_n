#ifndef FALL_N_ALGORITHMS_CULTURAL_POPULATION_SPACE_HH
#define FALL_N_ALGORITHMS_CULTURAL_POPULATION_SPACE_HH
// =============================================================================
//  PopulationSpace.hh  --  src/algorithms/cultural
//
//  Population-space models. The Cultural Algorithm is a META-heuristic
//  (Reynolds): the population space can host any evolutionary substrate --
//  evolutionary programming, a genetic algorithm, evolution strategies, ... --
//  while the belief space stays the same. A PopulationModel supplies the two
//  population-side operators the generational loop needs:
//
//    propose(pop, rng)        -> traits of a new candidate (parent selection
//                                and, if the model uses it, recombination);
//    vary(traits, rng, space) -> the model's own variation operator, applied
//                                when the belief space does NOT influence the
//                                candidate this time.
//
//  Provided models:
//    EvolutionaryProgrammingModel -- binary tournament + Gaussian perturbation
//                                    of every trait (the classical CA pairing;
//                                    default, and the historical behaviour of
//                                    this module).
//    GeneticRecombinationModel    -- two-tournament BLX-alpha blend + Gaussian
//                                    perturbation; the "genetic algorithm"
//                                    reading is one instantiation, not the
//                                    framework.
// =============================================================================

#include "src/algorithms/cultural/Individual.hh"

#include <cstddef>
#include <random>
#include <span>
#include <vector>

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

/// Add a zero-mean Gaussian step to every trait, sigma scaled by the box extent.
template <class Rng, class Space>
void gaussian_perturbation(std::span<double> t, double sigma_fraction, Rng& rng,
                           const Space& space) {
    for (std::size_t i = 0; i < t.size(); ++i) {
        const double reach = space.upper(i) - space.lower(i);
        const double sigma = sigma_fraction * ((reach > 0.0) ? reach : 1.0);
        std::normal_distribution<double> step(0.0, sigma);
        t[i] += step(rng);
    }
}

/// Contract of a population-space model (checked with concrete types by the
/// unit tests; CulturalAlgorithm is duck-typed on the same shape).
template <class M, class Space, class Rng>
concept PopulationModel =
    requires(const M m, std::span<const Individual> pop, const Space& sp,
             std::span<double> t, Rng& rng) {
        { m.propose(pop, rng) } -> std::same_as<std::vector<double>>;
        { m.vary(t, rng, sp) }  -> std::same_as<void>;
    };

// -----------------------------------------------------------------------------
struct EvolutionaryProgrammingModel {
    double sigma_fraction = 0.1;

    template <class Rng>
    [[nodiscard]] std::vector<double> propose(std::span<const Individual> pop,
                                              Rng& rng) const {
        return binary_tournament(pop, rng).traits;
    }

    template <class Rng, class Space>
    void vary(std::span<double> t, Rng& rng, const Space& space) const {
        gaussian_perturbation(t, sigma_fraction, rng, space);
    }
};

// -----------------------------------------------------------------------------
struct GeneticRecombinationModel {
    double sigma_fraction = 0.05;
    double blend_alpha    = 0.5;   // BLX-alpha interval extension

    template <class Rng>
    [[nodiscard]] std::vector<double> propose(std::span<const Individual> pop,
                                              Rng& rng) const {
        const Individual& p1 = binary_tournament(pop, rng);
        const Individual& p2 = binary_tournament(pop, rng);
        std::vector<double> child(p1.traits);
        const std::size_t n = std::min(child.size(), p2.traits.size());
        for (std::size_t i = 0; i < n; ++i) {
            const double a = child[i], b = p2.traits[i];
            const double lo = std::min(a, b), hi = std::max(a, b);
            const double ext = blend_alpha * (hi - lo);
            std::uniform_real_distribution<double> blend(lo - ext, hi + ext);
            child[i] = blend(rng);
        }
        return child;
    }

    template <class Rng, class Space>
    void vary(std::span<double> t, Rng& rng, const Space& space) const {
        gaussian_perturbation(t, sigma_fraction, rng, space);
    }
};

}  // namespace fall_n::algorithms::cultural

#endif  // FALL_N_ALGORITHMS_CULTURAL_POPULATION_SPACE_HH
