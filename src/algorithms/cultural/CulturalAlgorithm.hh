#ifndef FALL_N_ALGORITHMS_CULTURAL_CULTURAL_ALGORITHM_HH
#define FALL_N_ALGORITHMS_CULTURAL_CULTURAL_ALGORITHM_HH
// =============================================================================
//  CulturalAlgorithm.hh  --  src/algorithms/cultural
//
//  Reynolds' Cultural Algorithm over a bounded real search space. A population
//  evolves under variation biased by a belief space that is itself updated from
//  the accepted elite each generation.
//
//  The objective is MAXIMISED. The run is fully deterministic given
//  CulturalConfig::seed (the RNG is the only source of randomness and it is
//  seeded from the config).
//
//  Structure:
//    population  ->  rank  ->  belief.accept(elite)  ->  variation(belief)  ->
//    elitism  ->  next population.
//
//  Future hook (control-device optimisation, not implemented here): the same
//  maximize() drives device tuning if the objective closes over a
//  DynamicAnalysis. The genome carries TMD/TID parameters; the objective sets
//  them through set_internal_residual_hook, runs the analysis from a captured
//  checkpoint under a ground motion, and returns -peak_drift. No FEM dependency
//  leaks into this header: the coupling lives entirely in the objective the
//  caller supplies.
// =============================================================================

#include "src/algorithms/optimization/Concepts.hh"
#include "src/algorithms/cultural/BeliefSpace.hh"
#include "src/algorithms/cultural/Individual.hh"
#include "src/algorithms/cultural/PopulationSpace.hh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <utility>
#include <vector>

namespace fall_n::algorithms::cultural {

struct CulturalConfig {
    std::size_t   population_size      = 24;
    std::size_t   max_generations      = 50;
    double        acceptance_fraction  = 0.25;
    double        influence_probability = 0.6;
    double        mutation_sigma_frac  = 0.1;
    std::uint64_t seed                 = 0xC0FFEEULL;
    std::size_t   stall_generations    = 0;   // 0 => disabled
    double        target_fitness       = std::numeric_limits<double>::infinity();
};

struct CulturalResult {
    Individual          best{};
    std::size_t         generations = 0;
    std::vector<double> best_fitness_history{};
    // Per-generation trace for offline tuning reports (ca_history.csv): the
    //  population mean and the best-so-far genome, aligned index-by-index with
    //  best_fitness_history.
    std::vector<double>              mean_fitness_history{};
    std::vector<std::vector<double>> best_genome_history{};
};

template <fall_n::algorithms::SearchSpace Space, class... KnowledgeSources>
class CulturalAlgorithm {
    Space          space_;
    CulturalConfig cfg_;

    [[nodiscard]] static const Individual& fittest(const std::vector<Individual>& pop) {
        return *std::max_element(
            pop.begin(), pop.end(),
            [](const Individual& a, const Individual& b) { return a.fitness < b.fitness; });
    }

public:
    CulturalAlgorithm(Space space, CulturalConfig cfg)
        : space_(std::move(space)), cfg_(cfg) {}

    template <fall_n::algorithms::ObjectiveFunction Objective,
              class Rng = std::mt19937_64>
    [[nodiscard]] CulturalResult maximize(Objective&& objective) {
        const std::size_t pop_size = (cfg_.population_size < 2) ? 2 : cfg_.population_size;

        Rng rng(cfg_.seed);
        CulturalResult result;

        auto evaluate = [&](std::vector<double> genome) -> Individual {
            space_.clamp(std::span<double>(genome));
            const double f =
                static_cast<double>(objective(std::span<const double>(genome)));
            return Individual{std::move(genome), f};
        };

        std::vector<Individual> pop;
        pop.reserve(pop_size);
        for (std::size_t i = 0; i < pop_size; ++i) {
            pop.push_back(evaluate(space_.sample(rng)));
        }

        BeliefSpace<KnowledgeSources...> belief{
            TopFractionAcceptance{cfg_.acceptance_fraction}};

        Individual best = fittest(pop);
        std::size_t stall = 0;
        std::uniform_real_distribution<double> unit(0.0, 1.0);

        std::size_t gen = 0;
        for (; gen < cfg_.max_generations; ++gen) {
            std::sort(pop.begin(), pop.end(),
                      [](const Individual& a, const Individual& b) {
                          return a.fitness > b.fitness;   // descending
                      });

            belief.accept(std::span<const Individual>(pop), space_);

            std::vector<Individual> next;
            next.reserve(pop_size);
            next.push_back(best);   // elitism
            while (next.size() < pop_size) {
                std::vector<double> child =
                    binary_tournament(std::span<const Individual>(pop), rng).genome;
                if (unit(rng) < cfg_.influence_probability) {
                    belief.influence(std::span<double>(child), rng, space_);
                } else {
                    gaussian_mutation(std::span<double>(child),
                                      cfg_.mutation_sigma_frac, rng, space_);
                }
                next.push_back(evaluate(std::move(child)));
            }
            pop.swap(next);

            const Individual& gbest = fittest(pop);
            if (gbest.fitness > best.fitness) { best = gbest; stall = 0; }
            else                              { ++stall; }

            result.best_fitness_history.push_back(best.fitness);
            double fitness_sum = 0.0;
            for (const auto& ind : pop) { fitness_sum += ind.fitness; }
            result.mean_fitness_history.push_back(
                fitness_sum / static_cast<double>(pop.size()));
            result.best_genome_history.push_back(best.genome);

            if (best.fitness >= cfg_.target_fitness) { ++gen; break; }
            if (cfg_.stall_generations > 0 && stall >= cfg_.stall_generations) {
                ++gen;
                break;
            }
        }

        result.best        = best;
        result.generations = gen;
        return result;
    }
};

}  // namespace fall_n::algorithms::cultural

#endif  // FALL_N_ALGORITHMS_CULTURAL_CULTURAL_ALGORITHM_HH
