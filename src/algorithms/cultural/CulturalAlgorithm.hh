#ifndef FALL_N_ALGORITHMS_CULTURAL_CULTURAL_ALGORITHM_HH
#define FALL_N_ALGORITHMS_CULTURAL_CULTURAL_ALGORITHM_HH
// =============================================================================
//  CulturalAlgorithm.hh  --  src/algorithms/cultural
//
//  Reynolds' Cultural Algorithm over a bounded real search space. A population
//  evolves under variation biased by a belief space that is itself updated from
//  the accepted elite each generation. The algorithm is a META-heuristic: the
//  population space is an interchangeable model (evolutionary programming by
//  default; a genetic-recombination model is provided as one alternative
//  instantiation -- see PopulationSpace.hh), and the belief space aggregates
//  any subset of the literature's knowledge sources (normative, situational,
//  domain, history, topographic -- see KnowledgeSources.hh).
//
//  The objective is MAXIMISED. The run is fully deterministic given
//  CulturalConfig::seed (the RNG is the only source of randomness and it is
//  seeded from the config).
//
//  Structure (dual inheritance loop):
//    population  ->  rank  ->  belief.accept(elite)  ->  variation, each child
//    influenced by the belief space with probability p_influence or varied by
//    the population model otherwise  ->  elitism  ->  next population.
//
//  Future hook (control-device optimisation, not implemented here): the same
//  maximize() drives device tuning if the objective closes over a
//  DynamicAnalysis. The traits carry TMD/TID parameters; the objective sets
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
#include <type_traits>
#include <utility>
#include <vector>

namespace fall_n::algorithms::cultural {

struct CulturalConfig {
    std::size_t   population_size      = 24;
    std::size_t   max_generations      = 50;
    double        acceptance_fraction  = 0.25;
    double        influence_probability = 0.6;
    //  Sigma of the population model's own variation operator, as a fraction
    //  of each trait's box extent (formerly "mutation_sigma_frac" -- renamed:
    //  the operator belongs to the population MODEL, not to a genetic
    //  instantiation in particular).
    double        variation_sigma_frac = 0.1;
    std::uint64_t seed                 = 0xC0FFEEULL;
    std::size_t   stall_generations    = 0;   // 0 => disabled
    double        target_fitness       = std::numeric_limits<double>::infinity();
};

struct CulturalResult {
    Individual          best{};
    std::size_t         generations = 0;
    std::vector<double> best_fitness_history{};
    // Per-generation trace for offline tuning reports (ca_history.csv): the
    //  population mean and the best-so-far traits, aligned index-by-index with
    //  best_fitness_history.
    std::vector<double>              mean_fitness_history{};
    std::vector<std::vector<double>> best_traits_history{};
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
              class Rng = std::mt19937_64,
              class PopModel = EvolutionaryProgrammingModel>
        requires PopulationModel<PopModel, Space, Rng>
    [[nodiscard]] CulturalResult maximize(Objective&& objective,
                                          PopModel pop_model = {}) {
        const std::size_t pop_size = (cfg_.population_size < 2) ? 2 : cfg_.population_size;

        //  The default-constructed EP model takes its sigma from the config so
        //  the historical single-knob interface keeps working; an explicitly
        //  supplied model is used as-is.
        if constexpr (std::is_same_v<PopModel, EvolutionaryProgrammingModel>) {
            pop_model.sigma_fraction = cfg_.variation_sigma_frac;
        }

        Rng rng(cfg_.seed);
        CulturalResult result;

        auto evaluate = [&](std::vector<double> traits) -> Individual {
            space_.clamp(std::span<double>(traits));
            const double f =
                static_cast<double>(objective(std::span<const double>(traits)));
            return Individual{std::move(traits), f};
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
                    pop_model.propose(std::span<const Individual>(pop), rng);
                if (unit(rng) < cfg_.influence_probability) {
                    belief.influence(std::span<double>(child), rng, space_);
                } else {
                    pop_model.vary(std::span<double>(child), rng, space_);
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
            result.best_traits_history.push_back(best.traits);

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
