#ifndef FALL_N_ALGORITHMS_CULTURAL_KNOWLEDGE_SOURCES_HH
#define FALL_N_ALGORITHMS_CULTURAL_KNOWLEDGE_SOURCES_HH
// =============================================================================
//  KnowledgeSources.hh  --  src/algorithms/cultural
//
//  Belief-space knowledge sources. Each source distils the accepted elite into
//  a bias and injects it back into a child genome. Sources are duck-typed
//  policies; the KnowledgeSource concept documents the contract.
//
//  Provided sources:
//    NormativeKnowledge   -- per-gene interval of the elite; resamples a gene
//                            uniformly inside it.
//    SituationalKnowledge -- best genome seen so far; nudges a gene toward it
//                            with a Gaussian step.
//
//  Extension sources (Domain / History / Topographic) follow the same two-method
//  shape: update(elite, space) and influence(genome, rng, space).
// =============================================================================

#include "src/algorithms/cultural/Individual.hh"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <limits>
#include <random>
#include <span>
#include <vector>

namespace fall_n::algorithms::cultural {

/// Contract every knowledge source honours (checked with concrete types by the
/// belief space and the unit tests).
template <class K, class Space, class Rng>
concept KnowledgeSource =
    requires(K k, const K ck, std::span<const Individual> elite, const Space& sp,
             std::span<double> g, Rng& rng) {
        { k.update(elite, sp) }     -> std::same_as<void>;
        { ck.influence(g, rng, sp) } -> std::same_as<void>;
    };

// -----------------------------------------------------------------------------
class NormativeKnowledge {
    std::vector<double> lo_;
    std::vector<double> hi_;
    bool initialised_ = false;

public:
    template <class Space>
    void update(std::span<const Individual> elite, const Space& space) {
        const std::size_t dim = space.dimension();
        lo_.assign(dim,  std::numeric_limits<double>::infinity());
        hi_.assign(dim, -std::numeric_limits<double>::infinity());
        for (const Individual& ind : elite) {
            const std::size_t n = std::min(dim, ind.genome.size());
            for (std::size_t i = 0; i < n; ++i) {
                lo_[i] = std::min(lo_[i], ind.genome[i]);
                hi_[i] = std::max(hi_[i], ind.genome[i]);
            }
        }
        initialised_ = !elite.empty();
    }

    template <class Rng, class Space>
    void influence(std::span<double> g, Rng& rng, const Space& space) const {
        if (!initialised_ || g.empty()) return;
        std::uniform_int_distribution<std::size_t> pick(0, g.size() - 1);
        const std::size_t i = pick(rng);
        double a = (i < lo_.size()) ? lo_[i] : space.lower(i);
        double b = (i < hi_.size()) ? hi_[i] : space.upper(i);
        if (!(b > a)) { a = space.lower(i); b = space.upper(i); }
        std::uniform_real_distribution<double> dist(a, b);
        g[i] = dist(rng);
    }
};

// -----------------------------------------------------------------------------
class SituationalKnowledge {
    std::vector<double> best_;
    double best_fitness_ = -std::numeric_limits<double>::infinity();
    bool has_best_ = false;

public:
    template <class Space>
    void update(std::span<const Individual> elite, const Space& /*space*/) {
        for (const Individual& ind : elite) {
            if (ind.fitness > best_fitness_) {
                best_fitness_ = ind.fitness;
                best_ = ind.genome;
                has_best_ = true;
            }
        }
    }

    template <class Rng, class Space>
    void influence(std::span<double> g, Rng& rng, const Space& space) const {
        if (!has_best_ || g.empty()) return;
        std::uniform_int_distribution<std::size_t> pick(0, g.size() - 1);
        const std::size_t i = pick(rng);
        if (i >= best_.size()) return;
        const double reach = space.upper(i) - space.lower(i);
        std::normal_distribution<double> step(0.0, (reach > 0.0) ? 0.25 * reach : 1.0);
        g[i] += 0.5 * (best_[i] - g[i]) + step(rng);
    }
};

}  // namespace fall_n::algorithms::cultural

#endif  // FALL_N_ALGORITHMS_CULTURAL_KNOWLEDGE_SOURCES_HH
