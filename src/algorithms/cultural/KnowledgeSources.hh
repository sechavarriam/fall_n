#ifndef FALL_N_ALGORITHMS_CULTURAL_KNOWLEDGE_SOURCES_HH
#define FALL_N_ALGORITHMS_CULTURAL_KNOWLEDGE_SOURCES_HH
// =============================================================================
//  KnowledgeSources.hh  --  src/algorithms/cultural
//
//  Belief-space knowledge sources. Each source distils the accepted elite into
//  a bias and injects it back into a candidate's traits. Sources are
//  duck-typed policies; the KnowledgeSource concept documents the contract.
//
//  The five sources follow the cultural-algorithm literature (Reynolds;
//  Reynolds & Peng; the comprehensive survey of Maheri et al.), which is also
//  the basis of a tuned-inerter-damper optimisation study
//  (Lara-Valencia, Echavarria-Montana & Valencia-Gonzalez, 2024):
//
//    NormativeKnowledge   -- per-trait interval of the elite; resamples a
//                            trait uniformly inside it.
//    SituationalKnowledge -- best exemplar seen so far; nudges a trait toward
//                            it with a Gaussian step.
//    DomainKnowledge      -- fitness-weighted centroid of the elite (the
//                            problem-side consensus); pulls a trait toward it.
//    HistoryKnowledge     -- temporal trend of the per-generation best
//                            exemplars; steps along the recent displacement
//                            (useful on drifting / deceptive landscapes).
//    TopographicKnowledge -- per-trait occupancy histogram of the elite;
//                            resamples a trait inside the most-visited cell
//                            (region-level exploitation).
// =============================================================================

#include "src/algorithms/cultural/Individual.hh"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <deque>
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
             std::span<double> t, Rng& rng) {
        { k.update(elite, sp) }     -> std::same_as<void>;
        { ck.influence(t, rng, sp) } -> std::same_as<void>;
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
            const std::size_t n = std::min(dim, ind.traits.size());
            for (std::size_t i = 0; i < n; ++i) {
                lo_[i] = std::min(lo_[i], ind.traits[i]);
                hi_[i] = std::max(hi_[i], ind.traits[i]);
            }
        }
        initialised_ = !elite.empty();
    }

    template <class Rng, class Space>
    void influence(std::span<double> t, Rng& rng, const Space& space) const {
        if (!initialised_ || t.empty()) return;
        std::uniform_int_distribution<std::size_t> pick(0, t.size() - 1);
        const std::size_t i = pick(rng);
        double a = (i < lo_.size()) ? lo_[i] : space.lower(i);
        double b = (i < hi_.size()) ? hi_[i] : space.upper(i);
        if (!(b > a)) { a = space.lower(i); b = space.upper(i); }
        std::uniform_real_distribution<double> dist(a, b);
        t[i] = dist(rng);
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
                best_ = ind.traits;
                has_best_ = true;
            }
        }
    }

    template <class Rng, class Space>
    void influence(std::span<double> t, Rng& rng, const Space& space) const {
        if (!has_best_ || t.empty()) return;
        std::uniform_int_distribution<std::size_t> pick(0, t.size() - 1);
        const std::size_t i = pick(rng);
        if (i >= best_.size()) return;
        const double reach = space.upper(i) - space.lower(i);
        std::normal_distribution<double> step(0.0, (reach > 0.0) ? 0.25 * reach : 1.0);
        t[i] += 0.5 * (best_[i] - t[i]) + step(rng);
    }
};

// -----------------------------------------------------------------------------
//  Fitness-weighted consensus of the elite. Weights are the rank-shifted
//  fitness (min-max normalised within the elite) so the source stays
//  scale-free; a degenerate elite (all equal fitness) degrades to the plain
//  centroid.
class DomainKnowledge {
    std::vector<double> centroid_;
    bool initialised_ = false;

public:
    template <class Space>
    void update(std::span<const Individual> elite, const Space& space) {
        const std::size_t dim = space.dimension();
        centroid_.assign(dim, 0.0);
        if (elite.empty()) { initialised_ = false; return; }
        double fmin = elite.front().fitness, fmax = fmin;
        for (const Individual& ind : elite) {
            fmin = std::min(fmin, ind.fitness);
            fmax = std::max(fmax, ind.fitness);
        }
        const double span = fmax - fmin;
        double wsum = 0.0;
        for (const Individual& ind : elite) {
            const double w = (span > 0.0) ? (ind.fitness - fmin) / span + 0.1 : 1.0;
            wsum += w;
            const std::size_t n = std::min(dim, ind.traits.size());
            for (std::size_t i = 0; i < n; ++i) centroid_[i] += w * ind.traits[i];
        }
        for (double& c : centroid_) c /= wsum;
        initialised_ = true;
    }

    template <class Rng, class Space>
    void influence(std::span<double> t, Rng& rng, const Space& /*space*/) const {
        if (!initialised_ || t.empty()) return;
        std::uniform_int_distribution<std::size_t> pick(0, t.size() - 1);
        const std::size_t i = pick(rng);
        if (i >= centroid_.size()) return;
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        t[i] += unit(rng) * (centroid_[i] - t[i]);
    }
};

// -----------------------------------------------------------------------------
//  Temporal trend of the per-generation best exemplar over a sliding window.
//  The influence steps a trait along the average recent displacement, which
//  helps track drifting optima and escape deceptive basins.
class HistoryKnowledge {
    std::deque<std::vector<double>> recent_;
    std::vector<double> trend_;
    std::size_t window_ = 5;
    bool has_trend_ = false;

public:
    HistoryKnowledge() = default;
    explicit HistoryKnowledge(std::size_t window) : window_(window ? window : 1) {}

    template <class Space>
    void update(std::span<const Individual> elite, const Space& space) {
        if (elite.empty()) return;
        recent_.push_back(elite.front().traits);   // ranked: front = generation best
        while (recent_.size() > window_) recent_.pop_front();
        const std::size_t dim = space.dimension();
        trend_.assign(dim, 0.0);
        has_trend_ = recent_.size() >= 2;
        if (!has_trend_) return;
        const auto& oldest = recent_.front();
        const auto& newest = recent_.back();
        const std::size_t n =
            std::min(dim, std::min(oldest.size(), newest.size()));
        const double steps = static_cast<double>(recent_.size() - 1);
        for (std::size_t i = 0; i < n; ++i) {
            trend_[i] = (newest[i] - oldest[i]) / steps;
        }
    }

    template <class Rng, class Space>
    void influence(std::span<double> t, Rng& rng, const Space& /*space*/) const {
        if (!has_trend_ || t.empty()) return;
        std::uniform_int_distribution<std::size_t> pick(0, t.size() - 1);
        const std::size_t i = pick(rng);
        if (i >= trend_.size()) return;
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        t[i] += unit(rng) * trend_[i];
    }
};

// -----------------------------------------------------------------------------
//  Per-trait occupancy histogram of the elite over the box. The influence
//  resamples a trait uniformly inside its most-visited cell: exploitation of
//  the promising region at marginal (per-dimension) level.
class TopographicKnowledge {
    std::size_t bins_ = 10;
    std::vector<std::vector<double>> counts_;   // [dim][bin]
    bool initialised_ = false;

public:
    TopographicKnowledge() = default;
    explicit TopographicKnowledge(std::size_t bins) : bins_(bins ? bins : 1) {}

    template <class Space>
    void update(std::span<const Individual> elite, const Space& space) {
        const std::size_t dim = space.dimension();
        counts_.assign(dim, std::vector<double>(bins_, 0.0));
        if (elite.empty()) { initialised_ = false; return; }
        for (std::size_t r = 0; r < elite.size(); ++r) {
            //  Rank weight: the elite is ranked by descending fitness.
            const double w =
                static_cast<double>(elite.size() - r) /
                static_cast<double>(elite.size());
            const auto& tr = elite[r].traits;
            const std::size_t n = std::min(dim, tr.size());
            for (std::size_t i = 0; i < n; ++i) {
                const double lo = space.lower(i), hi = space.upper(i);
                if (!(hi > lo)) continue;
                double u = (tr[i] - lo) / (hi - lo);
                u = std::clamp(u, 0.0, 1.0);
                auto b = static_cast<std::size_t>(u * static_cast<double>(bins_));
                if (b >= bins_) b = bins_ - 1;
                counts_[i][b] += w;
            }
        }
        initialised_ = true;
    }

    template <class Rng, class Space>
    void influence(std::span<double> t, Rng& rng, const Space& space) const {
        if (!initialised_ || t.empty()) return;
        std::uniform_int_distribution<std::size_t> pick(0, t.size() - 1);
        const std::size_t i = pick(rng);
        if (i >= counts_.size()) return;
        const auto& row = counts_[i];
        const std::size_t bbest = static_cast<std::size_t>(
            std::distance(row.begin(), std::max_element(row.begin(), row.end())));
        const double lo = space.lower(i), hi = space.upper(i);
        if (!(hi > lo)) return;
        const double w = (hi - lo) / static_cast<double>(bins_);
        std::uniform_real_distribution<double> dist(
            lo + static_cast<double>(bbest) * w,
            lo + static_cast<double>(bbest + 1) * w);
        t[i] = dist(rng);
    }
};

}  // namespace fall_n::algorithms::cultural

#endif  // FALL_N_ALGORITHMS_CULTURAL_KNOWLEDGE_SOURCES_HH
