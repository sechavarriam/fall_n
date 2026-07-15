#ifndef FALL_N_ALGORITHMS_CULTURAL_BELIEF_SPACE_HH
#define FALL_N_ALGORITHMS_CULTURAL_BELIEF_SPACE_HH
// =============================================================================
//  BeliefSpace.hh  --  src/algorithms/cultural
//
//  The belief space aggregates a compile-time pack of knowledge sources and the
//  acceptance policy that selects which individuals feed them.
// =============================================================================

#include "src/algorithms/cultural/Individual.hh"

#include <cstddef>
#include <span>
#include <tuple>

namespace fall_n::algorithms::cultural {

/// Accept the top `fraction` of the ranked population as the elite (>= 1).
struct TopFractionAcceptance {
    double fraction = 0.25;

    [[nodiscard]] std::size_t count(std::size_t population_size) const noexcept {
        if (population_size == 0) return 0;
        auto k = static_cast<std::size_t>(fraction * static_cast<double>(population_size));
        if (k < 1) k = 1;
        if (k > population_size) k = population_size;
        return k;
    }
};

template <class... KnowledgeSources>
class BeliefSpace {
    std::tuple<KnowledgeSources...> sources_{};
    TopFractionAcceptance acceptance_{};

public:
    BeliefSpace() = default;
    explicit BeliefSpace(TopFractionAcceptance acceptance) : acceptance_(acceptance) {}

    /// Distil knowledge from the elite. `ranked` must be sorted by DESCENDING
    /// fitness.
    template <class Space>
    void accept(std::span<const Individual> ranked, const Space& space) {
        const std::size_t k = acceptance_.count(ranked.size());
        const std::span<const Individual> elite = ranked.subspan(0, k);
        std::apply([&](auto&... src) { (src.update(elite, space), ...); }, sources_);
    }

    /// Let every source bias the child genome in turn.
    template <class Rng, class Space>
    void influence(std::span<double> g, Rng& rng, const Space& space) const {
        std::apply([&](const auto&... src) { (src.influence(g, rng, space), ...); }, sources_);
    }
};

}  // namespace fall_n::algorithms::cultural

#endif  // FALL_N_ALGORITHMS_CULTURAL_BELIEF_SPACE_HH
