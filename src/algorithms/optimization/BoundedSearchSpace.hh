#ifndef FALL_N_ALGORITHMS_OPTIMIZATION_BOUNDED_SEARCH_SPACE_HH
#define FALL_N_ALGORITHMS_OPTIMIZATION_BOUNDED_SEARCH_SPACE_HH
// =============================================================================
//  BoundedSearchSpace.hh  --  src/algorithms/optimization
//
//  Axis-aligned box search space [lo_i, hi_i]^d. Models the SearchSpace concept.
// =============================================================================

#include "src/algorithms/optimization/Concepts.hh"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <random>
#include <span>
#include <utility>
#include <vector>

namespace fall_n::algorithms {

class BoundedSearchSpace {
    std::vector<double> lo_;
    std::vector<double> hi_;

public:
    BoundedSearchSpace() = default;

    BoundedSearchSpace(std::vector<double> lower, std::vector<double> upper)
        : lo_(std::move(lower)), hi_(std::move(upper)) {
        assert(lo_.size() == hi_.size() && "bound vectors must share dimension");
    }

    [[nodiscard]] std::size_t dimension() const noexcept { return lo_.size(); }
    [[nodiscard]] double lower(std::size_t i) const { return lo_[i]; }
    [[nodiscard]] double upper(std::size_t i) const { return hi_[i]; }
    [[nodiscard]] double extent(std::size_t i) const { return hi_[i] - lo_[i]; }

    /// Project a genome back into the box, gene by gene, in place.
    void clamp(std::span<double> g) const {
        const std::size_t n = std::min(g.size(), lo_.size());
        for (std::size_t i = 0; i < n; ++i) {
            g[i] = std::clamp(g[i], lo_[i], hi_[i]);
        }
    }

    /// Draw a uniformly-random admissible genome.
    template <class Rng>
    [[nodiscard]] std::vector<double> sample(Rng& rng) const {
        std::vector<double> g(lo_.size());
        for (std::size_t i = 0; i < lo_.size(); ++i) {
            std::uniform_real_distribution<double> dist(lo_[i], hi_[i]);
            g[i] = dist(rng);
        }
        return g;
    }
};

static_assert(SearchSpace<BoundedSearchSpace>,
              "BoundedSearchSpace must satisfy the SearchSpace concept");

}  // namespace fall_n::algorithms

#endif  // FALL_N_ALGORITHMS_OPTIMIZATION_BOUNDED_SEARCH_SPACE_HH
