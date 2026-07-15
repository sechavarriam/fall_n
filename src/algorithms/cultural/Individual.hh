#ifndef FALL_N_ALGORITHMS_CULTURAL_INDIVIDUAL_HH
#define FALL_N_ALGORITHMS_CULTURAL_INDIVIDUAL_HH
// =============================================================================
//  Individual.hh  --  src/algorithms/cultural
//
//  A candidate solution: a real genome plus its (maximised) fitness.
// =============================================================================

#include <limits>
#include <vector>

namespace fall_n::algorithms::cultural {

struct Individual {
    std::vector<double> genome{};
    double fitness = -std::numeric_limits<double>::infinity();
};

}  // namespace fall_n::algorithms::cultural

#endif  // FALL_N_ALGORITHMS_CULTURAL_INDIVIDUAL_HH
