#ifndef FALL_N_ALGORITHMS_CULTURAL_INDIVIDUAL_HH
#define FALL_N_ALGORITHMS_CULTURAL_INDIVIDUAL_HH
// =============================================================================
//  Individual.hh  --  src/algorithms/cultural
//
//  A candidate solution: its behavioural TRAITS (a real vector in the search
//  space) plus the (maximised) fitness. "Traits" is Reynolds' cultural-
//  algorithm vocabulary: the framework is agnostic to the population-space
//  model underneath (EP, GA, ES, ...), so the solution vector is not a
//  "genome" -- that name belongs to one particular instantiation.
// =============================================================================

#include <limits>
#include <vector>

namespace fall_n::algorithms::cultural {

struct Individual {
    std::vector<double> traits{};
    double fitness = -std::numeric_limits<double>::infinity();
};

}  // namespace fall_n::algorithms::cultural

#endif  // FALL_N_ALGORITHMS_CULTURAL_INDIVIDUAL_HH
