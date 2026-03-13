#ifndef FALL_N_SERENDIPITY_SIMPLEX_ELEMENT_HH
#define FALL_N_SERENDIPITY_SIMPLEX_ELEMENT_HH

#include "../../geometry/SimplexCell.hh"
#include "SimplexElement.hh"

namespace geometry::simplex {

template <std::size_t D, std::size_t Order>
    requires (Order == 1 || Order == 2)
using SerendipitySimplexCell = SimplexCell<D, Order>;

} // namespace geometry::simplex

template <std::size_t Dim, std::size_t TopDim, std::size_t Order>
    requires (Order == 1 || Order == 2)
using SerendipitySimplexElement = SimplexElement<Dim, TopDim, Order>;

#endif // FALL_N_SERENDIPITY_SIMPLEX_ELEMENT_HH
