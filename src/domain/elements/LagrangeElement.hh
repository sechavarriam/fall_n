#ifndef FALL_N_LAGRANGIAN_FINITE_ELEMENT  
#define FALL_N_LAGRANGIAN_FINITE_ELEMENT

#include <array>
#include <cstddef>


#include "../Node.h"

//#include "../../geometry/ReferenceElement.h"
//#include "../../numerics/Interpolation/LagrangeInterpolation.h"
//#include "../../numerics/numerical_integration/Quadrature.h"
//#include "../../numerics/numerical_integration/GaussLegendreNodes.h"
//#include "../../numerics/numerical_integration/GaussLegendreWeights.h"
//#include "../../numerics/Polynomial.h"
//#include "../../numerics/Vector.h"



template<std::size_t... N>
class LagrangeElement
{
static inline constexpr std::size_t dim       = sizeof...(N);
static inline constexpr std::size_t num_nodes = (... * N);

std::size_t id_;

std::array<Node<dim>, num_nodes> nodes_;

  public:
    LagrangeElement() = default;

    LagrangeElement(std::array<Node<dim>, num_nodes> nodes) : nodes_(nodes){};


    ~LagrangeElement() = default;

};


#endif