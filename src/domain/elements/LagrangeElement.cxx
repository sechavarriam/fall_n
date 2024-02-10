#include <iostream>

#include "LagrangeElement.hh"

#include "../../numerics/numerical_integration/Quadrature.hh"

namespace element::LagrangeElement{


template <std::size_t... N>
class GaussIntegrator{

    using  CellIntegrator = GaussLegendre::CellIntegrator<N...>;

    CellIntegrator integrator_;

    //MOVE POINTS TO ELEMENT (OR COPY)?
    constexpr auto set_integration_points(is_LagrangeElement auto& element) const noexcept {
        element.set_integration_points(integrator_.evalPoints_);
        }



    };

} // namespace element::LagrangeElement
    





//auto LagrangeElement_integrator(const is_LagrangeElement auto& element, std::invocable auto&& function) -> double {
//  std::cout << "Integrating Lagrange Element" << std::endl;
//
//  for (auto i_point : element.integration_points_) {
//    std::cout << "Integration Point: " << i_point.coord(0) << " " << i_point.coord(1) << " " << i_point.coord(2) << std::endl;
//  }
//
//
//  return 0.0;
//};

} // namespace LagrangeElement
