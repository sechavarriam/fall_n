#include <Eigen/Dense>

#include"header_files.h"
#include "src/domain/IntegrationPoint.h"
#include "src/domain/elements/ElementBase.h"

#include "src/domain/DoF.h"

#include <array>
#include <cwchar>
#include <functional>
#include <iostream> 


#include "src/numerics/Tensor.h"

#include "src/numerics/numerical_integration/Quadrature.h"
#include "src/numerics/numerical_integration/GaussLegendreNodes.h"
#include "src/numerics/numerical_integration/GaussLegendreWeights.h"

int main(){

    constexpr int dim = 3;

    Domain<dim> D; //Domain Aggregator Object
    //D.preallocate_node_capacity(100);
 
    D.add_node( Node<dim>(1, 0.0, 0.0, 0.0) );
    D.make_element<BeamColumn_Euler<dim>>(1, 0, 8, 1.0, 1.0, 1.0);

    std::function<double(double)> Fn = [](double x){return x*x;};
    for (int i = 0; ++i, i<10;)std::cout << i << ' ' << Fn(i) << std::endl;


    constexpr short order = 15;
    Quadrature<1,order> GaussOrder3(GaussLegendre::Weights1D<order>(),GaussLegendre::evalPoints1D<order>());
    
    std::cout << GaussOrder3([](double x){return x*x;}) << std::endl;
    std::cout << GaussOrder3(Fn) << std::endl;

    std::cout << "____________________________" << std::endl;
    auto W_2D = GaussLegendre::Weights2D<5>();
    for(auto&& i:W_2D) std::cout << i << std::endl;

};

