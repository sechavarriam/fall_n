#include"header_files.h"
#include "src/domain/IntegrationPoint.h"
#include "src/domain/elements/ElementBase.h"

#include "src/domain/DoF.h"

#include <functional>
#include <iostream> 
#include <Eigen/Dense>

#include "src/numerics/Tensor.h"

#include "src/numerics/numerical_integration/Quadrature.h"

int main(){

    constexpr int dim = 3;

    Domain<dim> D; //Domain Aggregator Object
    //D.preallocate_node_capacity(100);
 
    D.add_node( Node<dim>(1, 0.0, 0.0, 0.0) );
    D.make_element<BeamColumn_Euler<dim>>(1, 0, 8, 1.0, 1.0, 1.0);

    std::function<double(double)> F = [](double x){return x*x;};
    for (int i = 0; ++i, i<10;)std::cout << i << ' ' << F(i) << std::endl;
    


    Quadrature<std::array<double,3>,std::function<double(double)>> Gauss3{{0.555556,0.888889,0.555556},F};

    
    std::cout << "Int_F (-1,1) := "<< Gauss3({-0.774597,0,0.774597}) << std::endl;
    
    //IntegrationPoint<dim> a;

    


};
