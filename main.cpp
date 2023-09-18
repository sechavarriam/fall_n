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


    //Quadrature<std::array<double,3>, std::function<double>>

    //IntegrationPoint<dim> a;

    


};
