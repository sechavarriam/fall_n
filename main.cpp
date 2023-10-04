#include <Eigen/Dense>

#include"header_files.h"
#include "src/domain/IntegrationPoint.h"
#include "src/domain/elements/ContinuumElement.h"
#include "src/domain/elements/ElementBase.h"

#include "src/domain/DoF.h"

#include <array>
#include <cwchar>
#include <functional>
#include <iostream> 

#include "src/numerics/Tensor.h"
#include "src/numerics/InterpolationFunction.h"

#include "src/numerics/numerical_integration/Quadrature.h"
#include "src/numerics/numerical_integration/GaussLegendreNodes.h"
#include "src/numerics/numerical_integration/GaussLegendreWeights.h"


typedef unsigned short ushort;
typedef unsigned int   uint  ;

int main(){

    constexpr int dim = 3;

    Domain<dim> D; //Domain Aggregator Object
    //D.preallocate_node_capacity(100);
 
    D.add_node( Node<dim>(1, 0.0, 0.0, 0.0) );
    D.add_node( Node<dim>(2, 1.0, 0.0, 0.0) );
    D.add_node( Node<dim>(3, 1.0, 1.0, 0.0) );
    D.add_node( Node<dim>(4, 0.0, 1.0, 0.0) );
    D.add_node( Node<dim>(5, 0.0, 0.0, 1.0) );
    D.add_node( Node<dim>(6, 1.0, 0.0, 1.0) );
    D.add_node( Node<dim>(7, 1.0, 1.0, 1.0) );
    D.add_node( Node<dim>(8, 0.0, 1.0, 1.0) );


//

    auto Casted = static_cast<ElementBase<dim,4>*>(D.elements_[0].get()); 

    
    ElementBase<dim,4> e(0,{0,1,2,3});

    std::cout << "_____________________________________________________" << std::endl;
    for (auto& n: static_cast<ElementBase<dim,4>*>(D.elements_[0].get())->nodes_ ){
        std::cout << n << std::endl;
    }
    std::cout << "_____________________________________________________" << std::endl;
    for (auto& n: dynamic_cast<ElementBase<dim,4>*>(D.elements_[1].get())->nodes_ ){
        std::cout << n << std::endl;
    }
    std::cout << "_____________________________________________________" << std::endl;
    for (auto& n: reinterpret_cast<ElementBase<dim,4>*>(D.elements_[0].get())->nodes_ ){
        std::cout << n << std::endl;
    }
    std::cout << "_____________________________________________________" << std::endl;

    
};

