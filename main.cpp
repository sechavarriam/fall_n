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

    //D.make_element<BeamColumn_Euler<dim>>(1, 0, 8, 1.0, 1.0, 1.0);

    D.make_element<ElementBase<dim,4>>(0,{0,1,2,3});
    D.make_element<ElementBase<dim,4>>(42,std::array<ushort,4>{10,11,12,13});


};

