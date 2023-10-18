#include <Eigen/Dense>

#include"header_files.h"
#include "src/domain/IntegrationPoint.h"
#include "src/domain/elements/ContinuumElement.h"
#include "src/domain/elements/Element.h"
#include "src/domain/elements/ElementBase.h"
#include "src/domain/elements/ContinuumElement.h"

#include "src/domain/DoF.h"

#include <array>
#include <functional>
#include <iostream> 
#include <vector>

#include "src/numerics/Tensor.h"
#include "src/numerics/Interpolation/GenericInterpolant.h"




#include "src/numerics/numerical_integration/Quadrature.h"
#include "src/numerics/numerical_integration/GaussLegendreNodes.h"
#include "src/numerics/numerical_integration/GaussLegendreWeights.h"

#include "src/numerics/Polynomial.h"
#include "src/numerics/Vector.h"

typedef unsigned short ushort;
typedef unsigned int   uint  ;

int main(){

    Polynomial<double, 0.0,0.0,1.0> f; // f(x) = 2x² +2.0x +5.0
    std::function<double(double)> F = f;
    std::function<double(double)> F2 = Polynomial<double, 0.0,0.0,1.0>{};

    std::cout << f(3.5) << std::endl;
    std::cout << F(3.5) << std::endl;
    std::cout << F2(3.5) << std::endl;
    std::cout << poly_eval(3.5, 0, 0, 1.0) << std::endl;


    Vector<10> v1{1,2,3,4,5,6,7,8,9,10};

    auto v2 = f(v1);

    for (auto const& i: f(v1))std::cout << i << " " ; std::cout << std::endl;


    //for (auto const& i: F(v1))std::cout << i << " " ; std::cout << std::endl;


    constexpr int dim = 3;

    Domain<dim> D; //Domain Aggregator Object
    D.preallocate_node_capacity(20);
 
    D.add_node( Node<dim>(1, 0.0, 0.0, 0.0) );
    D.add_node( Node<dim>(2, 1.0, 0.0, 0.0) );
    D.add_node( Node<dim>(3, 1.0, 1.0, 0.0) );
    D.add_node( Node<dim>(4, 0.0, 1.0, 0.0) );
    D.add_node( Node<dim>(5, 0.0, 0.0, 1.0) );
    D.add_node( Node<dim>(6, 1.0, 0.0, 1.0) );
    D.add_node( Node<dim>(7, 1.0, 1.0, 1.0) );
    D.add_node( Node<dim>(8, 0.0, 1.0, 1.0) );
    D.add_node( Node<dim>(9, 0.5, 0.5, 0.5) );

    D.make_element<ElementBase<dim,8 ,42> > (1 , {1,2,3,4,5,6,7,8});   
    D.make_element<ElementBase<dim,3 ,42> > (2 , {1,2,3});
    D.make_element<ElementBase<dim,5 ,42> > (4 , {1,2,3,4,5});
    D.make_element<ElementBase<dim,7 ,42> > (5 , {1,2,3,4,5,6,7});
    D.make_element<ElementBase<dim,14   > > (7 , {1,2,3,4,5,6,7,8,9,10,11,12,13,14});
    D.make_element<ElementBase<dim,5    > > (16, {1,2,3,4,5});
    D.make_element<ContinuumElement<dim,9>> (2 , {1,2,3,4,5,6,7,8,9});
    D.make_element<ElementBase<dim,5    > > (42, {1,2,3,4,5}); 
    D.make_element<ElementBase<dim,9    > > (44, {1,2,3,4,5,6,7,8,9}); 
    D.make_element<ElementBase<dim,13   > > (50, {1,2,3,4,5,6,7,8,9,10,11,12,13});
                        
    //for (auto const& e: D.elements_){
    //    std::cout << id(e)<< " " << num_nodes(e)<< " " << num_dof(e)<< " ";
    //        for (auto const& n: nodes(e)){
    //            std::cout << n << " ";
    //        }
    //    std::cout << std::endl;
    //}

};

