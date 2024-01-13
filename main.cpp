//#include <Eigen/Dense>
#include <array>
#include <concepts>
#include <functional>
#include <iostream> 
#include <vector>
#include <numeric>
#include <algorithm>
#include <ranges>

#include"header_files.h"

#include "src/GeneralConcepts.h"


#include "src/domain/IntegrationPoint.h"
#include "src/domain/elements/ContinuumElement.h"
#include "src/domain/elements/Element.h"
#include "src/domain/elements/ElementBase.h"
#include "src/domain/elements/ContinuumElement.h"

#include "src/domain/DoF.h"


#include "src/geometry/geometry.h"
#include "src/geometry/Topology.h"
#include "src/geometry/Cell.h"
#include "src/geometry/Point.h"
#include "src/geometry/ReferenceElement.h"

#include "src/numerics/Tensor.h"


#include "src/numerics/Interpolation/GenericInterpolant.h"
#include "src/numerics/Interpolation/LagrangeInterpolation.h"


#include "src/numerics/numerical_integration/Quadrature.h"
#include "src/numerics/numerical_integration/GaussLegendreNodes.h"
#include "src/numerics/numerical_integration/GaussLegendreWeights.h"

#include "src/numerics/Polynomial.h"
#include "src/numerics/Vector.h"

#include "src/utils/constexpr_function.h"

typedef unsigned short ushort;
typedef unsigned int   uint  ;


int main(){
    static constexpr int dim = 3;

    //constexpr geometry::cell::LagrangianCell<dim, 1, 1, 2> C1;

    //geometry::Cell<dim,1>     C1;
    //geometry::Cell<dim,2,2,2> C2;

    static constexpr uint nx = 5;
    static constexpr uint ny = 4;    
    static constexpr uint nz = 3;

    domain::Domain<dim> D; //Domain Aggregator Object
    
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

    auto integrationScheme = [](auto const & e){/**integrate*/};
    
    //ElementBase<type,dim,9,42> test{1, {1,2,3,4,5,6,7,8,9}}

    constexpr geometry::cell::LagrangianCell<dim, nx,ny,nz> test_cell;


    Element test1{ElementBase<dim,10,42>{1, {1,2,3,4,5,6,7,8,9,10}}, integrationScheme};

    interpolation::LagrangeBasis_ND<2,3> L2_3({0.0,1.0},{-1.0, 0.0, 1.0});

    std::cout << "---------------------------------------" << std::endl;
    for (auto const& i: std::get<0>(L2_3.coordinates_i)){
        std::cout << i << " ";
    }
    std::cout << std::endl;
    std::cout << "---------------------------------------" << std::endl;
    for (auto const& i: std::get<1>(L2_3.coordinates_i)){
        std::cout << i << " ";
    }
    std::cout << std::endl;
    std::cout << "---------------------------------------" << std::endl;
    //std::cout << std::get<0>(L2_3.coordinates_i)[0] << std::endl;

    std::cout << std::get<0>(L2_3.Li)[0](0.5) << std::endl;
    std::cout << std::get<1>(L2_3.Li)[0](0.5) << std::endl;
    std::cout << "---------------------------------------" << std::endl;

    interpolation::LagrangeBasis_1D<2> L2({0.0,1.0});

    std::cout << L2[0](0.5) << std::endl;
    std::cout << L2[1](0.5) << std::endl;

    std::cout << L2[0](0.0) << std::endl;
    std::cout << L2[1](0.0) << std::endl;

    interpolation::LagrangeBasis_1D<3> L3({-1.0, 0.0, 1.0});

    std::cout << L3[0](0.5) << std::endl;
    std::cout << L3[1](0.5) << std::endl;
    std::cout << L3[2](0.5) << std::endl;

    

    //std::cout << test_cell.L(0,0.5) << std::endl;



    //std::cout << LinearInterpolant({0.0,1.0}, 0.5) << std::endl;
    
    //D.make_element<ElementBase<dim,9,5> >(integrationScheme, 1, {1,2,3,4,5,6,7,8,9});   

    //Element test2{ElementBase<dim,8>{2, {1,2,3,4,5,6,7,8  }}, integrationScheme};
    //Element test3{ElementBase<dim,7>{3, {1,2,3,4,5,6,7    }}, integrationScheme};
    //Element test4{ElementBase<dim,6>{4, {1,2,3,4,5,6      }}, integrationScheme};
    //
    //ElementConstRef test5 = ElementConstRef(test1);
    //Element test6(test2 );
    //
    //integrate(test1);
    //integrate(test2);
    //integrate(test3);
    //integrate(test4);
    //integrate(test5);
    //integrate(test6);

    //D.make_element<ElementBase<dim,3 ,42> > (2 , {1,2,3});
    //D.make_element<ElementBase<dim,5 ,42> > (4 , {1,2,3,4,5});
    //D.make_element<ElementBase<dim,7 ,42> > (5 , {1,2,3,4,5,6,7});
    //D.make_element<ElementBase<dim,14   > > (7 , {1,2,3,4,5,6,7,8,9,10,11,12,13,14});
    //D.make_element<ElementBase<dim,5    > > (16, {1,2,3,4,5});
    //D.make_element<ContinuumElement<dim,9>> (2 , {1,2,3,4,5,6,7,8,9});
    //D.make_element<ElementBase<dim,5    > > (42, {1,2,3,4,5}); 
    //D.make_element<ElementBase<dim,9    > > (44, {1,2,3,4,5,6,7,8,9}); 
    //D.make_element<ElementBase<dim,13   > > (50, {1,2,3,4,5,6,7,8,9,10,11,12,13});
                        
    //for (auto const& e: D.elements_){
    //    std::cout << id(e)<< " " << num_nodes(e)<< " " << num_dof(e)<< " ";
    //        for (auto const& n: nodes(e)){
    //            std::cout << n << " ";
    //        }
    //    std::cout << std::endl;
    //}

};

