//#include <Eigen/Dense>
#include <array>
#include <concepts>
#include <functional>
#include <iostream> 
#include <vector>
#include <numeric>
#include <algorithm>
#include <ranges>
#include<tuple>
#include<utility>

#include"header_files.hh"

#include "src/domain/IntegrationPoint.hh"
#include "src/domain/Node.hh"
#include "src/domain/elements/ContinuumElement.hh"
#include "src/domain/elements/Element.hh"
#include "src/domain/elements/ElementBase.hh"
#include "src/domain/elements/ContinuumElement.hh"
#include "src/domain/elements/LagrangeElement.hh"

#include "src/domain/DoF.hh"

#include "src/geometry/geometry.hh"
#include "src/geometry/Topology.hh"
#include "src/geometry/Cell.hh"
#include "src/geometry/Point.hh"
#include "src/geometry/ReferenceElement.hh"

#include "src/numerics/Polynomial.hh"
#include "src/numerics/Tensor.hh"

#include "src/numerics/Interpolation/GenericInterpolant.hh"
#include "src/numerics/Interpolation/LagrangeInterpolation.hh"

#include "src/numerics/numerical_integration/Quadrature.hh"
#include "src/numerics/numerical_integration/GaussLegendreNodes.hh"
#include "src/numerics/numerical_integration/GaussLegendreWeights.hh"

#include "src/numerics/linear_algebra/Matrix.hh"
#include "src/numerics/linear_algebra/Vector.hh"
#include "src/numerics/linear_algebra/LinalgOperations.hh"

#include "src/analysis/Analysis.hh"
#include "src/mesh/Mesh.hh"
#include "src/mesh/gmsh/ReadGmsh.hh"


#include "src/model/Model.hh"
#include "src/integrator/MaterialIntegrator.hh"

#include "src/materials/ConstitutiveRelation.hh"
#include "src/materials/LinealRelation.hh"
#include "src/materials/IsotropicRelation.hh"

#include "src/materials/Material.hh"

#include "src/materials/MaterialState.hh"
#include "src/materials/LinealElasticMaterial.hh"

#include "src/materials/Stress.hh"
#include "src/materials/Strain.hh"

//#include <matplot/matplot.h>

#include <petsc.h>
//#include <petscksp.h>
#include <petscsys.h>


int main(int argc, char **args){
PetscInitialize(&argc, &args, nullptr, nullptr);{ // PETSc Scope starts here

    static constexpr int dim = 3;

    std::array dataA{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};    
    std::array data1{1.0, 2.0, 3.0};
    std::array data2{0.0, 0.0, 1.0};

    Matrix A{dataA, 3,3};
    Vector x{data1};
    auto y = A*x;

    ContinuumIsotropicRelation steel3D{200.0, 0.3};
    
    UniaxialIsotropicRelation steel1D{200.0};

    steel3D.compliance_matrix.print_content();

    ContinuumIsotropicElasticMaterial steel_mat3D{200.0, 0.3};
    UniaxialIsotropicElasticMaterial  steel_mat1D{200.0};

    

    domain::Domain<dim> D; //Domain Aggregator Object
    
    D.preallocate_node_capacity(20);
    D.add_node( Node<dim>(0 ,  2.0, 2.0, 4.0) );
    D.add_node( Node<dim>(1 ,  4.0, 3.0, 3.0) );
    D.add_node( Node<dim>(2 ,  9.0, 3.0, 3.0) );
    D.add_node( Node<dim>(3 ,  2.0, 4.0, 3.0) );
    D.add_node( Node<dim>(4 ,  4.0, 5.0, 2.0) );
    D.add_node( Node<dim>(5 ,  9.0, 4.0, 2.0) );
    D.add_node( Node<dim>(6 ,  2.0, 6.0, 2.0) );
    D.add_node( Node<dim>(7 ,  4.0, 7.0, 1.0) );
    D.add_node( Node<dim>(8 ,  9.0, 6.0, 1.0) );
    D.add_node( Node<dim>(9 ,  2.0, 8.0, 1.0) );
    D.add_node( Node<dim>(10,  5.0, 2.0, 7.0) );
    D.add_node( Node<dim>(11,  9.0, 2.0, 8.5) );
    D.add_node( Node<dim>(12,  4.0, 4.0, 8.5) );
    D.add_node( Node<dim>(13,  6.0, 4.0, 8.0) );
    D.add_node( Node<dim>(14,  9.0, 4.5, 9.0) );
    D.add_node( Node<dim>(15,  3.0, 6.0, 8.0) );
    D.add_node( Node<dim>(16,  6.0, 7.5, 7.0) );
    D.add_node( Node<dim>(17,  9.0, 6.0, 8.0) );

    auto integrationScheme = [](auto const & e){/**integrate*/};
    Element test1{ElementBase<dim,10,42>{1, {1,2,3,4,5,6,7,8,9,10}}, integrationScheme};

    //ElementBase<type,dim,9,42> test{1, {1,2,3,4,5,6,7,8,9}}

    /*
    static constexpr uint nx = 3;
    static constexpr uint ny = 3;    
    static constexpr uint nz = 2;
    constexpr geometry::cell::LagrangianCell<nx,ny,nz> test_cell;

    LagrangeElement<nx,ny,nz> E1 {{D.node_p(0 ),D.node_p(1 ),D.node_p(2 ),
                                   D.node_p(3 ),D.node_p(4 ),D.node_p(5 ),
                                   D.node_p(6 ),D.node_p(7 ),D.node_p(8 ),
                                   D.node_p(9 ),D.node_p(10),D.node_p(11),
                                   D.node_p(12),D.node_p(13),D.node_p(14),
                                   D.node_p(15),D.node_p(16),D.node_p(17)}};
    
    Node<dim> N1{1,0.0,0.0,0.0};
    Node<dim> N2{2,1.0,0.0,0.0};
    Node<dim> N3{3,1.0,1.0,0.0};
    Node<dim> N4{4,0.0,1.0,0.0};
    Node<dim> N5{5,0.0,0.0,1.0};
    Node<dim> N6{6,1.0,0.0,1.0};
    Node<dim> N7{7,1.0,1.0,1.0};
    Node<dim> N8{8,0.0,1.0,1.0};
    Node<dim> N9{9,0.5,0.5,0.5};

    Element test2{ElementBase<dim,8>{2, {1,2,3,4,5,6,7,8  }}, integrationScheme};
    Element test3{ElementBase<dim,7>{3, {1,2,3,4,5,6,7    }}, integrationScheme};
    Element test4{ElementBase<dim,6>{4, {1,2,3,4,5,6      }}, integrationScheme};


    

    Element testL{LagrangeElement<nx,ny,nz>{{D.node_p(0 ),D.node_p(1 ),D.node_p(2 ),
                                             D.node_p(3 ),D.node_p(4 ),D.node_p(5 ),
                                             D.node_p(6 ),D.node_p(7 ),D.node_p(8 ),
                                             D.node_p(9 ),D.node_p(10),D.node_p(11),
                                             D.node_p(12),D.node_p(13),D.node_p(14),
                                             D.node_p(15),D.node_p(16),D.node_p(17)}}, integrationScheme};


    //GaussIntegrator<2,2,2> LagElem_integrator(E1);

    static_assert(is_LagrangeElement<LagrangeElement<nx,ny,nz>>);

    ElementConstRef test5 = ElementConstRef(test1);
    Element test6(test2 );
    
    integrate(test1); integrate(test2); integrate(test3);
    integrate(test4); integrate(test5); integrate(testL);


    IntegrationPoint<dim> IP_1 {0.0,5.0,0.0};
    IntegrationPoint<dim> IP_2 {0.0,0.0,0.0};
    IntegrationPoint<dim> IP_3 {0.0,0.0,0.0};
    IntegrationPoint<dim> IP_4 {0.0,0.0,0.0};
    IntegrationPoint<dim> IP_5 {0.0,0.0,0.0};
    IntegrationPoint<dim> IP_6 {0.0,0.0,0.0};
    IntegrationPoint<dim> IP_7 {0.0,0.0,0.0};
    IntegrationPoint<dim> IP_8 {0.0,0.0,0.0};
    IntegrationPoint<dim> IP_9 {0.0,0.0,0.0};
    IntegrationPoint<dim> IP_10{0.0,0.0,0.0};
    IntegrationPoint<dim> IP_11{0.0,0.0,0.0};

    IntegrationPoint<dim> IP_12(N3);
 

    std::cout << "Integration Points:" << std::endl;
    std::cout << IP_1.coord(0) << " " << IP_1.coord(1) << " " << IP_1.coord(2) << std::endl;
    std::cout << IP_12.coord(0) << " " << IP_12.coord(1) << " " << IP_12.coord(2) << std::endl;

    std::vector<IntegrationPoint<dim>>  IP_list{IP_1,IP_2,IP_3,IP_4,IP_5,IP_6,IP_7,IP_8,IP_12};

*/

    //std::vector<geometry::Point<dim>*> IP_list_p{&IP_1,&IP_2,&IP_3,&IP_4,&IP_5,&IP_6,&IP_7,&IP_8,&IP_9,&IP_10,&IP_11,&IP_12};
    //std::vector<IntegrationPoint<dim>>* IP_list_p = &IP_list;

    //auto J = [](auto const& point){return 1.0;};

    //std::cout << J(IP_1) << std::endl;


    //CellIntegrator<3, 2, 2> CellQuad(IP_list_p);

    //auto Cell_Volume = CellQuad(J);
    
    //std::cout << "Cell Volume: " << Cell_Volume << std::endl;

    ////CellIntegrator<2,2,2> CellQuad;
    //    
    //std::cout << "Weights:" << std::endl;
    //for (auto w : std::get<0>(CellQuad.dir_weights)) std::cout << w << " "; std::cout << std::endl; 
    //for (auto w : std::get<1>(CellQuad.dir_weights)) std::cout << w << " "; std::cout << std::endl;
    //for (auto w : std::get<2>(CellQuad.dir_weights)) std::cout << w << " "; std::cout << std::endl;
    //std::cout << "____________________________________________________________________________" << std::endl;
    //
    //for (auto w : CellQuad.weights_) std::cout << w << " "; std::cout << std::endl;


    //D.make_element<ElementBase<dim,3 ,42> > (2 , {1,2,3});
    //D.make_element<ElementBase<dim,5 ,42> > (4 , {1,2,3,4,5});
    //D.make_element<ElementBase<dim,7 ,42> > (5 , {1,2,3,4,5,6,7});
    //D.make_element<ElementBase<dim,14   > > (7 , {1,2,3,4,5,6,7,8,9,10,11,12,13,14});
    //D.make_element<ElementBase<dim,5    > > (16, {1,2,3,4,5});
    //D.make_element<ContinuumElement<dim,9>> (2 , {1,2,3,4,5,6,7,8,9});
    //D.make_element<ElementBase<dim,5    > > (42, {1,2,3,4,5}); 
    //D.make_element<ElementBase<dim,9    > > (44, {1,2,3,4,5,6,7,8,9}); 
    //D.make_element<ElementBase<dim,13   > > (50, {1,2,3,4,5,6,7,8,9,10,11,12,13});
    //                    
    //for (auto const& e: D.elements_){
    //    std::cout << id(e)<< " " << num_nodes(e)<< " " << num_dof(e)<< " ";
    //        for (auto const& n: nodes(e)){
    //            std::cout << n << " ";
    //        }
    //    std::cout << std::endl;
    //}

}// PETSc Scope ends here
PetscFinalize(); //This is necessary to avoid memory leaks and MPI errors.
}; 

