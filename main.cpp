//#include <Eigen/Dense>
#include <array>
#include <concepts>
#include <functional>
#include <iostream> 
#include <vector>
#include <numeric>
#include <algorithm>
#include <ranges>
#include <tuple>
#include <utility>

#include <charconv>
#include <string>
#include <string_view>
#include <fstream>
#include <filesystem>

#include"header_files.hh"

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

#include "src/model/Model.hh"
#include "src/model/ModelBuilder.hh"
#include "src/model/Graph.hh"

#include "src/mesh/Mesh.hh"

#include "src/mesh/gmsh/ReadGmsh.hh"
#include "src/mesh/gmsh/GmshDomainBuilder.hh"

#include "src/graph/AdjacencyList.hh"
#include "src/graph/AdjacencyMatrix.hh"


#include "src/domain/IntegrationPoint.hh"
#include "src/domain/MaterialPoint.hh"


//#include <matplot/matplot.h>

#include <petsc.h>
//#include <petscksp.h>
#include <petscsys.h>



int main(int argc, char **args){
PetscInitialize(&argc, &args, nullptr, nullptr);{ // PETSc Scope starts here

    static constexpr int dim = 3;
    Domain<dim> D; //Domain Aggregator Object

    // Mesh File Location
    std::string mesh_file = "/home/sechavarriam/MyLibs/fall_n/data/input/box.msh";

    GmshDomainBuilder domain_constructor(mesh_file, D);

    //GmshDomainBuilder_3D domain_constructor(mesh_file);
    for (auto const& element : D.elements()){
        std::cout << "element " << id(element) << std::endl;
        print_nodes_info(element);
    }

    auto integrator = GaussLegendreCellIntegrator<5,5,5>{};
    

    auto TestElement = LagrangeElement<2,2,2>{{D.node_p(6),D.node_p(2),D.node_p(4),D.node_p(0),D.node_p(7),D.node_p(3),D.node_p(5),D.node_p(1)}};




    static_assert(std::invocable<decltype(integrator),decltype(TestElement),std::function<double(geometry::Point<dim>)>>);

    auto one_ = [](geometry::Point<3> point){return (point.coord(0)-point.coord(0))+1.0;};

    // Element Jacobian
    auto X = geometry::Point<dim>{1.0, 2.0, 3.0};
    auto Jx = TestElement.evaluate_jacobian(X);

    std::cout << "Jacobian for X: " << std::endl;
    for(std::size_t i = 0; i<dim; ++i){
        for(std::size_t j = 0; j<dim; ++j){
            printf("%f ", Jx[i][j]);
        }
        std::cout << std::endl;
    }

    std::cout << "determinant: " << TestElement.detJ(X) << std::endl;

    // Element Volume
    auto volume = integrator(TestElement,one_);
    std::cout << "Volume: " << volume << std::endl;


    //TestElement.reference_element_.print_node_coords();

    
    //auto volume = integrator(TestElement,one_);



    // Desired sintax
    //integrate(element,function);
    // ó
    //element.integrate(function);

     //std::function<double(double)> Fn = [](double x){return x*x;};
    //for (int i = 0; ++i, i<10;)std::cout << i << ' ' << Fn(i) << std::endl;
    //constexpr short order = 15;
    //Quadrature<1,order> GaussOrder3(GaussLegendre::Weights1D<order>(),GaussLegendre::evalPoints1D<order>());
    //std::cout << GaussOrder3([](double x){return x*x;}) << std::endl;
    //std::cout << GaussOrder3(Fn) << std::endl;
    //std::cout << "____________________________" << std::endl;
    //auto W_2D = GaussLegendre::Weights<1,3>();
    //for(auto&& i:W_2D) std::cout << i << std::endl;



/*
    std::array dataA{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};    
    std::array data1{1.0, 2.0, 3.0};


    Matrix A{dataA, 3,3};
    Vector x{data1};
    auto y = A*x;

    ContinuumIsotropicRelation steel3D{200.0, 0.3};    
    UniaxialIsotropicRelation  steel1D{200.0};

    ContinuumIsotropicElasticMaterial steel_mat3D{200.0, 0.3};
    UniaxialIsotropicElasticMaterial  steel_mat1D{200.0};
*/

    //steel_mat3D.print_material_parameters();
    //steel_mat1D.print_material_parameters();
    
    //auto integrationScheme = [](auto const & e){/**integrate*/};
    //Element test1{ElementBase<dim,10,42>{1, {1,2,3,4,5,6,7,8,9,10}}, integrationScheme};


    //Model<dim> model;
    //std::size_t num_dofs = 6;
    //ModelBuilder<dim> model_builder (model,D,num_dofs);
    
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

/*
    Node<dim> N1{1,0.0,0.0,0.0};
    Node<dim> N2{2,1.0,0.0,0.0};
    Node<dim> N3{3,1.0,1.0,0.0};
    Node<dim> N4{4,0.0,1.0,0.0};
    Node<dim> N5{5,0.0,0.0,1.0};
    Node<dim> N6{6,1.0,0.0,1.0};
    Node<dim> N7{7,1.0,1.0,1.0};
    Node<dim> N8{8,0.0,1.0,1.0};
    Node<dim> N9{9,0.5,0.5,0.5};

    N1.set_num_dof(6);
    N2.set_num_dof(3);

    std::cout << N1.num_dof()   << " " << N2.num_dof()   << std::endl;
    std::cout << N1.num_dof_h() << " " << N2.num_dof_h() << std::endl;
*/

    //Element test_L0{LagrangeElement<nx,ny,nz>{{D.node_p(0 ),D.node_p(1 ),D.node_p(2 ),
    //                                           D.node_p(3 ),D.node_p(4 ),D.node_p(5 ),
    //                                           D.node_p(6 ),D.node_p(7 ),D.node_p(8 ),
    //                                           D.node_p(9 ),D.node_p(10),D.node_p(11),
    //                                           D.node_p(12),D.node_p(13),D.node_p(14),
    //                                           D.node_p(15),D.node_p(16),D.node_p(17)}}, integrationScheme};
    //Element test_L1{E1, integrationScheme};
    //GaussIntegrator<2,2,2> LagElem_integrator(E1);
    //static_assert(is_LagrangeElement<LagrangeElement<nx,ny,nz>>); //Concept test.

    /*
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
    //std::cout << "Weights:" << std::endl;
    //for (auto w : std::get<0>(CellQuad.dir_weights)) std::cout << w << " "; std::cout << std::endl; 
    //for (auto w : std::get<1>(CellQuad.dir_weights)) std::cout << w << " "; std::cout << std::endl;
    //for (auto w : std::get<2>(CellQuad.dir_weights)) std::cout << w << " "; std::cout << std::endl;
    //std::cout << "____________________________________________________________________________" << std::endl;
    //
    //for (auto w : CellQuad.weights_) std::cout << w << " "; std::cout << std::endl;

}// PETSc Scope ends here
PetscFinalize(); //This is necessary to avoid memory leaks and MPI errors.
}; 

