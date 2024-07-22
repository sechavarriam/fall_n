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
#include "src/domain/elements/ElementGeometry.hh"
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

int main(int argc, char **args){
PetscInitialize(&argc, &args, nullptr, nullptr);{ // PETSc Scope starts here

    std::string mesh_file = "/home/sechavarriam/MyLibs/fall_n/data/input/box.msh";

    static constexpr int dim  = 3;
    static constexpr int ndof = 3; 

    Domain<dim> D; //Domain Aggregator Object
    GmshDomainBuilder domain_constructor(mesh_file, D);
    
    //Compute Domain Volume for Testing integration
    
    auto _1 = []([[maybe_unused]] const std::array<double,dim>& x)->double {return 1.0;};
    auto __1 = std::function<double(const std::array<double,dim>&)>(_1);

    auto _1vec = []([[maybe_unused]] const std::array<double,dim>& x){
        Vector aux; 
        aux = Vector{std::array{1.0, 2.0, 3.0}};
        return aux;
    };

    auto __1vec = std::function<Vector(const std::array<double,dim>&)>(_1vec);

    auto _1mat = []([[maybe_unused]] const std::array<double,dim>& x)->Matrix {
        Matrix aux;
        aux = Matrix{std::array{1.0, 2.0, 3.0, 2.0, 1.0, 3.0, 3.0, 2.0, 1.0}, 3,3};
        return aux;
    };

    auto __1mat = std::function<Matrix(const std::array<double,dim>&)>(_1mat);

    std::cout << "Testing Matrix Function: " << std::endl;
    //Vector ones = _1vec(std::array{1.0, 2.0, 3.0});
    Matrix ones = _1mat(std::array{1.0, 2.0, 3.0});
    ones.print_content();
    std::cout << "::::::::::::::::::::::::::::::::::::::::::::::::::::::::::" << std::endl;



    double volume = 0.0;
    auto volume_vec = Vector{std::array{0.0, 0.0, 0.0}};
    auto volume_mat = Matrix{std::array{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 3,3};

    for (auto const& element : D.elements()){
        volume     += element.integrate(__1);
        volume_vec += element.integrate(_1vec);
        volume_mat += element.integrate(_1mat);
    }

    std::cout << "Domain Volume: " << volume << std::endl;
    

    volume_vec.print_content();
    volume_mat.print_content();




    Model<LinealElastic3D,ndof> M{D}; //Model Aggregator Object
    //          ^            
    //          | 
    //    Constitutive Relation Type (Policy) and Dimension implicitly.

    std::array dataA{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};    
    std::array data1{1.0, 2.0, 3.0};

    Matrix A{dataA, 3,3};
    Vector x{data1};
    auto y = A*x;

    y.print_content();

    auto yy = y*10.0;
    auto yyy = 100*y;

    yy.print_content();
    yyy.print_content();

    ContinuumIsotropicRelation steel3D{200.0, 0.3};    
    UniaxialIsotropicRelation  steel1D{200.0}     ;

    ContinuumIsotropicElasticMaterial steel_mat3D{200.0, 0.3};
    UniaxialIsotropicElasticMaterial  steel_mat1D{200.0};

    steel_mat3D.print_material_parameters();
    steel_mat1D.print_material_parameters();
    
    ContinuumElement<ContinuumIsotropicElasticMaterial,ndof> brick{&D.elements()[0]};//, steel_mat3D);
    



}// PETSc Scope ends here
PetscFinalize(); //This is necessary to avoid memory leaks and MPI errors.
}; 

