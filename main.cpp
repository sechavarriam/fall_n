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

#include "src/materials/ConstitutiveRelation.hh"

#include "src/materials/constitutive_models/lineal/ElasticRelation.hh"
#include "src/materials/constitutive_models/lineal/IsotropicRelation.hh"

#include "src/materials/Material.hh"

#include "src/materials/MaterialState.hh"
#include "src/materials/LinealElasticMaterial.hh"

#include "src/materials/Stress.hh"
#include "src/materials/Strain.hh"

#include "src/model/Model.hh"
#include "src/model/ModelBuilder.hh"

#include "src/mesh/Mesh.hh"

#include "src/mesh/gmsh/ReadGmsh.hh"
#include "src/mesh/gmsh/GmshDomainBuilder.hh"

#include "src/graph/AdjacencyList.hh"
#include "src/graph/AdjacencyMatrix.hh"

#include "src/domain/IntegrationPoint.hh"

//#include <matplot/matplot.h>

#include <petsc.h>

int main(int argc, char **args){
PetscInitialize(&argc, &args, nullptr, nullptr);{ // PETSc Scope starts here

    std::string mesh_file = "/home/sechavarriam/MyLibs/fall_n/data/input/box.msh";

    static constexpr std::size_t dim  = 3;
    static constexpr std::size_t ndof = 6; 

    Domain<dim> D; //Domain Aggregator Object
    GmshDomainBuilder domain_constructor(mesh_file, D);
    
    //Compute Domain Volume for Testing integration
    Model<LinealElastic3D,ndof> M{D}; //Model Aggregator Object
    //          ^            
    //          | 
    //    Constitutive Relation Type (Policy) and Dimension implicitly.

    ContinuumIsotropicRelation steel3D{200.0, 0.3};    
    UniaxialIsotropicRelation  steel1D{200.0}     ; 

    ContinuumIsotropicElasticMaterial steel_mat3D{200.0, 0.3};
    UniaxialIsotropicElasticMaterial  steel_mat1D{200.0};

    steel_mat3D.print_material_parameters();
    steel_mat1D.print_material_parameters();
    
    ContinuumElement<ContinuumIsotropicElasticMaterial,ndof> brick{&D.elements()[0]};//, steel_mat3D);

    Matrix K;
    K = brick.K(steel_mat3D);
    K.print_content();


}// PETSc Scope ends here
PetscFinalize(); //This is necessary to avoid memory leaks and MPI errors.
}; 

