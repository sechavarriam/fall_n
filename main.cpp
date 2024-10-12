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

#include "src/materials/constitutive_models/non_lineal/InelasticRelation.hh"

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
    
    Model<LinealElastic3D,ndof> M{D}; //Model Aggregator Object
    //          ^            
    //          | 
    //    Constitutive Relation Type (Policy) and Dimension implicitly.

    Strain<6> e0 {0.01, 0.02, 0.03, 0.04, 0.05, 0.06};

    MaterialState<ElasticState,Strain<6>> sv0{e0};
    MaterialState<MemoryState ,Strain<6>> sv1{e0};

    //auto updateStrategy = [](){std::cout << "TEST: e.g. Linear Update Strategy" << std::endl;};

    UniaxialIsotropicElasticMaterial  steel_mat1D{200.0};
    ContinuumIsotropicElasticMaterial steel_mat3D{200.0, 0.3};

    //Material<UniaxialMaterial>         mat1D(steel_mat1D, updateStrategy);
    Material<ThreeDimensionalMaterial> mat3D(steel_mat3D, 0);

    // Testing Material Wrapper interface.
    // Printing Material Parameters (Not YET)
    // Printing Material State

    //auto s1 = mat1D.get_state();
    //auto s2 = mat3D.get_state();





    /*
    ContinuumIsotropicRelation steel3D{200.0, 0.3};    
    UniaxialIsotropicRelation  steel1D{200.0}     ; 
    */    

    
    

    steel_mat3D.print_material_parameters();
    steel_mat1D.print_material_parameters();

    // INTENT:
    // Material mat(base_material, stress_update_strategy);
    // e.g. 

    // Material mat(steel3D, ElasticUpdateStrategy::Linear{});
    // Material mat(steel1D, ElasticUpdateStrategy::Linear{});

    // Material mat(steel3D, InelasticUpdateStrategy::ReturnMapping);
    // Material mat(steel3D, InelasticUpdateStrategy::FullImplicitBackwardEuler);
    // Material mat(steel3D, InelasticUpdateStrategy::SemiImplicitBackwardEuler);
    // Material mat(steel3D, InelasticUpdateStrategy::RateTanget);
    // Material mat(steel3D, InelasticUpdateStrategy::IncrementallyObjective);


    //ContinuumElement<ContinuumIsotropicElasticMaterial,ndof> brick{&D.elements()[0]};//, steel_mat3D);
    //Matrix K;
    //K = brick.K(steel_mat3D);
    //K.print_content();


}// PETSc Scope ends here
PetscFinalize(); //This is necessary to avoid memory leaks and MPI errors.
}; 

