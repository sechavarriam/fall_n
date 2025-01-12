// #include <Eigen/Dense>
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

#include "header_files.hh"

#include "src/elements/Node.hh"

#include "src/elements/FEM_Element.hh"
#include "src/elements/ContinuumElement.hh"

#include "src/elements/element_geometry/ElementGeometry.hh"
#include "src/elements/element_geometry/LagrangeElement.hh"

#include "src/model/DoF.hh"

#include "src/geometry/geometry.hh"
#include "src/geometry/Topology.hh"
#include "src/geometry/Cell.hh"
#include "src/geometry/Point.hh"

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


#include "src/post-processing/VTK/VTKheaders.hh"
#include "src/post-processing/VTK/VTKwriter.hh"


// #include <matplot/matplot.h>

#include <petsc.h>

int main(int argc, char **args)
{
    PetscInitialize(&argc, &args, nullptr, nullptr);
    { // PETSc Scope starts here

        std::string mesh_file = "/home/sechavarriam/MyLibs/fall_n/data/input/box.msh";

        static constexpr std::size_t dim = 3;
        static constexpr std::size_t ndof = dim; // 6;

        Domain<dim> D; // Domain Aggregator Object
        GmshDomainBuilder domain_constructor(mesh_file, D);

        auto updateStrategy = []()
        { std::cout << "TEST: e.g. Linear Update Strategy" << std::endl; };

        Model<ThreeDimensionalMaterial, ndof> M{D, Material<ThreeDimensionalMaterial>{ContinuumIsotropicElasticMaterial{200.0, 0.3}, updateStrategy}};

        Strain<6> e0{0.01, 0.02, 0.03, 0.04, 0.05, 0.06};

        MaterialState<ElasticState, Strain<6>> sv0{e0};
        MaterialState<MemoryState, Strain<6>> sv1{e0};

        UniaxialIsotropicElasticMaterial steel_mat1D{200.0};
        ContinuumIsotropicElasticMaterial steel_mat3D{200.0, 0.3};

        // Material<UniaxialMaterial>         mat1D(steel_mat1D, updateStrategy);
        Material<ThreeDimensionalMaterial> mat3D(steel_mat3D, updateStrategy);

        // Testing Material Wrapper interface.
        // Printing Material Parameters (Not YET)
        // Printing Material State

        /*
        // auto s1 = mat1D.get_state();
        auto s2 = mat3D.get_state();

        // mat1D.update_state(e0);
        mat3D.update_state(e0);

        auto s3 = mat3D.get_state();

        for (auto i = 0; i < 6; i++)
            std::cout << "s2[" << i << "] = " << s2[i] << std::endl;
        for (auto i = 0; i < 6; i++)
            std::cout << "s3[" << i << "] = " << s3[i] << std::endl;

        steel_mat3D.print_material_parameters();
        */

        //M.boundary_constraining_begin();

        VTKDataContainer view;
        view.load_domain(M.get_domain());
        view.load_gauss_points(M.get_domain());

        //M.fix_node(0);
        //M.fix_node(1);
        //M.fix_node(4);
        //M.fix_node(5);

        M.fix_x(0.0);

        M.setup();
        //M.boundary_constraining_end(); //seting up sieve layout to have correct sizes in the mesh (and perform processor communication).

        //M.apply_node_force(385, 0.0, 0.0, -1.0);
        M.apply_node_force(6, 0.0, 0.0, -0.5);
        M.apply_node_force(4, 0.0, 0.0, -0.5);


        LinearAnalysis analisis_obj{&M};
        
        analisis_obj.solve();
        analisis_obj.record_solution(view);

        view.write_vtu("/home/sechavarriam/MyLibs/fall_n/data/output/structure.vtu");
        view.write_gauss_vtu("/home/sechavarriam/MyLibs/fall_n/data/output/structure_gauss.vtu");

        //NLAnalysis nl_analisis_obj{&M};
        //nl_analisis_obj.solve();


    } // PETSc Scope ends here
    PetscFinalize(); // This is necessary to avoid memory leaks and MPI errors.
};
