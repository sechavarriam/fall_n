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

#include "src/domain/IntegrationPoint.hh"

#include "src/post-processing/VTK/VTKheaders.hh"
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

        Analysis analisis_obj{&M};

        M.apply_node_force(7, 1.0, 1.0, 1.0);
        
        M.fix_node(0);
        M.fix_node(1);
        M.fix_node(4);
        M.fix_node(5);

        analisis_obj.solve();

        //M.fix_node_dofs(0, 0,2);
        ////////M.solve();

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
        
        vtkNew<vtkNamedColors>            colors;
        vtkNew<vtkCylinderSource>         cylinder;
        vtkNew<vtkPolyDataMapper>         cylinderMapper;
        vtkNew<vtkActor>                  cylinderActor;
        vtkNew<vtkRenderer>               renderer;
        vtkNew<vtkRenderWindow>           renderWindow;
        vtkNew<vtkRenderWindowInteractor> renderWindowInteractor;

        std::array<unsigned char, 4> bkg{{26, 51, 102, 255}}; // Set the background color.
        colors->SetColor("BkgColor", bkg.data());

        cylinder->SetResolution(8);
        cylinderMapper->SetInputConnection(cylinder->GetOutputPort());
        
        cylinderActor->SetMapper(cylinderMapper);
        cylinderActor->GetProperty()->SetColor(colors->GetColor4d("Tomato").GetData());
        cylinderActor->RotateX(30.0);
        cylinderActor->RotateY(-45.0);

        renderer->AddActor(cylinderActor);
        renderer->SetBackground(colors->GetColor3d("BkgColor").GetData());

        renderer->ResetCamera();
        renderer->GetActiveCamera()->Zoom(1.5);

        renderWindow->SetSize(300, 300);
        renderWindow->AddRenderer(renderer);
        renderWindow->SetWindowName("Cylinder");
        
        //enderWindowInteractor->SetRenderWindow(renderWindow);
        //enderWindow->Render();
        //enderWindowInteractor->Start();

    } // PETSc Scope ends here
    PetscFinalize(); // This is necessary to avoid memory leaks and MPI errors.
};
