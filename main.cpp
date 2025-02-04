
#include "header_files.hh"

int main(int argc, char **args)
{
    PetscInitialize(&argc, &args, nullptr, nullptr);
    { // PETSc Scope starts here

        std::string mesh_file = "/home/sechavarriam/MyLibs/fall_n/data/input/box.msh";

        static constexpr std::size_t dim  = 3;
        static constexpr std::size_t ndof = dim; // 6;

        Domain<dim> D1; // Domain Aggregator Object
        GmshDomainBuilder domain_constructor(mesh_file, D1);

        auto updateStrategy = [](){std::cout << "TEST: e.g. Linear Update Strategy" << std::endl;};
        

        Domain<dim> D2; // Domain Aggregator Object (Second Domain)

        double H0 = 0.0, H1 = 4.0;
        double L0 = 0.0, L1 = 5.0;
        double B0 = 0.0, B1 = 6.0;

        //double H0 = 1.0, H1 = 1.0;
        //double L0 = 1.0, L1 = 1.0;
        //double W0 = 1.0, W1 = 1.0;

        D2.preallocate_node_capacity(8);
        D2.add_node(0,  B0,  L0, H0);
        D2.add_node(1,  B1,  L0, H0);
        D2.add_node(2,  B0,  L1, H0);
        D2.add_node(3,  B1,  L1, H0);
        D2.add_node(4,  B0,  L0, H1);
        D2.add_node(5,  B1,  L0, H1);
        D2.add_node(6,  B0,  L1, H1);
        D2.add_node(7,  B1,  L1, H1);

        D2.make_element<LagrangeElement<2,2,2>>(GaussLegendreCellIntegrator<2,2,2>{}, 0, std::array{0,1,2,3,4,5,6,7}.data());
        D2.assemble_sieve();


        Model<ThreeDimensionalMaterial, ndof> M1{D1, Material<ThreeDimensionalMaterial>{ContinuumIsotropicElasticMaterial{200.0, 0.3}, updateStrategy}};
        Model<ThreeDimensionalMaterial, ndof> M2{D2, Material<ThreeDimensionalMaterial>{ContinuumIsotropicElasticMaterial{200.0, 0.3}, updateStrategy}};        

        std::cout << " ========================== GMSH DOMAIN =============================== " << std::endl;   
        for (auto& e : M1.get_domain().elements()) e.print_info();
        std::cout << " ========================== MANUAL DOMAIN ============================= " << std::endl;
        for (auto& e : M2.get_domain().elements()) e.print_info();
        std::cout << " ====================================================================== " << std::endl;

        VTKDataContainer view1;
        view1.load_domain(      M1.get_domain());
        view1.load_gauss_points(M1.get_domain());

        VTKDataContainer view2;
        view2.load_domain(      M2.get_domain());
        view2.load_gauss_points(M2.get_domain());

        const int _x = 0;

        M1.fix_x(B0);
        M1.setup();
        M1._force_orthogonal_plane(_x, B1, 1.0, 0.0, 0.0);

        M2.fix_x(B0);
        M2.setup();
        M2._force_orthogonal_plane(_x, B1, 1.0, 0.0, 0.0);


        LinearAnalysis analisis_obj1{&M1};
        analisis_obj1.solve(); 
        analisis_obj1.record_solution(view1);

        LinearAnalysis analisis_obj2{&M2};
        analisis_obj2.solve();
        analisis_obj2.record_solution(view2);

        view1.write_vtu("/home/sechavarriam/MyLibs/fall_n/data/output/test1.vtu");
        view1.write_gauss_vtu("/home/sechavarriam/MyLibs/fall_n/data/output/gauss_test1.vtu");

        view2.write_vtu("/home/sechavarriam/MyLibs/fall_n/data/output/test2.vtu");
        view2.write_gauss_vtu("/home/sechavarriam/MyLibs/fall_n/data/output/gauss_test2.vtu");

        //NLAnalysis nl_analisis_obj{&M1};
        //nl_analisis_obj.solve();

    } // PETSc Scope ends here
    PetscFinalize(); // This is necessary to avoid memory leaks and MPI errors.
};


/*
        // Testing Material Wrapper interface.
        // Printing Material Parameters (Not YET)
        // Printing Material StateC

        Strain<6> e0{0.01, 0.02, 0.03, 0.04, 0.05, 0.06};

        MaterialState<ElasticState, Strain<6>> sv0{e0};
        MaterialState<MemoryState , Strain<6>> sv1{e0};

        UniaxialIsotropicElasticMaterial  steel_mat1D{200.0};
        ContinuumIsotropicElasticMaterial steel_mat3D{200.0, 0.3};

        // Material<UniaxialMaterial>         mat1D(steel_mat1D, updateStrategy);
        Material<ThreeDimensionalMaterial> mat3D(steel_mat3D, updateStrategy);        
        //auto s1 = mat1D.get_state();
        auto s2 = mat3D.get_state();

        // mat1D.update_state(e0);
        mat3D.update_state(e0);

        auto s3 = mat3D.get_state();

        //for (auto i = 0; i < 6; i++)
        //    std::cout << "s2[" << i << "] = " << s2[i] << std::endl;
        //for (auto i = 0; i < 6; i++)
        //    std::cout << "s3[" << i << "] = " << s3[i] << std::endl;
        //steel_mat3D.print_material_parameters();

*/
