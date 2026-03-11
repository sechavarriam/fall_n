
#include "header_files.hh"
#include "src/continuum/HyperelasticRelation.hh"

#include "src/geometry/Point.hh"

#include <iostream>
#include <iomanip>
#include <string>

// =============================================================================
//  Nonlinear showcase — 7 cases on the same Gmsh geometry
// =============================================================================
//
//  Mesh:  data/input/box.msh  (20 hex27,  315 nodes,  [0,10]×[0,0.4]×[0,0.8])
//
//  BCs:   Fixed face: x = 0   (all DOFs clamped)
//         Load face:  x = 10  (surface traction via Gauss quadrature)
//
//  Each case exports:
//     data/output/case_N_mesh.vtu   — nodal fields (displacement)
//     data/output/case_N_gauss.vtu  — Gauss-point fields (strain, stress, ...)
//
//  Cases:
//    1. SmallStrain + Linear Elastic          (reference)
//    2. TotalLagrangian + SVK                 (small load, single step)
//    3. TotalLagrangian + SVK incremental     (large load, 5 steps)
//    4. TotalLagrangian + Neo-Hookean incr.   (large load, 5 steps)
//    5. UpdatedLagrangian + SVK               (small load, single step)
//    6. UpdatedLagrangian + Neo-Hookean incr. (large load, 5 steps)
//    7. SmallStrain + J2 Plasticity incr.     (beyond yield, 20 steps)
//
// =============================================================================

static constexpr std::size_t DIM  = 3;
static constexpr std::size_t NDOF = DIM;

static constexpr double E_mod  = 200.0;
static constexpr double nu     = 0.3;
//static constexpr double sigma_y= 0.250;   // Initial yield stress (J2)
//static constexpr double H_hard = 10.0;    // Isotropic hardening modulus

static const std::string MESH =
    "/home/sechavarriam/MyLibs/fall_n/data/input/box.msh";
static const std::string OUT =
    "/home/sechavarriam/MyLibs/fall_n/data/output/";

// ── Helper: export model to a pair of VTU files ─────────────────────────────
template <typename ModelT>
static void export_vtk(ModelT& M, const std::string& tag) {
    M.update_elements_state();

    fall_n::vtk::VTKModelExporter exporter(M);
    exporter.set_displacement();
    exporter.compute_material_fields();

    exporter.write_mesh       (OUT + tag + "_mesh.vtu");
    exporter.write_gauss_points(OUT + tag + "_gauss.vtu");

    std::cout << "    → " << tag << "_mesh.vtu  +  " << tag << "_gauss.vtu\n";
}

// ── Helper: max displacement magnitude ──────────────────────────────────────
static double max_disp(auto& M) {
    const PetscScalar* arr;
    PetscInt n;
    VecGetLocalSize(M.state_vector(), &n);
    VecGetArrayRead (M.state_vector(), &arr);
    double mx = 0.0;
    for (PetscInt i = 0; i < n; ++i)
        mx = std::max(mx, std::abs(arr[i]));
    VecRestoreArrayRead(M.state_vector(), &arr);
    return mx;
}

int main(int argc, char** args)
{
    PetscInitialize(&argc, &args, nullptr, nullptr);

    // SNES options (direct solver for moderate-size problem)
    PetscOptionsSetValue(nullptr, "-ksp_type",  "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type",   "lu");
    PetscOptionsSetValue(nullptr, "-snes_rtol", "1e-8");
    PetscOptionsSetValue(nullptr, "-snes_atol", "1e-10");
    PetscOptionsSetValue(nullptr, "-snes_max_it", "50");
    PetscOptionsSetValue(nullptr, "-snes_converged_reason", nullptr);
    PetscOptionsSetValue(nullptr, "-snes_linesearch_type", "basic");

    {
        double small_load = 0.05;  // within linear regime
       // double large_load = 0.50;  // significant geometric nonlinearity
        //double plast_load = 1.00;  // well beyond yield

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "============================================================\n"
                  << "  Nonlinear Showcase — 7 cases on box.msh\n"
                  << "============================================================\n\n";

        // ═════════════════════════════════════════════════════════════════
        //  Case 1: SmallStrain + Linear Elastic (reference)
        // ═════════════════════════════════════════════════════════════════
        {
            std::cout << "── Case 1: SmallStrain + Linear Elastic ──\n";

            Domain<DIM> D;  GmshDomainBuilder b(MESH, D);

            ContinuumIsotropicElasticMaterial mat_inst{E_mod, nu};
            Material<ThreeDimensionalMaterial> mat{mat_inst, ElasticUpdate{}};

            Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF> M{D, mat};
            M.fix_x(0.0);
            M.setup();

            D.create_boundary_from_plane("Load", 0, 10.0);
            M.apply_surface_traction("Load", 0.0, small_load, -small_load);

            LinearAnalysis<ThreeDimensionalMaterial> solver{&M};
            solver.solve();

            std::cout << "    Max |u| = " << max_disp(M) << "\n";
            export_vtk(M, "case1_SS_elastic");
        }
/*^
        // ═════════════════════════════════════════════════════════════════
        //  Case 2: TotalLagrangian + SVK (small load, 1 step)
        // ═════════════════════════════════════════════════════════════════
        {
            std::cout << "\n── Case 2: TotalLagrangian + SVK (small load) ──\n";

            Domain<DIM> D;  GmshDomainBuilder b(MESH, D);

            auto svk = continuum::SaintVenantKirchhoff<3>::from_E_nu(E_mod, nu);
            MaterialInstance<continuum::SVKRelation<3>> inst{svk};
            Material<ThreeDimensionalMaterial> mat{inst, ElasticUpdate{}};

            Model<ThreeDimensionalMaterial, continuum::TotalLagrangian, NDOF> M{D, mat};
            M.fix_x(0.0);
            M.setup();

            D.create_boundary_from_plane("Load", 0, 10.0);
            M.apply_surface_traction("Load", 0.0, small_load, -small_load);

            NonlinearAnalysis<ThreeDimensionalMaterial, continuum::TotalLagrangian> nl{&M};
            nl.solve();

            std::cout << "    SNES iterations: " << nl.num_iterations()
                      << "  reason: " << static_cast<int>(nl.converged_reason()) << "\n";
            std::cout << "    Max |u| = " << max_disp(M) << "\n";
            export_vtk(M, "case2_TL_SVK_small");
        }

        // ═════════════════════════════════════════════════════════════════
        //  Case 3: TotalLagrangian + SVK incremental (large load)
        // ═════════════════════════════════════════════════════════════════
        {
            std::cout << "\n── Case 3: TotalLagrangian + SVK incremental ──\n";

            Domain<DIM> D;  GmshDomainBuilder b(MESH, D);

            auto svk = continuum::SaintVenantKirchhoff<3>::from_E_nu(E_mod, nu);
            MaterialInstance<continuum::SVKRelation<3>> inst{svk};
            Material<ThreeDimensionalMaterial> mat{inst, ElasticUpdate{}};

            Model<ThreeDimensionalMaterial, continuum::TotalLagrangian, NDOF> M{D, mat};
            M.fix_x(0.0);
            M.setup();

            D.create_boundary_from_plane("Load", 0, 10.0);
            M.apply_surface_traction("Load", 0.0, large_load, -large_load);

            NonlinearAnalysis<ThreeDimensionalMaterial, continuum::TotalLagrangian> nl{&M};
            nl.solve_incremental(5);

            std::cout << "    reason: " << static_cast<int>(nl.converged_reason()) << "\n";
            std::cout << "    Max |u| = " << max_disp(M) << "\n";
            export_vtk(M, "case3_TL_SVK_large");
        }

        // ═════════════════════════════════════════════════════════════════
        //  Case 4: TotalLagrangian + Neo-Hookean incremental
        // ═════════════════════════════════════════════════════════════════
        {
            std::cout << "\n── Case 4: TotalLagrangian + Neo-Hookean incremental ──\n";

            Domain<DIM> D;  GmshDomainBuilder b(MESH, D);

            auto nh = continuum::CompressibleNeoHookean<3>::from_E_nu(E_mod, nu);
            MaterialInstance<continuum::NeoHookeanRelation<3>> inst{nh};
            Material<ThreeDimensionalMaterial> mat{inst, ElasticUpdate{}};

            Model<ThreeDimensionalMaterial, continuum::TotalLagrangian, NDOF> M{D, mat};
            M.fix_x(0.0);
            M.setup();

            D.create_boundary_from_plane("Load", 0, 10.0);
            M.apply_surface_traction("Load", 0.0, large_load, -large_load);

            NonlinearAnalysis<ThreeDimensionalMaterial, continuum::TotalLagrangian> nl{&M};
            nl.solve_incremental(5);

            std::cout << "    reason: " << static_cast<int>(nl.converged_reason()) << "\n";
            std::cout << "    Max |u| = " << max_disp(M) << "\n";
            export_vtk(M, "case4_TL_NH_large");
        }

        // ═════════════════════════════════════════════════════════════════
        //  Case 5: UpdatedLagrangian + SVK (small load, 1 step)
        // ═════════════════════════════════════════════════════════════════
        {
            std::cout << "\n── Case 5: UpdatedLagrangian + SVK (small load) ──\n";

            Domain<DIM> D;  GmshDomainBuilder b(MESH, D);

            auto svk = continuum::SaintVenantKirchhoff<3>::from_E_nu(E_mod, nu);
            MaterialInstance<continuum::SVKRelation<3>> inst{svk};
            Material<ThreeDimensionalMaterial> mat{inst, ElasticUpdate{}};

            Model<ThreeDimensionalMaterial, continuum::UpdatedLagrangian, NDOF> M{D, mat};
            M.fix_x(0.0);
            M.setup();

            D.create_boundary_from_plane("Load", 0, 10.0);
            M.apply_surface_traction("Load", 0.0, small_load, -small_load);

            NonlinearAnalysis<ThreeDimensionalMaterial, continuum::UpdatedLagrangian> nl{&M};
            nl.solve();

            std::cout << "    SNES iterations: " << nl.num_iterations()
                      << "  reason: " << static_cast<int>(nl.converged_reason()) << "\n";
            std::cout << "    Max |u| = " << max_disp(M) << "\n";
            export_vtk(M, "case5_UL_SVK_small");
        }

        // ═════════════════════════════════════════════════════════════════
        //  Case 6: UpdatedLagrangian + Neo-Hookean incremental
        // ═════════════════════════════════════════════════════════════════
        {
            std::cout << "\n── Case 6: UpdatedLagrangian + Neo-Hookean incremental ──\n";

            Domain<DIM> D;  GmshDomainBuilder b(MESH, D);

            auto nh = continuum::CompressibleNeoHookean<3>::from_E_nu(E_mod, nu);
            MaterialInstance<continuum::NeoHookeanRelation<3>> inst{nh};
            Material<ThreeDimensionalMaterial> mat{inst, ElasticUpdate{}};

            Model<ThreeDimensionalMaterial, continuum::UpdatedLagrangian, NDOF> M{D, mat};
            M.fix_x(0.0);
            M.setup();

            D.create_boundary_from_plane("Load", 0, 10.0);
            M.apply_surface_traction("Load", 0.0, large_load, -large_load);

            NonlinearAnalysis<ThreeDimensionalMaterial, continuum::UpdatedLagrangian> nl{&M};
            nl.solve_incremental(5);

            std::cout << "    reason: " << static_cast<int>(nl.converged_reason()) << "\n";
            std::cout << "    Max |u| = " << max_disp(M) << "\n";
            export_vtk(M, "case6_UL_NH_large");
        }

        // ═════════════════════════════════════════════════════════════════
        //  Case 7: SmallStrain + J2 Plasticity incremental
        // ═════════════════════════════════════════════════════════════════
        {
            std::cout << "\n── Case 7: SmallStrain + J2 Plasticity incremental ──\n";

            Domain<DIM> D;  GmshDomainBuilder b(MESH, D);

            J2PlasticMaterial3D j2_inst{E_mod, nu, sigma_y, H_hard};
            Material<ThreeDimensionalMaterial> mat{j2_inst, InelasticUpdate{}};

            Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF> M{D, mat};
            M.fix_x(0.0);
            M.setup();

            D.create_boundary_from_plane("Load", 0, 10.0);
            M.apply_surface_traction("Load", 0.0, plast_load, -plast_load);

            NonlinearAnalysis<ThreeDimensionalMaterial, continuum::SmallStrain> nl{&M};
            nl.solve_incremental(20);

            std::cout << "    reason: " << static_cast<int>(nl.converged_reason()) << "\n";
            std::cout << "    Max |u| = " << max_disp(M) << "\n";
            export_vtk(M, "case7_SS_J2_plastic");
        }
*/
        std::cout << "\n============================================================\n"
                  << "  All 7 cases exported to " << OUT << "\n"
                  << "============================================================\n";

    } // PETSc scope
    PetscFinalize();
    return 0;
}
