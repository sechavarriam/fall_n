
#include "header_files.hh"
#include "src/continuum/HyperelasticRelation.hh"

#include "src/geometry/Point.hh"

#include <iostream>
#include <iomanip>
#include <string>

// =============================================================================
//  Nonlinear showcase — 7 cases × 2 element types (HEX27 + TET10)
// =============================================================================
//
//  Geometry: cantilever box [0,10] × [0,0.4] × [0,0.8]
//
//  Meshes:
//    HEX27: data/input/box.msh          (20 hex27, 315 nodes)
//    TET10: data/input/box_tet10.msh    (TET10, ~2840 nodes)
//
//  BCs:  Fixed face:  x = 0   (all DOFs clamped)
//        Load face:   x = 10  (surface traction via Gauss quadrature)
//
//  Post-processing:
//    - SPR volume-weighted averaging for Gauss-point → nodal projection
//      (unconditionally robust: all weights positive, no checkerboard)
//    - Dual VTU export: primary mesh + Gauss-point cloud
//
//  Cases:
//    1. SmallStrain + Linear Elastic         (reference)
//    2. TotalLagrangian + SVK                (small load, 1 step)
//    3. TotalLagrangian + SVK incremental    (large load, 5 steps)
//    4. TotalLagrangian + Neo-Hookean incr.  (large load, 5 steps)
//    5. UpdatedLagrangian + SVK              (small load, 1 step)
//    6. UpdatedLagrangian + Neo-Hookean incr.(large load, 5 steps)
//    7. SmallStrain + J2 Plasticity incr.    (beyond yield, 20 steps)
//
// =============================================================================

static constexpr std::size_t DIM  = 3;
static constexpr std::size_t NDOF = DIM;

// ── Material properties ─────────────────────────────────────────────────
static constexpr double E_mod   = 200.0;   // Young's modulus
static constexpr double nu      = 0.3;     // Poisson's ratio
static constexpr double sigma_y = 0.250;   // Initial yield stress (J2)
static constexpr double H_hard  = 10.0;    // Isotropic hardening modulus

// ── Mesh files ──────────────────────────────────────────────────────────
static const std::string BASE =
    "/home/sechavarriam/MyLibs/fall_n/";

static const std::string MESH_HEX27  = BASE + "data/input/box.msh";
static const std::string MESH_TET10  = BASE + "data/input/box_tet10.msh";
static const std::string OUT         = BASE + "data/output/";


// ── Helper: export model to VTU with SPR projection ─────────────────────
//
//  Uses spr_average_to_nodes() for all Gauss-point fields: this avoids
//  the checkerboard artefact that can occur with lumped L2 projection
//  when quadrature rules have negative weights or when vertex shape
//  functions go negative at interior quadrature points.
//
template <typename ModelT>
static void export_vtk(ModelT& M, const std::string& tag) {
    M.update_elements_state();

    fall_n::vtk::VTKModelExporter exporter(M);
    exporter.set_displacement();
    exporter.compute_material_fields_spr();

    exporter.write_mesh       (OUT + tag + "_mesh.vtu");
    exporter.write_gauss_points(OUT + tag + "_gauss.vtu");

    std::cout << "    -> " << tag << "_mesh.vtu  +  " << tag << "_gauss.vtu\n";
}

// ── Helper: max displacement magnitude ──────────────────────────────────
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


// ═════════════════════════════════════════════════════════════════════════
//  Case runners — parameterized by mesh file and output prefix
// ═════════════════════════════════════════════════════════════════════════

static void case1_small_strain_elastic(const std::string& mesh,
                                       const std::string& prefix)
{
    std::cout << "-- Case 1: SmallStrain + Linear Elastic (" << prefix << ") --\n";

    Domain<DIM> D;  GmshDomainBuilder b(mesh, D);

    ContinuumIsotropicElasticMaterial mat_inst{E_mod, nu};
    Material<ThreeDimensionalMaterial> mat{mat_inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF> M{D, mat};
    M.fix_x(0.0);
    M.setup();

    D.create_boundary_from_plane("Load", 0, 10.0);
    M.apply_surface_traction("Load", 0.0, 0.05, -0.05);

    LinearAnalysis<ThreeDimensionalMaterial> solver{&M};
    solver.solve();

    std::cout << "    Max |u| = " << max_disp(M) << "\n";
    export_vtk(M, prefix + "_case1_SS_elastic");
}

[[maybe_unused]] static void case2_TL_SVK_small(const std::string& mesh,
                                const std::string& prefix)
{
    std::cout << "\n-- Case 2: TotalLagrangian + SVK small (" << prefix << ") --\n";

    Domain<DIM> D;  GmshDomainBuilder b(mesh, D);

    auto svk = continuum::SaintVenantKirchhoff<3>::from_E_nu(E_mod, nu);
    MaterialInstance<continuum::SVKRelation<3>> inst{svk};
    Material<ThreeDimensionalMaterial> mat{inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::TotalLagrangian, NDOF> M{D, mat};
    M.fix_x(0.0);
    M.setup();

    D.create_boundary_from_plane("Load", 0, 10.0);
    M.apply_surface_traction("Load", 0.0, 0.05, -0.05);

    NonlinearAnalysis<ThreeDimensionalMaterial, continuum::TotalLagrangian> nl{&M};
    nl.solve();

    std::cout << "    SNES iterations: " << nl.num_iterations()
              << "  reason: " << static_cast<int>(nl.converged_reason()) << "\n";
    std::cout << "    Max |u| = " << max_disp(M) << "\n";
    export_vtk(M, prefix + "_case2_TL_SVK_small");
}

[[maybe_unused]] static void case3_TL_SVK_large(const std::string& mesh,
                                const std::string& prefix)
{
    std::cout << "\n-- Case 3: TotalLagrangian + SVK incremental (" << prefix << ") --\n";

    Domain<DIM> D;  GmshDomainBuilder b(mesh, D);

    auto svk = continuum::SaintVenantKirchhoff<3>::from_E_nu(E_mod, nu);
    MaterialInstance<continuum::SVKRelation<3>> inst{svk};
    Material<ThreeDimensionalMaterial> mat{inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::TotalLagrangian, NDOF> M{D, mat};
    M.fix_x(0.0);
    M.setup();

    D.create_boundary_from_plane("Load", 0, 10.0);
    M.apply_surface_traction("Load", 0.0, 0.50, -0.50);

    NonlinearAnalysis<ThreeDimensionalMaterial, continuum::TotalLagrangian> nl{&M};
    nl.solve_incremental(5);

    std::cout << "    reason: " << static_cast<int>(nl.converged_reason()) << "\n";
    std::cout << "    Max |u| = " << max_disp(M) << "\n";
    export_vtk(M, prefix + "_case3_TL_SVK_large");
}

[[maybe_unused]] static void case4_TL_NH_large(const std::string& mesh,
                               const std::string& prefix)
{
    std::cout << "\n-- Case 4: TotalLagrangian + Neo-Hookean incr. (" << prefix << ") --\n";

    Domain<DIM> D;  GmshDomainBuilder b(mesh, D);

    auto nh = continuum::CompressibleNeoHookean<3>::from_E_nu(E_mod, nu);
    MaterialInstance<continuum::NeoHookeanRelation<3>> inst{nh};
    Material<ThreeDimensionalMaterial> mat{inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::TotalLagrangian, NDOF> M{D, mat};
    M.fix_x(0.0);
    M.setup();

    D.create_boundary_from_plane("Load", 0, 10.0);
    M.apply_surface_traction("Load", 0.0, 0.50, -0.50);

    NonlinearAnalysis<ThreeDimensionalMaterial, continuum::TotalLagrangian> nl{&M};
    nl.solve_incremental(5);

    std::cout << "    reason: " << static_cast<int>(nl.converged_reason()) << "\n";
    std::cout << "    Max |u| = " << max_disp(M) << "\n";
    export_vtk(M, prefix + "_case4_TL_NH_large");
}

[[maybe_unused]] static void case5_UL_SVK_small(const std::string& mesh,
                                const std::string& prefix)
{
    std::cout << "\n-- Case 5: UpdatedLagrangian + SVK small (" << prefix << ") --\n";

    Domain<DIM> D;  GmshDomainBuilder b(mesh, D);

    auto svk = continuum::SaintVenantKirchhoff<3>::from_E_nu(E_mod, nu);
    MaterialInstance<continuum::SVKRelation<3>> inst{svk};
    Material<ThreeDimensionalMaterial> mat{inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::UpdatedLagrangian, NDOF> M{D, mat};
    M.fix_x(0.0);
    M.setup();

    D.create_boundary_from_plane("Load", 0, 10.0);
    M.apply_surface_traction("Load", 0.0, 0.05, -0.05);

    NonlinearAnalysis<ThreeDimensionalMaterial, continuum::UpdatedLagrangian> nl{&M};
    nl.solve();

    std::cout << "    SNES iterations: " << nl.num_iterations()
              << "  reason: " << static_cast<int>(nl.converged_reason()) << "\n";
    std::cout << "    Max |u| = " << max_disp(M) << "\n";
    export_vtk(M, prefix + "_case5_UL_SVK_small");
}

[[maybe_unused]] static void case6_UL_NH_large(const std::string& mesh,
                               const std::string& prefix)
{
    std::cout << "\n-- Case 6: UpdatedLagrangian + Neo-Hookean incr. (" << prefix << ") --\n";

    Domain<DIM> D;  GmshDomainBuilder b(mesh, D);

    auto nh = continuum::CompressibleNeoHookean<3>::from_E_nu(E_mod, nu);
    MaterialInstance<continuum::NeoHookeanRelation<3>> inst{nh};
    Material<ThreeDimensionalMaterial> mat{inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::UpdatedLagrangian, NDOF> M{D, mat};
    M.fix_x(0.0);
    M.setup();

    D.create_boundary_from_plane("Load", 0, 10.0);
    M.apply_surface_traction("Load", 0.0, 0.50, -0.50);

    NonlinearAnalysis<ThreeDimensionalMaterial, continuum::UpdatedLagrangian> nl{&M};
    nl.solve_incremental(5);

    std::cout << "    reason: " << static_cast<int>(nl.converged_reason()) << "\n";
    std::cout << "    Max |u| = " << max_disp(M) << "\n";
    export_vtk(M, prefix + "_case6_UL_NH_large");
}

[[maybe_unused]] static void case7_SS_J2(const std::string& mesh,
                        const std::string& prefix)
{
    std::cout << "\n-- Case 7: SmallStrain + J2 Plasticity incr. (" << prefix << ") --\n";

    Domain<DIM> D;  GmshDomainBuilder b(mesh, D);

    J2PlasticMaterial3D j2_inst{E_mod, nu, sigma_y, H_hard};
    Material<ThreeDimensionalMaterial> mat{j2_inst, InelasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF> M{D, mat};
    M.fix_x(0.0);
    M.setup();

    D.create_boundary_from_plane("Load", 0, 10.0);
    M.apply_surface_traction("Load", 0.0, 1.0, -1.0);

    NonlinearAnalysis<ThreeDimensionalMaterial, continuum::SmallStrain> nl{&M};
    nl.solve_incremental(20);

    std::cout << "    reason: " << static_cast<int>(nl.converged_reason()) << "\n";
    std::cout << "    Max |u| = " << max_disp(M) << "\n";
    export_vtk(M, prefix + "_case7_SS_J2");
}


// ═════════════════════════════════════════════════════════════════════════
//  main
// ═════════════════════════════════════════════════════════════════════════

int main(int argc, char** args)
{
    PetscInitialize(&argc, &args, nullptr, nullptr);

    // SNES options (direct solver for moderate-size problems)
    PetscOptionsSetValue(nullptr, "-ksp_type",  "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type",   "lu");
    PetscOptionsSetValue(nullptr, "-snes_rtol", "1e-8");
    PetscOptionsSetValue(nullptr, "-snes_atol", "1e-10");
    PetscOptionsSetValue(nullptr, "-snes_max_it", "50");
    PetscOptionsSetValue(nullptr, "-snes_converged_reason", nullptr);
    PetscOptionsSetValue(nullptr, "-snes_linesearch_type", "basic");

    {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "============================================================\n"
                  << "  fall_n -- Nonlinear Showcase\n"
                  << "  Cases on HEX27 + TET10 with SPR projection\n"
                  << "============================================================\n";

        // ─────────────────────────────────────────────────────────────
        //  HEX27 element suite (Gauss-Legendre 3x3x3 integration)
        // ─────────────────────────────────────────────────────────────
        std::cout << "\n========================================================\n"
                  << "  HEX27 -- 27-node triquadratic hexahedral elements\n"
                  << "========================================================\n\n";

        case1_small_strain_elastic(MESH_HEX27, "hex27");
        // Nonlinear cases (uncomment as solver is tuned):
        // case2_TL_SVK_small       (MESH_HEX27, "hex27");
        // case3_TL_SVK_large       (MESH_HEX27, "hex27");
        // case4_TL_NH_large        (MESH_HEX27, "hex27");
        // case5_UL_SVK_small       (MESH_HEX27, "hex27");
        // case6_UL_NH_large        (MESH_HEX27, "hex27");
        // case7_SS_J2              (MESH_HEX27, "hex27");

        // ─────────────────────────────────────────────────────────────
        //  TET10 element suite (Hammer-Marlowe-Stroud 4-point, all
        //  positive weights -- avoids negative-weight checkerboard)
        // ─────────────────────────────────────────────────────────────
        std::cout << "\n========================================================\n"
                  << "  TET10 -- 10-node quadratic tetrahedral elements\n"
                  << "  Quadrature: 4-pt HMS (degree 2, all w_i > 0)\n"
                  << "========================================================\n\n";

        case1_small_strain_elastic(MESH_TET10, "tet10");
        // Nonlinear cases (uncomment as solver is tuned):
        // case2_TL_SVK_small       (MESH_TET10, "tet10");
        // case3_TL_SVK_large       (MESH_TET10, "tet10");
        // case4_TL_NH_large        (MESH_TET10, "tet10");
        // case5_UL_SVK_small       (MESH_TET10, "tet10");
        // case6_UL_NH_large        (MESH_TET10, "tet10");
        // case7_SS_J2              (MESH_TET10, "tet10");

        std::cout << "\n============================================================\n"
                  << "  Cases exported to " << OUT << "\n"
                  << "============================================================\n";

    } // PETSc scope
    PetscFinalize();
    return 0;
}
