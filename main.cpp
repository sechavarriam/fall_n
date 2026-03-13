#include "header_files.hh"

#include "src/geometry/Point.hh"
#include "src/elements/TimoshenkoBeamN.hh"
#include "src/elements/BeamElement.hh"

#include <Eigen/Dense>

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

// =============================================================================
//  Linear Elastic Cantilever Beam — 4-mesh + Timoshenko reference
// =============================================================================
//
//  Domain:   [0, 10] × [0, 0.4] × [0, 0.8]
//
//  Meshes (3D continuum):
//    1. HEX8   — Beam_LagCell_Ord1_30x6x3   (trilinear, 2×2×2 Gauss)
//    2. HEX27  — Beam_LagCell_Ord2_30x6x3   (triquadratic, 3×3×3 Gauss)
//    3. TET4   — Beam_LagSimplex_Ord1_Fine   (linear, 4-pt simplex rule)
//    4. TET10  — Beam_LagSimplex_Ord2_Fine     (quadratic, StroudConicalProduct<3,3>)
//
//  Reference:
//    5. Timoshenko beam N=3 (quadratic, 5 elements) — "exact" 1D solution
//
//  BCs:
//    Fixed end  x = 0   (all DOFs clamped)
//    Load  face x = 10  (uniform surface traction: ty = 0.05, tz = -0.05)
//
//  Post-processing:
//    L2 lumped projection (Gauss-point → nodal).  All quadrature rules
//    in this comparison have strictly positive weights (Gauss-Legendre for
//    hex cells, HMS-4 or StroudConical for tets), so L2 is well-conditioned.
//
// =============================================================================

static constexpr std::size_t DIM  = 3;
static constexpr std::size_t NDOF = DIM;

// ── Material ─────────────────────────────────────────────────────────────────
static constexpr double E_mod = 200.0;
static constexpr double nu    = 0.3;
static constexpr double G_mod = E_mod / (2.0 * (1.0 + nu));  // ≈ 76.923

// ── Traction on the tip face ──────────────────────────────────────────────────
static constexpr double TY =  0.05;
static constexpr double TZ = -0.05;

// ── Geometry ──────────────────────────────────────────────────────────────────
static constexpr double L_BEAM  = 10.0;  // span   (x)
static constexpr double B_BEAM  = 0.40;  // width  (y)
static constexpr double H_BEAM  = 0.80;  // height (z)

static constexpr double X_FIXED = 0.0;
static constexpr double X_TIP   = L_BEAM;

// ── Section properties (rectangular) ──────────────────────────────────────────
static constexpr double A_sec   = B_BEAM * H_BEAM;                // 0.32
static constexpr double Iy_sec  = B_BEAM * H_BEAM*H_BEAM*H_BEAM / 12.0; // 0.017067
static constexpr double Iz_sec  = H_BEAM * B_BEAM*B_BEAM*B_BEAM / 12.0; // 0.004267
static constexpr double kappa   = 5.0 / 6.0;
static constexpr double b_min   = B_BEAM;
static constexpr double h_max   = H_BEAM;
static constexpr double J_tor   = (b_min*b_min*b_min * h_max / 3.0)
                                  * (1.0 - 0.63 * b_min / h_max);

// ── Resultant forces from the surface traction ────────────────────────────────
static constexpr double Fy_tip = TY * A_sec;   // 0.016
static constexpr double Fz_tip = TZ * A_sec;   // -0.016

// ── Structural shell example geometry ───────────────────────────────────────
static constexpr double L_SHELL = 4.0;
static constexpr double W_SHELL = 1.0;
static constexpr double T_SHELL = 0.08;
static constexpr int    NX_SHELL = 8;
static constexpr int    NY_SHELL = 2;
static constexpr double FZ_SHELL_TOTAL = -2.0e-4;

// ── Analytical Timoshenko deflections (tip point load) ────────────────────────
//  δ_y (from Fy→ bending about z → I_z):  PL³/(3EI_z) + PL/(κGA)
//  δ_z (from Fz→ bending about y → I_y):  PL³/(3EI_y) + PL/(κGA)
static constexpr double delta_y_analytical() {
    double P = std::abs(Fy_tip);
    return P * L_BEAM*L_BEAM*L_BEAM / (3.0 * E_mod * Iz_sec)
         + P * L_BEAM / (kappa * G_mod * A_sec);
}
static constexpr double delta_z_analytical() {
    double P = std::abs(Fz_tip);
    return P * L_BEAM*L_BEAM*L_BEAM / (3.0 * E_mod * Iy_sec)
         + P * L_BEAM / (kappa * G_mod * A_sec);
}

// ── Paths ─────────────────────────────────────────────────────────────────────
static const std::string BASE = "/home/sechavarriam/MyLibs/fall_n/";
static const std::string IN   = BASE + "data/input/";
static const std::string OUT  = BASE + "data/output/";

// =============================================================================
//  Result record
// =============================================================================
struct Result {
    std::string label;
    std::size_t n_nodes;
    std::size_t n_elems;
    double      max_disp;   // max |u| overall
    double      max_uy;     // max |u_y| (weak axis deflection)
    double      max_uz;     // max |u_z| (strong axis deflection)
};

// =============================================================================
//  Helpers — continuum models (3 DOFs per node)
// =============================================================================

static double max_abs_disp(auto& M) {
    const PetscScalar* arr; PetscInt n;
    VecGetLocalSize(M.state_vector(), &n);
    VecGetArrayRead(M.state_vector(), &arr);
    double mx = 0.0;
    for (PetscInt i = 0; i < n; ++i) mx = std::max(mx, std::abs(arr[i]));
    VecRestoreArrayRead(M.state_vector(), &arr);
    return mx;
}

static double max_uy_disp(auto& M, std::size_t ndof = 3) {
    const PetscScalar* arr; PetscInt n;
    VecGetLocalSize(M.state_vector(), &n);
    VecGetArrayRead(M.state_vector(), &arr);
    double mx = 0.0;
    for (PetscInt i = 1; i < n; i += static_cast<PetscInt>(ndof))
        mx = std::max(mx, std::abs(arr[i]));
    VecRestoreArrayRead(M.state_vector(), &arr);
    return mx;
}

static double max_uz_disp(auto& M, std::size_t ndof = 3) {
    const PetscScalar* arr; PetscInt n;
    VecGetLocalSize(M.state_vector(), &n);
    VecGetArrayRead(M.state_vector(), &arr);
    double mx = 0.0;
    for (PetscInt i = 2; i < n; i += static_cast<PetscInt>(ndof))
        mx = std::max(mx, std::abs(arr[i]));
    VecRestoreArrayRead(M.state_vector(), &arr);
    return mx;
}

// Export two ParaView-oriented VTUs with adaptive nodal projection.
// `*_mesh.vtu` carries the continuum mesh and nodal fields only.
// `*_gauss.vtu` carries the material-point cloud and the same displacement
// field interpolated to the Gauss points so Warp By Vector can be applied
// directly to the quadrature cloud. Lumped L2 is used only when the local
// nodal lumping remains strictly positive; otherwise it falls back to
// polynomial patch recovery.
template <typename ModelT>
static void export_vtk(ModelT& M, const std::string& tag) {
    M.update_elements_state();
    fall_n::vtk::VTKModelExporter exporter(M);
    exporter.set_displacement();
    exporter.compute_material_fields();        // <── adaptive: L2 when safe, patch recovery otherwise
    exporter.write_mesh(OUT + tag + "_mesh.vtu");
    exporter.write_gauss_points(OUT + tag + "_gauss.vtu");
    std::cout << "       VTU : " << tag
              << "_mesh.vtu, " << tag
              << "_gauss.vtu  [split mesh + gauss cloud]\n";
}

template <typename ModelT, typename BeamProfileT, typename ThicknessProfileT>
static void export_structural_vtm(
    ModelT& M,
    const std::string& filename,
    BeamProfileT beam_profile,
    ThicknessProfileT thickness_profile)
{
    fall_n::vtk::StructuralVTMExporter exporter(
        M,
        std::move(beam_profile),
        std::move(thickness_profile));
    exporter.write(filename);
    std::cout << "       VTM : " << filename << "\n";
}

// =============================================================================
//  3D continuum case runner
// =============================================================================
static Result run_continuum(const std::string& mesh_file,
                            const std::string& label,
                            const std::string& prefix)
{
    std::cout << "\n  >> " << label << "\n";

    Domain<DIM> D;
    GmshDomainBuilder b(mesh_file, D);

    ContinuumIsotropicElasticMaterial mat_inst{E_mod, nu};
    Material<ThreeDimensionalMaterial> mat{mat_inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial, continuum::SmallStrain, NDOF> M{D, mat};
    M.fix_x(X_FIXED);
    M.setup();

    D.create_boundary_from_plane("Load", 0, X_TIP);
    M.apply_surface_traction("Load", 0.0, TY, TZ);

    LinearAnalysis<ThreeDimensionalMaterial> solver{&M};
    solver.solve();

    const double mx = max_abs_disp(M);
    const double uy = max_uy_disp(M);
    const double uz = max_uz_disp(M);
    const auto   nn = D.num_nodes();
    const auto   ne = D.num_elements();

    std::cout << "       Nodes: " << nn << " | Elements: " << ne << "\n"
              << "       Max |u| = " << mx
              << "   |uy| = " << uy
              << "   |uz| = " << uz << "\n";

    export_vtk(M, prefix);
    return { label, nn, ne, mx, uy, uz };
}

// =============================================================================
//  Timoshenko beam reference (N=3 quadratic, 5 elements along span)
// =============================================================================
static constexpr std::size_t NDOF_BEAM = 6;

using BeamN3Elem   = TimoshenkoBeamN<3>;
using BeamN3Policy = SingleElementPolicy<BeamN3Elem>;
using BeamN3Model  = Model<TimoshenkoBeam3D, continuum::SmallStrain,
                           NDOF_BEAM, BeamN3Policy>;

static Result run_timoshenko_reference()
{
    static constexpr int NEL  = 5;
    static constexpr int NNOD = 2 * NEL + 1;  // 11 nodes for 5 quadratic elements

    std::cout << "\n  >> Timoshenko Beam N=3 (5 quadratic elements)\n";

    Domain<DIM> D;
    D.preallocate_node_capacity(NNOD);

    double dx = L_BEAM / (2 * NEL);   // 1.0 m between consecutive nodes
    for (int i = 0; i < NNOD; ++i)
        D.add_node(i, i * dx, 0.0, 0.0);

    for (int e = 0; e < NEL; ++e) {
        std::array<int, 3> conn = {2*e, 2*e+1, 2*e+2};
        D.template make_element<LagrangeElement3D<3>>(
            GaussLegendreCellIntegrator<2>{}, e, conn.data());
    }
    D.assemble_sieve();

    TimoshenkoBeamMaterial3D mat_inst{E_mod, G_mod, A_sec, Iy_sec, Iz_sec,
                                      J_tor, kappa, kappa};
    Material<TimoshenkoBeam3D> mat{mat_inst, ElasticUpdate{}};

    BeamN3Model M{D, mat};
    M.fix_x(0.0);
    M.setup();

    // Apply the resultant tip forces (equivalent to the surface traction)
    M.apply_node_force(NNOD - 1,
                       0.0, Fy_tip, Fz_tip,    // fx, fy, fz
                       0.0, 0.0,    0.0);       // mx, my, mz

    LinearAnalysis<TimoshenkoBeam3D, continuum::SmallStrain,
                   NDOF_BEAM, BeamN3Policy> solver{&M};
    solver.solve();

    const double uy = max_uy_disp(M, NDOF_BEAM);
    const double uz = max_uz_disp(M, NDOF_BEAM);
    const double mx = std::sqrt(uy*uy + uz*uz);   // resultant tip deflection

    std::cout << "       Nodes: " << NNOD << " | Elements: " << NEL << "\n"
              << "       Max |u| = " << mx
              << "   |uy| = " << uy
              << "   |uz| = " << uz << "\n"
              << "       Analytical δ_y = " << delta_y_analytical()
              << "   δ_z = " << delta_z_analytical() << "\n";

    export_structural_vtm(
        M,
        OUT + "timoshenko_ref_structural.vtm",
        fall_n::reconstruction::RectangularSectionProfile<2>{B_BEAM, H_BEAM},
        fall_n::reconstruction::ShellThicknessProfile<3>{});

    return { "Timoshenko N=3  (ref)", static_cast<std::size_t>(NNOD),
             static_cast<std::size_t>(NEL), mx, uy, uz };
}

// =============================================================================
//  MITC4 shell cantilever example (structural VTM export)
// =============================================================================
static void run_shell_reference()
{
    using ShellElemPolicy =
        SingleElementPolicy<ShellElement<MindlinReissnerShell3D>>;
    using ShellModel =
        Model<MindlinReissnerShell3D, continuum::SmallStrain, 6, ShellElemPolicy>;

    std::cout << "\n  >> MITC4 Shell Cantilever (structural VTM export)\n";

    Domain<DIM> D;
    D.preallocate_node_capacity((NX_SHELL + 1) * (NY_SHELL + 1));

    const double dx = L_SHELL / static_cast<double>(NX_SHELL);
    const double dy = W_SHELL / static_cast<double>(NY_SHELL);

    auto node_id = [&](int i, int j) {
        return static_cast<PetscInt>(j * (NX_SHELL + 1) + i);
    };

    for (int j = 0; j <= NY_SHELL; ++j) {
        const double y = -0.5 * W_SHELL + static_cast<double>(j) * dy;
        for (int i = 0; i <= NX_SHELL; ++i) {
            D.add_node(node_id(i, j), static_cast<double>(i) * dx, y, 0.0);
        }
    }

    std::size_t elem_id = 0;
    for (int j = 0; j < NY_SHELL; ++j) {
        for (int i = 0; i < NX_SHELL; ++i) {
            PetscInt conn[4] = {
                node_id(i,     j),
                node_id(i + 1, j),
                node_id(i,     j + 1),
                node_id(i + 1, j + 1)
            };
            D.template make_element<LagrangeElement3D<2, 2>>(
                GaussLegendreCellIntegrator<2, 2>{}, elem_id++, conn);
        }
    }
    D.assemble_sieve();

    MindlinShellMaterial mat_inst{E_mod, nu, T_SHELL};
    Material<MindlinReissnerShell3D> mat{mat_inst, ElasticUpdate{}};

    ShellModel M{D, mat};
    M.fix_x(0.0);
    M.setup();

    std::vector<std::size_t> tip_nodes;
    tip_nodes.reserve(NY_SHELL + 1);
    for (const auto& node : D.nodes()) {
        if (std::abs(node.coord_ref()[0] - L_SHELL) < 1.0e-12) {
            tip_nodes.push_back(node.id());
        }
    }

    const double nodal_load = FZ_SHELL_TOTAL / static_cast<double>(tip_nodes.size());
    for (const auto id : tip_nodes) {
        M.apply_node_force(id, 0.0, 0.0, nodal_load, 0.0, 0.0, 0.0);
    }

    LinearAnalysis<MindlinReissnerShell3D, continuum::SmallStrain,
                   6, ShellElemPolicy> solver{&M};
    solver.solve();

    const double mx = max_abs_disp(M);
    const double uz = max_uz_disp(M, 6);

    std::cout << "       Nodes: " << D.num_nodes()
              << " | Elements: " << D.num_elements() << "\n"
              << "       Max |u| = " << mx
              << "   |uz| = " << uz << "\n";

    export_structural_vtm(
        M,
        OUT + "shell_cantilever_structural.vtm",
        fall_n::reconstruction::RectangularSectionProfile<1>{1.0, 1.0},
        fall_n::reconstruction::ShellThicknessProfile<5>{});
}

// =============================================================================
//  Comparison table
// =============================================================================
static void print_comparison(const std::vector<Result>& R) {
    const int w1 = 40, w2 = 8, w3 = 8, w4 = 12, w5 = 12, w6 = 12, w7 = 8;
    const auto sep = std::string(w1 + w2 + w3 + w4 + w5 + w6 + w7, '-');

    std::cout << "\n\n"
              << "================================================================\n"
              << "  RESULTS COMPARISON — Linear Elastic Cantilever Beam\n"
              << "  E = " << E_mod << "  nu = " << nu
              << "  ty = " << TY << "  tz = " << TZ << "\n"
              << "  Domain: [0," << L_BEAM << "] x [0," << B_BEAM
              << "] x [0," << H_BEAM << "]\n"
              << "================================================================\n\n";

    std::cout << std::left  << std::setw(w1) << "  Model"
              << std::right << std::setw(w2) << "Nodes"
                            << std::setw(w3) << "Elems"
                            << std::setw(w4) << "|u|_max"
                            << std::setw(w5) << "|uy|_max"
                            << std::setw(w6) << "|uz|_max"
                            << std::setw(w7) << "  err%"
              << "\n" << sep << "\n";

    // Reference: first entry = Timoshenko beam
    const double u_ref = R.front().max_disp;

    for (const auto& r : R) {
        double err = (u_ref > 0.0)
                     ? std::abs(r.max_disp - u_ref) / u_ref * 100.0
                     : 0.0;
        std::cout << std::left  << std::setw(w1) << ("  " + r.label)
                  << std::right << std::setw(w2) << r.n_nodes
                                << std::setw(w3) << r.n_elems
                  << std::fixed << std::setprecision(6)
                                << std::setw(w4) << r.max_disp
                                << std::setw(w5) << r.max_uy
                                << std::setw(w6) << r.max_uz
                  << std::fixed << std::setprecision(2)
                                << std::setw(w7) << err
                  << "\n";
    }

    std::cout << sep << "\n"
              << "  Reference = Timoshenko beam:  |u| = " << u_ref << "\n"
              << "  Analytical:  δ_y = " << std::fixed << std::setprecision(6)
              << delta_y_analytical() << "   δ_z = " << delta_z_analytical()
              << "\n"
              << "  VTU outputs:  " << OUT << "\n"
              << "================================================================\n";
}

// =============================================================================
//  main
// =============================================================================
int main(int argc, char** args)
{
    PetscInitialize(&argc, &args, nullptr, nullptr);

    PetscOptionsSetValue(nullptr, "-ksp_type", "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type",  "lu");

    {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "================================================================\n"
                  << "  fall_n — Cantilever Beam Study (linear elastic)\n"
                  << "  4 continuum meshes + Timoshenko beam reference\n"
                  << "================================================================\n";

        std::vector<Result> R;

        // ── Reference: Timoshenko beam (should be first for comparison) ──
        R.push_back(run_timoshenko_reference());

        // ── 3D continuum meshes ──────────────────────────────────────────
        R.push_back(run_continuum(
            IN + "Beam_LagCell_Ord1_30x6x3.msh",
            "HEX8  (Ord.1, 30x6x3, GL 2x2x2)",
            "hex8_beam"));

        R.push_back(run_continuum(
            IN + "Beam_LagCell_Ord2_30x6x3.msh",
            "HEX27 (Ord.2, 30x6x3, GL 3x3x3)",
            "hex27_beam"));

        R.push_back(run_continuum(
            IN + "Beam_LagSimplex_Ord1_Fine.msh",
            "TET4  (Ord.1, Fine, 4-pt simplex)",
            "tet4_beam"));

        R.push_back(run_continuum(
            IN + "Beam_LagSimplex_Ord2_Fine.msh",
            "TET10 (Ord.2, Fine, StroudCP 27-pt)",
            "tet10_beam"));

        print_comparison(R);
        run_shell_reference();
    }

    PetscFinalize();
    return 0;
}
