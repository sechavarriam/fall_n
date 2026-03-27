// =============================================================================
//  main_rc_beam_validation.cpp
//
//  Validation of the Ko-Bathe concrete 3D model used in reinforced
//  continuum sub-models.  A doubly-clamped prismatic RC beam is loaded
//  by imposing an incremental rotation on one face (MaxZ), reproducing
//  the boundary-condition pattern that arises in the sub-model pipeline
//  when beam kinematics change between global analysis steps.
//
//  Two cases are solved and compared:
//    Case A — Plain concrete (hex8 only)
//    Case B — Reinforced concrete (hex8 + embedded truss rebar)
//
//  The imposed rotation θ about the local Y-axis (strong axis) linearly
//  ramps from 0 to θ_max.  At each converged increment, the reaction
//  moment at the MaxZ face is extracted from the assembled internal-force
//  vector, producing a moment–rotation curve that can be compared to
//  analytical or experimental references.
//
//  Output:
//    data/output/rc_beam_validation/moment_rotation_{plain,reinforced}.csv
//    data/output/rc_beam_validation/vtk/{plain,reinforced}_mesh.vtu
//    data/output/rc_beam_validation/vtk/{plain,reinforced}_gauss.vtu
//
//  Build:
//    cmake --build build --target fall_n_rc_beam_validation
//
// =============================================================================

#include "header_files.hh"

#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <petsc.h>

using namespace fall_n;


// =============================================================================
//  Configuration
// =============================================================================

struct BeamValidationConfig {
    // Geometry — typical RC column section
    double width   = 0.30;    // cross-section width  (X) [m]
    double height  = 0.40;    // cross-section height (Y) [m]
    double length  = 2.00;    // beam span            (Z) [m]

    // Mesh
    int nx = 4;               // elements in X
    int ny = 4;               // elements in Y
    int nz = 16;              // elements in Z (along beam axis)

    // Material — concrete
    double fc = 30.0;         // compressive strength f'c [MPa]

    // Material — rebar (Menegotto-Pinto)
    double E_s  = 200000.0;   // Young's modulus  [MPa]
    double fy   = 420.0;      // Yield stress     [MPa]
    double b_s  = 0.01;       // Hardening ratio
    double R0   = 20.0;
    double cR1  = 18.5;
    double cR2  = 0.15;

    // Rebar layout: 4 corner bars + 4 mid-face bars
    double bar_diameter = 0.020;  // 20 mm bars

    // Loading
    double theta_max = 0.005;     // max rotation [rad] ≈ 0.29°

    // Solver
    int num_steps     = 40;       // increments for p ∈ [0,1]
    int max_bisection = 8;
};


// =============================================================================
//  Rebar layout helper
// =============================================================================
//
//  Places rebar bars in the cross-section grid.  Corner bars sit at
//  grid nodes (1, 1), (nx-1, 1), (1, ny-1), (nx-1, ny-1) (one element
//  inset from the edge to approximate cover).  Mid-face bars sit at
//  the midpoint of each face.

static RebarSpec make_rebar_layout(const BeamValidationConfig& cfg)
{
    const double A_bar = M_PI / 4.0 * cfg.bar_diameter * cfg.bar_diameter;

    // Corner bars (inset by one element)
    RebarSpec rebar;
    rebar.bars = {
        {1,           1,           A_bar, "Rebar"},   // bottom-left
        {cfg.nx - 1,  1,           A_bar, "Rebar"},   // bottom-right
        {1,           cfg.ny - 1,  A_bar, "Rebar"},   // top-left
        {cfg.nx - 1,  cfg.ny - 1,  A_bar, "Rebar"},   // top-right
        // Mid-face bars
        {cfg.nx / 2,  1,           A_bar, "Rebar"},   // bottom-mid
        {cfg.nx / 2,  cfg.ny - 1,  A_bar, "Rebar"},   // top-mid
        {1,           cfg.ny / 2,  A_bar, "Rebar"},   // left-mid
        {cfg.nx - 1,  cfg.ny / 2,  A_bar, "Rebar"},   // right-mid
    };

    return rebar;
}


// =============================================================================
//  compute_face_rotation_bcs — rotation about Y-axis at a prism face
// =============================================================================
//
//  For a rotation θ about the local Y-axis at a face, each face node
//  gets a displacement pattern consistent with rigid-section kinematics:
//
//      u_x(x, y, z) = 0
//      u_y(x, y, z) = 0
//      u_z(x, y, z) = −θ · (x − x_centroid)
//
//  (Small-rotation approximation: the face translates along Z
//   proportionally to the distance from the section centroid in X.)
//
//  This is exactly the kind of BC that the sub-model pipeline imposes
//  when a beam section rotates around its neutral axis.

static std::vector<std::pair<std::size_t, Eigen::Vector3d>>
compute_face_rotation_bcs(
    const Domain<3>& domain,
    const PrismaticGrid& grid,
    PrismFace face,
    double theta_y,
    double x_centroid = 0.0)
{
    auto face_nodes = grid.nodes_on_face(face);
    std::vector<std::pair<std::size_t, Eigen::Vector3d>> bcs;
    bcs.reserve(face_nodes.size());

    for (auto nid : face_nodes) {
        const auto& node = domain.node(static_cast<std::size_t>(nid));
        double x = node.coord(0);

        // Small-rotation: u_z = -θ_y * (x - x_c)
        Eigen::Vector3d u{0.0, 0.0, -theta_y * (x - x_centroid)};
        bcs.emplace_back(static_cast<std::size_t>(nid), u);
    }

    return bcs;
}


// =============================================================================
//  extract_reaction_moment — moment about centroid from internal forces
// =============================================================================
//
//  Computes the resultant force and moment of the internal forces at the
//  specified face nodes, about the face centroid.  The moment component
//  M_y (about Y-axis) is the primary output for bending validation.

struct ReactionResult {
    Eigen::Vector3d force  = Eigen::Vector3d::Zero();
    Eigen::Vector3d moment = Eigen::Vector3d::Zero();
};

template <typename ModelT>
ReactionResult extract_face_reactions(
    const ModelT& model,
    const Domain<3>& domain,
    const std::vector<PetscInt>& face_node_ids)
{
    // Assemble internal forces at converged state
    Vec f_int;
    VecDuplicate(model.state_vector(), &f_int);
    VecSet(f_int, 0.0);

    // Need mutable ref for compute_internal_forces
    auto& mut_model = const_cast<ModelT&>(model);
    for (auto& elem : mut_model.elements())
        elem.compute_internal_forces(model.state_vector(), f_int);
    VecAssemblyBegin(f_int);
    VecAssemblyEnd(f_int);

    // Compute centroid of the face for moment arm
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (auto nid : face_node_ids) {
        const auto& node = domain.node(static_cast<std::size_t>(nid));
        centroid += Eigen::Vector3d{node.coord(0), node.coord(1), node.coord(2)};
    }
    centroid /= static_cast<double>(face_node_ids.size());

    // Sum forces and compute moment about centroid
    ReactionResult result;
    for (auto nid : face_node_ids) {
        const auto& node = domain.node(static_cast<std::size_t>(nid));
        auto dofs = node.dof_index();
        PetscScalar vals[3];
        PetscInt idx[3] = {
            static_cast<PetscInt>(dofs[0]),
            static_cast<PetscInt>(dofs[1]),
            static_cast<PetscInt>(dofs[2])
        };
        VecGetValues(f_int, 3, idx, vals);

        Eigen::Vector3d f{vals[0], vals[1], vals[2]};
        result.force += f;

        Eigen::Vector3d r{
            node.coord(0) - centroid[0],
            node.coord(1) - centroid[1],
            node.coord(2) - centroid[2]
        };
        result.moment += r.cross(f);
    }

    VecDestroy(&f_int);
    return result;
}


// =============================================================================
//  run_validation — solve a single case (plain or reinforced)
// =============================================================================

struct StepRecord {
    int    step;
    double p;
    double theta;         // rotation [rad]
    double M_y;           // reaction moment about Y [MN·m]
    double max_disp;      // max |u| [m]
    double F_z;           // axial reaction force [MN]
};

template <typename ElemPolicyT, typename ModelT>
std::vector<StepRecord> run_case(
    ModelT& model,
    Domain<3>& domain,
    const PrismaticGrid& grid,
    const BeamValidationConfig& cfg,
    const std::string& vtk_dir,
    const std::string& label)
{
    // ── PETSc solver options ────────────────────────────────────────
    PetscOptionsSetValue(nullptr, "-snes_linesearch_type", "basic");
    PetscOptionsSetValue(nullptr, "-snes_max_it", "100");
    PetscOptionsSetValue(nullptr, "-ksp_type",  "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type",   "lu");

    NonlinearAnalysis<ThreeDimensionalMaterial, continuum::SmallStrain, 3,
                      ElemPolicyT> nl{&model};

    // Capture full imposed-solution snapshot
    Vec u_full;
    VecDuplicate(model.imposed_solution(), &u_full);
    VecCopy(model.imposed_solution(), u_full);

    // Records collected at each converged step
    std::vector<StepRecord> records;
    records.reserve(static_cast<std::size_t>(cfg.num_steps + 1));

    // Initial record (unloaded)
    records.push_back({0, 0.0, 0.0, 0.0, 0.0, 0.0});

    // MaxZ face nodes for reaction extraction
    auto face_max = grid.nodes_on_face(PrismFace::MaxZ);

    // Step callback: extract reactions at each converged increment
    nl.set_step_callback(
        [&](int step, double p, const ModelT& m) {
            auto rxn = extract_face_reactions(m, domain, face_max);
            double theta = p * cfg.theta_max;

            // Max displacement
            const PetscScalar* arr;
            PetscInt n;
            VecGetLocalSize(m.state_vector(), &n);
            VecGetArrayRead(m.state_vector(), &arr);
            double max_disp = 0.0;
            for (PetscInt i = 0; i < n; ++i)
                max_disp = std::max(max_disp, std::abs(arr[i]));
            VecRestoreArrayRead(m.state_vector(), &arr);

            records.push_back({step, p, theta, rxn.moment[1],
                               max_disp, rxn.force[2]});

            PetscPrintf(PETSC_COMM_WORLD,
                "  [%s] step=%d  p=%.4f  θ=%.6f rad  "
                "My=%.6e MN·m  max|u|=%.4e m\n",
                label.c_str(), step, p, theta,
                rxn.moment[1], max_disp);
        });

    // Displacement control: scale imposed BCs by p
    auto scheme = make_control(
        [&u_full](double p, Vec /*f_full*/, Vec f_ext, auto* m) {
            VecSet(f_ext, 0.0);                         // no body forces
            VecCopy(u_full, m->imposed_solution());      // restore full BCs
            VecScale(m->imposed_solution(), p);           // scale to p·u_bc
        });

    PetscPrintf(PETSC_COMM_WORLD,
        "\n══════════════════════════════════════════════════\n"
        "  RC beam validation: %s\n"
        "  θ_max = %.4f rad  (%d increments, bisection=%d)\n"
        "══════════════════════════════════════════════════\n\n",
        label.c_str(), cfg.theta_max, cfg.num_steps, cfg.max_bisection);

    bool ok = nl.solve_incremental(cfg.num_steps, cfg.max_bisection, scheme);

    VecDestroy(&u_full);

    PetscPrintf(PETSC_COMM_WORLD,
        "\n  %s: %s  (%d converged records)\n",
        label.c_str(),
        ok ? "COMPLETED" : "ABORTED (some steps failed)",
        static_cast<int>(records.size()));

    // ── VTK export at final state ──────────────────────────────────
    if (!vtk_dir.empty()) {
        std::filesystem::create_directories(vtk_dir);
        fall_n::vtk::VTKModelExporter exporter{model};
        exporter.set_displacement();
        exporter.compute_material_fields();
        exporter.write_mesh(vtk_dir + "/" + label + "_mesh.vtu");
        exporter.write_gauss_points(vtk_dir + "/" + label + "_gauss.vtu");
        PetscPrintf(PETSC_COMM_WORLD,
            "  VTK written to %s/\n", vtk_dir.c_str());
    }

    return records;
}


// =============================================================================
//  write_csv — export moment-rotation records
// =============================================================================

static void write_csv(const std::string& path,
                      const std::vector<StepRecord>& records)
{
    std::ofstream ofs(path);
    ofs << "step,p,theta_rad,My_MNm,max_disp_m,Fz_MN\n";
    ofs << std::scientific << std::setprecision(8);
    for (const auto& r : records) {
        ofs << r.step << ","
            << r.p << ","
            << r.theta << ","
            << r.M_y << ","
            << r.max_disp << ","
            << r.F_z << "\n";
    }
    PetscPrintf(PETSC_COMM_WORLD,
        "  CSV written: %s  (%d records)\n",
        path.c_str(), static_cast<int>(records.size()));
}


// =============================================================================
//  main
// =============================================================================

int main(int argc, char** argv)
{
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    PetscOptionsSetValue(nullptr, "-snes_monitor_cancel", "");
    PetscOptionsSetValue(nullptr, "-ksp_monitor_cancel",  "");

    int exit_code = 0;
    try {
        exit_code = [&]() -> int {

    // ── Configuration ─────────────────────────────────────────────────
    BeamValidationConfig cfg;

    // Output directory
    std::string base_dir = std::string(FALL_N_SOURCE_DIR)
                         + "/data/output/rc_beam_validation";
    std::string vtk_dir  = base_dir + "/vtk";
    std::filesystem::create_directories(vtk_dir);

    PetscPrintf(PETSC_COMM_WORLD,
        "\n╔══════════════════════════════════════════════════════╗\n"
          "║   RC Beam 3D Validation — Ko-Bathe Concrete Model   ║\n"
          "╠══════════════════════════════════════════════════════╣\n"
          "║  Geometry: %.2f × %.2f × %.2f m                    ║\n"
          "║  Mesh:     %d × %d × %d hex8 elements              ║\n"
          "║  f'c:      %.0f MPa                                 ║\n"
          "║  θ_max:    %.4f rad                                ║\n"
          "╚══════════════════════════════════════════════════════╝\n\n",
        cfg.width, cfg.height, cfg.length,
        cfg.nx, cfg.ny, cfg.nz,
        cfg.fc,
        cfg.theta_max);


    // =================================================================
    //  Case A: Plain concrete (hex8 only)
    // =================================================================
    {
        PetscPrintf(PETSC_COMM_WORLD,
            "\n────────────────────────────────────────────────────\n"
              "  Case A: Plain concrete\n"
              "────────────────────────────────────────────────────\n");

        PrismaticSpec spec{
            .width  = cfg.width,
            .height = cfg.height,
            .length = cfg.length,
            .nx = cfg.nx, .ny = cfg.ny, .nz = cfg.nz,
        };

        auto [domain, grid] = make_prismatic_domain(spec);

        // Material
        InelasticMaterial<KoBatheConcrete3D> mat_inst{cfg.fc};
        Material<ThreeDimensionalMaterial> mat{mat_inst, InelasticUpdate{}};

        // Model
        using Policy = ThreeDimensionalMaterial;
        Model<Policy, continuum::SmallStrain, 3> model{domain, mat};

        // BCs: fix MinZ face, impose rotation on MaxZ face
        auto face_min = grid.nodes_on_face(PrismFace::MinZ);
        for (auto nid : face_min)
            model.constrain_node(static_cast<std::size_t>(nid),
                                 {0.0, 0.0, 0.0});

        auto rotation_bcs = compute_face_rotation_bcs(
            domain, grid, PrismFace::MaxZ, cfg.theta_max);
        for (const auto& [nid, u] : rotation_bcs)
            model.constrain_node(nid, {u[0], u[1], u[2]});

        model.setup();

        auto records = run_case<SingleElementPolicy<
            ContinuumElement<Policy, 3, continuum::SmallStrain>>>(
                model, domain, grid, cfg, vtk_dir, "plain");

        write_csv(base_dir + "/moment_rotation_plain.csv", records);
    }

    // =================================================================
    //  Case B: Reinforced concrete (hex8 + truss rebar)
    // =================================================================
    {
        PetscPrintf(PETSC_COMM_WORLD,
            "\n────────────────────────────────────────────────────\n"
              "  Case B: Reinforced concrete\n"
              "────────────────────────────────────────────────────\n");

        PrismaticSpec spec{
            .width  = cfg.width,
            .height = cfg.height,
            .length = cfg.length,
            .nx = cfg.nx, .ny = cfg.ny, .nz = cfg.nz,
        };

        auto rebar = make_rebar_layout(cfg);

        auto rd = make_reinforced_prismatic_domain(spec, rebar);
        auto& domain     = rd.domain;
        auto& grid       = rd.grid;
        auto& rebar_range = rd.rebar_range;

        // ── Build heterogeneous element vector ──────────────────────
        InelasticMaterial<KoBatheConcrete3D> concrete_inst{cfg.fc};
        Material<ThreeDimensionalMaterial> concrete_mat{
            concrete_inst, InelasticUpdate{}};

        RebarSteelConfig steel_cfg{cfg.E_s, cfg.fy, cfg.b_s,
                                   cfg.R0, cfg.cR1, cfg.cR2};
        InelasticMaterial<MenegottoPintoSteel> steel_inst{
            steel_cfg.E_s, steel_cfg.fy, steel_cfg.b,
            steel_cfg.R0, steel_cfg.cR1, steel_cfg.cR2};
        Material<UniaxialMaterial> rebar_mat{steel_inst, InelasticUpdate{}};

        using HexElem = ContinuumElement<ThreeDimensionalMaterial, 3,
                                         continuum::SmallStrain>;

        std::vector<FEM_Element> elements;
        elements.reserve(domain.num_elements());

        // Hex8 concrete elements
        for (std::size_t i = 0; i < rebar_range.first; ++i) {
            elements.emplace_back(
                HexElem{&domain.element(i), concrete_mat});
        }

        // Truss rebar elements
        const auto nz_sz = static_cast<std::size_t>(cfg.nz);
        const double A_bar = M_PI / 4.0 * cfg.bar_diameter * cfg.bar_diameter;
        for (std::size_t i = rebar_range.first; i < rebar_range.last; ++i) {
            std::size_t bar_idx = (i - rebar_range.first) / nz_sz;
            double area = rebar.bars[bar_idx].area;
            elements.emplace_back(
                TrussElement<3>{&domain.element(i), rebar_mat, area});
        }

        // ── Build model with MultiElementPolicy ────────────────────
        Model<ThreeDimensionalMaterial, continuum::SmallStrain, 3,
              MultiElementPolicy> model{domain, std::move(elements)};

        // BCs: fix MinZ face, impose rotation on MaxZ face
        auto face_min = grid.nodes_on_face(PrismFace::MinZ);
        for (auto nid : face_min)
            model.constrain_node(static_cast<std::size_t>(nid),
                                 {0.0, 0.0, 0.0});

        auto rotation_bcs = compute_face_rotation_bcs(
            domain, grid, PrismFace::MaxZ, cfg.theta_max);
        for (const auto& [nid, u] : rotation_bcs)
            model.constrain_node(nid, {u[0], u[1], u[2]});

        model.setup();

        PetscPrintf(PETSC_COMM_WORLD,
            "  Elements: %d hex8 + %d truss = %d total\n"
            "  Rebar: %d bars x %d z-segments = %d line elements\n"
            "  A_bar = %.4e m2  (d=%.0f mm)\n\n",
            (int)rebar_range.first,
            (int)(rebar_range.last - rebar_range.first),
            (int)domain.num_elements(),
            (int)rebar.bars.size(),
            cfg.nz,
            (int)(rebar_range.last - rebar_range.first),
            A_bar,
            cfg.bar_diameter * 1000.0);

        auto records = run_case<MultiElementPolicy>(
            model, domain, grid, cfg, vtk_dir, "reinforced");

        write_csv(base_dir + "/moment_rotation_reinforced.csv", records);
    }

    // =================================================================
    //  Summary
    // =================================================================

    PetscPrintf(PETSC_COMM_WORLD,
        "\n══════════════════════════════════════════════════\n"
          "  Validation complete.  Output in:\n"
          "    %s/\n"
          "══════════════════════════════════════════════════\n\n",
        base_dir.c_str());

    return 0;

        }();
    } catch (const std::exception& e) {
        std::cerr << "\n*** EXCEPTION: " << e.what() << " ***\n";
        PetscFinalize();
        return 1;
    } catch (...) {
        std::cerr << "\n*** UNKNOWN EXCEPTION ***\n";
        PetscFinalize();
        return 1;
    }

    PetscFinalize();
    return exit_code;
}
