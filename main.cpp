#include "header_files.hh"

#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <filesystem>
#include <numbers>
#include <print>
#include <string>
#include <vector>

namespace {

// =============================================================================
//  Corotational dynamic 5-story RC building example
// =============================================================================
//
//  Units:
//    Force   : MN
//    Length  : m
//    Stress  : MPa = MN/m²
//    Mass    : MN·s²/m
//    Density : MN·s²/m⁴   (2400 kg/m³ = 2.4e-3)
//
//  Structural idealisation:
//    - Columns and beams: 3D Timoshenko frame elements, **corotational**
//    - Frame sections    : nonlinear fiber sections (Kent-Park + Menegotto-Pinto)
//    - Slabs             : MITC4 / Mindlin-Reissner shells (elastic)
//
//  Dynamic analysis:
//    - PETSc TS (Generalized-α, TSALPHA2)
//    - Consistent mass matrix from beam density + shell density
//    - Rayleigh damping  C = αₘ·M + βₖ·K₀,  ξ = 5 %
//    - Cyclic lateral loading at each story level
//    - VTK time-series output via PVDWriter + StructuralVTMExporter
//
//  Geometric nonlinearity:
//    The beam::Corotational kinematic policy removes the rigid-body
//    rotation from each element, allowing large displacements and
//    rotations while preserving the small-strain B-matrix locally.
//
// =============================================================================

static constexpr std::size_t NDOF = 6;

// ── Output path ───────────────────────────────────────────────────────────────
static const std::string BASE = "/home/sechavarriam/MyLibs/fall_n/";
static const std::string OUT  = BASE + "data/output/dynamic_rc_building/";

// ── Grid definition ───────────────────────────────────────────────────────────
static constexpr int NUM_AXES_X = 4;
static constexpr int NUM_AXES_Y = 3;
static constexpr int NUM_STORIES = 5;

static constexpr std::array<double, NUM_AXES_X> X_GRID = {0.0, 6.0, 12.0, 18.0};
static constexpr std::array<double, NUM_AXES_Y> Y_GRID = {0.0, 5.0, 10.0};
static constexpr double STORY_HEIGHT = 3.20;

// ── Member sizes ──────────────────────────────────────────────────────────────
static constexpr double COLUMN_B = 0.45;
static constexpr double COLUMN_H = 0.45;
static constexpr double COLUMN_COVER = 0.05;
static constexpr double COLUMN_BAR_D = 0.025;
static constexpr double COLUMN_TIE_SPACING = 0.10;

static constexpr double BEAM_B = 0.30;
static constexpr double BEAM_H = 0.60;
static constexpr double BEAM_COVER = 0.05;
static constexpr double BEAM_BAR_D = 0.020;

static constexpr double SLAB_T = 0.15;

// ── Material data (RC frame + elastic slab) ──────────────────────────────────
static constexpr double NU_RC = 0.20;

static constexpr double COLUMN_FPC = 35.0;      // MPa
static constexpr double BEAM_FPC   = 30.0;      // MPa
static constexpr double STEEL_E    = 200000.0;  // MPa
static constexpr double STEEL_FY   = 420.0;     // MPa
static constexpr double STEEL_B    = 0.01;      // strain-hardening ratio
static constexpr double TIE_FY     = 420.0;     // MPa

static constexpr double SLAB_E  = 28000.0;      // MPa

// ── Mass + damping ────────────────────────────────────────────────────────────
//  2400 kg/m³ = 2.4e-3 MN·s²/m⁴  (consistent with MN / m / MPa unit system)
static constexpr double RC_DENSITY = 2.4e-3;

//  Rayleigh damping: 5% at T₁ ≈ 0.50 s  and  T₃ ≈ 0.10 s
static constexpr double XI_DAMP   = 0.05;
static constexpr double OMEGA_1   = 2.0 * std::numbers::pi / 0.50;  // ≈ 12.57 rad/s
static constexpr double OMEGA_3   = 2.0 * std::numbers::pi / 0.10;  // ≈ 62.83 rad/s

// ── Gravity ───────────────────────────────────────────────────────────────────
//  g = 9.81 m/s²  (no unit conversion needed — consistent with MN / m / s)
static constexpr double GRAVITY_ACCEL = 9.81;

static constexpr double COLUMN_AREA = COLUMN_B * COLUMN_H;   // 0.2025 m²
static constexpr double BEAM_AREA   = BEAM_B   * BEAM_H;     // 0.18   m²

// ── Dynamic loading ───────────────────────────────────────────────────────────
//  Cyclic lateral load: sinusoidal at near-fundamental period.
static constexpr double LATERAL_AMPLITUDE      =  0.050;   // MN = 50 kN per node
static constexpr double EXCITATION_PERIOD      =  0.50;    // s (near T₁ → resonance amplification)
static constexpr double GRAVITY_RAMP_TIME      =  1.00;    // s (ramp gravity to avoid dynamic shock)
static constexpr double LATERAL_RAMP_TIME      =  0.50;    // s (ramp lateral envelope)
static constexpr double T_FINAL                =  5.00;    // s
static constexpr double DT                     =  0.001;   // s  → 5000 steps

static constexpr int VTK_SNAPSHOT_INTERVAL = 100;  // write VTK every 100 steps

// ── Type aliases ──────────────────────────────────────────────────────────────
using StructuralPolicy = SingleElementPolicy<StructuralElement>;
using StructuralModel  = Model<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, StructuralPolicy>;
using DynamicSolver    = DynamicAnalysis<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, StructuralPolicy>;

// =============================================================================
//  Dynamic analysis driver
// =============================================================================
static void run_corotational_dynamic_rc_building() {
    std::println("================================================================");
    std::println("  fall_n — Corotational Dynamic 5-Story RC Building");
    std::println("================================================================\n");

    // ─────────────────────────────────────────────────────────────────────
    //  1. Build the geometric domain from a building specification
    // ─────────────────────────────────────────────────────────────────────
    auto [domain, grid] = fall_n::make_building_domain({
        .x_axes       = {X_GRID.begin(), X_GRID.end()},
        .y_axes       = {Y_GRID.begin(), Y_GRID.end()},
        .num_stories  = NUM_STORIES,
        .story_height = STORY_HEIGHT,
    });

    // ─────────────────────────────────────────────────────────────────────
    //  2. Create formulation-specific materials
    // ─────────────────────────────────────────────────────────────────────
    const auto column_material = fall_n::make_rc_column_section({
        .b            = COLUMN_B,
        .h            = COLUMN_H,
        .cover        = COLUMN_COVER,
        .bar_diameter = COLUMN_BAR_D,
        .tie_spacing  = COLUMN_TIE_SPACING,
        .fpc          = COLUMN_FPC,
        .nu           = NU_RC,
        .steel_E      = STEEL_E,
        .steel_fy     = STEEL_FY,
        .steel_b      = STEEL_B,
        .tie_fy       = TIE_FY,
    });

    const auto beam_material = fall_n::make_rc_beam_section({
        .b            = BEAM_B,
        .h            = BEAM_H,
        .cover        = BEAM_COVER,
        .bar_diameter = BEAM_BAR_D,
        .fpc          = BEAM_FPC,
        .nu           = NU_RC,
        .steel_E      = STEEL_E,
        .steel_fy     = STEEL_FY,
        .steel_b      = STEEL_B,
    });

    MindlinShellMaterial slab_relation{SLAB_E, NU_RC, SLAB_T};
    const auto slab_material = Material<MindlinReissnerShell3D>{slab_relation, ElasticUpdate{}};

    // ─────────────────────────────────────────────────────────────────────
    //  3. Build structural elements via physical-group mapping
    // ─────────────────────────────────────────────────────────────────────
    std::vector<const ElementGeometry<3>*> shell_geometries;

    auto elements = fall_n::StructuralModelBuilder{}
        .set_frame_material("Columns", column_material)
        .set_frame_material("Beams",   beam_material)
        .set_shell_material("Slabs",   slab_material)
        .build_elements(domain, &shell_geometries);

    StructuralModel model{domain, std::move(elements)};

    // ─────────────────────────────────────────────────────────────────────
    //  4. Boundary conditions (base fixed)
    // ─────────────────────────────────────────────────────────────────────
    model.fix_z(0.0);
    model.setup();

    // ─────────────────────────────────────────────────────────────────────
    //  5. Pre-assemble constant gravity load vector
    // ─────────────────────────────────────────────────────────────────────
    DM dm = model.get_plex();

    fall_n::apply_building_self_weight(model, grid, {
        .density        = RC_DENSITY,
        .gravity        = GRAVITY_ACCEL,
        .column_area    = COLUMN_AREA,
        .beam_area      = BEAM_AREA,
        .slab_thickness = SLAB_T,
    });

    petsc::OwnedVec f_gravity;
    DMCreateGlobalVector(dm, f_gravity.ptr());
    VecSet(f_gravity, 0.0);
    DMLocalToGlobal(dm, model.force_vector(), ADD_VALUES, f_gravity);
    VecSet(model.force_vector(), 0.0);

    // ── Diagnostic: verify gravity forces ──
    {
        PetscReal norm;
        VecNorm(f_gravity, NORM_2, &norm);
        PetscPrintf(PETSC_COMM_WORLD, "  ||f_gravity|| = %e MN\n", norm);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  6. Pre-assemble unit lateral force pattern (triangular, +X)
    // ─────────────────────────────────────────────────────────────────────
    fall_n::apply_triangular_lateral_load(model, grid, 0, LATERAL_AMPLITUDE);

    petsc::OwnedVec f_lateral;
    DMCreateGlobalVector(dm, f_lateral.ptr());
    VecSet(f_lateral, 0.0);
    DMLocalToGlobal(dm, model.force_vector(), ADD_VALUES, f_lateral);
    VecSet(model.force_vector(), 0.0);

    // ── Diagnostic: verify lateral forces ──
    {
        PetscReal norm;
        VecNorm(f_lateral, NORM_2, &norm);
        PetscPrintf(PETSC_COMM_WORLD, "  ||f_lateral|| = %e MN\n", norm);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  7. Configure dynamic solver
    // ─────────────────────────────────────────────────────────────────────
    DynamicSolver solver{&model};

    // Mass: set uniform density on all elements (RC)
    solver.set_density(RC_DENSITY);

    // Damping: Rayleigh from two target frequencies and damping ratio
    solver.set_rayleigh_damping(OMEGA_1, OMEGA_3, XI_DAMP);

    // External force:  f_ext(t) = g(t)·f_gravity + h(t)·sin(2πt/T)·f_lateral
    //  g(t): smooth gravity ramp over GRAVITY_RAMP_TIME
    //  h(t): lateral amplitude ramp starting at t=0 alongside gravity
    solver.set_force_function(
        [&](double t, Vec f_ext) {
            // Ramp gravity smoothly to avoid dynamic shock
            const double g_ramp = std::min(t / GRAVITY_RAMP_TIME, 1.0);
            VecAXPY(f_ext, g_ramp, f_gravity);

            // Cyclic lateral with simultaneous ramp envelope
            const double envelope =
                std::min(t / LATERAL_RAMP_TIME, 1.0);
            const double lateral_scale =
                envelope * std::sin(2.0 * std::numbers::pi * t / EXCITATION_PERIOD);
            VecAXPY(f_ext, lateral_scale, f_lateral);
        });

    // ─────────────────────────────────────────────────────────────────────
    //  8. Observer pipeline (replaces monolithic monitor lambda)
    // ─────────────────────────────────────────────────────────────────────
    std::filesystem::create_directories(OUT);

    // Roof corner node for recording
    const auto roof_node = grid.node_id(NUM_AXES_X - 1, NUM_AXES_Y - 1, NUM_STORIES);

    // VTK exporter factory (captures beam/shell profile settings)
    auto vtk_factory = [](const StructuralModel& m) {
        return fall_n::vtk::StructuralVTMExporter(
            m,
            fall_n::reconstruction::RectangularSectionProfile<2>{BEAM_B, BEAM_H},
            fall_n::reconstruction::ShellThicknessProfile<5>{});
    };

    // Compose observers:
    //   1. Console progress (every 10 steps)
    //   2. VTK snapshots + PVD time series (every 100 steps)
    //   3. Node recorder at roof corner (ux, uz every step)
    //   4. Peak response (envelope) tracker
    auto observers = fall_n::make_composite_observer<StructuralModel>(
        fall_n::ConsoleProgressObserver<StructuralModel>{10},
        fall_n::make_vtk_observer<StructuralModel>(
            OUT, "building_dynamic", VTK_SNAPSHOT_INTERVAL,
            vtk_factory),
        fall_n::NodeRecorder<StructuralModel>{
            {{static_cast<std::size_t>(roof_node), 0},    // ux
             {static_cast<std::size_t>(roof_node), 2}},   // uz
            1  // every step
        },
        fall_n::MaxResponseTracker<StructuralModel>{1}
    );

    solver.set_observer(observers);

    // ─────────────────────────────────────────────────────────────────────
    //  9. Solve
    // ─────────────────────────────────────────────────────────────────────
    std::println("Dynamic parameters");
    std::println("  Density (RC)       : {} MN s^2/m^4",   RC_DENSITY);
    std::println("  Damping ratio      : {} %",              XI_DAMP * 100);
    std::println("  Rayleigh omega_1   : {} rad/s (T1={} s)", OMEGA_1, 2*std::numbers::pi/OMEGA_1);
    std::println("  Rayleigh omega_3   : {} rad/s (T3={} s)", OMEGA_3, 2*std::numbers::pi/OMEGA_3);
    std::println("  Gravity accel.     : {} m/s^2",          GRAVITY_ACCEL);
    std::println("  Lateral amplitude  : {} MN per node",    LATERAL_AMPLITUDE);
    std::println("  Excitation period  : {} s",              EXCITATION_PERIOD);
    std::println("  Gravity ramp       : {} s",              GRAVITY_RAMP_TIME);
    std::println("  Lateral ramp       : {} s",              LATERAL_RAMP_TIME);
    std::println("  Total time         : {} s",              T_FINAL);
    std::println("  Time step          : {} s",              DT);
    std::println("  Formulation        : Corotational (large displacements)\n");

    // ─────────────────────────────────────────────────────────────────────
    //  9. Setup, then override TS options programmatically
    // ─────────────────────────────────────────────────────────────────────
    //  setup() calls TSSetFromOptions(), which consumes the option database.
    //  After that, we override TS-specific options via the C API to ensure
    //  they are not reset.
    solver.setup();

    {
        TS ts = solver.get_ts();
        TSAlpha2SetRadius(ts, 0.9);

        TSAdapt adapt;
        TSGetAdapt(ts, &adapt);
        TSAdaptSetType(adapt, TSADAPTNONE);

        TSSetMaxSNESFailures(ts, 5);
    }

    const bool ok = solver.solve(T_FINAL, DT);

    // ─────────────────────────────────────────────────────────────────────
    //  10. Final report  (uses fall_n::query:: + std::println)
    // ─────────────────────────────────────────────────────────────────────
    const auto roof_corner = fall_n::query::nodal_translation(
        model,
        grid.node_id(NUM_AXES_X - 1, NUM_AXES_Y - 1, NUM_STORIES));

    const double max_u  = fall_n::query::max_translation_norm(model);
    const double max_ux = fall_n::query::max_component_abs(model, 0);
    const double max_uz = fall_n::query::max_component_abs(model, 2);
    const double roof_drift_ratio = roof_corner[0] / grid.z_levels.back();

    std::println("\nGeometry / topology");
    std::println("  Nodes              : {}", domain.num_nodes());
    std::println("  Columns (corot.)   : {}", grid.num_columns());
    std::println("  Beams   (corot.)   : {}", grid.num_beams());
    std::println("  Slabs   (elastic)  : {}", grid.num_slabs());
    std::println("  Total elements     : {}\n", domain.num_elements());

    std::println("Materials");
    std::println("  Columns            : RC fiber section (Kent-Park + Menegotto-Pinto)");
    std::println("  Beams              : RC fiber section (Kent-Park + Menegotto-Pinto)");
    std::println("  Slabs              : elastic Mindlin shell");
    std::println("  Steel fy           : {} MPa\n", STEEL_FY);

    std::println("Solver outcome");
    std::println("  Converged          : {}\n", ok ? "yes" : "no");

    std::println("Final-state response");
    std::println("  Max |u|            : {:.6f} m", max_u);
    std::println("  Max |ux|           : {:.6f} m", max_ux);
    std::println("  Max |uz|           : {:.6f} m", max_uz);
    std::println("  Roof corner ux     : {:.6f} m", roof_corner[0]);
    std::println("  Roof corner uy     : {:.6f} m", roof_corner[1]);
    std::println("  Roof corner uz     : {:.6f} m", roof_corner[2]);
    std::println("  Roof drift ratio   : {:.6f}\n", roof_drift_ratio);

    // VTK is written automatically by VTKSnapshotObserver::on_analysis_end
    std::println("VTK output written to: {}", OUT);
}

} // namespace

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    // ── PETSc KSP/PC/SNES options (TS options set programmatically) ──
    PetscOptionsSetValue(nullptr, "-ksp_type",        "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type",         "lu");
    PetscOptionsSetValue(nullptr, "-snes_type",       "newtonls");
    PetscOptionsSetValue(nullptr, "-snes_linesearch_type", "bt");

    run_corotational_dynamic_rc_building();

    PetscFinalize();
    return 0;
}
