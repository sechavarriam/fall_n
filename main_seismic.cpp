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
//  Seismic Dynamic Analysis — 5-Story RC Building under El Centro 1940
// =============================================================================
//
//  Units:
//    Force   : MN
//    Length  : m
//    Stress  : MPa = MN/m²
//    Mass    : MN·s²/m
//    Density : MN·s²/m⁴   (2400 kg/m³ = 2.4e-3)
//    Time    : s
//    Accel.  : m/s²
//
//  Structural idealisation:
//    - Columns and beams: 3D Timoshenko frame elements, **corotational**
//    - Frame sections    : nonlinear fiber (Kent-Park concrete + Menegotto-Pinto steel)
//    - Slabs             : CorotationalMITC4 Mindlin-Reissner shells (elastic)
//
//  Seismic input:
//    - El Centro 1940 NS component — first 10 seconds
//    - Uniform base acceleration in X direction
//    - PGA ≈ 0.3194 g ≈ 3.13 m/s²
//
//  Analysis features:
//    - PETSc TS (Generalized-α, TSALPHA2)
//    - Consistent mass + Rayleigh damping (5%)
//    - DamageTracker with MaxStrainDamageCriterion (injection point)
//    - FiberHysteresisRecorder for extreme concrete/steel fibers
//    - VTK time-series output
//    - Roof displacement history
//
// =============================================================================

static constexpr std::size_t NDOF = 6;

// ── Project path ──────────────────────────────────────────────────────────────
static const std::string BASE = "/home/sechavarriam/MyLibs/fall_n/";
static const std::string OUT  = BASE + "data/output/seismic_rc_building/";
static const std::string EQ_FILE = BASE + "data/input/earthquakes/el_centro_1940_ns.dat";

// ── Grid definition ───────────────────────────────────────────────────────────
static constexpr int NUM_AXES_X  = 4;
static constexpr int NUM_AXES_Y  = 3;
static constexpr int NUM_STORIES = 5;

static constexpr std::array<double, NUM_AXES_X> X_GRID = {0.0, 6.0, 12.0, 18.0};
static constexpr std::array<double, NUM_AXES_Y> Y_GRID = {0.0, 5.0, 10.0};
static constexpr double STORY_HEIGHT = 3.20;

// ── Member sizes ──────────────────────────────────────────────────────────────
static constexpr double COLUMN_B = 0.50;
static constexpr double COLUMN_H = 0.50;
static constexpr double COLUMN_COVER = 0.04;
static constexpr double COLUMN_BAR_D = 0.025;
static constexpr double COLUMN_TIE_SPACING = 0.10;

static constexpr double BEAM_B = 0.30;
static constexpr double BEAM_H = 0.60;
static constexpr double BEAM_COVER = 0.04;
static constexpr double BEAM_BAR_D = 0.020;

static constexpr double SLAB_T = 0.15;

// ── Material data ─────────────────────────────────────────────────────────────
static constexpr double NU_RC = 0.20;

static constexpr double COLUMN_FPC = 35.0;      // MPa
static constexpr double BEAM_FPC   = 28.0;      // MPa
static constexpr double STEEL_E    = 200000.0;  // MPa
static constexpr double STEEL_FY   = 420.0;     // MPa
static constexpr double STEEL_B    = 0.01;      // strain-hardening ratio
static constexpr double TIE_FY     = 420.0;     // MPa

static constexpr double SLAB_E  = 25000.0;      // MPa (elastic shell)

// ── Mass + damping ────────────────────────────────────────────────────────────
static constexpr double RC_DENSITY = 2.4e-3;  // MN·s²/m⁴ (= 2400 kg/m³)

//  Rayleigh damping: 5% at T₁ ≈ 0.60 s  and  T₃ ≈ 0.12 s
static constexpr double XI_DAMP = 0.05;
static constexpr double OMEGA_1 = 2.0 * std::numbers::pi / 0.60;  // ≈ 10.47 rad/s
static constexpr double OMEGA_3 = 2.0 * std::numbers::pi / 0.12;  // ≈ 52.36 rad/s

// ── Gravity + superimposed loads ──────────────────────────────────────────────
static constexpr double GRAVITY_ACCEL = 9.81;  // m/s²
static constexpr double COLUMN_AREA = COLUMN_B * COLUMN_H;
static constexpr double BEAM_AREA   = BEAM_B   * BEAM_H;

// ── Reference strain for damage criterion ─────────────────────────────────────
//    Steel yield strain: ε_y = f_y / E = 420/200000 = 0.0021
static constexpr double EPS_YIELD = STEEL_FY / STEEL_E;

// ── Time integration parameters ───────────────────────────────────────────────
static constexpr double GRAVITY_RAMP_TIME = 1.00;    // s
static constexpr double T_FINAL           = 10.0;    // s (10s of El Centro)
static constexpr double DT                = 0.005;   // s → 2000 steps
static constexpr int    VTK_INTERVAL      = 50;      // write VTK every 50 steps
static constexpr int    DAMAGE_INTERVAL   = 10;      // evaluate damage every 10 steps

// ── Type aliases ──────────────────────────────────────────────────────────────
using StructuralPolicy = SingleElementPolicy<StructuralElement>;
using StructuralModel  = Model<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, StructuralPolicy>;
using DynamicSolver    = DynamicAnalysis<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, StructuralPolicy>;


// =============================================================================
//  Seismic analysis driver
// =============================================================================
static void run_seismic_rc_building() {
    std::println("================================================================");
    std::println("  fall_n — Seismic Analysis: 5-Story RC Building");
    std::println("           El Centro 1940 NS — Corotational Formulation");
    std::println("================================================================\n");

    // ─────────────────────────────────────────────────────────────────────
    //  1. Parse earthquake record
    // ─────────────────────────────────────────────────────────────────────
    auto eq_record = fall_n::GroundMotionRecord::from_file(
        EQ_FILE, GRAVITY_ACCEL);  // scale g → m/s²

    std::println("Earthquake record: {}", eq_record.name());
    std::println("  Points         : {}", eq_record.num_points());
    std::println("  Duration       : {} s", eq_record.duration());
    std::println("  dt             : {} s", eq_record.dt());
    std::println("  PGA            : {:.4f} m/s² ({:.4f} g)",
                 eq_record.pga(), eq_record.pga() / GRAVITY_ACCEL);
    std::println("  Time of PGA    : {:.2f} s\n", eq_record.time_of_pga());

    // ─────────────────────────────────────────────────────────────────────
    //  2. Build the geometric domain
    // ─────────────────────────────────────────────────────────────────────
    auto [domain, grid] = fall_n::make_building_domain({
        .x_axes       = {X_GRID.begin(), X_GRID.end()},
        .y_axes       = {Y_GRID.begin(), Y_GRID.end()},
        .num_stories  = NUM_STORIES,
        .story_height = STORY_HEIGHT,
    });

    // ─────────────────────────────────────────────────────────────────────
    //  3. Materials
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
    //  4. Build structural elements (corotational beams + MITC4 shells)
    // ─────────────────────────────────────────────────────────────────────
    std::vector<const ElementGeometry<3>*> shell_geometries;

    auto elements = fall_n::StructuralModelBuilder<
        BeamElement<TimoshenkoBeam3D, 3, beam::Corotational>,
        CorotationalMITC4Shell<>,
        TimoshenkoBeam3D,
        MindlinReissnerShell3D>{}
        .set_frame_material("Columns", column_material)
        .set_frame_material("Beams",   beam_material)
        .set_shell_material("Slabs",   slab_material)
        .build_elements(domain, &shell_geometries);

    StructuralModel model{domain, std::move(elements)};

    // ─────────────────────────────────────────────────────────────────────
    //  5. Boundary conditions: base fixed + ground motion
    // ─────────────────────────────────────────────────────────────────────
    model.fix_z(0.0);
    model.setup();

    BoundaryConditionSet<3> bcs;
    bcs.add_ground_motion({0, eq_record.as_time_function()});  // X-direction

    // ─────────────────────────────────────────────────────────────────────
    //  6. Gravity loads (pre-assembled, ramped)
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

    {
        PetscReal norm;
        VecNorm(f_gravity, NORM_2, &norm);
        PetscPrintf(PETSC_COMM_WORLD, "  ||f_gravity|| = %e MN\n", norm);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  7. Configure dynamic solver
    // ─────────────────────────────────────────────────────────────────────
    DynamicSolver solver{&model};

    solver.set_density(RC_DENSITY);
    solver.set_rayleigh_damping(OMEGA_1, OMEGA_3, XI_DAMP);

    // Gravity: ramped over GRAVITY_RAMP_TIME
    // Ground motion: applied via BoundaryConditionSet (pseudo-force M·ĝ·ag(t))
    solver.set_force_function(
        [&](double t, Vec f_ext) {
            const double g_ramp = std::min(t / GRAVITY_RAMP_TIME, 1.0);
            VecAXPY(f_ext, g_ramp, f_gravity);
        });

    solver.set_boundary_conditions(bcs);

    // ─────────────────────────────────────────────────────────────────────
    //  8. Observer pipeline — with damage tracking + hysteresis
    // ─────────────────────────────────────────────────────────────────────
    std::filesystem::create_directories(OUT);

    const auto roof_node = grid.node_id(NUM_AXES_X - 1, NUM_AXES_Y - 1, NUM_STORIES);

    // VTK factory
    auto vtk_factory = [](const StructuralModel& m) {
        return fall_n::vtk::StructuralVTMExporter(
            m,
            fall_n::reconstruction::RectangularSectionProfile<2>{BEAM_B, BEAM_H},
            fall_n::reconstruction::ShellThicknessProfile<5>{});
    };

    // ── Damage criterion (injection point) ──────────────────────
    //  Users can swap this for:
    //    - ParkAngDamageCriterion(eps_u, beta, Fy_cum)
    //    - make_damage_criterion("Custom", [](auto& e, size_t, Vec){...})
    //    - Any subclass of DamageCriterion
    fall_n::MaxStrainDamageCriterion damage_criterion{EPS_YIELD};

    // ── Fiber classifier (concrete vs steel) ────────────────────
    //  Heuristic: fibers with area < 0.001 m² are rebar (steel);
    //  larger patches are concrete.  Users can inject a more precise
    //  classifier based on element/section knowledge.
    fall_n::FiberClassifier fiber_classifier =
        [](std::size_t /*elem*/, std::size_t /*gp*/, std::size_t /*fi*/,
           double /*y*/, double /*z*/, double area) -> fall_n::FiberMaterialClass
        {
            return (area < 0.001)
                ? fall_n::FiberMaterialClass::Steel
                : fall_n::FiberMaterialClass::Concrete;
        };

    // ── Compose observers ───────────────────────────────────────
    auto observers = fall_n::make_composite_observer<StructuralModel>(
        fall_n::ConsoleProgressObserver<StructuralModel>{20},
        fall_n::make_vtk_observer<StructuralModel>(
            OUT, "seismic_building", VTK_INTERVAL, vtk_factory),
        fall_n::NodeRecorder<StructuralModel>{
            {{static_cast<std::size_t>(roof_node), 0},    // ux (seismic dir)
             {static_cast<std::size_t>(roof_node), 1},    // uy
             {static_cast<std::size_t>(roof_node), 2}},   // uz
            1  // every step
        },
        fall_n::MaxResponseTracker<StructuralModel>{1},
        fall_n::DamageTracker<StructuralModel>{damage_criterion, DAMAGE_INTERVAL, 10},
        fall_n::FiberHysteresisRecorder<StructuralModel>{
            damage_criterion, fiber_classifier, {}, 5, 1}
    );

    solver.set_observer(observers);

    // ─────────────────────────────────────────────────────────────────────
    //  9. Print analysis parameters
    // ─────────────────────────────────────────────────────────────────────
    std::println("\nAnalysis parameters");
    std::println("  Density (RC)       : {} MN·s²/m⁴", RC_DENSITY);
    std::println("  Damping ratio      : {}%",          XI_DAMP * 100);
    std::println("  Rayleigh ω₁       : {:.2f} rad/s (T₁={:.2f} s)", OMEGA_1, 2*std::numbers::pi/OMEGA_1);
    std::println("  Rayleigh ω₃       : {:.2f} rad/s (T₃={:.2f} s)", OMEGA_3, 2*std::numbers::pi/OMEGA_3);
    std::println("  Gravity accel.     : {} m/s²", GRAVITY_ACCEL);
    std::println("  Gravity ramp       : {} s", GRAVITY_RAMP_TIME);
    std::println("  Total time         : {} s", T_FINAL);
    std::println("  Time step          : {} s", DT);
    std::println("  Damage criterion   : {} (ε_ref = {:.6f})", damage_criterion.name(), EPS_YIELD);
    std::println("  Formulation        : Corotational (large displacements)\n");

    // ─────────────────────────────────────────────────────────────────────
    //  10. Setup + solve
    // ─────────────────────────────────────────────────────────────────────
    solver.setup();

    {
        TS ts = solver.get_ts();
        TSAlpha2SetRadius(ts, 0.9);

        TSAdapt adapt;
        TSGetAdapt(ts, &adapt);
        TSAdaptSetType(adapt, TSADAPTNONE);

        TSSetMaxSNESFailures(ts, 10);
    }

    const bool ok = solver.solve(T_FINAL, DT);

    // ─────────────────────────────────────────────────────────────────────
    //  11. Final report
    // ─────────────────────────────────────────────────────────────────────
    const auto roof_corner = fall_n::query::nodal_translation(
        model,
        grid.node_id(NUM_AXES_X - 1, NUM_AXES_Y - 1, NUM_STORIES));

    const double max_u  = fall_n::query::max_translation_norm(model);
    const double max_ux = fall_n::query::max_component_abs(model, 0);
    const double roof_drift_ratio = roof_corner[0] / grid.z_levels.back();

    std::println("\n════════════════════════════════════════════════════════════════");
    std::println("  SEISMIC ANALYSIS REPORT");
    std::println("════════════════════════════════════════════════════════════════");

    std::println("\nGeometry / topology");
    std::println("  Nodes              : {}", domain.num_nodes());
    std::println("  Columns (corot.)   : {}", grid.num_columns());
    std::println("  Beams   (corot.)   : {}", grid.num_beams());
    std::println("  Slabs   (MITC4)    : {}", grid.num_slabs());
    std::println("  Total elements     : {}", domain.num_elements());

    std::println("\nMaterials");
    std::println("  Columns            : RC fiber section (Kent-Park + Menegotto-Pinto)");
    std::println("  Beams              : RC fiber section (Kent-Park + Menegotto-Pinto)");
    std::println("  Slabs              : CorotationalMITC4 elastic Mindlin-Reissner shell");
    std::println("  Steel fy           : {} MPa", STEEL_FY);
    std::println("  Column f'c         : {} MPa", COLUMN_FPC);
    std::println("  Beam f'c           : {} MPa", BEAM_FPC);

    std::println("\nSolver outcome");
    std::println("  Converged          : {}", ok ? "yes" : "no");

    std::println("\nFinal-state response");
    std::println("  Max |u|            : {:.6f} m", max_u);
    std::println("  Max |ux|           : {:.6f} m (seismic direction)", max_ux);
    std::println("  Roof corner ux     : {:.6f} m", roof_corner[0]);
    std::println("  Roof corner uy     : {:.6f} m", roof_corner[1]);
    std::println("  Roof corner uz     : {:.6f} m", roof_corner[2]);
    std::println("  Roof drift ratio   : {:.6f}", roof_drift_ratio);

    // ── Export hysteresis CSVs ──────────────────────────────────────
    auto& hysteresis_obs = observers.template get<5>();
    hysteresis_obs.write_hysteresis_csv(OUT + "hysteresis");

    auto& node_rec = observers.template get<2>();
    node_rec.write_csv(OUT + "roof_displacement.csv");

    std::println("\nOutput");
    std::println("  VTK                : {}seismic_building_*.vtm", OUT);
    std::println("  Roof displacement  : {}roof_displacement.csv", OUT);
    std::println("  Hysteresis (conc.) : {}hysteresis_concrete.csv", OUT);
    std::println("  Hysteresis (steel) : {}hysteresis_steel.csv", OUT);

    std::println("\n════════════════════════════════════════════════════════════════\n");
}

} // namespace

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    // PETSc solver options
    PetscOptionsSetValue(nullptr, "-ksp_type",        "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type",         "lu");
    PetscOptionsSetValue(nullptr, "-snes_type",       "newtonls");
    PetscOptionsSetValue(nullptr, "-snes_linesearch_type", "bt");

    run_seismic_rc_building();

    PetscFinalize();
    return 0;
}
