// =============================================================================
//  main_lshaped_multiscale_16storey.cpp
// =============================================================================
//
//  L-shaped 16-story RC building — height-irregular, 3-component seismic
//  multiscale FE² analysis.
//
//  This example extends the 4-story reference to demonstrate:
//    - Height irregularity: three column-section ranges reducing with height
//      (0.50×0.50 base, 0.40×0.40 mid, 0.30×0.30 top); different f'c per range.
//    - Full iterated two-way FE² coupling on ALL critical sub-models.
//    - Per-range Hex27 sub-models with KoBatheConcrete3D.
//    - Parallel sub-model solving via OpenMP (within MultiscaleAnalysis).
//    - VTK evolution series (global frame + per-sub-model crack glyphs).
//    - CSV output for roof displacement, fiber hysteresis, crack evolution.
//    - Python postprocessing invocation from C++.
//
//  Outputs
//  -------
//  data/output/lshaped_multiscale_16/
//    yield_state.vtm               -- global frame at first yield
//    evolution/frame_NNNNNN.vtm    -- global frame time series
//    evolution/frame.pvd           -- ParaView PVD for frame animation
//    evolution/sub_models/         -- per-element PVD series
//    recorders/
//      roof_displacement.csv
//      fiber_hysteresis_concrete.csv
//      fiber_hysteresis_steel.csv
//      crack_evolution.csv
//
//  Units: [m, MN, MPa = MN/m², MN·s²/m⁴, s]
//
// =============================================================================

#include "header_files.hh"
#include "src/utils/PythonPlotter.hh"

#include <Eigen/Dense>

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numbers>
#include <print>
#include <string>
#include <unordered_map>
#include <vector>

using namespace fall_n;


// =============================================================================
//  Constants
// =============================================================================
namespace {

[[nodiscard]] double csv_scalar_or_nan(bool available, double value)
{
    return available ? value : std::numeric_limits<double>::quiet_NaN();
}

// ── I/O paths ────────────────────────────────────────────────────────────────
#ifdef FALL_N_SOURCE_DIR
static const std::string BASE = std::string(FALL_N_SOURCE_DIR) + "/";
#else
static const std::string BASE = "./";
#endif

static const std::string EQ_DIR = BASE + "data/input/earthquakes/Japan2011/"
                                         "Tsukidate-MYG004/";
static const std::string EQ_NS  = EQ_DIR + "MYG0041103111446.NS";
static const std::string EQ_EW  = EQ_DIR + "MYG0041103111446.EW";
static const std::string EQ_UD  = EQ_DIR + "MYG0041103111446.UD";
static const std::string OUT    = BASE + "data/output/lshaped_multiscale_16/";

// ── L-shaped frame geometry ─────────────────────────────────────────────────
//  4 bays in X (5 m each),  3 bays in Y (4 m each)
//  Cutout removes bays ix∈[2,4) iy∈[2,3) → L-shape in plan
static constexpr std::array<double, 5> X_GRID = {0.0, 5.0, 10.0, 15.0, 20.0};
static constexpr std::array<double, 4> Y_GRID = {0.0, 4.0, 8.0, 12.0};
static constexpr int    NUM_STORIES  = 16;
static constexpr double STORY_HEIGHT = 3.2;   // m (uniform — API limitation)

// Cutout: remove upper-right 2×1 bays at all stories → L-shape
static constexpr int CUTOUT_X_START = 1;
static constexpr int CUTOUT_X_END   = 4;
static constexpr int CUTOUT_Y_START = 1;
static constexpr int CUTOUT_Y_END   = 3;
static constexpr int CUTOUT_ABOVE   = 0;  // all stories

// ── Story ranges for height irregularity ────────────────────────────────────
//  Range 0 (Lower):  stories 1–5   — bottom z < STORY_BREAK_1 * STORY_HEIGHT
//  Range 1 (Mid):    stories 6–11  — bottom z < STORY_BREAK_2 * STORY_HEIGHT
//  Range 2 (Upper):  stories 12–16 — remainder
static constexpr int STORY_BREAK_1 = 5;
static constexpr int STORY_BREAK_2 = 11;
static constexpr int NUM_RANGES    = 3;

// ── Column sections per range ───────────────────────────────────────────────
static constexpr double COL_B[]   = {0.50,  0.40,  0.30};   // width  [m]
static constexpr double COL_H[]   = {0.50,  0.40,  0.30};   // height [m]
static constexpr double COL_FPC[] = {35.0,  28.0,  21.0};   // f'c    [MPa]
static constexpr double COL_CVR   = 0.04;
static constexpr double COL_BAR   = 0.020;
static constexpr double COL_TIE   = 0.10;

// ── Beam section 0.30 × 0.60 m (uniform across all stories) ────────────────
static constexpr double BM_B   = 0.30;
static constexpr double BM_H   = 0.60;
static constexpr double BM_CVR = 0.04;
static constexpr double BM_BAR = 0.016;
static constexpr double BM_FPC = 25.0;

// ── RC steel ─────────────────────────────────────────────────────────────────
static constexpr double STEEL_E  = 200000.0;  // Es [MPa]
static constexpr double STEEL_FY = 420.0;     // fy [MPa]
static constexpr double STEEL_B  = 0.01;      // hardening ratio
static constexpr double TIE_FY   = 420.0;     // tie yield [MPa]
static constexpr double NU_RC    = 0.20;

// Effective elastic moduli per range (ACI 318: Ec = 4700√f'c)
static const double EC_RANGE[] = {
    4700.0 * std::sqrt(COL_FPC[0]),
    4700.0 * std::sqrt(COL_FPC[1]),
    4700.0 * std::sqrt(COL_FPC[2]),
};
static const double GC_RANGE[] = {
    EC_RANGE[0] / (2.0 * (1.0 + NU_RC)),
    EC_RANGE[1] / (2.0 * (1.0 + NU_RC)),
    EC_RANGE[2] / (2.0 * (1.0 + NU_RC)),
};

// Story range group names
static const std::string COL_GROUPS[] = {
    "ColumnsLower", "ColumnsMid", "ColumnsUpper"};

// ── Damage / transition ──────────────────────────────────────────────────────
static constexpr double EPS_YIELD = STEEL_FY / STEEL_E;  // ≈ 0.0021

// ── Mass + Rayleigh damping  (5 % at T₁≈1.6 s, T₃≈0.40 s) ─────────────────
//  Empirical T₁ ≈ 0.1·N for RC frames → 0.1×16 = 1.6 s
static constexpr double RC_DENSITY = 2.4e-3;   // MN·s²/m⁴
static constexpr double XI_DAMP    = 0.05;
static const double OMEGA_1 = 2.0 * std::numbers::pi / 1.60;
static const double OMEGA_3 = 2.0 * std::numbers::pi / 0.40;

// ── Time integration ─────────────────────────────────────────────────────────
static constexpr double DT       = 0.02;     // s
static constexpr double T_SKIP   = 40.0;     // skip to strong-motion onset
static constexpr double T_MAX    = 1.5;      // window [40,41.5]s — elastic + post-yield
static constexpr double EQ_SCALE = 5.0;      // amplified for lab-scale demonstration

// ── Sub-model mesh ───────────────────────────────────────────────────────────
static constexpr int SUB_NX = 2;
static constexpr int SUB_NY = 2;
static constexpr int SUB_NZ = 8;   // refined longitudinally for convergence

// ── Sub-model evolution ──────────────────────────────────────────────────────
static constexpr int EVOL_VTK_INTERVAL   = 5;     // Sub-model VTK every 5 steps
static constexpr int EVOL_PRINT_INTERVAL = 1;     // print every Phase-2 step
static constexpr int FRAME_VTK_INTERVAL  = 10;    // frame VTK every 10 Phase-2 steps

// ── FE² two-way coupling ────────────────────────────────────────────────────
static constexpr int    MAX_STAGGERED_ITER  = 6;
static constexpr double STAGGERED_TOL       = 0.03;
static constexpr double STAGGERED_RELAX     = 0.7;
static constexpr int    COUPLING_START_STEP = 5;

// ── Number of critical elements for sub-model analysis ──────────────────────
static constexpr std::size_t N_CRITICAL = 3;

// ── Type aliases ─────────────────────────────────────────────────────────────
static constexpr std::size_t NDOF = 6;
using StructPolicy = SingleElementPolicy<StructuralElement>;
using StructModel  = Model<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, StructPolicy>;
using DynSolver    = DynamicAnalysis<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, StructPolicy>;
using BeamElemT    = BeamElement<TimoshenkoBeam3D, 3, beam::SmallRotation>;
using ShellElemT   = MITC4Shell<>;

} // anonymous namespace


// =============================================================================
//  Helper: separator
// =============================================================================
static void sep(char c = '=', int n = 72) {
    std::cout << std::string(n, c) << '\n';
}


// =============================================================================
//  Helper: determine story range from bottom z-coordinate
// =============================================================================
static int story_range(double z_bottom) {
    if (z_bottom < STORY_BREAK_1 * STORY_HEIGHT - 0.01) return 0;
    if (z_bottom < STORY_BREAK_2 * STORY_HEIGHT - 0.01) return 1;
    return 2;
}


// =============================================================================
//  Main analysis
// =============================================================================
int main(int argc, char* argv[]) {

    setvbuf(stdout, nullptr, _IONBF, 0);

    double eq_scale = EQ_SCALE;
    if (argc >= 2) {
        eq_scale = std::stod(argv[1]);
        if (eq_scale <= 0.0)
            throw std::invalid_argument("Scale factor must be positive.");
    }

    PetscInitialize(&argc, &argv, nullptr, nullptr);
    PetscOptionsSetValue(nullptr, "-snes_monitor_cancel", "");
    PetscOptionsSetValue(nullptr, "-ksp_monitor_cancel",  "");

    sep('=');
    std::println("  fall_n — 16-Story L-Shaped RC Building: Height-Irregular");
    std::println("  Multiscale FE² Seismic Analysis");
    std::println("  Tohoku 2011 (MYG004 NS+EW+UD) + Fiber Sections + Ko-Bathe 3D");
    sep('=');

    // ─────────────────────────────────────────────────────────────────────
    //  1. Parse 3-component earthquake records
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[1] Loading 3-component earthquake records...");

    auto eq_ns_full = GroundMotionRecord::from_knet(EQ_NS);
    auto eq_ew_full = GroundMotionRecord::from_knet(EQ_EW);
    auto eq_ud_full = GroundMotionRecord::from_knet(EQ_UD);

    auto eq_ns = eq_ns_full.trim(T_SKIP, T_SKIP + T_MAX);
    auto eq_ew = eq_ew_full.trim(T_SKIP, T_SKIP + T_MAX);
    auto eq_ud = eq_ud_full.trim(T_SKIP, T_SKIP + T_MAX);

    std::println("  Station       : MYG004 (Tsukidate, Miyagi) — near-fault");
    std::println("  Event         : Tohoku 2011-03-11 Mw 9.0");
    std::println("  Window        : [{:.0f} s, {:.0f} s]", T_SKIP, T_SKIP + T_MAX);
    std::println("  PGA (NS)      : {:.3f} m/s² ({:.3f} g)",
                 eq_ns.pga(), eq_ns.pga() / 9.81);
    std::println("  PGA (EW)      : {:.3f} m/s² ({:.3f} g)",
                 eq_ew.pga(), eq_ew.pga() / 9.81);
    std::println("  PGA (UD)      : {:.3f} m/s² ({:.3f} g)",
                 eq_ud.pga(), eq_ud.pga() / 9.81);
    std::println("  Scale factor  : {:.2f}", eq_scale);

    // ─────────────────────────────────────────────────────────────────────
    //  2. Building domain: 16-story L-shaped RC frame
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[2] Building L-shaped structural domain (16 stories)...");

    auto [domain, grid] = make_building_domain({
        .x_axes          = {X_GRID.begin(), X_GRID.end()},
        .y_axes          = {Y_GRID.begin(), Y_GRID.end()},
        .num_stories     = NUM_STORIES,
        .story_height    = STORY_HEIGHT,
        .cutout_x_start  = CUTOUT_X_START,
        .cutout_x_end    = CUTOUT_X_END,
        .cutout_y_start  = CUTOUT_Y_START,
        .cutout_y_end    = CUTOUT_Y_END,
        .cutout_above_story = CUTOUT_ABOVE,
        .include_slabs   = false,
    });

    std::println("  Plan          : {} bays X × {} bays Y, L-shape with cutout",
                 X_GRID.size()-1, Y_GRID.size()-1);
    std::println("  Stories       : {} × {:.1f} m = {:.1f} m total height",
                 NUM_STORIES, STORY_HEIGHT, NUM_STORIES * STORY_HEIGHT);
    std::println("  Nodes         : {}", domain.num_nodes());
    std::println("  Columns       : {}", grid.num_columns());
    std::println("  Beams         : {}", grid.num_beams());

    // ─────────────────────────────────────────────────────────────────────
    //  3. Post-process: reassign column physical groups by story range
    //     This enables per-range material assignment for height irregularity.
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[3] Assigning column groups by story range (height irregularity)...");

    std::unordered_map<std::size_t, int> elem_to_range;
    std::array<int, NUM_RANGES> col_count_per_range = {0, 0, 0};

    {
        std::size_t idx = 0;
        for (auto& geom : domain.elements()) {
            if (geom.physical_group() == "Columns") {
                const double z_bot = geom.point_p(0).coord(2);
                const int r = story_range(z_bot);
                geom.set_physical_group(COL_GROUPS[r]);
                elem_to_range[idx] = r;
                ++col_count_per_range[static_cast<std::size_t>(r)];
            }
            ++idx;
        }
    }

    for (int r = 0; r < NUM_RANGES; ++r) {
        std::println("  {} : {} cols, {}×{} m, f'c={} MPa",
                     COL_GROUPS[r], col_count_per_range[r],
                     COL_B[r], COL_H[r], COL_FPC[r]);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  4. RC fiber sections (one per range + one for beams)
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[4] Building RC fiber sections (3 column ranges + beams)...");

    std::vector<decltype(make_rc_column_section({}))> col_mats;
    col_mats.reserve(NUM_RANGES);
    for (int r = 0; r < NUM_RANGES; ++r) {
        col_mats.push_back(make_rc_column_section({
            .b            = COL_B[r],
            .h            = COL_H[r],
            .cover        = COL_CVR,
            .bar_diameter = COL_BAR,
            .tie_spacing  = COL_TIE,
            .fpc          = COL_FPC[r],
            .nu           = NU_RC,
            .steel_E      = STEEL_E,
            .steel_fy     = STEEL_FY,
            .steel_b      = STEEL_B,
            .tie_fy       = TIE_FY,
        }));
    }

    const auto bm_mat = make_rc_beam_section({
        .b            = BM_B,
        .h            = BM_H,
        .cover        = BM_CVR,
        .bar_diameter = BM_BAR,
        .fpc          = BM_FPC,
        .nu           = NU_RC,
        .steel_E      = STEEL_E,
        .steel_fy     = STEEL_FY,
        .steel_b      = STEEL_B,
    });

    std::println("  Beams : {}×{} m, f'c={} MPa", BM_B, BM_H, BM_FPC);

    // ─────────────────────────────────────────────────────────────────────
    //  5. Structural model
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[5] Assembling structural model (frame-only)...");

    std::vector<const ElementGeometry<3>*> shell_geoms;

    auto builder = StructuralModelBuilder<
        BeamElemT, ShellElemT,
        TimoshenkoBeam3D, MindlinReissnerShell3D>{};
    for (int r = 0; r < NUM_RANGES; ++r)
        builder.set_frame_material(COL_GROUPS[r], col_mats[static_cast<std::size_t>(r)]);
    builder.set_frame_material("Beams", bm_mat);

    std::vector<StructuralElement> elements =
        builder.build_elements(domain, &shell_geoms);

    StructModel model{domain, std::move(elements)};
    model.fix_z(0.0);
    model.setup();

    std::println("  Total structural elements : {}", model.elements().size());

    // ─────────────────────────────────────────────────────────────────────
    //  6. Dynamic solver: density, Rayleigh damping, 3-component ground motion
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[6] Configuring dynamic solver (3-component)...");

    DynSolver solver{&model};
    solver.set_density(RC_DENSITY);
    solver.set_rayleigh_damping(OMEGA_1, OMEGA_3, XI_DAMP);
    solver.set_force_function([](double, Vec) {});

    BoundaryConditionSet<3> bcs;
    bcs.add_ground_motion({0, eq_ns.as_time_function()}, eq_scale);  // X (NS)
    bcs.add_ground_motion({1, eq_ew.as_time_function()}, eq_scale);  // Y (EW)
    bcs.add_ground_motion({2, eq_ud.as_time_function()}, eq_scale);  // Z (UD)
    solver.set_boundary_conditions(bcs);

    std::println("  Density          : {} MN·s²/m⁴", RC_DENSITY);
    std::println("  Damping          : {}%", XI_DAMP * 100.0);
    std::println("  T₁ (approx.)    : {:.2f} s", 2.0 * std::numbers::pi / OMEGA_1);
    std::println("  T₃ (approx.)    : {:.2f} s", 2.0 * std::numbers::pi / OMEGA_3);
    std::println("  Time step        : {} s", DT);
    std::println("  Duration         : {} s", T_MAX);

    // ─────────────────────────────────────────────────────────────────────
    //  7. Observers
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[7] Setting up observers...");

    MaxStrainDamageCriterion damage_crit{EPS_YIELD};

    DamageTracker<StructModel> damage_tracker{damage_crit, 1, 10};

    auto fiber_classifier = [](std::size_t, std::size_t, std::size_t,
                               double, double, double area) -> FiberMaterialClass {
        return (area < 0.001) ? FiberMaterialClass::Steel
                              : FiberMaterialClass::Concrete;
    };
    FiberHysteresisRecorder<StructModel> hysteresis_rec{
        damage_crit, fiber_classifier, {}, 5, 1};

    const int top_level = NUM_STORIES;
    std::vector<NodeRecorder<StructModel>::Channel> disp_channels;
    for (int ix = 0; ix < static_cast<int>(X_GRID.size()); ++ix) {
        for (int iy = 0; iy < static_cast<int>(Y_GRID.size()); ++iy) {
            if (!grid.is_node_active(ix, iy, top_level)) continue;
            auto nid = static_cast<std::size_t>(grid.node_id(ix, iy, top_level));
            disp_channels.push_back({nid, 0});  // X
            disp_channels.push_back({nid, 1});  // Y
            disp_channels.push_back({nid, 2});  // Z
        }
    }
    NodeRecorder<StructModel> node_rec{disp_channels, 1};

    auto composite = make_composite_observer<StructModel>(
        std::move(damage_tracker), std::move(hysteresis_rec), std::move(node_rec));
    solver.set_observer(composite);

    std::println("  DamageTracker          : top-10, every step");
    std::println("  FiberHysteresisRecorder: top-5/material, every step");
    std::println("  NodeRecorder           : {} channels (roof nodes)", disp_channels.size());

    // ─────────────────────────────────────────────────────────────────────
    //  8. Transition director
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[8] Configuring transition director...");

    auto [director, transition_report] =
        make_damage_threshold_director<StructModel>(damage_crit, 1.0);

    std::println("  Criterion  : MaxStrain (ε_ref = {:.6f})", EPS_YIELD);
    std::println("  Threshold  : damage_index > 1.0 (ε > ε_y)");

    // ── Create output directories upfront ────────────────────────────────
    std::filesystem::create_directories(OUT);
    std::filesystem::create_directories(OUT + "evolution/sub_models/");
    std::filesystem::create_directories(OUT + "recorders/");

    // ── Global history CSV (elastic + post-yield, every step) ────────────
    std::ofstream global_csv(OUT + "recorders/global_history.csv");
    global_csv << "time,step,phase,u_inf,peak_damage\n";

    // ─────────────────────────────────────────────────────────────────────
    //  9. Phase 1: Global nonlinear dynamic analysis until first yield
    // ─────────────────────────────────────────────────────────────────────
    sep('=');
    std::println("\n[9] PHASE 1: Global fiber-section dynamic analysis");
    std::println("    Running until first steel fiber reaches yield...");

    PetscOptionsSetValue(nullptr, "-snes_max_it",          "50");
    PetscOptionsSetValue(nullptr, "-snes_rtol",             "1e-2");
    PetscOptionsSetValue(nullptr, "-snes_atol",             "1e-6");
    PetscOptionsSetValue(nullptr, "-ksp_type",              "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type",               "lu");
    PetscOptionsSetValue(nullptr, "-snes_linesearch_type",  "bt");

    solver.setup();
    solver.set_time_step(DT);

    {
        TS ts = solver.get_ts();
        TSAlpha2SetRadius(ts, 0.9);
        TSAdapt adapt;
        TSGetAdapt(ts, &adapt);
        TSAdaptSetType(adapt, TSADAPTBASIC);
        TSAdaptSetStepLimits(adapt, DT * 0.01, DT);
        TSSetTimeStep(ts, DT);
        TSSetMaxSNESFailures(ts, -1);
        SNES snes;
        TSGetSNES(ts, &snes);
        SNESSetTolerances(snes, 1e-6, 1e-2, PETSC_DETERMINE, 50, PETSC_DETERMINE);
        KSP ksp;
        SNESGetKSP(snes, &ksp);
        KSPSetTolerances(ksp, PETSC_DETERMINE, PETSC_DETERMINE, PETSC_DETERMINE, 200);
    }

    double peak_damage_global = 0.0;
    fall_n::StepDirector<StructModel> phase1_director =
        [&director, &peak_damage_global, &damage_crit, &global_csv]
        (const fall_n::StepEvent& ev, const StructModel& m) -> fall_n::StepVerdict
    {
        double max_d = 0.0;
        for (std::size_t e = 0; e < m.elements().size(); ++e) {
            auto info = damage_crit.evaluate_element(m.elements()[e], e, m.state_vector());
            max_d = std::max(max_d, info.damage_index);
        }
        peak_damage_global = std::max(peak_damage_global, max_d);

        PetscReal u_norm = 0.0;
        VecNorm(ev.displacement, NORM_INFINITY, &u_norm);

        // Every step → global history CSV
        global_csv << std::fixed << std::setprecision(6) << ev.time
                   << "," << ev.step
                   << ",1,"
                   << std::scientific << std::setprecision(6)
                   << static_cast<double>(u_norm)
                   << "," << peak_damage_global
                   << "\n" << std::flush;

        if (ev.step % 5 == 0) {
            std::println("    t={:.4f} s  step={:4d}  |u|={:.3e} m  damage={:.6e}",
                         ev.time, ev.step, static_cast<double>(u_norm),
                         peak_damage_global);
            std::cout << std::flush;
        }
        return director(ev, m);
    };

    solver.step_to(T_MAX, phase1_director);

    sep('-');
    if (!transition_report->triggered) {
        std::println("[!] No fiber yielding detected within {} s.", T_MAX);
        std::println("    Peak damage = {:.4f} — try larger scale.", peak_damage_global);
        global_csv.close();
        PetscFinalize();
        return 0;
    }

    std::println("\n[*] YIELDING DETECTED");
    std::println("    Time of first yield   : {:.4f} s", transition_report->trigger_time);
    std::println("    Critical element      : {}", transition_report->critical_element);
    std::println("    Damage index          : {:.6f}", transition_report->metric_value);

    // ─────────────────────────────────────────────────────────────────────
    //  10. Identify top-N critical column elements
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[10] Identifying {} most-damaged column elements...", N_CRITICAL);

    std::vector<ElementDamageInfo> current_damages;
    current_damages.reserve(model.elements().size());
    for (std::size_t i = 0; i < model.elements().size(); ++i) {
        auto info = damage_crit.evaluate_element(
            model.elements()[i], i, model.state_vector());
        current_damages.push_back(info);
    }
    std::sort(current_damages.begin(), current_damages.end(), std::greater<>{});

    std::vector<std::size_t> crit_elem_ids;
    for (const auto& di : current_damages) {
        const auto& se = model.elements()[di.element_index];
        if (se.as<BeamElemT>() && di.damage_index > 0.8) {
            crit_elem_ids.push_back(di.element_index);
            if (crit_elem_ids.size() >= N_CRITICAL) break;
        }
    }
    if (crit_elem_ids.empty()) {
        crit_elem_ids.push_back(transition_report->critical_element);
    }

    std::println("  Critical elements for sub-model analysis:");
    for (auto eid : crit_elem_ids) {
        double di = 0;
        for (const auto& d : current_damages)
            if (d.element_index == eid) { di = d.damage_index; break; }
        int r = elem_to_range.count(eid) ? elem_to_range.at(eid) : 0;
        std::println("    element {}  —  damage_index = {:.6f}  range = {} ({})",
                     eid, di, r, COL_GROUPS[r]);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  11. Export global frame VTK at yield time
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[11] Exporting global frame VTK at first yield...");

    {
        fall_n::vtk::StructuralVTMExporter vtm{
            model,
            fall_n::reconstruction::RectangularSectionProfile<2>{COL_B[0], COL_H[0]},
            fall_n::reconstruction::ShellThicknessProfile<5>{}
        };
        vtm.set_displacement(model.state_vector());
        vtm.set_yield_strain(EPS_YIELD);
        vtm.write(OUT + "yield_state.vtm");
        std::println("  Written: {}yield_state.vtm", OUT);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  12. Extract element kinematics + build sub-models (per range)
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[12] Extracting kinematics and building Hex27 sub-models...");

    auto extract_beam_kinematics = [&](std::size_t e_idx) -> ElementKinematics {
        const auto& se = model.elements()[e_idx];
        const auto* beam = se.as<BeamElemT>();
        const auto u_e = se.extract_element_dofs(model.state_vector());

        int r = elem_to_range.count(e_idx) ? elem_to_range.at(e_idx) : 0;

        auto kin_A = extract_section_kinematics(*beam, u_e, -1.0);
        auto kin_B = extract_section_kinematics(*beam, u_e, +1.0);
        kin_A.E = EC_RANGE[r]; kin_A.G = GC_RANGE[r]; kin_A.nu = NU_RC;
        kin_B.E = EC_RANGE[r]; kin_B.G = GC_RANGE[r]; kin_B.nu = NU_RC;

        ElementKinematics ek;
        ek.element_id   = e_idx;
        ek.kin_A        = kin_A;
        ek.kin_B        = kin_B;
        ek.endpoint_A   = beam->geometry().map_local_point(std::array{-1.0});
        ek.endpoint_B   = beam->geometry().map_local_point(std::array{+1.0});
        ek.up_direction = std::array<double,3>{1.0, 0.0, 0.0};
        return ek;
    };

    // Group critical elements by story range
    std::map<int, std::vector<std::size_t>> crit_by_range;
    for (auto eid : crit_elem_ids)
        crit_by_range[elem_to_range.count(eid) ? elem_to_range.at(eid) : 0].push_back(eid);

    // Build one coordinator per active range
    std::map<int, MultiscaleCoordinator> coordinators;
    for (auto& [range, eids] : crit_by_range) {
        auto& coord = coordinators[range];
        for (auto eid : eids) {
            if (!model.elements()[eid].as<BeamElemT>()) continue;
            coord.add_critical_element(extract_beam_kinematics(eid));
        }

        // Build rebar bar layout mirroring the structural fiber section.
        // 8 bars: 4 corner + 4 mid-face, matching make_rc_column_section.
        const double cvr = COL_CVR;
        const double bar_d = COL_BAR;
        const double bar_area = std::numbers::pi / 4.0 * bar_d * bar_d;
        const double b = COL_B[range], h = COL_H[range];
        const double y0 = -b / 2.0 + cvr + bar_d / 2.0;
        const double y1 =  b / 2.0 - cvr - bar_d / 2.0;
        const double z0 = -h / 2.0 + cvr + bar_d / 2.0;
        const double z1 =  h / 2.0 - cvr - bar_d / 2.0;

        std::vector<SubModelSpec::RebarBar> bars = {
            {y0, z0, bar_area, bar_d}, {y1, z0, bar_area, bar_d},
            {y0, z1, bar_area, bar_d}, {y1, z1, bar_area, bar_d},
            {0.0, z0, bar_area, bar_d}, {0.0, z1, bar_area, bar_d},
            {y0, 0.0, bar_area, bar_d}, {y1, 0.0, bar_area, bar_d},
        };

        coord.build_sub_models(SubModelSpec{
            .section_width  = COL_B[range],
            .section_height = COL_H[range],
            .nx = SUB_NX,
            .ny = SUB_NY,
            .nz = SUB_NZ,
            .hex_order = HexOrder::Quadratic,
            .rebar_bars = std::move(bars),
            .rebar_E  = STEEL_E,
            .rebar_fy = STEEL_FY,
            .rebar_b  = STEEL_B,
        });
        const auto rpt = coord.report();
        std::println("  Range {} ({}) : {} sub-models, {} nodes, {} hex elements",
                     range, COL_GROUPS[range],
                     rpt.num_elements, rpt.total_nodes, rpt.total_elements);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  13. Create nonlinear sub-model evolvers
    // ─────────────────────────────────────────────────────────────────────
    sep('=');
    std::println("\n[13] Creating nonlinear sub-model evolvers...");

    const std::string evol_sub_dir = OUT + "evolution/sub_models";

    std::vector<NonlinearSubModelEvolver> nl_evolvers;
    for (auto& [range, coord] : coordinators) {
        for (auto& sub : coord.sub_models()) {
            nl_evolvers.emplace_back(
                sub, COL_FPC[range], evol_sub_dir, EVOL_VTK_INTERVAL);
            nl_evolvers.back().set_incremental_params(8, 6);
        }
    }

    std::println("  Nonlinear evolvers : {}", nl_evolvers.size());
    std::println("  Constitutive model : KoBatheConcrete3D (per-range f'c)");
    std::println("  Material state     : PERSISTENT across all time steps");
    std::println("  VTK interval       : every {} steps ({:.2f} s)",
                 EVOL_VTK_INTERVAL, EVOL_VTK_INTERVAL * DT);

    // ── Assemble MultiscaleModel + MultiscaleAnalysis ────────────────────
    using MacroBridge = BeamMacroBridge<StructModel, BeamElemT>;
    MultiscaleModel<MacroBridge, NonlinearSubModelEvolver> ms_model{
        MacroBridge{model}};

    for (auto& ev : nl_evolvers) {
        const auto eid = ev.parent_element_id();
        ms_model.register_local_model(
            ms_model.macro_bridge().default_site(eid),
            std::move(ev));
    }

    // Section dimensions for homogenization scaling: use first critical element
    const int first_range = crit_by_range.begin()->first;

    MultiscaleAnalysis<
        DynSolver,
        MacroBridge,
        NonlinearSubModelEvolver,
        OpenMPExecutor> analysis(
        solver,
        std::move(ms_model),
        std::make_unique<IteratedTwoWayFE2>(MAX_STAGGERED_ITER),
        std::make_unique<ForceAndTangentConvergence>(
            STAGGERED_TOL, STAGGERED_TOL),
        std::make_unique<ConstantRelaxation>(STAGGERED_RELAX),
        OpenMPExecutor{});
    analysis.set_coupling_start_step(COUPLING_START_STEP);
    analysis.set_section_dimensions(COL_B[first_range], COL_H[first_range]);

    std::println("  MultiscaleAnalysis : IteratedTwoWayFE2, max_iter={}, "
                 "force/tangent tol={:.2f}, relax={:.2f}",
                 MAX_STAGGERED_ITER, STAGGERED_TOL, STAGGERED_RELAX);
    std::println("  Section dims (hom) : {:.2f} × {:.2f} m (range {})",
                 COL_B[first_range], COL_H[first_range], first_range);

    // ─────────────────────────────────────────────────────────────────────
    //  14. Initial sub-model solve at yield time
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[14] Initial nonlinear solve at t={:.4f} s...",
                 transition_report->trigger_time);

    const bool init_ok = analysis.initialize_local_models();
    std::println("  Local-state initialization : {} (failed sub-models = {})",
                 init_ok ? "READY" : "FAILED",
                 analysis.last_report().failed_submodels);
    if (!init_ok) {
        PetscFinalize();
        return 1;
    }

    // ─────────────────────────────────────────────────────────────────────
    //  15. Phase 2: Resume global + evolve sub-models step-by-step
    // ─────────────────────────────────────────────────────────────────────
    sep('=');
    std::println("\n[15] PHASE 2: Sub-model evolution through the earthquake");
    std::println("     Evolving global + {} nonlinear sub-models simultaneously...",
                 analysis.model().num_local_models());

    const std::string evol_frame_dir = OUT + "evolution";
    PVDWriter pvd_global(evol_frame_dir + "/frame");

    auto beam_profile  = fall_n::reconstruction::RectangularSectionProfile<2>{COL_B[0], COL_H[0]};
    auto shell_profile = fall_n::reconstruction::ShellThicknessProfile<5>{};

    int    evol_step       = 0;
    double evol_max_damage = peak_damage_global;
    int    total_cracks    = 0;

    // ── Crack evolution CSV ────────────────────────────────────────────
    std::ofstream crack_csv(OUT + "recorders/crack_evolution.csv");
    crack_csv << "time,total_cracked_gps,total_cracks,damage_scalar_available,max_damage_scalar,max_opening";
    for (std::size_t i = 0; i < analysis.model().num_local_models(); ++i)
        crack_csv << ",sub" << i << "_cracked_gps"
                  << ",sub" << i << "_cracks"
                  << ",sub" << i << "_damage_scalar_available"
                  << ",sub" << i << "_max_damage_scalar";
    crack_csv << "\n";

    // Phase 2: reset dt to nominal and disable adaptation for stable FE² coupling.
    {
        TS ts = solver.get_ts();
        TSSetMaxTime(ts, T_MAX);
        TSSetTimeStep(ts, DT);
        TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER);
        TSAdapt adapt;
        TSGetAdapt(ts, &adapt);
        TSAdaptSetType(adapt, TSADAPTNONE);
        PetscReal dt_current;
        TSGetTimeStep(ts, &dt_current);
        std::println("  [TS] Phase 2 dt reset to {:.6f} s", static_cast<double>(dt_current));
        std::cout << std::flush;
    }

    for (;;) {
        PetscReal t_current;
        TSGetTime(solver.get_ts(), &t_current);
        if (static_cast<double>(t_current) >= T_MAX - 1e-14) break;

        // Advance one global step
        if (!analysis.step()) {
            std::println("  [!] Multiscale step failed at t={:.4f} s",
                         solver.current_time());
            break;
        }

        ++evol_step;
        const double t = solver.current_time();

        // ── Iterated two-way FE² coupling ───────────────────────────
        const int staggered_iters = analysis.last_staggered_iterations();

        // ── End-of-step: crack collection + VTK (once per global step) ─

        // ── Collect crack summaries ─────────────────────────────────
        int    step_cracked_gps          = 0;
        int    step_total_cracks         = 0;
        bool   step_damage_scalar_available = false;
        double step_max_crack_damage =
            std::numeric_limits<double>::quiet_NaN();
        double step_max_opening          = 0.0;
        std::vector<CrackSummary> step_summaries;
        for (auto& ev : analysis.model().local_models()) {
            auto cs = ev.crack_summary();
            step_summaries.push_back(cs);
            step_cracked_gps  += cs.num_cracked_gps;
            step_total_cracks += cs.total_cracks;
            if (cs.damage_scalar_available) {
                if (!step_damage_scalar_available) {
                    step_max_crack_damage = cs.max_damage_scalar;
                } else {
                    step_max_crack_damage = std::max(
                        step_max_crack_damage, cs.max_damage_scalar);
                }
                step_damage_scalar_available = true;
            }
            step_max_opening      = std::max(step_max_opening, cs.max_opening);
        }
        total_cracks = step_total_cracks;

        // ── Write crack evolution CSV row ────────────────────────────
        crack_csv << std::fixed << std::setprecision(4) << t
                  << "," << step_cracked_gps
                  << "," << step_total_cracks
                  << "," << (step_damage_scalar_available ? 1 : 0)
                  << std::scientific << std::setprecision(6)
                  << "," << csv_scalar_or_nan(
                                 step_damage_scalar_available,
                                 step_max_crack_damage)
                  << "," << step_max_opening;
        for (const auto& cs : step_summaries)
            crack_csv << "," << cs.num_cracked_gps
                      << "," << cs.total_cracks
                      << "," << (cs.damage_scalar_available ? 1 : 0)
                      << "," << csv_scalar_or_nan(
                                     cs.damage_scalar_available,
                                     cs.max_damage_scalar);
        crack_csv << "\n" << std::flush;

        // ── Track peak damage ───────────────────────────────────────
        double step_max_damage = 0.0;
        for (std::size_t e = 0; e < model.elements().size(); ++e) {
            auto info = damage_crit.evaluate_element(
                model.elements()[e], e, model.state_vector());
            step_max_damage = std::max(step_max_damage, info.damage_index);
        }
        evol_max_damage = std::max(evol_max_damage, step_max_damage);

        // ── Global VTK snapshot (expensive; rarely) ────────────────────
        if (evol_step % FRAME_VTK_INTERVAL == 0) {
            const auto vtm_file = std::format("{}/frame_{:06d}.vtm",
                                              evol_frame_dir, evol_step);
            fall_n::vtk::StructuralVTMExporter vtm{model, beam_profile, shell_profile};
            vtm.set_displacement(model.state_vector());
            vtm.set_yield_strain(EPS_YIELD);
            vtm.write(vtm_file);
            pvd_global.add_timestep(t, vtm_file);
        }

        // ── Global history CSV + progress (Phase 2) ────────────────
        PetscReal u_norm2 = 0.0;
        VecNorm(model.state_vector(), NORM_INFINITY, &u_norm2);

        global_csv << std::fixed << std::setprecision(6) << t
                   << "," << evol_step
                   << ",2,"
                   << std::scientific << std::setprecision(6)
                   << static_cast<double>(u_norm2)
                   << "," << evol_max_damage
                   << "\n" << std::flush;

        if (evol_step % EVOL_PRINT_INTERVAL == 0 || evol_step <= 3) {
            std::println("    [FE²] step={:4d}  t={:.3f} s  |u|={:.3e} m  "
                         "damage={:.4f}  cracks={}  stag_iter={}",
                         evol_step, t, static_cast<double>(u_norm2),
                         evol_max_damage, total_cracks, staggered_iters);
            std::cout << std::flush;
        }
    }

    crack_csv.close();
    global_csv.close();

    // ─────────────────────────────────────────────────────────────────────
    //  16. Final frame VTK + finalize
    // ─────────────────────────────────────────────────────────────────────
    // One global frame at the final evolution state:
    {
        PetscReal t_end;
        TSGetTime(solver.get_ts(), &t_end);
        const auto vtm_file = std::format("{}/frame_final.vtm", evol_frame_dir);
        fall_n::vtk::StructuralVTMExporter vtm{model, beam_profile, shell_profile};
        vtm.set_displacement(model.state_vector());
        vtm.set_yield_strain(EPS_YIELD);
        vtm.write(vtm_file);
        pvd_global.add_timestep(static_cast<double>(t_end), vtm_file);
        std::println("\n  [VTK] Final frame written: {}", vtm_file);
        std::cout << std::flush;
    }

    pvd_global.write();
    for (auto& ev : analysis.model().local_models())
        ev.finalize();

    const std::string rec_dir = OUT + "recorders/";
    composite.template get<2>().write_csv(rec_dir + "roof_displacement.csv");
    composite.template get<1>().write_hysteresis_csv(rec_dir + "fiber_hysteresis");

    PetscReal t_final;
    TSGetTime(solver.get_ts(), &t_final);

    // ─────────────────────────────────────────────────────────────────────
    //  17. Python postprocessing
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[17] Running Python postprocessing...");
    {
        fall_n::PythonPlotter plotter(BASE + "scripts/falln_postprocess.py");
        int rc = plotter.plot(rec_dir, BASE + "doc/figures/lshaped_multiscale_16/");
        if (rc == 0)
            std::println("  Plots generated successfully.");
        else
            std::println("  [!] Python plotter returned code {}", rc);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  18. Summary
    // ─────────────────────────────────────────────────────────────────────
    sep('=');
    std::println("\n  16-STORY L-SHAPED MULTISCALE SEISMIC ANALYSIS — SUMMARY");
    sep('-');
    std::println("  Earthquake:      Tohoku 2011, MYG004 (NS+EW+UD), scale={:.2f}", eq_scale);
    std::println("  Structure:       {}-story L-shaped RC frame, {} × {} m plan",
                 NUM_STORIES, X_GRID.back(), Y_GRID.back());
    std::println("  Height irreg.:   Lower {}×{} / Mid {}×{} / Upper {}×{}",
                 COL_B[0], COL_H[0], COL_B[1], COL_H[1], COL_B[2], COL_H[2]);
    std::println("  Beams:           {}×{} m, f'c={} MPa", BM_B, BM_H, BM_FPC);
    std::println("  First yield:     t = {:.4f} s  (element {})",
                 transition_report->trigger_time,
                 transition_report->critical_element);
    std::println("  Sub-models:      {} (Hex27, KoBatheConcrete3D, per-range f'c)",
                 analysis.model().num_local_models());
    std::println("  Evolution:       {} steps, {:.1f} s — t_final = {:.4f} s",
                 evol_step, evol_step * DT, static_cast<double>(t_final));
    std::println("  Peak damage:     {:.6f}", evol_max_damage);
    std::println("  Active cracks:   {} (across all sub-models at final step)", total_cracks);
    std::println("  Output dir:      {}", OUT);
    sep('=');

    PetscFinalize();
    return 0;
}
