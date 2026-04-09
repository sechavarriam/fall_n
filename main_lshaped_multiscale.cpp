// =============================================================================
//  main_lshaped_multiscale.cpp
// =============================================================================
//
//  L-shaped 4-story RC building — 3-component seismic multiscale analysis
//
//  This example demonstrates the full fall_n multiscale seismic pipeline
//  for a doctoral-level case study:
//
//    Phase 1 (global, nonlinear fiber):
//      A 4-story L-shaped RC frame is subjected to the 2011 Tohoku earthquake
//      (K-NET station MYG004, Tsukidate) using all three components (NS, EW,
//      UD).  Columns and beams use fibered RC sections (Kent-Park concrete
//      + Menegotto-Pinto steel).  A TransitionDirector monitors damage at
//      every step and pauses the instant any steel fiber reaches yield.
//
//    Phase 2 (local, nonlinear RC continuum sub-model evolution):
//      After the transition, the top-3 most-damaged column elements are
//      identified via DamageTracker::peak_ranking().  For each critical
//      RC column, a prismatic Hex27 sub-model is built by MultiscaleCoordinator.
//      The KoBatheConcrete3D constitutive model captures cracking, crushing,
//      and damage evolution of the concrete matrix, while longitudinal and
//      transverse reinforcement effects operate at the structural scale
//      through the fiber sections.  The sub-models are evolved through the
//      rest of the earthquake using NonlinearSubModelEvolver, which persists
//      material state across all time steps so that plastic strains, crack
//      patterns, and damage accumulate realistically.
//
//  Outputs
//  -------
//  data/output/lshaped_multiscale/
//    yield_state.vtm               -- global frame at first yield
//    evolution/frame_NNNNNN.vtm    -- global frame time series
//    evolution/frame.pvd           -- ParaView PVD for frame animation
//    evolution/sub_models/         -- per-element PVD series:
//      nlsub_{id}_mesh.pvd           mesh (deformed shape + material fields)
//      nlsub_{id}_gauss.pvd          Gauss-point cloud (stress, strain, cracks)
//      nlsub_{id}_cracks.pvd         crack plane glyphs
//    recorders/
//      roof_displacement.csv         nodal displacement histories
//      fiber_hysteresis_concrete.csv top-5 concrete fiber hysteresis
//      fiber_hysteresis_steel.csv    top-5 steel fiber hysteresis
//      crack_evolution.csv           crack count/damage vs time (per sub-model)
//
//  Units: [m, MN, MPa = MN/m², MN·s²/m⁴, s]
//
// =============================================================================

#include "header_files.hh"

#include <Eigen/Dense>

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <print>
#include <string>
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
static const std::string OUT    = BASE + "data/output/lshaped_multiscale/";

// ── L-shaped frame geometry ─────────────────────────────────────────────────
//  3 bays in X (5 m each), 2 bays in Y (4 m each)
//  Cutout removes upper-right bay at all stories → L-shape in plan
static constexpr std::array<double, 4> X_GRID = {0.0, 5.0, 10.0, 15.0};
static constexpr std::array<double, 3> Y_GRID = {0.0, 4.0, 8.0};
static constexpr int    NUM_STORIES  = 4;
static constexpr double STORY_HEIGHT = 3.2;   // m

// Cutout: bay indices (ix=2, iy=1) → removes upper-right corner at all stories
static constexpr int CUTOUT_X_START = 2;
static constexpr int CUTOUT_X_END   = 3;
static constexpr int CUTOUT_Y_START = 1;
static constexpr int CUTOUT_Y_END   = 2;
static constexpr int CUTOUT_ABOVE   = 0;  // from story 0 upward = all stories

// ── Column section 0.40 × 0.40 m ───────────────────────────────────────────
static constexpr double COL_B    = 0.40;
static constexpr double COL_H    = 0.40;
static constexpr double COL_CVR  = 0.04;
static constexpr double COL_BAR  = 0.020;
static constexpr double COL_TIE  = 0.10;

// ── Beam section 0.30 × 0.50 m ─────────────────────────────────────────────
static constexpr double BM_B   = 0.30;
static constexpr double BM_H   = 0.50;
static constexpr double BM_CVR = 0.04;
static constexpr double BM_BAR = 0.016;

// ── RC materials ─────────────────────────────────────────────────────────────
static constexpr double COL_FPC  = 28.0;    // f'c columns [MPa]
static constexpr double BM_FPC   = 25.0;    // f'c beams   [MPa]
static constexpr double STEEL_E  = 200000.0;// Es [MPa]
static constexpr double STEEL_FY = 420.0;   // fy [MPa]
static constexpr double STEEL_B  = 0.01;    // hardening ratio
static constexpr double TIE_FY   = 420.0;   // tie yield [MPa]
static constexpr double NU_RC    = 0.20;

// Effective elastic modulus (ACI 318: Ec = 4700√f'c)
static const double EC_COL = 4700.0 * std::sqrt(COL_FPC);
static const double GC_COL = EC_COL / (2.0 * (1.0 + NU_RC));

// ── Damage / transition ──────────────────────────────────────────────────────
static constexpr double EPS_YIELD = STEEL_FY / STEEL_E;  // ≈ 0.0021

// ── Mass + Rayleigh damping  (5 % at T₁≈0.60 s, T₃≈0.15 s) ────────────────
static constexpr double RC_DENSITY = 2.4e-3;   // MN·s²/m⁴
static constexpr double XI_DAMP    = 0.05;
static const double OMEGA_1 = 2.0 * std::numbers::pi / 0.60;
static const double OMEGA_3 = 2.0 * std::numbers::pi / 0.15;

// ── Time integration ─────────────────────────────────────────────────────────
static constexpr double DT      = 0.02;    // s
static constexpr double T_SKIP  = 25.0;    // skip pre-event noise
static constexpr double T_MAX   = 275.0;   // analysis window after skip (full strong motion)
static constexpr double EQ_SCALE = 1.0;    // no scaling — full PGA for realism

// ── Sub-model mesh ───────────────────────────────────────────────────────────
static constexpr int SUB_NX = 4;
static constexpr int SUB_NY = 4;
static constexpr int SUB_NZ = 8;

// ── Sub-model evolution ────────────────────────────────────────────────────────
static constexpr int EVOL_VTK_INTERVAL   = 10;    // VTK every 10 steps (= 0.2 s)
static constexpr int EVOL_PRINT_INTERVAL = 100;   // progress every 100 steps (= 2.0 s)

// ── FE² two-way coupling ─────────────────────────────────────────────────────
static constexpr int    MAX_STAGGERED_ITER  = 4;      // max coupling iterations per step
static constexpr double STAGGERED_TOL       = 0.05;   // 5% relative Frobenius norm
static constexpr double STAGGERED_RELAX     = 0.7;    // relaxation factor ω ∈ (0,1]
static constexpr int    COUPLING_START_STEP = 10;      // begin coupling after 10 evol steps

// ── Type aliases (small-strain, NDOF=6) ─────────────────────────────────────
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
    std::println("  fall_n — L-Shaped RC Building: 3-Component Seismic Multiscale");
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
    std::println("  Window        : [{:.0f} s, {:.0f} s] of original {:.1f} s",
                 T_SKIP, T_SKIP + T_MAX, eq_ns_full.duration());
    std::println("  PGA (NS)      : {:.3f} m/s² ({:.3f} g) at t={:.2f} s",
                 eq_ns.pga(), eq_ns.pga() / 9.81, eq_ns.time_of_pga());
    std::println("  PGA (EW)      : {:.3f} m/s² ({:.3f} g) at t={:.2f} s",
                 eq_ew.pga(), eq_ew.pga() / 9.81, eq_ew.time_of_pga());
    std::println("  PGA (UD)      : {:.3f} m/s² ({:.3f} g) at t={:.2f} s",
                 eq_ud.pga(), eq_ud.pga() / 9.81, eq_ud.time_of_pga());
    std::println("  Scale factor  : {:.2f}", eq_scale);

    // ─────────────────────────────────────────────────────────────────────
    //  2. Building domain: 4-story L-shaped RC frame
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[2] Building L-shaped structural domain...");

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
        .include_slabs   = false,   // frame-only for computational speed
    });

    std::println("  Plan          : {} bays X × {} bays Y, L-shape with cutout",
                 X_GRID.size()-1, Y_GRID.size()-1);
    std::println("  Stories       : {} × {:.1f} m = {:.1f} m total height",
                 NUM_STORIES, STORY_HEIGHT, NUM_STORIES * STORY_HEIGHT);
    std::println("  Nodes         : {}", domain.num_nodes());
    std::println("  Columns       : {}", grid.num_columns());
    std::println("  Beams         : {}", grid.num_beams());

    // ─────────────────────────────────────────────────────────────────────
    //  3. RC fiber sections
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[3] Building RC fiber sections...");

    const auto col_mat = make_rc_column_section({
        .b            = COL_B,
        .h            = COL_H,
        .cover        = COL_CVR,
        .bar_diameter = COL_BAR,
        .tie_spacing  = COL_TIE,
        .fpc          = COL_FPC,
        .nu           = NU_RC,
        .steel_E      = STEEL_E,
        .steel_fy     = STEEL_FY,
        .steel_b      = STEEL_B,
        .tie_fy       = TIE_FY,
    });

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

    std::println("  Columns : {}×{} m, f'c={} MPa, confined Kent-Park + M-P steel",
                 COL_B, COL_H, COL_FPC);
    std::println("  Beams   : {}×{} m, f'c={} MPa, unconfined + M-P steel",
                 BM_B, BM_H, BM_FPC);

    // ─────────────────────────────────────────────────────────────────────
    //  4. Structural model
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[4] Assembling structural model (frame-only)...");

    std::vector<const ElementGeometry<3>*> shell_geoms;
    std::vector<StructuralElement> elements =
        StructuralModelBuilder<
            BeamElemT, ShellElemT,
            TimoshenkoBeam3D, MindlinReissnerShell3D>{}
            .set_frame_material("Columns", col_mat)
            .set_frame_material("Beams",   bm_mat)
            .build_elements(domain, &shell_geoms);

    StructModel model{domain, std::move(elements)};
    model.fix_z(0.0);
    model.setup();

    std::println("  Total structural elements : {}", model.elements().size());

    // ─────────────────────────────────────────────────────────────────────
    //  5. Dynamic solver: density, Rayleigh damping, 3-component ground motion
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[5] Configuring dynamic solver (3-component)...");

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
    std::println("  Components       : NS (X) + EW (Y) + UD (Z)");

    // ─────────────────────────────────────────────────────────────────────
    //  6. Observers: damage tracker + fiber hysteresis recorder + node recorder
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[6] Setting up observers...");

    MaxStrainDamageCriterion damage_crit{EPS_YIELD};

    // Damage tracker: keep top-10 elements
    DamageTracker<StructModel> damage_tracker{damage_crit, 1, 10};

    // Fiber hysteresis recorder: top-5 per material class
    // Classifier: steel fibers have area < 0.001 m² (typical bar ~300 mm²)
    auto fiber_classifier = [](std::size_t, std::size_t, std::size_t,
                               double, double, double area) -> FiberMaterialClass {
        return (area < 0.001) ? FiberMaterialClass::Steel
                              : FiberMaterialClass::Concrete;
    };
    FiberHysteresisRecorder<StructModel> hysteresis_rec{
        damage_crit, fiber_classifier, {}, 5, 1};

    // Node recorder: roof displacement at top corner nodes
    //  The top level = NUM_STORIES, so z = NUM_STORIES * STORY_HEIGHT
    //  We track DOFs 0 (X), 1 (Y) at a few top nodes
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

    // Composite observer
    auto composite = make_composite_observer<StructModel>(
        std::move(damage_tracker), std::move(hysteresis_rec), std::move(node_rec));
    solver.set_observer(composite);

    std::println("  DamageTracker         : top-10, every step");
    std::println("  FiberHysteresisRecorder : top-5/material, every step");
    std::println("  NodeRecorder           : {} channels (roof nodes)", disp_channels.size());

    // ─────────────────────────────────────────────────────────────────────
    //  7. Transition director: pause at first fiber yield
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[7] Configuring damage-threshold transition director...");

    auto [director, transition_report] =
        make_damage_threshold_director<StructModel>(damage_crit, 1.0);

    std::println("  Criterion  : MaxStrain (ε_ref = {:.6f})", EPS_YIELD);
    std::println("  Threshold  : damage_index > 1.0 (i.e. ε > ε_y)");

    // ─────────────────────────────────────────────────────────────────────
    //  8. Phase 1: Global nonlinear dynamic analysis until first yield
    // ─────────────────────────────────────────────────────────────────────
    sep('=');
    std::println("\n[8] PHASE 1: Global fiber-section dynamic analysis");
    std::println("    Running until first steel fiber reaches yield...");

    PetscOptionsSetValue(nullptr, "-snes_max_it", "100");
    PetscOptionsSetValue(nullptr, "-snes_rtol",   "1e-3");
    PetscOptionsSetValue(nullptr, "-snes_atol",   "1e-6");
    PetscOptionsSetValue(nullptr, "-ksp_max_it",  "200");

    solver.setup();
    solver.set_time_step(DT);

    {
        TS ts = solver.get_ts();
        TSAlpha2SetRadius(ts, 0.9);
        TSAdapt adapt;
        TSGetAdapt(ts, &adapt);
        TSAdaptSetType(adapt, TSADAPTNONE);
        TSSetTimeStep(ts, DT);
        TSSetMaxSNESFailures(ts, -1);
        SNES snes;
        TSGetSNES(ts, &snes);
        SNESSetTolerances(snes, 1e-6, 1e-3, PETSC_DETERMINE, 100, PETSC_DETERMINE);
        KSP ksp;
        SNESGetKSP(snes, &ksp);
        KSPSetTolerances(ksp, PETSC_DETERMINE, PETSC_DETERMINE, PETSC_DETERMINE, 200);
    }

    // Diagnostic wrapper for the transition director
    double peak_damage_global = 0.0;
    fall_n::StepDirector<StructModel> phase1_director =
        [&director, &peak_damage_global, &damage_crit]
        (const fall_n::StepEvent& ev, const StructModel& m) -> fall_n::StepVerdict
    {
        double max_d = 0.0;
        for (std::size_t e = 0; e < m.elements().size(); ++e) {
            auto info = damage_crit.evaluate_element(m.elements()[e], e, m.state_vector());
            max_d = std::max(max_d, info.damage_index);
        }
        peak_damage_global = std::max(peak_damage_global, max_d);

        if (ev.step % 500 == 0) {
            PetscReal u_norm = 0.0;
            VecNorm(ev.displacement, NORM_INFINITY, &u_norm);
            std::println("    t={:.2f} s  step={}  |u|_inf={:.3e} m  peak_damage={:.6e}",
                         ev.time, ev.step, u_norm, peak_damage_global);
        }
        return director(ev, m);
    };

    solver.step_to(T_MAX, phase1_director);

    sep('-');
    if (!transition_report->triggered) {
        std::println("[!] No fiber yielding detected within {} s.", T_MAX);
        std::println("    Peak damage = {:.4f} — try larger scale.", peak_damage_global);
        PetscFinalize();
        return 0;
    }

    std::println("\n[*] YIELDING DETECTED");
    std::println("    Time of first yield   : {:.4f} s", transition_report->trigger_time);
    std::println("    Critical element      : {}", transition_report->critical_element);
    std::println("    Damage index          : {:.6f}", transition_report->metric_value);

    // ─────────────────────────────────────────────────────────────────────
    //  9. Identify top-3 critical column elements
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[9] Identifying most-damaged elements...");

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
            if (crit_elem_ids.size() >= 3) break;
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
        std::println("    element {}  —  damage_index = {:.6f}", eid, di);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  10. Export global frame VTK at yield time
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[10] Exporting global frame VTK at first yield...");
    std::filesystem::create_directories(OUT);
    std::filesystem::create_directories(OUT + "evolution/sub_models/");
    std::filesystem::create_directories(OUT + "recorders/");

    {
        fall_n::vtk::StructuralVTMExporter vtm{
            model,
            fall_n::reconstruction::RectangularSectionProfile<2>{COL_B, COL_H},
            fall_n::reconstruction::ShellThicknessProfile<5>{}
        };
        vtm.set_displacement(model.state_vector());
        vtm.set_yield_strain(EPS_YIELD);
        vtm.write(OUT + "yield_state.vtm");
        std::println("  Written: {}yield_state.vtm", OUT);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  11. Extract element kinematics + build sub-models
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[11] Extracting kinematics and building Hex27 (RC) sub-models...");

    auto extract_beam_kinematics = [&](std::size_t e_idx) -> ElementKinematics {
        const auto& se = model.elements()[e_idx];
        const auto* beam = se.as<BeamElemT>();
        const auto u_e = se.extract_element_dofs(model.state_vector());

        auto kin_A = extract_section_kinematics(*beam, u_e, -1.0);
        auto kin_B = extract_section_kinematics(*beam, u_e, +1.0);
        kin_A.E = EC_COL; kin_A.G = GC_COL; kin_A.nu = NU_RC;
        kin_B.E = EC_COL; kin_B.G = GC_COL; kin_B.nu = NU_RC;

        ElementKinematics ek;
        ek.element_id  = e_idx;
        ek.kin_A       = kin_A;
        ek.kin_B       = kin_B;
        ek.endpoint_A  = beam->geometry().map_local_point(std::array{-1.0});
        ek.endpoint_B  = beam->geometry().map_local_point(std::array{+1.0});
        ek.up_direction = std::array<double,3>{1.0, 0.0, 0.0};
        return ek;
    };

    MultiscaleCoordinator coordinator;
    for (auto e_idx : crit_elem_ids) {
        if (!model.elements()[e_idx].as<BeamElemT>()) continue;
        coordinator.add_critical_element(extract_beam_kinematics(e_idx));
    }

    coordinator.build_sub_models(SubModelSpec{
        .section_width  = COL_B,
        .section_height = COL_H,
        .nx = SUB_NX,
        .ny = SUB_NY,
        .nz = SUB_NZ,
        .hex_order = HexOrder::Quadratic,  // Hex27
    });

    const auto ms_report = coordinator.report();
    std::println("  Sub-models built   : {}", ms_report.num_elements);
    std::println("  Total nodes        : {}", ms_report.total_nodes);
    std::println("  Total elements     : {}", ms_report.total_elements);
    std::println("  Element type       : Hex27 (triquadratic Lagrange)");

    // ─────────────────────────────────────────────────────────────────────
    //  12. Create NONLINEAR sub-model evolvers (persistent material state)
    // ─────────────────────────────────────────────────────────────────────
    sep('=');
    std::println("\n[12] Creating nonlinear sub-model evolvers...");

    const std::string evol_sub_dir = OUT + "evolution/sub_models";

    std::vector<NonlinearSubModelEvolver> nl_evolvers;
    {
        auto& subs = coordinator.sub_models();
        for (auto& sub : subs) {
            nl_evolvers.emplace_back(
                sub, COL_FPC, evol_sub_dir, EVOL_VTK_INTERVAL);
            nl_evolvers.back().set_incremental_params(15, 6);
        }
    }

    std::println("  Nonlinear evolvers : {}", nl_evolvers.size());
    std::println("  Constitutive model : KoBatheConcrete3D (f'c={} MPa)", COL_FPC);
    std::println("  RC treatment       : concrete matrix in 3D, rebar at fiber-section scale");
    std::println("  Material state     : PERSISTENT across all time steps");
    std::println("  VTK interval       : every {} steps ({:.2f} s)",
                 EVOL_VTK_INTERVAL, EVOL_VTK_INTERVAL * DT);

    // ── Assemble MultiscaleModel + MultiscaleAnalysis orchestrator ───────
    using MacroBridge = BeamMacroBridge<StructModel, BeamElemT>;
    MultiscaleModel<MacroBridge, NonlinearSubModelEvolver> ms_model{
        MacroBridge{model}};

    for (auto& ev : nl_evolvers) {
        const auto eid = ev.parent_element_id();
        ms_model.register_local_model(
            ms_model.macro_bridge().default_site(eid),
            std::move(ev));
    }

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
    analysis.set_section_dimensions(COL_B, COL_H);

    std::println("  MultiscaleAnalysis : IteratedTwoWayFE2, max_iter={}, "
                 "force/tangent tol={:.2f}, relax={:.2f}",
                 MAX_STAGGERED_ITER, STAGGERED_TOL, STAGGERED_RELAX);

    // ─────────────────────────────────────────────────────────────────────
    //  13. Initial sub-model solve at yield time
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[13] Initial nonlinear solve at t={:.4f} s...",
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
    //  14. Phase 2: Resume global + evolve sub-models step-by-step
    // ─────────────────────────────────────────────────────────────────────
    sep('=');
    std::println("\n[14] PHASE 2: Sub-model evolution through the earthquake");
    std::println("    Evolving global + {} nonlinear sub-models simultaneously...",
                 analysis.model().num_local_models());

    const std::string evol_frame_dir = OUT + "evolution";
    PVDWriter pvd_global(evol_frame_dir + "/frame");

    auto beam_profile  = fall_n::reconstruction::RectangularSectionProfile<2>{COL_B, COL_H};
    auto shell_profile = fall_n::reconstruction::ShellThicknessProfile<5>{};

    int    evol_step       = 0;
    double evol_max_damage = peak_damage_global;
    int    total_cracks    = 0;

    // ── Crack evolution CSV ────────────────────────────────────────────────────
    std::ofstream crack_csv(OUT + "recorders/crack_evolution.csv");
    crack_csv << "time,total_cracked_gps,total_cracks,damage_scalar_available,max_damage_scalar,max_opening";
    for (std::size_t i = 0; i < analysis.model().num_local_models(); ++i)
        crack_csv << ",sub" << i << "_cracked_gps"
                  << ",sub" << i << "_cracks"
                  << ",sub" << i << "_damage_scalar_available"
                  << ",sub" << i << "_max_damage_scalar";
    crack_csv << "\n";

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

        // ── Iterated two-way FE² coupling (delegated to MultiscaleAnalysis) ──
        const int staggered_iters = analysis.last_staggered_iterations();

        // ── Collect crack summaries (post-step) ──────────────────
        int step_cracked_gps    = 0;
        int step_total_cracks   = 0;
        bool step_damage_scalar_available = false;
        double step_max_crack_damage =
            std::numeric_limits<double>::quiet_NaN();
        double step_max_opening      = 0.0;
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

        // ── Write crack evolution CSV row ────────────────────────
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
        crack_csv << "\n";

        // ── Track peak damage ───────────────────────────────────────
        double step_max_damage = 0.0;
        for (std::size_t e = 0; e < model.elements().size(); ++e) {
            auto info = damage_crit.evaluate_element(
                model.elements()[e], e, model.state_vector());
            step_max_damage = std::max(step_max_damage, info.damage_index);
        }
        evol_max_damage = std::max(evol_max_damage, step_max_damage);

        // ── Global VTK snapshot ─────────────────────────────────────
        if (evol_step % EVOL_VTK_INTERVAL == 0) {
            const auto vtm_file = std::format("{}/frame_{:06d}.vtm",
                                              evol_frame_dir, evol_step);
            fall_n::vtk::StructuralVTMExporter vtm{model, beam_profile, shell_profile};
            vtm.set_displacement(model.state_vector());
            vtm.set_yield_strain(EPS_YIELD);
            vtm.write(vtm_file);
            pvd_global.add_timestep(t, vtm_file);
        }

        // ── Progress ────────────────────────────────────────────────
        if (evol_step % EVOL_PRINT_INTERVAL == 0) {
            std::println("  [Evolution] t={:.2f} s  step={:5d}  "
                         "peak_damage={:.4f}  cracks={}  FE2_iters={}",
                         t, evol_step, evol_max_damage, total_cracks,
                         staggered_iters);
        }
    }

    crack_csv.close();

    // ─────────────────────────────────────────────────────────────────────
    //  15. Finalize: write PVD files, export recorder CSV files
    // ─────────────────────────────────────────────────────────────────────
    pvd_global.write();
    for (auto& ev : analysis.model().local_models())
        ev.finalize();

    // Export recorder data
    const std::string rec_dir = OUT + "recorders/";
    // CompositeObserver stores: [0]=DamageTracker, [1]=FiberHysteresisRecorder, [2]=NodeRecorder
    composite.template get<2>().write_csv(rec_dir + "roof_displacement.csv");
    composite.template get<1>().write_hysteresis_csv(rec_dir + "fiber_hysteresis");

    PetscReal t_final;
    TSGetTime(solver.get_ts(), &t_final);

    // ─────────────────────────────────────────────────────────────────────
    //  16. Summary
    // ─────────────────────────────────────────────────────────────────────
    sep('=');
    std::println("\n  L-SHAPED MULTISCALE SEISMIC ANALYSIS — SUMMARY");
    sep('-');
    std::println("  Earthquake:    Tohoku 2011, MYG004 (NS+EW+UD), PGA(NS)={:.3f}g (scale={:.2f})",
                 eq_scale * eq_ns.pga() / 9.81, eq_scale);
    std::println("  Structure:     {}-story L-shaped RC frame, {} × {} m plan",
                 NUM_STORIES, X_GRID.back(), Y_GRID.back());
    std::println("  Columns:       {}×{} m, f'c={} MPa", COL_B, COL_H, COL_FPC);
    std::println("  Beams:         {}×{} m, f'c={} MPa", BM_B, BM_H, BM_FPC);
    std::println("  First yield:   t = {:.4f} s  (element {})",
                 transition_report->trigger_time,
                 transition_report->critical_element);
    std::println("  Sub-models:    {} (Hex27, KoBatheConcrete3D)", analysis.model().num_local_models());
    std::println("  Evolution:     {} steps, {:.1f} s — t_final = {:.4f} s",
                 evol_step, evol_step * DT, static_cast<double>(t_final));
    std::println("  Peak damage:   {:.6f}", evol_max_damage);
    std::println("  Active cracks: {} (across all sub-models at final step)", total_cracks);
    std::println("  Output dir:    {}", OUT);
    sep('=');

    PetscFinalize();
    return 0;
}
