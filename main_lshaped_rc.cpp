// =============================================================================
//  main_lshaped_rc.cpp  —  L-shaped RC building, bidirectional seismic,
//                           reinforced multiscale sub-models
// =============================================================================
//
//  Demonstrates a realistic L-shaped reinforced concrete building under
//  bidirectional seismic excitation with multiscale analysis:
//
//    Phase 1 (global, nonlinear fiber):
//      An L-shaped RC frame (4 stories lower wing + 2 stories upper wing
//      forming a setback) is subjected to the 2011 Tohoku earthquake
//      (K-NET station MYG004, Tsukidate, Miyagi) simultaneously in X (NS)
//      and Y (EW) directions.  Columns and beams use fibered RC
//      cross-sections.  A MaxStrainDamageCriterion monitors all elements;
//      the TransitionDirector pauses at first steel yield.
//
//    Phase 2 (local, reinforced continuum sub-models):
//      The top-4 most-damaged column elements are downscaled to prismatic
//      3D sub-models with hex8 concrete (KoBatheConcrete3D) and embedded
//      truss rebar (Menegotto-Pinto).  Sub-models evolve through the
//      earthquake with continuously updated BCs from the global analysis.
//
//  Building geometry
//  -----------------
//    Plan: 3 bays in X (5 m each) × 2 bays in Y (6 m each)
//    Full grid: 4×3 nodes per floor
//    Cutout: bay [2,3) × [1,2) removed above story 2 → L-shape + setback
//    Stories: 4 stories at 3.2 m → 12.8 m tall (lower wing)
//             Only 2 stories above the cutout zone (upper wing: 6.4 m)
//
//  Ground motion
//  -------------
//    Station: MYG004 (Tsukidate, Miyagi), Tohoku 2011 Mw 9.0
//    NS component → X direction (PGA ≈ 2764 gal)
//    EW component → Y direction (PGA ≈ 1268 gal)
//    Window: [30 s, 90 s] of original record (strong-motion phase)
//
//  Units: [m, MN, MPa, MN*s²/m⁴, s]
//
// =============================================================================

#include "header_files.hh"

#include <Eigen/Dense>

#include <array>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <print>
#include <string>
#include <vector>

using namespace fall_n;

// =============================================================================
//  Constants
// =============================================================================
namespace {

// ── I/O paths ────────────────────────────────────────────────────────────────
#ifdef FALL_N_SOURCE_DIR
static const std::string BASE = std::string(FALL_N_SOURCE_DIR) + "/";
#else
static const std::string BASE = "./";
#endif
static const std::string EQ_NS  = BASE + "data/input/earthquakes/Japan2011/"
                                         "Tsukidate-MYG004/MYG0041103111446.NS";
static const std::string EQ_EW  = BASE + "data/input/earthquakes/Japan2011/"
                                         "Tsukidate-MYG004/MYG0041103111446.EW";
static const std::string OUT    = BASE + "data/output/lshaped_rc/";

// ── Frame geometry ───────────────────────────────────────────────────────────
//  3 bays in X: 0, 5, 10, 15 m
//  2 bays in Y: 0, 6, 12 m
//  4 stories, 3.2 m each
static constexpr std::array<double, 4> X_GRID = {0.0, 5.0, 10.0, 15.0};
static constexpr std::array<double, 3> Y_GRID = {0.0, 6.0, 12.0};
static constexpr int    NUM_STORIES  = 4;
static constexpr double STORY_HEIGHT = 3.2;   // m

// Cutout: remove bay [2,3) × [1,2) above story 2  →  L-shape with setback
// Bay indices are 0-based: X bay 2 = [10,15], Y bay 1 = [6,12]
static constexpr int CUTOUT_X_START   = 2;
static constexpr int CUTOUT_X_END     = 3;
static constexpr int CUTOUT_Y_START   = 1;
static constexpr int CUTOUT_Y_END     = 2;
static constexpr int CUTOUT_ABOVE     = 2;   // cutout only above story 2

// ── Column section 0.45×0.45 m ──────────────────────────────────────────────
static constexpr double COL_B    = 0.45;
static constexpr double COL_H    = 0.45;
static constexpr double COL_CVR  = 0.035;
static constexpr double COL_BAR  = 0.022;   // 22 mm rebar
static constexpr double COL_TIE  = 0.10;

// ── Beam section 0.30×0.50 m ────────────────────────────────────────────────
static constexpr double BM_B    = 0.30;
static constexpr double BM_H    = 0.50;
static constexpr double BM_CVR  = 0.035;
static constexpr double BM_BAR  = 0.019;    // 19 mm rebar

// ── Slab ─────────────────────────────────────────────────────────────────────
static constexpr double SLAB_T   = 0.15;    // thickness [m]
static constexpr double SLAB_E   = 25000.0; // [MPa] elastic

// ── RC materials ─────────────────────────────────────────────────────────────
static constexpr double COL_FPC  = 30.0;    // f'c columns [MPa]
static constexpr double BM_FPC   = 25.0;    // f'c beams   [MPa]
static constexpr double STEEL_E  = 200000.0;
static constexpr double STEEL_FY = 420.0;
static constexpr double STEEL_B  = 0.01;    // strain-hardening ratio
static constexpr double TIE_FY   = 420.0;
static constexpr double NU_RC    = 0.20;

// Effective elastic modulus (ACI)
static const double EC_COL = 4700.0 * std::sqrt(COL_FPC);
static const double GC_COL = EC_COL / (2.0 * (1.0 + NU_RC));

// ── Damage / transition ──────────────────────────────────────────────────────
static constexpr double EPS_YIELD = STEEL_FY / STEEL_E;  // ≈ 0.0021

// ── Mass + Rayleigh damping ─────────────────────────────────────────────────
static constexpr double RC_DENSITY = 2.4e-3;   // MN*s²/m⁴
static constexpr double XI_DAMP    = 0.05;
// Approx periods for a 4-story frame: T1≈0.5 s, T3≈0.10 s
static const double OMEGA_1 = 2.0 * std::numbers::pi / 0.50;
static const double OMEGA_3 = 2.0 * std::numbers::pi / 0.10;

// ── Time integration ─────────────────────────────────────────────────────────
static constexpr double DT       = 0.02;      // s
static constexpr double T_SKIP   = 30.0;      // skip P-wave zone
static constexpr double T_WINDOW = 20.0;      // analyse 20 s (strong-motion window)
static constexpr double T_MAX    = T_WINDOW;
static constexpr double EQ_SCALE = 1.5;       // scale up moderately to trigger yielding

// ── Sub-model mesh ───────────────────────────────────────────────────────────
static constexpr int SUB_NX = 4;              // cross-section X divisions
static constexpr int SUB_NY = 4;              // cross-section Y divisions
static constexpr int SUB_NZ = 8;              // along beam axis
static constexpr int MAX_SUB_MODELS = 4;

// ── Sub-model rebar: 8 bars (4 corner + 4 mid-face) ─────────────────────────
// Grid positions in a 4×4 cross-section (corners at 1,1 / 1,3 / 3,1 / 3,3)
// and mid-face at 2,1 / 2,3 / 1,2 / 3,2
static const double REBAR_AREA = std::numbers::pi / 4.0 * COL_BAR * COL_BAR;

// ── Sub-model evolution ──────────────────────────────────────────────────────
static constexpr int EVOL_VTK_INTERVAL   = 10;
static constexpr int EVOL_PRINT_INTERVAL = 10;
static constexpr int EVOL_SUB_UPDATE     = 10;  // update sub-models every N global steps
static constexpr int MAX_EVOL_STEPS      = 50;  // cap evolution to keep runtime practical

// ── Type aliases ─────────────────────────────────────────────────────────────
static constexpr std::size_t NDOF = 6;
using StructPolicy = SingleElementPolicy<StructuralElement>;
using StructModel  = Model<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, StructPolicy>;
using DynSolver    = DynamicAnalysis<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, StructPolicy>;
using BeamElemT    = BeamElement<TimoshenkoBeam3D, 3, beam::SmallRotation>;
using ShellElemT   = MITC4Shell<>;

} // anonymous namespace


// =============================================================================
//  Helpers
// =============================================================================
static void sep(char c = '=', int n = 72) {
    std::cout << std::string(n, c) << '\n';
}

static void print_sub_model_result(std::size_t idx,
                                   const SubModelSolverResult& res,
                                   const HomogenizedBeamSection& hs)
{
    sep('-', 60);
    std::cout << "Sub-model " << idx << "\n";
    std::cout << "  Converged            : " << (res.converged ? "YES" : "NO") << "\n";
    std::cout << "  Gauss points         : " << res.num_gp << "\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Max |u|        [mm]  : " << std::setw(12) << res.max_displacement * 1e3 << "\n";
    std::cout << "  Max sigma_VM   [MPa] : " << std::setw(12) << res.max_stress_vm << "\n";
    std::cout << "  N              [kN]  : " << std::setw(12) << hs.N  * 1e3 << "\n";
    std::cout << "  Vy             [kN]  : " << std::setw(12) << hs.Vy * 1e3 << "\n";
    std::cout << "  Mz           [kN*m]  : " << std::setw(12) << hs.Mz * 1e3 << "\n";
}


// =============================================================================
//  Build reinforced sub-models manually (bypasses MultiscaleCoordinator's
//  hex-only build to use make_reinforced_prismatic_domain)
// =============================================================================
static std::vector<MultiscaleSubModel> build_reinforced_sub_models(
    const std::vector<ElementKinematics>& critical_elements)
{
    std::vector<MultiscaleSubModel> subs;
    subs.reserve(critical_elements.size());

    // RebarSpec: 8 bars in a 4×4 grid
    // Corners: (1,1), (1,3), (3,1), (3,3)
    // Mid-face: (2,1), (2,3), (1,2), (3,2)
    RebarSpec rebar_spec;
    rebar_spec.bars = {
        {1, 1, REBAR_AREA, "Rebar"},
        {1, 3, REBAR_AREA, "Rebar"},
        {3, 1, REBAR_AREA, "Rebar"},
        {3, 3, REBAR_AREA, "Rebar"},
        {2, 1, REBAR_AREA, "Rebar"},
        {2, 3, REBAR_AREA, "Rebar"},
        {1, 2, REBAR_AREA, "Rebar"},
        {3, 2, REBAR_AREA, "Rebar"},
    };

    // Phase 1 (sequential): build reinforced prismatic domains
    for (const auto& ek : critical_elements) {
        auto pspec = align_to_beam(
            ek.endpoint_A, ek.endpoint_B, ek.up_direction,
            COL_B, COL_H, SUB_NX, SUB_NY, SUB_NZ);

        auto [domain, grid, rebar_range] =
            make_reinforced_prismatic_domain(pspec, rebar_spec);

        MultiscaleSubModel sub;
        sub.parent_element_id = ek.element_id;
        sub.domain = std::move(domain);
        sub.grid   = std::move(grid);
        sub.kin_A  = ek.kin_A;
        sub.kin_B  = ek.kin_B;
        subs.push_back(std::move(sub));
    }

    // Phase 2 (parallel): compute boundary displacements
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (std::size_t i = 0; i < critical_elements.size(); ++i) {
        const auto& ek  = critical_elements[i];
        auto&       sub = subs[i];

        auto face_A = sub.grid.nodes_on_face(PrismFace::MinZ);
        auto face_B = sub.grid.nodes_on_face(PrismFace::MaxZ);

        sub.bc_min_z = compute_boundary_displacements(
            ek.kin_A, sub.domain, face_A);
        sub.bc_max_z = compute_boundary_displacements(
            ek.kin_B, sub.domain, face_B);
    }

    return subs;
}


// =============================================================================
//  Main
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

    int exit_code = 0;
    try {
        exit_code = [&]() -> int {
            sep('=');
            std::println("  fall_n  —  L-shaped RC Building, Bidirectional Seismic");
            std::println("  Japan 2011 MYG004 (NS + EW) + Reinforced Multiscale");
            sep('=');

    // =====================================================================
    //  1. Load earthquake records (bidirectional)
    // =====================================================================
    std::println("\n[1] Loading earthquake records...");

    auto eq_ns_full = GroundMotionRecord::from_knet(EQ_NS);
    auto eq_ew_full = GroundMotionRecord::from_knet(EQ_EW);

    auto eq_ns = eq_ns_full.trim(T_SKIP, T_SKIP + T_WINDOW);
    auto eq_ew = eq_ew_full.trim(T_SKIP, T_SKIP + T_WINDOW);

    std::println("  Station   : MYG004 (Tsukidate, Miyagi)");
    std::println("  Event     : Tohoku 2011-03-11 Mw 9.0");
    std::println("  NS → X    : dt={:.4f} s, dur={:.1f} s, PGA(window)={:.1f} gal",
                 eq_ns.dt(), eq_ns.duration(), eq_ns.pga() / 9.81e-3);
    std::println("  EW → Y    : dt={:.4f} s, dur={:.1f} s, PGA(window)={:.1f} gal",
                 eq_ew.dt(), eq_ew.duration(), eq_ew.pga() / 9.81e-3);
    std::println("  Window    : [{:.0f}, {:.0f}] s of original record", T_SKIP, T_SKIP + T_WINDOW);
    std::println("  Scale     : {:.2f}", eq_scale);

    // =====================================================================
    //  2. Build L-shaped building domain
    // =====================================================================
    std::println("\n[2] Building L-shaped structural domain...");

    auto [domain, grid] = make_building_domain({
        .x_axes         = {X_GRID.begin(), X_GRID.end()},
        .y_axes         = {Y_GRID.begin(), Y_GRID.end()},
        .num_stories    = NUM_STORIES,
        .story_height   = STORY_HEIGHT,
        .cutout_x_start = CUTOUT_X_START,
        .cutout_x_end   = CUTOUT_X_END,
        .cutout_y_start = CUTOUT_Y_START,
        .cutout_y_end   = CUTOUT_Y_END,
        .cutout_above_story = CUTOUT_ABOVE,
        .include_slabs  = true,
    });

    std::println("  Grid       : {}×{} plan, {} stories ({:.1f} m)",
                 X_GRID.size(), Y_GRID.size(), NUM_STORIES,
                 NUM_STORIES * STORY_HEIGHT);
    std::println("  Cutout     : X bays [{},{}), Y bays [{},{}), above story {}",
                 CUTOUT_X_START, CUTOUT_X_END,
                 CUTOUT_Y_START, CUTOUT_Y_END, CUTOUT_ABOVE);
    std::println("  Nodes      : {}", domain.num_nodes());
    std::println("  Columns    : {}", grid.num_columns());
    std::println("  Beams      : {}", grid.num_beams());
    std::println("  Slabs      : {}", grid.num_slabs());

    // =====================================================================
    //  3. Build RC fiber section materials
    // =====================================================================
    std::println("\n[3] Building RC fiber section materials...");

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

    // Slab: elastic Mindlin-Reissner
    MindlinShellMaterial slab_relation{SLAB_E, NU_RC, SLAB_T};
    const auto slab_mat = Material<MindlinReissnerShell3D>{slab_relation, ElasticUpdate{}};

    std::println("  Columns : Kent-Park f'c={} MPa + Menegotto-Pinto fy={} MPa",
                 COL_FPC, STEEL_FY);
    std::println("  Beams   : Kent-Park f'c={} MPa + Menegotto-Pinto fy={} MPa",
                 BM_FPC, STEEL_FY);
    std::println("  Slabs   : elastic E={} MPa, t={} m", SLAB_E, SLAB_T);

    // =====================================================================
    //  4. Build structural model
    // =====================================================================
    std::println("\n[4] Assembling structural model...");

    std::vector<const ElementGeometry<3>*> shell_geoms;
    std::vector<StructuralElement> elements =
        StructuralModelBuilder<BeamElemT, ShellElemT,
                               TimoshenkoBeam3D, MindlinReissnerShell3D>{}
            .set_frame_material("Columns", col_mat)
            .set_frame_material("Beams",   bm_mat)
            .set_shell_material("Slabs",   slab_mat)
            .build_elements(domain, &shell_geoms);

    StructModel model{domain, std::move(elements)};
    model.fix_z(0.0);
    model.setup();

    std::println("  Total structural elements : {}", model.elements().size());

    // =====================================================================
    //  5. Configure dynamic solver (bidirectional ground motion)
    // =====================================================================
    std::println("\n[5] Configuring dynamic solver (bidirectional seismic)...");

    DynSolver solver{&model};
    solver.set_density(RC_DENSITY);
    solver.set_rayleigh_damping(OMEGA_1, OMEGA_3, XI_DAMP);
    solver.set_force_function([](double, Vec) {});

    BoundaryConditionSet<3> bcs;
    bcs.add_ground_motion({0, eq_ns.as_time_function()}, eq_scale);  // X = NS
    bcs.add_ground_motion({1, eq_ew.as_time_function()}, eq_scale);  // Y = EW
    solver.set_boundary_conditions(bcs);

    std::println("  Direction X: NS component, scaled by {:.2f}", eq_scale);
    std::println("  Direction Y: EW component, scaled by {:.2f}", eq_scale);
    std::println("  Damping {}% (Rayleigh), dt={} s, T_max={} s",
                 XI_DAMP * 100.0, DT, T_MAX);

    // =====================================================================
    //  6. Damage tracker + transition director + NodeRecorder
    // =====================================================================
    std::println("\n[6] Setting up observers and transition director...");

    MaxStrainDamageCriterion damage_crit{EPS_YIELD};
    DamageTracker<StructModel> damage_tracker{damage_crit, 1, 10};

    auto [director, transition_report] =
        make_damage_threshold_director<StructModel>(damage_crit, 0.80);

    // NodeRecorder: record ux and uy at roof corner nodes
    // Roof = level NUM_STORIES.  Pick 4 corner nodes of the full plan.
    const auto roof_n00 = grid.node_id(0, 0, NUM_STORIES);
    const auto roof_n30 = grid.node_id(3, 0, NUM_STORIES);
    const auto roof_n02 = grid.node_id(0, 2, NUM_STORIES);
    // Node (3,2) might be inactive above cutout — check and use (2,2) instead
    const auto roof_n22 = grid.node_id(2, 2, NUM_STORIES);

    NodeRecorder<StructModel> recorder({
        {static_cast<std::size_t>(roof_n00), 0},  // ux at (0,0,roof)
        {static_cast<std::size_t>(roof_n00), 1},  // uy at (0,0,roof)
        {static_cast<std::size_t>(roof_n30), 0},  // ux at (3,0,roof)
        {static_cast<std::size_t>(roof_n30), 1},  // uy at (3,0,roof)
        {static_cast<std::size_t>(roof_n02), 0},  // ux at (0,2,roof)
        {static_cast<std::size_t>(roof_n02), 1},  // uy at (0,2,roof)
        {static_cast<std::size_t>(roof_n22), 0},  // ux at (2,2,roof)
        {static_cast<std::size_t>(roof_n22), 1},  // uy at (2,2,roof)
    }, 1);  // every step

    // Composite observer: damage tracker + node recorder
    auto composite_obs = make_composite_observer<StructModel>(
        std::move(damage_tracker), std::move(recorder));

    solver.set_observer(composite_obs);

    // Alias to access recorder within composite (damage_tracker at index 0)
    [[maybe_unused]] auto& damage_tracker_ref = composite_obs.get<0>();
    auto& recorder_ref       = composite_obs.get<1>();

    std::println("  Criterion : MaxStrain (eps_ref={:.6f})", EPS_YIELD);
    std::println("  Recorder  : 8 channels (4 roof nodes × ux,uy)");

    // =====================================================================
    //  7. Phase 1: global dynamic analysis until first yield
    // =====================================================================
    sep('=');
    std::println("\n[7] Phase 1: Global nonlinear dynamic analysis (bidirectional)");
    std::println("    Running until first steel fiber yields...");

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

    double peak_damage_global = 0.0;
    double peak_fiber_strain_global = 0.0;

    fall_n::StepDirector<StructModel> diag_director =
        [&director, &peak_damage_global, &peak_fiber_strain_global,
         &damage_crit]
        (const fall_n::StepEvent& ev, const StructModel& m) -> fall_n::StepVerdict
    {
        double max_d   = 0.0;
        double max_eps = 0.0;
        for (std::size_t e = 0; e < m.elements().size(); ++e) {
            auto info = damage_crit.evaluate_element(
                m.elements()[e], e, m.state_vector());
            max_d = std::max(max_d, info.damage_index);
            for (const auto& snap : m.elements()[e].section_snapshots())
                for (const auto& f : snap.fibers)
                    max_eps = std::max(max_eps, std::abs(f.strain_xx));
        }
        if (max_d   > peak_damage_global)       peak_damage_global = max_d;
        if (max_eps > peak_fiber_strain_global)  peak_fiber_strain_global = max_eps;

        if (ev.step % 250 == 0) {
            PetscReal u_norm = 0.0;
            VecNorm(ev.displacement, NORM_INFINITY, &u_norm);
            std::println("    t={:.2f} s  step={}  |u|_inf={:.3e} m  "
                         "peak_dmg={:.4e}  peak_eps={:.4e}",
                         ev.time, ev.step, u_norm,
                         peak_damage_global, peak_fiber_strain_global);
        }
        return director(ev, m);
    };

    solver.step_to(T_MAX, diag_director);

    sep('-');
    if (!transition_report->triggered) {
        std::println("[!] No yielding detected within {} s.  "
                     "peak_damage={:.4f}", T_MAX, peak_damage_global);
        // Still export time histories even without yielding
        std::filesystem::create_directories(OUT);
        recorder_ref.write_csv(OUT + "roof_displacement.csv");
        std::println("  Roof displacement CSV: {}roof_displacement.csv", OUT);
        return 0;
    }

    std::println("\n[*] YIELDING DETECTED");
    std::println("    Time         : {:.4f} s", transition_report->trigger_time);
    std::println("    Element      : {}", transition_report->critical_element);
    std::println("    Damage index : {:.6f}", transition_report->metric_value);
    std::println("    Recorder     : {} samples", recorder_ref.num_samples());

    // =====================================================================
    //  8. Identify top-N critical column elements
    // =====================================================================
    std::println("\n[8] Ranking damaged elements...");

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
        if (se.as<BeamElemT>() && di.damage_index > 0.3) {
            crit_elem_ids.push_back(di.element_index);
            if (crit_elem_ids.size() >= MAX_SUB_MODELS) break;
        }
    }

    if (crit_elem_ids.empty()) {
        crit_elem_ids.push_back(transition_report->critical_element);
    }

    std::println("  Critical columns for sub-model analysis ({}):",
                 crit_elem_ids.size());
    for (auto eid : crit_elem_ids) {
        double di = 0.0;
        for (const auto& d : current_damages)
            if (d.element_index == eid) { di = d.damage_index; break; }
        std::println("    element {}  damage={:.6f}", eid, di);
    }

    // =====================================================================
    //  9. Export global VTK at first yield
    // =====================================================================
    std::println("\n[9] Exporting global frame VTK at yield point...");
    std::filesystem::create_directories(OUT);
    std::filesystem::create_directories(OUT + "sub_models/");

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

    // =====================================================================
    //  10. Extract element kinematics
    // =====================================================================
    std::println("\n[10] Extracting beam element kinematics...");

    auto extract_beam_kinematics = [&](std::size_t e_idx) -> ElementKinematics {
        const auto& se       = model.elements()[e_idx];
        const auto* beam_ptr = se.as<BeamElemT>();

        const auto u_e  = se.extract_element_dofs(model.state_vector());
        auto kin_A = extract_section_kinematics(*beam_ptr, u_e, -1.0);
        auto kin_B = extract_section_kinematics(*beam_ptr, u_e, +1.0);

        kin_A.E = EC_COL; kin_A.G = GC_COL; kin_A.nu = NU_RC;
        kin_B.E = EC_COL; kin_B.G = GC_COL; kin_B.nu = NU_RC;

        ElementKinematics ek;
        ek.element_id   = e_idx;
        ek.kin_A        = kin_A;
        ek.kin_B        = kin_B;
        ek.endpoint_A   = beam_ptr->geometry().map_local_point(
                              std::array<double,1>{-1.0});
        ek.endpoint_B   = beam_ptr->geometry().map_local_point(
                              std::array<double,1>{+1.0});
        ek.up_direction = {1.0, 0.0, 0.0};
        return ek;
    };

    std::vector<ElementKinematics> crit_kinematics;
    for (auto e_idx : crit_elem_ids) {
        if (!model.elements()[e_idx].as<BeamElemT>()) continue;
        auto ek = extract_beam_kinematics(e_idx);
        crit_kinematics.push_back(ek);
        std::println("  Element {}:  A=({:.2f},{:.2f},{:.2f})  "
                     "B=({:.2f},{:.2f},{:.2f})  eps_0(A)={:.4e}",
                     e_idx,
                     ek.endpoint_A[0], ek.endpoint_A[1], ek.endpoint_A[2],
                     ek.endpoint_B[0], ek.endpoint_B[1], ek.endpoint_B[2],
                     ek.kin_A.eps_0);
    }

    // =====================================================================
    //  11. Build reinforced prismatic sub-models
    // =====================================================================
    std::println("\n[11] Building reinforced prismatic sub-models (hex8 + truss rebar)...");

    auto sub_models = build_reinforced_sub_models(crit_kinematics);

    std::size_t total_nodes = 0, total_elems = 0;
    for (const auto& s : sub_models) {
        total_nodes += s.domain.num_nodes();
        total_elems += s.domain.num_elements();
    }
    std::println("  Sub-models        : {}", sub_models.size());
    std::println("  Total nodes       : {}", total_nodes);
    std::println("  Total elements    : {} (hex8 + truss)", total_elems);

    // =====================================================================
    //  12. Create reinforced sub-model evolvers
    // =====================================================================
    sep('=');
    std::println("\n[12] Creating reinforced sub-model evolvers...");

    const std::string evol_dir = OUT + "evolution/sub_models";
    std::filesystem::create_directories(evol_dir);

    // Rebar areas vector: 8 bars × nz elements each
    const std::vector<double> rebar_areas(8, REBAR_AREA);

    // Rebar element range for each sub-model
    // In a 4×4×8 grid, hex elements = 4*4*8 = 128, rebar starts at 128
    const std::size_t n_hex = SUB_NX * SUB_NY * SUB_NZ;
    const std::size_t n_rebar_per_bar = SUB_NZ;
    const std::size_t n_rebar_total = 8 * n_rebar_per_bar;
    RebarElementRange rebar_range{n_hex, n_hex + n_rebar_total};

    RebarSteelConfig steel{
        .E_s = STEEL_E,
        .fy  = STEEL_FY,
        .b   = STEEL_B,
    };

    std::vector<SubModelEvolver> evolvers;
    for (auto& sub : sub_models) {
        evolvers.emplace_back(sub, COL_FPC, evol_dir, EVOL_VTK_INTERVAL);
        evolvers.back().set_rebar(steel, rebar_range, rebar_areas, SUB_NZ);
    }

    std::println("  Evolvers         : {}", evolvers.size());
    std::println("  Rebar per sub    : 8 bars × {} truss elements = {}",
                 n_rebar_per_bar, n_rebar_total);
    std::println("  Rebar area/bar   : {:.2f} mm²", REBAR_AREA * 1e6);
    std::println("  Steel config     : E_s={} MPa, fy={} MPa, b={}",
                 STEEL_E, STEEL_FY, STEEL_B);

    // =====================================================================
    //  13. Initial sub-model solve at yield time
    // =====================================================================
    std::println("\n[13] Solving reinforced sub-models at yield time t={:.4f} s...",
                 transition_report->trigger_time);

    for (auto& ev : evolvers) {
        auto result = ev.solve_step(transition_report->trigger_time);
        auto hs = homogenize(result, ev.sub_model(), COL_B, COL_H);
        print_sub_model_result(ev.parent_element_id(), result, hs);
    }

    // =====================================================================
    //  14. Phase 2: resume global + evolve sub-models
    // =====================================================================
    sep('=');
    std::println("\n[14] Phase 2: Sub-model evolution through the earthquake");
    std::println("    Resuming global analysis step-by-step...");

    const std::string evol_frame_dir = OUT + "evolution";
    std::filesystem::create_directories(evol_frame_dir);
    PVDWriter pvd_global(evol_frame_dir + "/frame");

    auto beam_profile  = fall_n::reconstruction::RectangularSectionProfile<2>{COL_B, COL_H};
    auto shell_profile = fall_n::reconstruction::ShellThicknessProfile<5>{};

    int    evol_step       = 0;
    double evol_max_damage = peak_damage_global;

    for (;;) {
        if (evol_step >= MAX_EVOL_STEPS) {
            std::println("  [Evol] Reached evolution cap ({} steps), stopping.",
                         MAX_EVOL_STEPS);
            break;
        }

        PetscReal t_current;
        TSGetTime(solver.get_ts(), &t_current);
        if (static_cast<double>(t_current) >= T_MAX - 1e-14) break;

        auto verdict = solver.step_n(1, fall_n::StepDirector<StructModel>{});
        if (verdict == fall_n::StepVerdict::Stop) {
            std::println("  [!] Solver stopped at t={:.4f} s",
                         static_cast<double>(t_current));
            break;
        }

        PetscReal t_new;
        TSGetTime(solver.get_ts(), &t_new);
        ++evol_step;
        const double t = static_cast<double>(t_new);

        // Update sub-models at coarser interval (each solve is expensive)
        if (evol_step % EVOL_SUB_UPDATE == 0) {
            for (auto& ev : evolvers) {
                auto ek = extract_beam_kinematics(ev.parent_element_id());
                ev.update_kinematics(ek.kin_A, ek.kin_B);
                ev.solve_step(t);
            }
        }

        // Track peak damage
        double step_max_damage = 0.0;
        for (std::size_t e = 0; e < model.elements().size(); ++e) {
            auto info = damage_crit.evaluate_element(
                model.elements()[e], e, model.state_vector());
            step_max_damage = std::max(step_max_damage, info.damage_index);
        }
        evol_max_damage = std::max(evol_max_damage, step_max_damage);

        // Global VTK snapshot
        if (evol_step % EVOL_VTK_INTERVAL == 0) {
            const auto vtm_file = std::format("{}/frame_{:06d}.vtm",
                                              evol_frame_dir, evol_step);
            fall_n::vtk::StructuralVTMExporter vtm_exp{
                model, beam_profile, shell_profile};
            vtm_exp.set_displacement(model.state_vector());
            vtm_exp.set_yield_strain(EPS_YIELD);
            vtm_exp.write(vtm_file);
            pvd_global.add_timestep(t, vtm_file);
        }

        // Progress
        if (evol_step % EVOL_PRINT_INTERVAL == 0) {
            std::println("  [Evol] t={:.2f} s  step={}  peak_dmg={:.4f}  "
                         "subs={}", t, evol_step, evol_max_damage,
                         evolvers.size());
        }
    }

    // =====================================================================
    //  15. Finalize: PVD files + time-history CSV + summary
    // =====================================================================
    pvd_global.write();
    for (auto& ev : evolvers)
        ev.finalize();

    // Write roof displacement CSV
    recorder_ref.write_csv(OUT + "roof_displacement.csv");
    std::println("\n  Roof displacement CSV: {}roof_displacement.csv", OUT);

    PetscReal t_final;
    TSGetTime(solver.get_ts(), &t_final);

    sep('=');
    std::println("\nL-SHAPED RC BUILDING — MULTISCALE SEISMIC ANALYSIS SUMMARY");
    sep('-');
    std::println("  Earthquake    : Tohoku 2011, MYG004 NS+EW (scaled ×{:.2f})",
                 eq_scale);
    std::println("  Building      : L-shaped, {} stories, {:.1f}×{:.1f} m plan",
                 NUM_STORIES, X_GRID.back(), Y_GRID.back());
    std::println("  Cutout        : X[{},{}), Y[{},{}), above story {}",
                 CUTOUT_X_START, CUTOUT_X_END,
                 CUTOUT_Y_START, CUTOUT_Y_END, CUTOUT_ABOVE);
    std::println("  First yield   : t = {:.4f} s (element {})",
                 transition_report->trigger_time,
                 transition_report->critical_element);
    std::println("  Sub-models    : {} (reinforced: hex8 + truss rebar)",
                 evolvers.size());
    std::println("  Evolution     : {} steps, final t = {:.4f} s",
                 evol_step, static_cast<double>(t_final));
    std::println("  Peak damage   : {:.6f}", evol_max_damage);
    std::println("  Output dir    : {}", OUT);
    std::println("  Global PVD    : {}/frame.pvd", evol_frame_dir);
    std::println("  Roof CSV      : {}roof_displacement.csv", OUT);
    sep('=');

    return 0;
        }();
    } catch (...) {
        PetscFinalize();
        throw;
    }

    PetscFinalize();
    return exit_code;
}
