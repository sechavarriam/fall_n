// =============================================================================
//  main_table_multiscale.cpp
// =============================================================================
//
//  "4-legged table" — simple RC frame with MITC16 shell slab.
//  3-component seismic excitation (Tohoku 2011, MYG004).
//  Columns: RC fiber sections (global), embedded rebar sub-models (local).
//
//  Geometry
//  --------
//  4 vertical columns at the corners of a 5 × 5 m plan, height 3.2 m.
//  1 bicubic MITC16 shell slab at the top for mass distribution.
//
//  Column nodes (base):  0 – 3  at z = 0
//  Slab nodes (top):     4 – 19 (4×4 grid at z = H)
//    corners:  4 → (0,0),  7 → (L,0),  16 → (0,L),  19 → (L,L)
//
//  Column elements: 0 – 3  (2-node beams)
//  Shell element:   4       (16-node MITC16)
//
//  Outputs
//  -------
//  data/output/table_multiscale/
//    yield_state.vtm
//    evolution/frame_NNNNNN.vtm
//    evolution/frame.pvd
//    evolution/sub_models/
//    recorders/
//      roof_displacement.csv
//      fiber_hysteresis_concrete.csv
//      fiber_hysteresis_steel.csv
//      crack_evolution.csv
//      global_history.csv
//      rebar_strains.csv
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

static const std::string EQ_DIR = BASE + "data/input/earthquakes/Japan2011/"
                                         "Tsukidate-MYG004/";
static const std::string EQ_NS  = EQ_DIR + "MYG0041103111446.NS";
static const std::string EQ_EW  = EQ_DIR + "MYG0041103111446.EW";
static const std::string EQ_UD  = EQ_DIR + "MYG0041103111446.UD";
static const std::string OUT    = BASE + "data/output/table_multiscale/";

// ── Table geometry ──────────────────────────────────────────────────────────
static constexpr double LX = 5.0;     // plan dimension X [m]
static constexpr double LY = 5.0;     // plan dimension Y [m]
static constexpr double H  = 3.2;     // column height    [m]

// ── Column section 0.25 × 0.25 m ───────────────────────────────────────────
static constexpr double COL_B   = 0.25;
static constexpr double COL_H   = 0.25;
static constexpr double COL_CVR = 0.03;
static constexpr double COL_BAR = 0.016;
static constexpr double COL_TIE = 0.08;
static constexpr double COL_FPC = 30.0;   // f'c [MPa]

// ── Shell slab: 0.20 m thick ────────────────────────────────────────────────
static constexpr double SLAB_T  = 0.20;   // thickness   [m]
static const     double SLAB_E  = 4700.0 * std::sqrt(COL_FPC);  // Ec [MPa]

// ── RC steel ─────────────────────────────────────────────────────────────────
static constexpr double STEEL_E  = 200000.0;
static constexpr double STEEL_FY = 420.0;
static constexpr double STEEL_B  = 0.01;
static constexpr double TIE_FY   = 420.0;
static constexpr double NU_RC    = 0.20;

// ── Effective column elastic moduli ─────────────────────────────────────────
static const double EC_COL = 4700.0 * std::sqrt(COL_FPC);
static const double GC_COL = EC_COL / (2.0 * (1.0 + NU_RC));

// ── Damage / transition ──────────────────────────────────────────────────────
static constexpr double EPS_YIELD = STEEL_FY / STEEL_E;  // ≈ 0.0021

// ── Mass + Rayleigh damping (5 % at T₁≈0.55 s, T₃≈0.15 s) ─────────────────
//  Thinner 0.25 m columns ⇒ longer first period than original 0.40 m design
static constexpr double RC_DENSITY = 2.4e-3;
static constexpr double XI_DAMP    = 0.05;
static const double OMEGA_1 = 2.0 * std::numbers::pi / 0.55;
static const double OMEGA_3 = 2.0 * std::numbers::pi / 0.15;

// ── Time integration ─────────────────────────────────────────────────────────
static constexpr double DT       = 0.01;      // s
static constexpr double T_SKIP   = 40.0;      // skip to strong-motion onset
static constexpr double T_MAX    = 2.0;       // window [40, 42] s
static constexpr double EQ_SCALE = 80.0;      // amplified for lab-scale demonstration

// ── Sub-model mesh ───────────────────────────────────────────────────────────
static constexpr int SUB_NX = 2;
static constexpr int SUB_NY = 2;
static constexpr int SUB_NZ = 4;

// ── Sub-model evolution ──────────────────────────────────────────────────────
static constexpr int EVOL_VTK_INTERVAL   = 5;
static constexpr int EVOL_PRINT_INTERVAL = 1;
static constexpr int FRAME_VTK_INTERVAL  = 5;static constexpr int MAX_PHASE2_STEPS    = 20;   // cap for demo runtime
// ── FE² two-way coupling ────────────────────────────────────────────────────
static constexpr int    MAX_STAGGERED_ITER  = 6;
static constexpr double STAGGERED_TOL       = 0.03;
static constexpr double STAGGERED_RELAX     = 0.7;
static constexpr int    COUPLING_START_STEP = 3;

// ── Type aliases ─────────────────────────────────────────────────────────────
static constexpr std::size_t NDOF = 6;
using StructPolicy = SingleElementPolicy<StructuralElement>;
using StructModel  = Model<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, StructPolicy>;
using DynSolver    = DynamicAnalysis<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, StructPolicy>;
using BeamElemT    = BeamElement<TimoshenkoBeam3D, 3, beam::SmallRotation>;
using ShellElemT   = MITC16Shell<>;

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
    std::println("  fall_n — 4-Legged Table: MITC16 Shell + RC Multiscale FE²");
    std::println("  Tohoku 2011 (MYG004 NS+EW+UD) + Fiber Sections + Embedded Rebar");
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

    std::println("  Station       : MYG004 (Tsukidate, Miyagi)");
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
    //  2. Build table domain manually
    //     - 4 base nodes (z=0)
    //     - 16 slab nodes (4×4 grid at z=H)
    //     - 4 beam columns + 1 MITC16 shell
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[2] Building 4-legged table domain...");

    Domain<3> domain;
    PetscInt tag = 0;

    // Base nodes: 0–3
    domain.add_node(0, 0.0, 0.0, 0.0);
    domain.add_node(1,  LX, 0.0, 0.0);
    domain.add_node(2, 0.0,  LY, 0.0);
    domain.add_node(3,  LX,  LY, 0.0);

    // Slab nodes: 4–19  (4×4 tensor-product grid at z = H)
    //  Row-major: jy outer, jx inner → node_id = 4 + 4*jy + jx
    //  Parametric positions: ξ,η ∈ {−1, −1/3, +1/3, +1}
    //  Physical:  x ∈ {0, L/3, 2L/3, L},  y ∈ {0, L/3, 2L/3, L}
    {
        PetscInt nid = 4;
        for (int jy = 0; jy < 4; ++jy) {
            for (int jx = 0; jx < 4; ++jx) {
                const double x = LX * jx / 3.0;
                const double y = LY * jy / 3.0;
                domain.add_node(nid++, x, y, H);
            }
        }
    }

    std::println("  Total nodes   : {}", domain.num_nodes());

    // Columns: 2-node beam elements connecting base to slab corners
    //  Corner mapping:
    //    base 0 → slab 4  (0,0)    base 1 → slab 7  (L,0)
    //    base 2 → slab 16 (0,L)    base 3 → slab 19 (L,L)
    const std::array<std::pair<PetscInt,PetscInt>, 4> col_pairs = {{
        {0,  4}, {1,  7}, {2, 16}, {3, 19}
    }};

    for (auto [base, top] : col_pairs) {
        PetscInt conn[2] = {base, top};
        auto& geom = domain.make_element<LagrangeElement3D<2>>(
            GaussLegendreCellIntegrator<2>{}, tag++, conn);
        geom.set_physical_group("Columns");
    }

    // MITC16 slab: 16-node bicubic shell element
    {
        PetscInt slab_conn[16];
        for (int i = 0; i < 16; ++i) slab_conn[i] = 4 + i;
        auto& geom = domain.make_element<LagrangeElement3D<4, 4>>(
            GaussLegendreCellIntegrator<4, 4>{}, tag++, slab_conn);
        geom.set_physical_group("Slabs");
    }

    std::println("  Columns       : 4  (2-node beams)");
    std::println("  Shell slab    : 1  (16-node MITC16)");
    std::println("  Plan          : {} × {} m,  height = {} m", LX, LY, H);

    // Finalize DMPlex sieve from the manually-built domain
    domain.assemble_sieve();

    // ─────────────────────────────────────────────────────────────────────
    //  3. RC fiber section for columns
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[3] Building RC fiber section (columns)...");

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

    std::println("  Section : {}×{} m, f'c = {} MPa", COL_B, COL_H, COL_FPC);

    // ─────────────────────────────────────────────────────────────────────
    //  4. Elastic shell material for slab (mass contribution)
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[4] Building elastic shell material (slab)...");

    MindlinShellMaterial slab_relation{SLAB_E, NU_RC, SLAB_T};
    const auto slab_mat = Material<MindlinReissnerShell3D>{slab_relation, ElasticUpdate{}};

    std::println("  Thickness : {} m,  Ec = {:.0f} MPa", SLAB_T, SLAB_E);

    // ─────────────────────────────────────────────────────────────────────
    //  5. Structural model assembly
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[5] Assembling structural model...");

    std::vector<const ElementGeometry<3>*> shell_geoms;

    auto builder = StructuralModelBuilder<
        BeamElemT, ShellElemT,
        TimoshenkoBeam3D, MindlinReissnerShell3D>{};
    builder.set_frame_material("Columns", col_mat);
    builder.set_shell_material("Slabs",   slab_mat);

    std::vector<StructuralElement> elements =
        builder.build_elements(domain, &shell_geoms);

    StructModel model{domain, std::move(elements)};
    model.fix_z(0.0);
    model.setup();

    std::println("  Total structural elements : {}", model.elements().size());
    std::println("  Shell geometries          : {}", shell_geoms.size());

    // ─────────────────────────────────────────────────────────────────────
    //  6. Dynamic solver: density, Rayleigh damping, 3-component ground motion
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[6] Configuring dynamic solver (3-component)...");

    DynSolver solver{&model};
    solver.set_density(RC_DENSITY);
    solver.set_rayleigh_damping(OMEGA_1, OMEGA_3, XI_DAMP);
    solver.set_force_function([](double, Vec) {});

    BoundaryConditionSet<3> bcs;
    bcs.add_ground_motion({0, eq_ns.as_time_function()}, eq_scale);
    bcs.add_ground_motion({1, eq_ew.as_time_function()}, eq_scale);
    bcs.add_ground_motion({2, eq_ud.as_time_function()}, eq_scale);
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
    DamageTracker<StructModel> damage_tracker{damage_crit, 1, 4};

    auto fiber_classifier = [](std::size_t, std::size_t, std::size_t,
                               double, double, double area) -> FiberMaterialClass {
        return (area < 0.001) ? FiberMaterialClass::Steel
                              : FiberMaterialClass::Concrete;
    };
    FiberHysteresisRecorder<StructModel> hysteresis_rec{
        damage_crit, fiber_classifier, {}, 4, 1};

    // Record 4 slab corners (roof nodes): 4, 7, 16, 19
    std::vector<NodeRecorder<StructModel>::Channel> disp_channels;
    for (PetscInt nid : {4, 7, 16, 19}) {
        disp_channels.push_back({static_cast<std::size_t>(nid), 0});
        disp_channels.push_back({static_cast<std::size_t>(nid), 1});
        disp_channels.push_back({static_cast<std::size_t>(nid), 2});
    }
    NodeRecorder<StructModel> node_rec{disp_channels, 1};

    auto composite = make_composite_observer<StructModel>(
        std::move(damage_tracker), std::move(hysteresis_rec), std::move(node_rec));
    solver.set_observer(composite);

    std::println("  DamageTracker     : top-4, every step");
    std::println("  FiberHysteresis   : top-4/material, every step");
    std::println("  NodeRecorder      : {} channels (roof corners)", disp_channels.size());

    // ─────────────────────────────────────────────────────────────────────
    //  8. Transition director
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[8] Configuring transition director...");

    auto [director, transition_report] =
        make_damage_threshold_director<StructModel>(damage_crit, 0.05);

    std::println("  Criterion  : MaxStrain (\u03b5_ref = {:.6f})", EPS_YIELD);
    std::println("  Threshold  : damage_index > 0.05");

    // ── Create output directories ────────────────────────────────────────
    std::filesystem::create_directories(OUT);
    std::filesystem::create_directories(OUT + "evolution/sub_models/");
    std::filesystem::create_directories(OUT + "recorders/");

    // ── Global history CSV ───────────────────────────────────────────────
    std::ofstream global_csv(OUT + "recorders/global_history.csv");
    global_csv << "time,step,phase,u_inf,peak_damage\n";

    // ─────────────────────────────────────────────────────────────────────
    //  9. Phase 1: Global fiber-section dynamic analysis until first yield
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
        std::println("    Peak damage = {:.4f} — try larger scale factor.", peak_damage_global);
        global_csv.close();

        // Still export VTK of the deformed frame
        {
            fall_n::vtk::StructuralVTMExporter vtm{
                model,
                fall_n::reconstruction::RectangularSectionProfile<2>{COL_B, COL_H},
                fall_n::reconstruction::ShellThicknessProfile<3>{}};
            vtm.set_displacement(model.state_vector());
            vtm.set_yield_strain(EPS_YIELD);
            vtm.write(OUT + "elastic_final.vtm");
            std::println("  Written: {}elastic_final.vtm", OUT);
        }

        // Write recorders even if elastic
        const std::string rec_dir = OUT + "recorders/";
        composite.template get<2>().write_csv(rec_dir + "roof_displacement.csv");

        PetscFinalize();
        return 0;
    }

    std::println("\n[*] YIELDING DETECTED");
    std::println("    Time of first yield   : {:.4f} s", transition_report->trigger_time);
    std::println("    Critical element      : {}", transition_report->critical_element);
    std::println("    Damage index          : {:.6f}", transition_report->metric_value);

    // ─────────────────────────────────────────────────────────────────────
    //  10. Identify critical column elements (all 4 for this simple table)
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[10] Identifying critical column elements...");

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
        if (se.as<BeamElemT>() && di.damage_index > 0.90) {
            crit_elem_ids.push_back(di.element_index);
        }
    }
    if (crit_elem_ids.empty()) {
        crit_elem_ids.push_back(transition_report->critical_element);
    }

    std::println("  Critical elements ({}):", crit_elem_ids.size());
    for (auto eid : crit_elem_ids) {
        double di = 0;
        for (const auto& d : current_damages)
            if (d.element_index == eid) { di = d.damage_index; break; }
        std::println("    element {} — damage_index = {:.6f}", eid, di);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  11. Export global frame VTK at yield time
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[11] Exporting global frame VTK at first yield...");

    {
        fall_n::vtk::StructuralVTMExporter vtm{
            model,
            fall_n::reconstruction::RectangularSectionProfile<2>{COL_B, COL_H},
            fall_n::reconstruction::ShellThicknessProfile<3>{}};
        vtm.set_displacement(model.state_vector());
        vtm.set_yield_strain(EPS_YIELD);
        vtm.write(OUT + "yield_state.vtm");
        std::println("  Written: {}yield_state.vtm", OUT);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  12. Extract element kinematics + build sub-models with embedded rebar
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[12] Extracting kinematics and building sub-models...");

    auto extract_beam_kinematics = [&](std::size_t e_idx) -> ElementKinematics {
        const auto& se = model.elements()[e_idx];
        const auto* beam = se.as<BeamElemT>();
        const auto u_e = se.extract_element_dofs(model.state_vector());

        auto kin_A = extract_section_kinematics(*beam, u_e, -1.0);
        auto kin_B = extract_section_kinematics(*beam, u_e, +1.0);
        kin_A.E = EC_COL; kin_A.G = GC_COL; kin_A.nu = NU_RC;
        kin_B.E = EC_COL; kin_B.G = GC_COL; kin_B.nu = NU_RC;

        ElementKinematics ek;
        ek.element_id   = e_idx;
        ek.kin_A        = kin_A;
        ek.kin_B        = kin_B;
        ek.endpoint_A   = beam->geometry().map_local_point(std::array{-1.0});
        ek.endpoint_B   = beam->geometry().map_local_point(std::array{+1.0});
        ek.up_direction = std::array<double,3>{1.0, 0.0, 0.0};
        return ek;
    };

    MultiscaleCoordinator coordinator;
    for (auto eid : crit_elem_ids) {
        if (!model.elements()[eid].as<BeamElemT>()) continue;
        coordinator.add_critical_element(extract_beam_kinematics(eid));
    }

    // 8-bar rebar layout matching fiber section (4 corners + 4 mid-face)
    const double cvr   = COL_CVR;
    const double bar_d = COL_BAR;
    const double bar_a = std::numbers::pi / 4.0 * bar_d * bar_d;
    const double y0    = -COL_B / 2.0 + cvr + bar_d / 2.0;
    const double y1    =  COL_B / 2.0 - cvr - bar_d / 2.0;
    const double z0    = -COL_H / 2.0 + cvr + bar_d / 2.0;
    const double z1    =  COL_H / 2.0 - cvr - bar_d / 2.0;

    std::vector<SubModelSpec::RebarBar> bars = {
        {y0, z0, bar_a, bar_d}, {y1, z0, bar_a, bar_d},
        {y0, z1, bar_a, bar_d}, {y1, z1, bar_a, bar_d},
        {0.0, z0, bar_a, bar_d}, {0.0, z1, bar_a, bar_d},
        {y0, 0.0, bar_a, bar_d}, {y1, 0.0, bar_a, bar_d},
    };

    coordinator.build_sub_models(SubModelSpec{
        .section_width  = COL_B,
        .section_height = COL_H,
        .nx = SUB_NX,
        .ny = SUB_NY,
        .nz = SUB_NZ,
        .hex_order = HexOrder::Quadratic,
        .rebar_bars = std::move(bars),
        .rebar_E  = STEEL_E,
        .rebar_fy = STEEL_FY,
        .rebar_b  = STEEL_B,
    });

    {
        const auto rpt = coordinator.report();
        std::println("  Sub-models  : {}", rpt.num_elements);
        std::println("  Nodes/sub   : {}", rpt.total_nodes);
        std::println("  Hex27/sub   : {}", rpt.total_elements);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  13. Create nonlinear sub-model evolvers
    // ─────────────────────────────────────────────────────────────────────
    sep('=');
    std::println("\n[13] Creating nonlinear sub-model evolvers...");

    const std::string evol_sub_dir = OUT + "evolution/sub_models";

    std::vector<NonlinearSubModelEvolver> nl_evolvers;
    for (auto& sub : coordinator.sub_models()) {
        nl_evolvers.emplace_back(sub, COL_FPC, evol_sub_dir, EVOL_VTK_INTERVAL);
        nl_evolvers.back().set_incremental_params(60, 8);
        nl_evolvers.back().set_penalty_alpha(EC_COL * 10.0);  // ~2.6e5
        nl_evolvers.back().set_snes_params(100, 2.0, 1e-3);   // relaxed tols
    }

    std::println("  Nonlinear evolvers : {}", nl_evolvers.size());
    std::println("  Constitutive       : KoBatheConcrete3D (f'c = {} MPa)", COL_FPC);
    std::println("  Embedded rebar     : 8 bars × ∅{} mm, penalty α = 1e6", COL_BAR * 1000);
    std::println("  VTK interval       : every {} steps", EVOL_VTK_INTERVAL);

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

    auto beam_profile  = fall_n::reconstruction::RectangularSectionProfile<2>{COL_B, COL_H};
    auto shell_profile = fall_n::reconstruction::ShellThicknessProfile<3>{};

    int    evol_step       = 0;
    double evol_max_damage = peak_damage_global;
    int    total_cracks    = 0;

    // ── Crack evolution CSV ──────────────────────────────────────────────
    std::ofstream crack_csv(OUT + "recorders/crack_evolution.csv");
    crack_csv << "time,total_cracked_gps,total_cracks,max_damage,max_opening";
    for (std::size_t i = 0; i < analysis.model().num_local_models(); ++i)
        crack_csv << ",sub" << i << "_cracked_gps"
                  << ",sub" << i << "_cracks"
                  << ",sub" << i << "_max_damage";
    crack_csv << "\n";

    // ── Rebar strain CSV ─────────────────────────────────────────────────
    std::ofstream rebar_csv(OUT + "recorders/rebar_strains.csv");
    rebar_csv << "time";
    for (std::size_t i = 0; i < analysis.model().num_local_models(); ++i) {
        auto& ev = analysis.model().local_models()[i];
        auto strains = ev.extract_rebar_strains();
        for (std::size_t b = 0; b < strains.size(); ++b)
            rebar_csv << ",sub" << i << "_bar" << b;
    }
    rebar_csv << "\n";

    // Phase 2: stable dt, no adaptation for FE² coupling
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
        std::println("  [TS] Phase 2 dt = {:.6f} s", static_cast<double>(dt_current));
        std::cout << std::flush;
    }

    for (;;) {
        PetscReal t_current;
        TSGetTime(solver.get_ts(), &t_current);
        if (static_cast<double>(t_current) >= T_MAX - 1e-14) break;
        if (evol_step >= MAX_PHASE2_STEPS) {
            std::println("  [!] Phase 2 capped at {} steps (demo limit)", MAX_PHASE2_STEPS);
            break;
        }

        if (!analysis.step()) {
            std::println("  [!] Multiscale step failed at t={:.4f} s",
                         solver.current_time());
            break;
        }

        ++evol_step;
        const double t = solver.current_time();

        // ── Iterated two-way FE² coupling ───────────────────────────
        const int staggered_iters = analysis.last_staggered_iterations();

        // ── End-of-step: crack collection + VTK ─────────────────────

        // ── Crack summaries ─────────────────────────────────────────
        int    step_cracked_gps      = 0;
        int    step_total_cracks     = 0;
        double step_max_crack_damage = 0.0;
        double step_max_opening      = 0.0;
        std::vector<CrackSummary> step_summaries;
        for (auto& ev : analysis.model().local_models()) {
            auto cs = ev.crack_summary();
            step_summaries.push_back(cs);
            step_cracked_gps  += cs.num_cracked_gps;
            step_total_cracks += cs.total_cracks;
            step_max_crack_damage = std::max(step_max_crack_damage, cs.max_damage);
            step_max_opening      = std::max(step_max_opening, cs.max_opening);
        }
        total_cracks = step_total_cracks;

        // ── Crack evolution CSV row ─────────────────────────────────
        crack_csv << std::fixed << std::setprecision(4) << t
                  << "," << step_cracked_gps
                  << "," << step_total_cracks
                  << std::scientific << std::setprecision(6)
                  << "," << step_max_crack_damage
                  << "," << step_max_opening;
        for (const auto& cs : step_summaries)
            crack_csv << "," << cs.num_cracked_gps
                      << "," << cs.total_cracks
                      << "," << cs.max_damage;
        crack_csv << "\n" << std::flush;

        // ── Rebar strain CSV row ────────────────────────────────────
        rebar_csv << std::fixed << std::setprecision(4) << t;
        for (auto& ev : analysis.model().local_models()) {
            auto strains = ev.extract_rebar_strains();
            for (double s : strains)
                rebar_csv << "," << std::scientific << std::setprecision(6) << s;
        }
        rebar_csv << "\n" << std::flush;

        // ── Track peak damage ───────────────────────────────────────
        double step_max_damage = 0.0;
        for (std::size_t e = 0; e < model.elements().size(); ++e) {
            auto info = damage_crit.evaluate_element(
                model.elements()[e], e, model.state_vector());
            step_max_damage = std::max(step_max_damage, info.damage_index);
        }
        evol_max_damage = std::max(evol_max_damage, step_max_damage);

        // ── Global VTK snapshot ─────────────────────────────────────
        if (evol_step % FRAME_VTK_INTERVAL == 0) {
            const auto vtm_file = std::format("{}/frame_{:06d}.vtm",
                                              evol_frame_dir, evol_step);
            fall_n::vtk::StructuralVTMExporter vtm{model, beam_profile, shell_profile};
            vtm.set_displacement(model.state_vector());
            vtm.set_yield_strain(EPS_YIELD);
            vtm.write(vtm_file);
            pvd_global.add_timestep(t, vtm_file);
        }

        // ── Global history CSV + progress ───────────────────────────
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
    rebar_csv.close();
    global_csv.close();

    // ─────────────────────────────────────────────────────────────────────
    //  16. Final VTK + finalize
    // ─────────────────────────────────────────────────────────────────────
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
        fall_n::PythonPlotter plotter(BASE + "scripts/plot_table_multiscale.py");
        int rc = plotter.plot(rec_dir, BASE + "doc/figures/table_multiscale/");
        if (rc == 0)
            std::println("  Plots generated successfully.");
        else
            std::println("  [!] Python plotter returned code {}", rc);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  18. Summary
    // ─────────────────────────────────────────────────────────────────────
    sep('=');
    std::println("\n  4-LEGGED TABLE MULTISCALE SEISMIC ANALYSIS — SUMMARY");
    sep('-');
    std::println("  Earthquake:      Tohoku 2011, MYG004 (NS+EW+UD), scale={:.2f}", eq_scale);
    std::println("  Structure:       4-column table, {} × {} m plan, {:.1f} m height",
                 LX, LY, H);
    std::println("  Columns:         4 × {}×{} m, f'c={} MPa (fiber sections)",
                 COL_B, COL_H, COL_FPC);
    std::println("  Shell slab:      MITC16 bicubic, t={} m (elastic)", SLAB_T);
    std::println("  First yield:     t = {:.4f} s  (element {})",
                 transition_report->trigger_time,
                 transition_report->critical_element);
    std::println("  Sub-models:      {} (Hex27, KoBatheConcrete3D, 8-bar embedded rebar)",
                 analysis.model().num_local_models());
    std::println("  Evolution:       {} steps — t_final = {:.4f} s",
                 evol_step, static_cast<double>(t_final));
    std::println("  Peak damage:     {:.6f}", evol_max_damage);
    std::println("  Active cracks:   {}", total_cracks);
    std::println("  Output dir:      {}", OUT);
    sep('=');

    PetscFinalize();
    return 0;
}
