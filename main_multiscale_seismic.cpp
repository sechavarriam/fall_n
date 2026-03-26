// =============================================================================
//  main_multiscale_seismic.cpp  —  Phase 6: Fiber-section seismic multiscale
// =============================================================================
//
//  Demonstrates the complete fall_n multiscale seismic pipeline:
//
//    Phase 1 (global, nonlinear fiber):
//      A 3-story RC frame is subjected to the 2011 Tohoku earthquake
//      (K-NET station MYG012, Shiogama, NS component, PGA 758 gal).
//      Columns and beams use fibered RC cross-sections (Kent-Park concrete
//      + Menegotto-Pinto steel) for realistic nonlinear global response.
//      A MaxStrainDamageCriterion monitors all frame elements at every
//      step; the TransitionDirector pauses the analysis the moment the
//      first steel fiber reaches its yield strain (eps_y = fy/E = 0.0021).
//
//    Phase 2 (local, elastic continuum sub-model):
//      After the transition, the top-3 most-damaged elements are identified
//      via DamageTracker::peak_ranking().  For each critical column element
//      a prismatic 3-D hex8 sub-model is built by MultiscaleCoordinator
//      using the kinematic state extracted from the paused analysis via
//      StructuralElement::as<BeamElem>() + extract_section_kinematics().
//      Each sub-model is solved with SubModelSolver (elastic E_c, nu_c)
//      and the deformed mesh + Gauss-point stress cloud are exported to VTK.
//      Section resultants from the continuum sub-model are compared with
//      those computed directly from the fiber section (as a cross-check).
//
//  Physical scenario
//  -----------------
//  Frame geometry  : 1 bay in X (6 m), 1 bay in Y (5 m), 3 stories (3.5 m)
//  Columns (0.45x0.45 m) : confined Kent-Park core + Menegotto-Pinto rebars
//  Beams   (0.30x0.50 m) : unconfined Kent-Park + Menegotto-Pinto rebars
//  Slabs               : elastic Mindlin-Reissner (t = 0.15 m)
//  Ground motion       : Japan 2011 MYG012 NS, trimmed 0-120 s, K-NET
//  Integration         : generalised-alpha TSAlpha2, rho_inf = 0.9, dt=0.01 s
//
//  Output files (data/output/multiscale_seismic/):
//    yield_state.vtm             -- global frame at first yielding
//    sub_models/sub_{i}_mesh.vtu -- deformed continuum mesh, element i
//    sub_models/sub_{i}_gauss.vtu -- Gauss-point stress cloud, element i
//
//  Units throughout: [m, MN, MPa = MN/m^2, MN*s^2/m^4, s]
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
//  Project-wide constants
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
static const std::string OUT     = BASE + "data/output/multiscale_seismic/";

// ── Frame geometry ───────────────────────────────────────────────────────────
static constexpr std::array<double, 2> X_GRID = {0.0, 6.0};   // 1 bay, 6 m
static constexpr std::array<double, 2> Y_GRID = {0.0, 5.0};   // 1 bay, 5 m
static constexpr int    NUM_STORIES  = 3;
static constexpr double STORY_HEIGHT = 3.5;   // m

// ── Column section 0.45x0.45 m ──────────────────────────────────────────────
static constexpr double COL_B    = 0.45;    // section width  [m]
static constexpr double COL_H    = 0.45;    // section height [m]
static constexpr double COL_CVR  = 0.035;   // cover          [m]
static constexpr double COL_BAR  = 0.022;   // rebar diameter [m]
static constexpr double COL_TIE  = 0.10;    // tie spacing    [m]

// ── Beam section 0.30x0.50 m ─────────────────────────────────────────────────
static constexpr double BM_B    = 0.30;     // [m]
static constexpr double BM_H    = 0.50;     // [m]
static constexpr double BM_CVR  = 0.035;    // [m]
static constexpr double BM_BAR  = 0.019;    // [m]

// ── Slab ─────────────────────────────────────────────────────────────────────
static constexpr double SLAB_T   = 0.15;    // thickness [m]
static constexpr double SLAB_E   = 25000.0; // [MPa] elastic

// ── RC materials ─────────────────────────────────────────────────────────────
static constexpr double COL_FPC  = 30.0;    // f'c columns [MPa]
static constexpr double BM_FPC   = 25.0;    // f'c beams   [MPa]
static constexpr double STEEL_E  = 200000.0;// Es          [MPa]
static constexpr double STEEL_FY = 420.0;   // fy          [MPa]
static constexpr double STEEL_B  = 0.01;    // strain-hardening ratio
static constexpr double TIE_FY   = 420.0;   // ties        [MPa]
static constexpr double NU_RC    = 0.20;

// Effective initial modulus for sub-model (ACI formula)
static const double EC_COL  = 4700.0 * std::sqrt(COL_FPC);  // ~25740 MPa
static const double GC_COL  = EC_COL / (2.0 * (1.0 + NU_RC));

// ── Damage / transition ──────────────────────────────────────────────────────
static constexpr double EPS_YIELD = STEEL_FY / STEEL_E;  // = 0.0021

// ── Mass + Rayleigh damping 5% at T1~0.55 s, T3~0.12 s ───────────────────────
static constexpr double RC_DENSITY = 2.4e-3;   // MN*s^2/m^4
static constexpr double XI_DAMP    = 0.05;
static const double OMEGA_1 = 2.0 * std::numbers::pi / 0.55;
static const double OMEGA_3 = 2.0 * std::numbers::pi / 0.12;

// ── Time integration ─────────────────────────────────────────────────────────
static constexpr double DT      = 0.02;    // s  (adequate: T1/DT = 27.5)
static constexpr double T_SKIP  = 30.0;    // s  skip P-wave (MYG004 near-fault, S-wave ~t=50s)
static constexpr double T_MAX   = 100.0;   // s  analysis duration after T_SKIP (covers up to 130s)
static constexpr double EQ_SCALE = 1.0;    // MYG004 already delivers strong near-fault demand

// ── Sub-model mesh ───────────────────────────────────────────────────────────
static constexpr int SUB_NX = 4;   // column section: fibres in X
static constexpr int SUB_NY = 4;   // column section: fibres in Y
static constexpr int SUB_NZ = 8;   // along element axis

// ── Sub-model evolution ──────────────────────────────────────────────────────
static constexpr int EVOL_VTK_INTERVAL   = 50;    // VTK every 50 steps (=1.0 s at dt=0.02)
static constexpr int EVOL_PRINT_INTERVAL = 500;   // progress every 500 steps (=10.0 s)

// ── Type aliases (small-strain, NDOF=6) ─────────────────────────────────────
static constexpr std::size_t NDOF = 6;
using StructPolicy = SingleElementPolicy<StructuralElement>;
using StructModel  = Model<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, StructPolicy>;
using DynSolver    = DynamicAnalysis<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, StructPolicy>;

// Concrete beam element type stored inside StructuralElement
using BeamElemT = BeamElement<TimoshenkoBeam3D, 3, beam::SmallRotation>;
// Concrete shell element type stored inside StructuralElement
using ShellElemT = MITC4Shell<>;

} // anonymous namespace


// =============================================================================
//  Helper: print a separator
// =============================================================================
static void sep(char c = '=', int n = 64) {
    std::cout << std::string(n, c) << '\n';
}


// =============================================================================
//  Helper: print sub-model results
// =============================================================================
static void print_sub_model_result(std::size_t idx,
                                   const SubModelSolverResult& res,
                                   const HomogenizedBeamSection& hs)
{
    sep('-');
    std::cout << "Sub-model " << idx << "\n";
    std::cout << "  Converged            : " << (res.converged ? "YES" : "NO") << "\n";
    std::cout << "  Gauss points         : " << res.num_gp << "\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Max |u|        [mm]  : " << std::setw(12) << res.max_displacement * 1e3 << "\n";
    std::cout << "  Max sigma_VM   [MPa] : " << std::setw(12) << res.max_stress_vm << "\n";
    std::cout << "  E_eff          [GPa] : " << std::setw(12) << hs.E_eff / 1e3
              << "   (ref " << std::setw(7) << EC_COL / 1e3 << " GPa)\n";
    std::cout << "  G_eff          [GPa] : " << std::setw(12) << hs.G_eff / 1e3
              << "   (ref " << std::setw(7) << GC_COL / 1e3 << " GPa)\n";
    std::cout << "  N              [kN]  : " << std::setw(12) << hs.N  * 1e3 << "\n";
    std::cout << "  Vy             [kN]  : " << std::setw(12) << hs.Vy * 1e3 << "\n";
    std::cout << "  Mz           [kN*m]  : " << std::setw(12) << hs.Mz * 1e3 << "\n";
}


// =============================================================================
//  Main analysis
// =============================================================================
int main(int argc, char* argv[]) {

    // Make stdout unbuffered so output appears immediately even when redirected
    setvbuf(stdout, nullptr, _IONBF, 0);

    double eq_scale = EQ_SCALE;
    if (argc >= 2) {
        eq_scale = std::stod(argv[1]);
        if (eq_scale <= 0.0) {
            throw std::invalid_argument("Scale factor must be positive.");
        }
    }

    PetscInitialize(&argc, &argv, nullptr, nullptr);

    // Suppress PETSc solver noise
    PetscOptionsSetValue(nullptr, "-snes_monitor_cancel",  "");
    PetscOptionsSetValue(nullptr, "-ksp_monitor_cancel",   "");

    sep('=');
    std::println("  fall_n  --  Seismic Multiscale Analysis");
    std::println("  Japan 2011 (K-NET MYG004 NS) + RC Fiber Frame + Sub-model");
    sep('=');

    // ─────────────────────────────────────────────────────────────────────
    //  1. Parse Japan 2011 K-NET earthquake record
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[1] Loading earthquake record: {}", EQ_NS);

    auto eq_full = GroundMotionRecord::from_knet(EQ_NS);
    // Skip P-wave zone: trim to [T_SKIP, T_SKIP+T_MAX]; times reset to start at 0
    auto eq = eq_full.trim(T_SKIP, T_SKIP + T_MAX);

    std::println("  Station     : MYG004 (Tsukidate, Miyagi) -- near-fault, PGA=2.75g");
    std::println("  Event       : Tohoku 2011-03-11 Mw 9.0");
    std::println("  Component   : N-S");
    std::println("  dt          : {:.4f} s", eq.dt());
    std::println("  Duration    : {:.1f} s  (trimmed [{:.0f}s, {:.0f}s] of original {:.1f}s)",
                 eq.duration(), T_SKIP, T_SKIP + T_MAX, eq_full.duration());
    std::println("  PGA (window): {:.3f} m/s^2  ({:.3f} g)  at t={:.2f} s (eq. t={:.2f} s)",
                 eq.pga(), eq.pga() / 9.81, eq.time_of_pga(), eq.time_of_pga() + T_SKIP);
    std::println("  Scale factor: {:.2f}  -> effective PGA = {:.3f} m/s^2 ({:.3f} g)",
                 eq_scale, eq_scale * eq.pga(), eq_scale * eq.pga() / 9.81);

    // ─────────────────────────────────────────────────────────────────────
    //  2. Building domain: 1-bay x 1-bay x 3-story RC frame
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[2] Building structural domain...");

    auto [domain, grid] = make_building_domain({
        .x_axes       = {X_GRID.begin(), X_GRID.end()},
        .y_axes       = {Y_GRID.begin(), Y_GRID.end()},
        .num_stories  = NUM_STORIES,
        .story_height = STORY_HEIGHT,
        .include_slabs = false,   // frame-only: isolate beam/column behaviour
    });

    std::println("  Nodes         : {}", domain.num_nodes());
    std::println("  Columns       : {}", grid.num_columns());
    std::println("  Beams         : {}", grid.num_beams());
    std::println("  Slabs         : {}", grid.num_slabs());

    // ─────────────────────────────────────────────────────────────────────
    //  3. Build RC fiber section materials
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[3] Building RC fiber section materials...");

    // Columns: confined Kent-Park core + Menegotto-Pinto rebars
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

    // Beams: unconfined Kent-Park core + Menegotto-Pinto rebars
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

    // Slabs: disabled for frame-only analysis (isolating DMGlobalToLocal bug)
    // MindlinShellMaterial slab_relation{SLAB_E, NU_RC, SLAB_T};
    // const auto slab_mat = Material<MindlinReissnerShell3D>{slab_relation, ElasticUpdate{}};

    // Column section: 4 cover patches (8x2 + 8x2 + 2x4 + 2x4 = 48) + core (6x6 = 36) + 8 bars = 92 fibers
    // Beam section: 4 cover patches (6x2 + 6x2 + 2x6 + 2x6 = 48) + core (4x6 = 24) + 6 bars = 78 fibers
    constexpr int COL_NFIBERS = 92;
    constexpr int BM_NFIBERS  = 78;
    std::println("  Columns ({} fibers): Kent-Park f'c={} MPa + "
                 "Menegotto-Pinto fy={} MPa",
                 COL_NFIBERS, COL_FPC, STEEL_FY);
    std::println("  Beams   ({} fibers): Kent-Park f'c={} MPa + "
                 "Menegotto-Pinto fy={} MPa",
                 BM_NFIBERS, BM_FPC, STEEL_FY);
    std::println("  Slabs               : DISABLED (frame-only analysis)");
    std::println("  Steel yield strain  : {:.6f}", EPS_YIELD);

    // ─────────────────────────────────────────────────────────────────────
    //  4. Build structural model (fiber beam elements, frame-only)
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[4] Assembling structural model (frame-only, no slabs)...");

    std::vector<const ElementGeometry<3>*> shell_geoms;
    std::vector<StructuralElement> elements =
        StructuralModelBuilder<
            BeamElemT,
            ShellElemT,
            TimoshenkoBeam3D,
            MindlinReissnerShell3D>{}
            .set_frame_material("Columns", col_mat)
            .set_frame_material("Beams",   bm_mat)
            // No shell material — frame-only
            .build_elements(domain, &shell_geoms);

    StructModel model{domain, std::move(elements)};
    model.fix_z(0.0);
    model.setup();

    std::println("  Total structural elements : {}", model.elements().size());

    // ─────────────────────────────────────────────────────────────────────
    //  5. Dynamic solver: density, Rayleigh damping, ground motion in X
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[5] Configuring dynamic solver...");

    DynSolver solver{&model};
    solver.set_density(RC_DENSITY);
    solver.set_rayleigh_damping(OMEGA_1, OMEGA_3, XI_DAMP);
    solver.set_force_function([](double /*t*/, Vec /*f*/) {});  // seismic only

    BoundaryConditionSet<3> bcs;
    bcs.add_ground_motion({0, eq.as_time_function()}, eq_scale);
    solver.set_boundary_conditions(bcs);

    std::println("  Density     RC     : {} MN*s^2/m^4", RC_DENSITY);
    std::println("  Damping ratio      : {}%", XI_DAMP * 100.0);
    std::println("  Omega_1            : {:.2f} rad/s  (T1={:.2f} s)",
                 OMEGA_1, 2.0 * std::numbers::pi / OMEGA_1);
    std::println("  Omega_3            : {:.2f} rad/s  (T3={:.2f} s)",
                 OMEGA_3, 2.0 * std::numbers::pi / OMEGA_3);
    std::println("  Time step          : {} s", DT);
    std::println("  Max time           : {} s", T_MAX);

    // ─────────────────────────────────────────────────────────────────────
    //  6. Damage tracker + transition director
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[6] Setting up damage monitoring and transition director...");

    MaxStrainDamageCriterion damage_crit{EPS_YIELD};

    // DamageTracker observes every step, keeps top-10 ranked elements
    DamageTracker<StructModel> damage_tracker{damage_crit, 1, 10};
    solver.set_observer(damage_tracker);

    // TransitionDirector pauses at first steel yield (damage_index > 1.0)
    auto [director, transition_report] =
        make_damage_threshold_director<StructModel>(damage_crit, 1.0);

    std::println("  Criterion  : MaxStrain (eps_ref = {:.6f})", EPS_YIELD);
    std::println("  Threshold  : damage_index > 1.0  (i.e. eps > eps_yield)");

    // ─────────────────────────────────────────────────────────────────────
    //  7. Phase 1: global dynamic analysis until first fiber yield
    // ─────────────────────────────────────────────────────────────────────
    sep('=');
    std::println("\n[7] Phase 1: Global nonlinear fiber-section dynamic analysis");
    std::println("    Running until first steel fiber reaches yield strain...");

    // These options must be set before solver.setup(), which calls TSSetFromOptions().
    PetscOptionsSetValue(nullptr, "-snes_max_it", "100");
    PetscOptionsSetValue(nullptr, "-snes_rtol",   "1e-3");
    PetscOptionsSetValue(nullptr, "-snes_atol",   "1e-6");
    PetscOptionsSetValue(nullptr, "-ksp_max_it",  "200");

    solver.setup();
    solver.set_time_step(DT);  // dt = 0.02 s (must set explicitly for step_to)
    {
        TS ts = solver.get_ts();
        SNES snes = nullptr;
        KSP ksp = nullptr;
        TSAlpha2SetRadius(ts, 0.9);
        TSAdapt adapt;
        TSGetAdapt(ts, &adapt);
        TSAdaptSetType(adapt, TSADAPTNONE);  // NONE: no adaptation → fixed step
        TSSetTimeStep(ts, DT);
        TSSetMaxSNESFailures(ts, -1);
        TSGetSNES(ts, &snes);
        SNESSetTolerances(snes, 1e-6, 1e-3, PETSC_DETERMINE, 100, PETSC_DETERMINE);
        SNESGetKSP(snes, &ksp);
        KSPSetTolerances(ksp, PETSC_DETERMINE, PETSC_DETERMINE, PETSC_DETERMINE, 200);
    }

    // Wrap director to also print progress every 500 steps
    double peak_damage_global = 0.0;
    double peak_fiber_strain_global = 0.0;
    fall_n::StepDirector<StructModel> diag_director =
        [&director, &peak_damage_global, &peak_fiber_strain_global,
         &damage_crit]
        (const fall_n::StepEvent& ev, const StructModel& model) -> fall_n::StepVerdict
    {
        // Evaluate max damage
        double max_d = 0.0;
        double max_eps = 0.0;
        for (std::size_t e = 0; e < model.elements().size(); ++e) {
            auto info = damage_crit.evaluate_element(model.elements()[e], e, model.state_vector());
            max_d = std::max(max_d, info.damage_index);

            const auto snapshots = model.elements()[e].section_snapshots();
            for (const auto& snapshot : snapshots) {
                for (const auto& fiber : snapshot.fibers) {
                    max_eps = std::max(max_eps, std::abs(fiber.strain_xx));
                }
            }
        }
        if (max_d > peak_damage_global) peak_damage_global = max_d;
        if (max_eps > peak_fiber_strain_global) peak_fiber_strain_global = max_eps;

        // Print every 500 steps: show time, displacement norm, and damage
        if (ev.step % 500 == 0) {
            PetscReal u_norm = 0.0;
            PetscReal u_local_norm = 0.0;
            VecNorm(ev.displacement, NORM_INFINITY, &u_norm);
            VecNorm(model.state_vector(), NORM_INFINITY, &u_local_norm);
            std::println("    t={:.2f} s  step={}  |u|_inf={:.3e} m  |u_local|_inf={:.3e} m  peak_damage={:.6e}  peak_eps={:.6e}",
                ev.time, ev.step, u_norm, u_local_norm,
                peak_damage_global, peak_fiber_strain_global);
        }

        return director(ev, model);
    };

    solver.step_to(T_MAX, diag_director);

    sep('-');
    std::println("  Peak damage index reached: {:.6e}  (eps_max={:.6e}, eps_y={:.6e})",
                 peak_damage_global, peak_fiber_strain_global, EPS_YIELD);
    if (!transition_report->triggered) {
        std::println("[!] No fiber yielding detected within {} s.", T_MAX);
        std::println("    Peak damage = {:.4f} (need > 1.0 for yield)", peak_damage_global);
        std::println("    Try a larger scale factor or a longer analysis.");
        PetscFinalize();
        return 0;
    }

    std::println("\n[*] YIELDING DETECTED");
    std::println("    Time of first yield     : {:.4f} s",
                 transition_report->trigger_time);
    std::println("    Critical element index  : {}",
                 transition_report->critical_element);
    std::println("    Damage index at trigger : {:.6f}",
                 transition_report->metric_value);
    std::println("    Criterion               : {}",
                 transition_report->criterion_name);

    // ─────────────────────────────────────────────────────────────────────
    //  8. Identify top-3 critical column elements from peak_ranking
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[8] Identifying most-damaged elements...");

    // Re-evaluate all elements at the current (paused) state to get a fresh ranking
    std::vector<ElementDamageInfo> current_damages;
    current_damages.reserve(model.elements().size());
    for (std::size_t i = 0; i < model.elements().size(); ++i) {
        auto info = damage_crit.evaluate_element(
            model.elements()[i], i, model.state_vector());
        current_damages.push_back(info);
    }
    std::sort(current_damages.begin(), current_damages.end(), std::greater<>{});

    // Pick the top elements that are beam elements (not slabs)
    std::vector<std::size_t> crit_elem_ids;
    for (const auto& di : current_damages) {
        const auto& se = model.elements()[di.element_index];
        if (se.as<BeamElemT>() && di.damage_index > 0.8) {
            crit_elem_ids.push_back(di.element_index);
            if (crit_elem_ids.size() >= 3) break;
        }
    }

    if (crit_elem_ids.empty()) {
        std::println("  No beam elements with damage_index > 0.8 found.");
        std::println("  Using highest-damage element as fallback: {}",
                     transition_report->critical_element);
        crit_elem_ids.push_back(transition_report->critical_element);
    }

    std::println("  Critical column elements selected for sub-model analysis:");
    for (auto eid : crit_elem_ids) {
        double di = 0;
        for (const auto& d : current_damages)
            if (d.element_index == eid) { di = d.damage_index; break; }
        std::println("    element {}  --  damage_index = {:.6f}", eid, di);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  9. Export global structural state at yielding  (VTK)
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[9] Exporting global frame VTK at first yield...");
    std::filesystem::create_directories(OUT);
    std::filesystem::create_directories(OUT + "sub_models/");

    {
        fall_n::vtk::StructuralVTMExporter vtm_exp{
            model,
            fall_n::reconstruction::RectangularSectionProfile<2>{COL_B, COL_H},
            fall_n::reconstruction::ShellThicknessProfile<5>{}
        };
        vtm_exp.set_displacement(model.state_vector());
        vtm_exp.set_yield_strain(EPS_YIELD);
        vtm_exp.write(OUT + "yield_state.vtm");
        std::println("  Written: {}yield_state.vtm", OUT);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  10. Extract element kinematics from the paused analysis
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[10] Extracting element kinematics from paused model state...");

    // Helper: extract beam kinematics for a single element from current
    // model state.  Captures model + material constants by reference so
    // it always uses the latest converged displacement field.
    auto extract_beam_kinematics = [&](std::size_t e_idx) -> ElementKinematics {
        const auto& se       = model.elements()[e_idx];
        const auto* beam_ptr = se.as<BeamElemT>();

        const auto u_e  = se.extract_element_dofs(model.state_vector());
        auto kin_A = extract_section_kinematics(*beam_ptr, u_e, -1.0);
        auto kin_B = extract_section_kinematics(*beam_ptr, u_e, +1.0);

        kin_A.E = EC_COL; kin_A.G = GC_COL; kin_A.nu = NU_RC;
        kin_B.E = EC_COL; kin_B.G = GC_COL; kin_B.nu = NU_RC;

        const std::array<double,1> xi_end_A{-1.0};
        const std::array<double,1> xi_end_B{+1.0};

        ElementKinematics ek;
        ek.element_id   = e_idx;
        ek.kin_A        = kin_A;
        ek.kin_B        = kin_B;
        ek.endpoint_A   = beam_ptr->geometry().map_local_point(xi_end_A);
        ek.endpoint_B   = beam_ptr->geometry().map_local_point(xi_end_B);
        ek.up_direction = std::array<double,3>{1.0, 0.0, 0.0};
        return ek;
    };

    MultiscaleCoordinator coordinator;

    for (const auto e_idx : crit_elem_ids) {
        if (!model.elements()[e_idx].as<BeamElemT>()) {
            std::println("  [skip] element {} is not a BeamElemT (shell?)", e_idx);
            continue;
        }

        auto ek = extract_beam_kinematics(e_idx);
        coordinator.add_critical_element(ek);

        std::println("  Element {}:", e_idx);
        std::println("    endpoint A = ({:.2f}, {:.2f}, {:.2f})",
                     ek.endpoint_A[0], ek.endpoint_A[1], ek.endpoint_A[2]);
        std::println("    endpoint B = ({:.2f}, {:.2f}, {:.2f})",
                     ek.endpoint_B[0], ek.endpoint_B[1], ek.endpoint_B[2]);
        std::println("    eps_0(A)   = {:.6e}   kappa_z(A) = {:.6e}",
                     ek.kin_A.eps_0, ek.kin_A.kappa_z);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  11. Phase 2: build prismatic sub-models
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[11] Phase 2: Building prismatic continuum sub-models...");

    coordinator.build_sub_models(SubModelSpec{
        .section_width  = COL_B,
        .section_height = COL_H,
        .nx = SUB_NX,
        .ny = SUB_NY,
        .nz = SUB_NZ,
    });

    const auto ms_report = coordinator.report();
    std::println("  Sub-models built    : {}", ms_report.num_elements);
    std::println("  Total continuum nodes : {}", ms_report.total_nodes);
    std::println("  Total hex elements  : {}", ms_report.total_elements);
    std::println("  Max BC displacement : {:.6e} m", ms_report.max_displacement);

    // ─────────────────────────────────────────────────────────────────────
    //  12. Create sub-model evolvers for time-series analysis
    // ─────────────────────────────────────────────────────────────────────
    sep('=');
    std::println("\n[12] Creating sub-model evolvers for earthquake evolution...");

    const std::string evol_dir = OUT + "evolution/sub_models";
    std::filesystem::create_directories(evol_dir);

    std::vector<SubModelEvolver> evolvers;
    {
        auto& subs = coordinator.sub_models();
        for (auto& sub : subs)
            evolvers.emplace_back(sub, COL_FPC, evol_dir, EVOL_VTK_INTERVAL);
    }

    std::println("  Evolvers created : {}", evolvers.size());
    std::println("  VTK interval     : every {} steps ({:.2f} s)",
                 EVOL_VTK_INTERVAL, EVOL_VTK_INTERVAL * DT);

    // ─────────────────────────────────────────────────────────────────────
    //  13. Initial sub-model solve at yield time
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[13] Initial sub-model solve at yield time t={:.4f} s...",
                 transition_report->trigger_time);

    for (auto& ev : evolvers) {
        auto result = ev.solve_step(transition_report->trigger_time);
        auto hs = homogenize(result, ev.sub_model(), COL_B, COL_H);
        print_sub_model_result(ev.parent_element_id(), result, hs);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  14. Phase 2: Resume global + update sub-models at each step
    // ─────────────────────────────────────────────────────────────────────
    sep('=');
    std::println("\n[14] Phase 2: Sub-model time evolution through the earthquake");
    std::println("    Resuming global nonlinear dynamic analysis step-by-step...");
    std::println("    Global VTK every {} steps ({:.2f} s)",
                 EVOL_VTK_INTERVAL, EVOL_VTK_INTERVAL * DT);

    const std::string evol_frame_dir = OUT + "evolution";
    std::filesystem::create_directories(evol_frame_dir);
    PVDWriter pvd_global(evol_frame_dir + "/frame");

    auto beam_profile  = fall_n::reconstruction::RectangularSectionProfile<2>{COL_B, COL_H};
    auto shell_profile = fall_n::reconstruction::ShellThicknessProfile<5>{};

    int    evol_step       = 0;
    double evol_max_damage = peak_damage_global;

    for (;;) {
        PetscReal t_current;
        TSGetTime(solver.get_ts(), &t_current);
        if (static_cast<double>(t_current) >= T_MAX - 1e-14) break;

        // Advance one global time step (empty director → always Continue)
        auto verdict = solver.step_n(1, fall_n::StepDirector<StructModel>{});
        if (verdict == fall_n::StepVerdict::Stop) {
            std::println("  [!] Global solver stopped at t={:.4f} s",
                         static_cast<double>(t_current));
            break;
        }

        PetscReal t_new;
        TSGetTime(solver.get_ts(), &t_new);
        ++evol_step;
        const double t = static_cast<double>(t_new);

        // ── Update sub-models with new beam kinematics ────────────────
        for (auto& ev : evolvers) {
            auto ek = extract_beam_kinematics(ev.parent_element_id());
            ev.update_kinematics(ek.kin_A, ek.kin_B);
            ev.solve_step(t);
        }

        // ── Track peak damage ─────────────────────────────────────────
        double step_max_damage = 0.0;
        for (std::size_t e = 0; e < model.elements().size(); ++e) {
            auto info = damage_crit.evaluate_element(
                model.elements()[e], e, model.state_vector());
            step_max_damage = std::max(step_max_damage, info.damage_index);
        }
        evol_max_damage = std::max(evol_max_damage, step_max_damage);

        // ── Global VTK snapshot ───────────────────────────────────────
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

        // ── Progress ──────────────────────────────────────────────────
        if (evol_step % EVOL_PRINT_INTERVAL == 0) {
            std::println("  [Evolution] t={:.2f} s  step={}  "
                         "peak_damage={:.4f}  sub_models={}",
                         t, evol_step, evol_max_damage, evolvers.size());
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    //  15. Finalize PVD files + Summary
    // ─────────────────────────────────────────────────────────────────────
    pvd_global.write();
    for (auto& ev : evolvers)
        ev.finalize();

    PetscReal t_final;
    TSGetTime(solver.get_ts(), &t_final);

    sep('=');
    std::println("\nMULTISCALE SEISMIC ANALYSIS -- SUMMARY");
    sep('-');
    std::println("  Earthquake         : Tohoku 2011, MYG004-NS, PGA={:.3f} m/s^2",
                 eq.pga());
    std::println("  Frame              : {}-story RC, {}-bay x {}-bay",
                 NUM_STORIES, X_GRID.size()-1, Y_GRID.size()-1);
    std::println("  Fiber sections     : Kent-Park f'c (col={}, bm={} MPa) + "
                 "Menegotto-Pinto fy={} MPa",
                 COL_FPC, BM_FPC, STEEL_FY);
    std::println("  First yield at     : t = {:.4f} s",
                 transition_report->trigger_time);
    std::println("  Critical element   : index {}",
                 transition_report->critical_element);
    std::println("  Sub-models evolved : {}", evolvers.size());
    std::println("  Evolution steps    : {}", evol_step);
    std::println("  Final time         : {:.4f} s", static_cast<double>(t_final));
    std::println("  Peak damage index  : {:.6f}", evol_max_damage);
    std::println("  VTK output dir     : {}", OUT);
    std::println("  Global PVD         : {}/frame.pvd", evol_frame_dir);
    sep('=');

    PetscFinalize();
    return 0;
}
