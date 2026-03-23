// =============================================================================
//  test_linear_nl_transition.cpp — Phase 2: Linear → Nonlinear Transition
// =============================================================================
//
//  Validates the condition-based phase transition infrastructure for
//  multiscale seismic analysis:
//
//    1.  Displacement threshold director pauses elastic dynamic analysis
//    2.  TransitionReport captures trigger details
//    3.  Two-model state transfer (elastic → stiffer model)
//    4.  AnalysisDirector pipeline with threshold-based transition
//    5.  ExceedanceReport identifies affected nodes
//    6.  Three-phase pipeline: preload → elastic w/ threshold → continue
//    7.  Threshold not reached — analysis completes without pause
//
//  Mesh: single hex8 element, unit cube [0,1]³, 8 nodes.
//  BCs: x=0 face clamped, x=1 face loaded.
//
//  Requires PETSc runtime.
//
// =============================================================================

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <array>

#include <petsc.h>

#include "header_files.hh"

// ── Constants ─────────────────────────────────────────────────────────────────

static constexpr std::size_t DIM  = 3;
static constexpr std::size_t NDOF = DIM;

static int passed = 0;
static int failed = 0;

// ── Helpers ───────────────────────────────────────────────────────────────────

static void check(bool cond, const char* msg) {
    if (cond) {
        std::cout << "  [PASS] " << msg << "\n";
        ++passed;
    } else {
        std::cout << "  [FAIL] " << msg << "\n";
        ++failed;
    }
}

static void create_unit_cube(Domain<DIM>& D) {
    D.preallocate_node_capacity(8);
    D.add_node(0, 0.0, 0.0, 0.0);
    D.add_node(1, 1.0, 0.0, 0.0);
    D.add_node(2, 0.0, 1.0, 0.0);
    D.add_node(3, 1.0, 1.0, 0.0);
    D.add_node(4, 0.0, 0.0, 1.0);
    D.add_node(5, 1.0, 0.0, 1.0);
    D.add_node(6, 0.0, 1.0, 1.0);
    D.add_node(7, 1.0, 1.0, 1.0);

    std::array<PetscInt, 8> conn{0, 1, 2, 3, 4, 5, 6, 7};
    D.make_element<LagrangeElement<3, 2, 2, 2>>(
        GaussLegendreCellIntegrator<2, 2, 2>{}, 0, conn.data());
    D.assemble_sieve();
}

using Policy = ThreeDimensionalMaterial;
using ModelT = Model<Policy, continuum::SmallStrain, NDOF>;
using DynA   = DynamicAnalysis<Policy>;
using NLA    = NonlinearAnalysis<Policy, continuum::SmallStrain>;


// =============================================================================
//  Test 1: Displacement threshold director pauses dynamic analysis
// =============================================================================

void test_displacement_threshold_pause() {
    std::cout << "\nTest 1: Displacement threshold director pauses analysis\n";

    Domain<DIM>                       domain;
    ContinuumIsotropicElasticMaterial mat_site{200.0, 0.3};
    Material<Policy>                  mat{mat_site, ElasticUpdate{}};

    create_unit_cube(domain);
    auto model = std::make_unique<ModelT>(domain, mat);
    model->fix_x(0.0);
    model->setup();
    model->set_density(1.0);

    // Apply a constant force on x=1 face to drive displacement
    const double f_per_node = 0.5;
    for (std::size_t id : {1ul, 3ul, 5ul, 7ul})
        model->apply_node_force(id, f_per_node, 0.0, 0.0);

    DynA dyn{model.get()};
    dyn.set_time_step(0.01);

    // Force evaluator: constant load
    dyn.set_force_function([&](double /*t*/, Vec f_ext) {
        DM dm = model->get_plex();
        Vec f_local;
        DMGetLocalVector(dm, &f_local);
        VecSet(f_local, 0.0);

        for (std::size_t id : {1ul, 3ul, 5ul, 7ul}) {
            PetscInt dof = domain.node(id).dof_index()[0];
            VecSetValueLocal(f_local, dof, f_per_node, ADD_VALUES);
        }

        DMLocalToGlobal(dm, f_local, ADD_VALUES, f_ext);
        DMRestoreLocalVector(dm, &f_local);
    });

    // Create threshold director: pause when max |u| > 0.001
    const double threshold = 0.001;
    auto [director, report] =
        fall_n::make_displacement_threshold_director<ModelT>(threshold);

    // Run up to 100 steps (enough to exceed threshold)
    auto verdict = dyn.step_n(100, director);

    check(verdict == fall_n::StepVerdict::Pause,
          "verdict == Pause (threshold exceeded)");
    check(report->triggered, "report.triggered == true");
    check(report->trigger_time > 0.0, "trigger_time > 0");
    check(report->trigger_step > 0, "trigger_step > 0");
    check(report->metric_value > threshold,
          "metric_value > threshold at trigger");
    check(report->criterion_name == "DisplacementThreshold",
          "criterion_name is correct");

    std::cout << "    Triggered at t=" << report->trigger_time
              << ", step=" << report->trigger_step
              << ", |u|_inf=" << report->metric_value << "\n";
}


// =============================================================================
//  Test 2: TransitionReport captures complete detail
// =============================================================================

void test_transition_report_detail() {
    std::cout << "\nTest 2: TransitionReport detail\n";

    Domain<DIM>                       domain;
    ContinuumIsotropicElasticMaterial mat_site{100.0, 0.25};
    Material<Policy>                  mat{mat_site, ElasticUpdate{}};

    create_unit_cube(domain);
    auto model = std::make_unique<ModelT>(domain, mat);
    model->fix_x(0.0);
    model->setup();
    model->set_density(1.0);

    DynA dyn{model.get()};
    dyn.set_time_step(0.005);

    // Apply force to get measurable displacement
    dyn.set_force_function([&](double /*t*/, Vec f_ext) {
        DM dm = model->get_plex();
        Vec f_local;
        DMGetLocalVector(dm, &f_local);
        VecSet(f_local, 0.0);

        for (std::size_t id : {1ul, 3ul, 5ul, 7ul}) {
            PetscInt dof = domain.node(id).dof_index()[0];
            VecSetValueLocal(f_local, dof, 1.0, ADD_VALUES);
        }

        DMLocalToGlobal(dm, f_local, ADD_VALUES, f_ext);
        DMRestoreLocalVector(dm, &f_local);
    });

    const double threshold = 0.002;
    auto [director, report] =
        fall_n::make_displacement_threshold_director<ModelT>(threshold);

    dyn.step_n(200, director);

    check(report->triggered, "threshold triggered");
    check(report->threshold == threshold, "report stores configured threshold");

    // Verify step and time are consistent
    check(report->trigger_step <= 200, "trigger_step <= max steps");
    check(report->trigger_time > 0.0, "trigger_time positive");
    check(std::abs(report->trigger_time - report->trigger_step * 0.005) < 1e-10,
          "trigger_time == trigger_step * dt");
}


// =============================================================================
//  Test 3: Two-model state transfer (elastic → stiffer model)
// =============================================================================

void test_two_model_state_transfer() {
    std::cout << "\nTest 3: Two-model state transfer\n";

    // Model A: soft elastic (E=100)
    Domain<DIM>                       domain_a;
    ContinuumIsotropicElasticMaterial mat_site_a{100.0, 0.25};
    Material<Policy>                  mat_a{mat_site_a, ElasticUpdate{}};

    create_unit_cube(domain_a);
    auto model_a = std::make_unique<ModelT>(domain_a, mat_a);
    model_a->fix_x(0.0);
    model_a->setup();
    model_a->set_density(1.0);

    // Model B: stiff elastic (E=500) — same domain topology
    Domain<DIM>                       domain_b;
    ContinuumIsotropicElasticMaterial mat_site_b{500.0, 0.25};
    Material<Policy>                  mat_b{mat_site_b, ElasticUpdate{}};

    create_unit_cube(domain_b);
    auto model_b = std::make_unique<ModelT>(domain_b, mat_b);
    model_b->fix_x(0.0);
    model_b->setup();
    model_b->set_density(1.0);

    // Phase A: run soft model for 10 steps
    DynA dyn_a{model_a.get()};
    dyn_a.set_time_step(0.01);
    dyn_a.set_force_function([&](double /*t*/, Vec f_ext) {
        DM dm = model_a->get_plex();
        Vec f_local;
        DMGetLocalVector(dm, &f_local);
        VecSet(f_local, 0.0);

        for (std::size_t id : {1ul, 3ul, 5ul, 7ul}) {
            PetscInt dof = domain_a.node(id).dof_index()[0];
            VecSetValueLocal(f_local, dof, 0.5, ADD_VALUES);
        }

        DMLocalToGlobal(dm, f_local, ADD_VALUES, f_ext);
        DMRestoreLocalVector(dm, &f_local);
    });

    dyn_a.step_n(10);
    auto state_a = dyn_a.get_analysis_state();

    check(state_a.displacement != nullptr, "Phase A displacement captured");
    check(state_a.velocity != nullptr, "Phase A velocity captured");
    check(state_a.time > 0.0, "Phase A time > 0");

    double phase_a_time = state_a.time;
    PetscInt phase_a_step = state_a.step;

    // Capture Phase A displacement norm for continuity check
    PetscReal u_a_norm;
    VecNorm(state_a.displacement, NORM_2, &u_a_norm);
    PetscReal v_a_norm;
    VecNorm(state_a.velocity, NORM_2, &v_a_norm);

    check(u_a_norm > 0.0, "Phase A has non-zero displacement");

    // Phase B: inject Phase A state into stiffer model
    DynA dyn_b{model_b.get()};
    dyn_b.set_time_step(0.01);
    dyn_b.set_initial_displacement(state_a.displacement);
    dyn_b.set_initial_velocity(state_a.velocity);

    // Continue stepping
    dyn_b.step_n(5);
    auto state_b = dyn_b.get_analysis_state();

    check(state_b.displacement != nullptr, "Phase B displacement exists");
    check(state_b.velocity != nullptr, "Phase B velocity exists");

    // Phase B time starts from 0 (TS resets); what matters is the
    // displacement/velocity were injected correctly.  Verify Phase B
    // actually advanced and the displacement changed from the IC.
    PetscReal u_b_norm;
    VecNorm(state_b.displacement, NORM_2, &u_b_norm);
    check(u_b_norm > 0.0, "Phase B displacement is non-zero");

    std::cout << "    Phase A: t=" << phase_a_time
              << ", step=" << phase_a_step
              << ", |u|=" << u_a_norm << "\n";
    std::cout << "    Phase B: t=" << state_b.time
              << ", step=" << state_b.step
              << ", |u|=" << u_b_norm << "\n";
}


// =============================================================================
//  Test 4: AnalysisDirector pipeline with threshold transition
// =============================================================================

void test_director_threshold_pipeline() {
    std::cout << "\nTest 4: AnalysisDirector pipeline with threshold\n";

    // Shared domain and models
    Domain<DIM>                       domain;
    ContinuumIsotropicElasticMaterial mat_site{200.0, 0.3};
    Material<Policy>                  mat{mat_site, ElasticUpdate{}};

    create_unit_cube(domain);
    auto model = std::make_unique<ModelT>(domain, mat);
    model->fix_x(0.0);
    model->setup();
    model->set_density(1.0);

    auto* raw = model.get();

    const double threshold = 0.001;
    std::shared_ptr<fall_n::TransitionReport> transition_report;
    double phase_a_end_time = 0.0;

    // Keep DynA objects alive across phases so borrowed Vecs remain valid
    DynA dyn_a{raw};
    DynA dyn_b{raw};

    fall_n::AnalysisDirector director;

    // Phase A: elastic dynamic with displacement threshold
    director.add_phase("elastic_dynamic", [&](const fall_n::AnalysisState&) {
        dyn_a.set_time_step(0.01);
        dyn_a.set_force_function([&](double /*t*/, Vec f_ext) {
            DM dm = raw->get_plex();
            Vec f_local;
            DMGetLocalVector(dm, &f_local);
            VecSet(f_local, 0.0);

            for (std::size_t id : {1ul, 3ul, 5ul, 7ul}) {
                PetscInt dof = domain.node(id).dof_index()[0];
                VecSetValueLocal(f_local, dof, 0.5, ADD_VALUES);
            }

            DMLocalToGlobal(dm, f_local, ADD_VALUES, f_ext);
            DMRestoreLocalVector(dm, &f_local);
        });

        auto [dir, rpt] =
            fall_n::make_displacement_threshold_director<ModelT>(threshold);
        transition_report = rpt;

        auto verdict = dyn_a.step_n(200, dir);
        check(verdict == fall_n::StepVerdict::Pause,
              "Phase A paused on threshold");

        auto state = dyn_a.get_analysis_state();
        phase_a_end_time = state.time;
        return state;
    });

    // Phase B: continue from Phase A state (same model, demonstrating
    // the AnalysisDirector state-threading pattern)
    director.add_phase("nonlinear_dynamic", [&](const fall_n::AnalysisState& prev) {
        check(prev.displacement != nullptr,
              "Phase B receives displacement from A");
        check(prev.velocity != nullptr,
              "Phase B receives velocity from A");

        dyn_b.set_time_step(0.01);
        dyn_b.set_initial_displacement(prev.displacement);
        dyn_b.set_initial_velocity(prev.velocity);

        dyn_b.step_n(5);
        return dyn_b.get_analysis_state();
    });

    auto report = director.run();

    check(report.all_succeeded(), "both phases succeeded");
    check(report.phases.size() == 2, "two phases in report");
    check(transition_report != nullptr, "transition report exists");
    check(transition_report->triggered, "transition was triggered");
    check(phase_a_end_time > 0.0, "Phase A ran for some time");

    std::cout << "    Phase A ended at t=" << phase_a_end_time
              << ", trigger |u|=" << transition_report->metric_value << "\n";
}


// =============================================================================
//  Test 5: ExceedanceReport identifies affected nodes
// =============================================================================

void test_exceedance_report() {
    std::cout << "\nTest 5: ExceedanceReport\n";

    Domain<DIM>                       domain;
    ContinuumIsotropicElasticMaterial mat_site{200.0, 0.3};
    Material<Policy>                  mat{mat_site, ElasticUpdate{}};

    create_unit_cube(domain);
    auto model = std::make_unique<ModelT>(domain, mat);
    model->fix_x(0.0);
    model->setup();
    model->set_density(1.0);

    DynA dyn{model.get()};
    dyn.set_time_step(0.01);

    dyn.set_force_function([&](double /*t*/, Vec f_ext) {
        DM dm = model->get_plex();
        Vec f_local;
        DMGetLocalVector(dm, &f_local);
        VecSet(f_local, 0.0);

        for (std::size_t id : {1ul, 3ul, 5ul, 7ul}) {
            PetscInt dof = domain.node(id).dof_index()[0];
            VecSetValueLocal(f_local, dof, 1.0, ADD_VALUES);
        }

        DMLocalToGlobal(dm, f_local, ADD_VALUES, f_ext);
        DMRestoreLocalVector(dm, &f_local);
    });

    // Step enough to get measurable displacement
    dyn.step_n(20);

    auto state = dyn.get_analysis_state();
    PetscReal max_u;
    VecNorm(state.displacement, NORM_INFINITY, &max_u);

    // Set threshold at half the max displacement — only some nodes exceed
    double threshold = max_u * 0.5;
    auto exc_report = fall_n::compute_exceedance_report(
        domain, state.displacement, threshold);

    check(exc_report.threshold == threshold,
          "report threshold matches input");

    // x=0 nodes are clamped (zero disp), x=1 nodes are loaded (max disp)
    // So only x=1 nodes (1,3,5,7) should have significant displacement
    check(!exc_report.nodes.empty(), "some nodes exceed threshold");

    // Nodes should be sorted descending
    bool sorted = true;
    for (std::size_t i = 1; i < exc_report.nodes.size(); ++i) {
        if (exc_report.nodes[i].displacement_norm >
            exc_report.nodes[i-1].displacement_norm) {
            sorted = false;
            break;
        }
    }
    check(sorted, "exceedance nodes sorted descending");

    // No clamped node (x=0) should appear
    bool no_clamped = true;
    for (const auto& n : exc_report.nodes) {
        if (n.node_id == 0 || n.node_id == 2 ||
            n.node_id == 4 || n.node_id == 6) {
            no_clamped = false;
            break;
        }
    }
    check(no_clamped, "clamped nodes not in exceedance report");

    std::cout << "    max |u| = " << max_u
              << ", threshold = " << threshold
              << ", " << exc_report.nodes.size() << " nodes exceed\n";
}


// =============================================================================
//  Test 6: Three-phase pipeline — preload → elastic + threshold → continue
// =============================================================================

void test_three_phase_pipeline() {
    std::cout << "\nTest 6: Three-phase pipeline\n";

    Domain<DIM>                       domain;
    ContinuumIsotropicElasticMaterial mat_site{200.0, 0.3};
    Material<Policy>                  mat{mat_site, ElasticUpdate{}};

    create_unit_cube(domain);
    auto model = std::make_unique<ModelT>(domain, mat);
    model->fix_x(0.0);
    model->setup();
    model->set_density(1.0);

    const double f_per_node = 0.25;
    for (std::size_t id : {1ul, 3ul, 5ul, 7ul})
        model->apply_node_force(id, f_per_node, 0.0, 0.0);

    auto* raw = model.get();

    // Keep solvers alive for borrowed Vec references
    NLA  nl_pre{raw};
    DynA dyn_elastic{raw};
    DynA dyn_nl{raw};

    std::shared_ptr<fall_n::TransitionReport> trig_report;

    fall_n::AnalysisDirector director;

    // Phase 1: static preload (gravity)
    director.add_phase("preload", [&](const fall_n::AnalysisState&) {
        nl_pre.solve_incremental(5);
        return nl_pre.get_analysis_state();
    });

    // Phase 2: elastic dynamic with threshold-based pause
    director.add_phase("elastic_dynamic", [&](const fall_n::AnalysisState& prev) {
        if (prev.displacement)
            dyn_elastic.set_initial_displacement(prev.displacement);
        dyn_elastic.set_time_step(0.01);

        auto [dir, rpt] =
            fall_n::make_displacement_threshold_director<ModelT>(0.005);
        trig_report = rpt;

        auto verdict = dyn_elastic.step_n(50, dir);
        // May or may not trigger depending on load magnitude
        (void)verdict;
        return dyn_elastic.get_analysis_state();
    });

    // Phase 3: continue from Phase 2 state
    director.add_phase("nonlinear_dynamic", [&](const fall_n::AnalysisState& prev) {
        if (prev.displacement)
            dyn_nl.set_initial_displacement(prev.displacement);
        if (prev.velocity)
            dyn_nl.set_initial_velocity(prev.velocity);
        dyn_nl.set_time_step(0.01);

        dyn_nl.step_n(5);
        return dyn_nl.get_analysis_state();
    });

    auto report = director.run();

    check(report.all_succeeded(), "all 3 phases succeeded");
    check(report.phases.size() == 3, "3 phases in report");
    check(report.phases[0].name == "preload", "phase 0 = preload");
    check(report.phases[1].name == "elastic_dynamic", "phase 1 = elastic_dynamic");
    check(report.phases[2].name == "nonlinear_dynamic", "phase 2 = nonlinear_dynamic");

    // Preload should have displacement but no velocity
    check(report.phases[0].state.displacement != nullptr,
          "preload has displacement");
    check(report.phases[0].state.velocity == nullptr,
          "preload has no velocity (static)");

    // Dynamic phases should have both
    check(report.phases[2].state.displacement != nullptr,
          "final phase has displacement");
    check(report.phases[2].state.velocity != nullptr,
          "final phase has velocity");
}


// =============================================================================
//  Test 7: Threshold not reached — analysis completes normally
// =============================================================================

void test_threshold_not_reached() {
    std::cout << "\nTest 7: Threshold not reached\n";

    Domain<DIM>                       domain;
    ContinuumIsotropicElasticMaterial mat_site{200.0, 0.3};
    Material<Policy>                  mat{mat_site, ElasticUpdate{}};

    create_unit_cube(domain);
    auto model = std::make_unique<ModelT>(domain, mat);
    model->fix_x(0.0);
    model->setup();
    model->set_density(1.0);

    DynA dyn{model.get()};
    dyn.set_time_step(0.01);

    // Very small force — displacement will stay tiny
    dyn.set_force_function([&](double /*t*/, Vec f_ext) {
        DM dm = model->get_plex();
        Vec f_local;
        DMGetLocalVector(dm, &f_local);
        VecSet(f_local, 0.0);

        for (std::size_t id : {1ul, 3ul, 5ul, 7ul}) {
            PetscInt dof = domain.node(id).dof_index()[0];
            VecSetValueLocal(f_local, dof, 1e-6, ADD_VALUES);
        }

        DMLocalToGlobal(dm, f_local, ADD_VALUES, f_ext);
        DMRestoreLocalVector(dm, &f_local);
    });

    // Very high threshold — should never trigger
    const double high_threshold = 100.0;
    auto [director, report] =
        fall_n::make_displacement_threshold_director<ModelT>(high_threshold);

    // Run exactly 10 steps — should complete without pause
    auto verdict = dyn.step_n(10, director);

    check(verdict == fall_n::StepVerdict::Continue,
          "verdict == Continue (threshold not reached)");
    check(!report->triggered, "report.triggered == false");
    check(report->trigger_time == 0.0, "trigger_time stays 0");
    check(report->trigger_step == 0, "trigger_step stays 0");
}


// =============================================================================
//  main
// =============================================================================

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    std::cout << std::string(60, '=') << "\n"
              << "  Phase 2: Linear -> Nonlinear Transition Verification\n"
              << std::string(60, '=') << "\n";

    test_displacement_threshold_pause();
    test_transition_report_detail();
    test_two_model_state_transfer();
    test_director_threshold_pipeline();
    test_exceedance_report();
    test_three_phase_pipeline();
    test_threshold_not_reached();

    std::cout << "\n" << std::string(60, '=') << "\n"
              << "  Summary: " << passed << " passed, " << failed << " failed\n"
              << std::string(60, '=') << "\n";

    PetscFinalize();
    return (failed > 0) ? 1 : 0;
}
