// =============================================================================
//  test_analysis_director.cpp — Phase D: AnalysisDirector orchestrator tests
// =============================================================================
//
//  Validates:
//
//    1.  Empty director — run() returns empty report, all_succeeded() false
//    2.  Single phase (static) — basic execution and state output
//    3.  Two-phase sequence — state threading from phase 1 → phase 2
//    4.  Static preload → Dynamic excitation (multi-engine)
//    5.  Abort-on-failure — director stops after failing phase
//    6.  Continue-on-failure mode — director runs all phases
//    7.  Phase gate callback — conditional skip
//    8.  DirectorReport summary — num_succeeded/num_failed
//    9.  Three-phase pipeline — preload → excite → redistribute
//   10.  Phase timing — elapsed_seconds > 0 for real phases
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
using NLA    = NonlinearAnalysis<Policy, continuum::SmallStrain>;
using ModelT = Model<Policy, continuum::SmallStrain, NDOF>;
using DynA   = DynamicAnalysis<Policy>;

/// NL fixture: elastic unit cube, x=0 clamped, x=1 loaded.
struct NLFixture {
    Domain<DIM>                               domain;
    ContinuumIsotropicElasticMaterial         mat_site{200.0, 0.3};
    Material<Policy>                          mat{mat_site, ElasticUpdate{}};
    std::unique_ptr<ModelT>                   model;

    NLFixture() {
        create_unit_cube(domain);
        model = std::make_unique<ModelT>(domain, mat);
        model->fix_x(0.0);
        model->setup();

        const double f_per_node = 1.0 / 4.0;
        for (std::size_t id : {1ul, 3ul, 5ul, 7ul})
            model->apply_node_force(id, f_per_node, 0.0, 0.0);
    }
};

/// Dynamic fixture: elastic unit cube with mass, x=0 clamped.
struct DynFixture {
    Domain<DIM>                               domain;
    ContinuumIsotropicElasticMaterial         mat_site{1000.0, 0.0};
    Material<Policy>                          mat{mat_site, ElasticUpdate{}};
    std::unique_ptr<Model<Policy>>            model;

    DynFixture() {
        create_unit_cube(domain);
        model = std::make_unique<Model<Policy>>(domain, mat);
        model->fix_x(0.0);
        model->setup();
        model->set_density(1.0);
    }
};


// =============================================================================
//  Test 1: Empty director
// =============================================================================

void test_empty_director() {
    std::cout << "\nTest 1: Empty director\n";

    fall_n::AnalysisDirector director;
    check(director.num_phases() == 0, "num_phases() == 0");

    auto report = director.run();
    check(report.phases.empty(), "report.phases is empty");
    check(!report.all_succeeded(), "all_succeeded() is false for empty");
    check(report.num_succeeded() == 0, "num_succeeded() == 0");
    check(report.num_failed() == 0, "num_failed() == 0");
}


// =============================================================================
//  Test 2: Single phase (static)
// =============================================================================

void test_single_phase_static() {
    std::cout << "\nTest 2: Single phase (static)\n";

    NLFixture f;

    fall_n::AnalysisDirector director;
    director.add_phase("gravity", [&](const fall_n::AnalysisState&) {
        NLA nl{f.model.get()};
        bool ok = nl.solve_incremental(5);
        check(ok, "NL solve converged");
        return nl.get_analysis_state();
    });

    check(director.num_phases() == 1, "num_phases() == 1");

    auto report = director.run();
    check(report.all_succeeded(), "all_succeeded()");
    check(report.phases.size() == 1, "one phase in report");
    check(report.phases[0].status == fall_n::PhaseStatus::Succeeded,
          "phase status == Succeeded");
    check(report.phases[0].state.displacement != nullptr,
          "output displacement is not null");
    check(report.phases[0].state.velocity == nullptr,
          "output velocity is null (static)");
    check(std::abs(report.phases[0].state.time - 1.0) < 1e-12,
          "output time == 1.0 (full load)");
}


// =============================================================================
//  Test 3: Two-phase state threading
// =============================================================================

void test_two_phase_state_threading() {
    std::cout << "\nTest 3: Two-phase state threading\n";

    NLFixture f;

    // Phase 1 applies 50% load, phase 2 continues to 100%.
    // Both share the same model so state carries naturally.

    double phase1_time = 0.0;

    fall_n::AnalysisDirector director;
    director.add_phase("phase1", [&](const fall_n::AnalysisState&) {
        NLA nl{f.model.get()};
        nl.begin_incremental(10);  // dp = 0.1
        nl.step_to(0.5);
        auto st = nl.get_analysis_state();
        phase1_time = st.time;
        return st;
    });

    director.add_phase("phase2", [&](const fall_n::AnalysisState& prev) {
        check(prev.displacement != nullptr,
              "phase2 receives displacement from phase1");
        check(std::abs(prev.time - 0.5) < 1e-12,
              "phase2 receives time==0.5 from phase1");

        NLA nl{f.model.get()};
        nl.begin_incremental(10);
        // Continue from where phase1 left off (model state persists)
        nl.step_to(1.0);
        return nl.get_analysis_state();
    });

    auto report = director.run();
    check(report.all_succeeded(), "both phases succeeded");
    check(report.phases.size() == 2, "two phases in report");
    check(std::abs(phase1_time - 0.5) < 1e-12, "phase1 reached p=0.5");
    check(std::abs(report.phases[1].state.time - 1.0) < 1e-12,
          "phase2 reached p=1.0");
}


// =============================================================================
//  Test 4: Static preload → Dynamic excitation (multi-engine)
// =============================================================================

void test_static_then_dynamic() {
    std::cout << "\nTest 4: Static preload -> Dynamic excitation\n";

    // Both engines share the SAME domain/model so displacement vectors
    // are compatible for state transfer.
    //
    // IMPORTANT: The NLA solver must remain alive while phase 2 reads
    // its state, because AnalysisState holds borrowed Vec references.
    // We achieve this by constructing the NLA outside the callbacks.

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

    auto* raw_model = model.get();

    // Keep NLA alive across both phases so its Vec is valid in phase 2.
    NLA nl{raw_model};

    fall_n::AnalysisDirector director;

    director.add_phase("preload", [&](const fall_n::AnalysisState&) {
        bool ok = nl.solve_incremental(5);
        check(ok, "preload converged");
        return nl.get_analysis_state();
    });

    director.add_phase("dynamic", [&](const fall_n::AnalysisState& prev) {
        check(prev.displacement != nullptr,
              "dynamic phase receives displacement");

        DynA dyn{raw_model};
        dyn.set_initial_displacement(prev.displacement);
        dyn.set_time_step(0.01);

        dyn.step_n(5);

        auto state = dyn.get_analysis_state();
        check(state.displacement != nullptr, "dynamic displacement not null");
        check(state.velocity != nullptr, "dynamic velocity not null");
        check(state.time > 0.0, "dynamic time > 0");
        check(state.step == 5, "dynamic step == 5");
        return state;
    });

    auto report = director.run();
    check(report.all_succeeded(), "both phases succeeded");
    check(report.phases[0].state.velocity == nullptr,
          "static phase has no velocity");
    check(report.phases[1].state.velocity != nullptr,
          "dynamic phase has velocity");
}


// =============================================================================
//  Test 5: Abort-on-failure
// =============================================================================

void test_abort_on_failure() {
    std::cout << "\nTest 5: Abort-on-failure\n";

    fall_n::AnalysisDirector director;

    // Phase 1: fails (returns nullptr displacement)
    director.add_phase("failing", [](const fall_n::AnalysisState&) {
        return fall_n::AnalysisState{};  // displacement == nullptr → failure
    });

    // Phase 2: should never run
    bool phase2_ran = false;
    director.add_phase("skipped", [&](const fall_n::AnalysisState&) {
        phase2_ran = true;
        return fall_n::AnalysisState{};
    });

    auto report = director.run();
    check(!report.all_succeeded(), "not all succeeded");
    check(report.num_failed() == 1, "1 failed phase");
    check(report.phases.size() == 1, "only 1 phase recorded (aborted)");
    check(!phase2_ran, "phase2 did not run");
}


// =============================================================================
//  Test 6: Continue-on-failure mode
// =============================================================================

void test_continue_on_failure() {
    std::cout << "\nTest 6: Continue-on-failure\n";

    fall_n::AnalysisDirector director;
    director.set_continue_on_failure(true);

    // Phase 1: fails
    director.add_phase("failing", [](const fall_n::AnalysisState&) {
        return fall_n::AnalysisState{};
    });

    // Phase 2: succeeds
    NLFixture f;
    director.add_phase("recovery", [&](const fall_n::AnalysisState&) {
        NLA nl{f.model.get()};
        nl.solve_incremental(3);
        return nl.get_analysis_state();
    });

    auto report = director.run();
    check(!report.all_succeeded(), "not all succeeded");
    check(report.phases.size() == 2, "both phases recorded");
    check(report.num_failed() == 1, "1 failed");
    check(report.num_succeeded() == 1, "1 succeeded");
    check(report.phases[0].status == fall_n::PhaseStatus::Failed,
          "first phase failed");
    check(report.phases[1].status == fall_n::PhaseStatus::Succeeded,
          "second phase succeeded despite first failure");
}


// =============================================================================
//  Test 7: Phase gate callback — conditional skip
// =============================================================================

void test_phase_gate() {
    std::cout << "\nTest 7: Phase gate callback\n";

    NLFixture f;

    fall_n::AnalysisDirector director;

    director.add_phase("preload", [&](const fall_n::AnalysisState&) {
        NLA nl{f.model.get()};
        nl.solve_incremental(3);
        return nl.get_analysis_state();
    });

    // Phase 2: gate checks if preload reached p >= 0.9.
    // Since the preload is 3 increments → p=1.0, gate passes.
    bool gated_ran = false;
    director.add_phase(
        "optional_analysis",
        [&](const fall_n::AnalysisState& prev) {
            gated_ran = true;
            return prev;  // just pass through
        },
        [](const fall_n::PhaseOutcome& prev_outcome) {
            return prev_outcome.state.time >= 0.9;  // gate: only if p >= 0.9
        });

    auto report = director.run();
    check(report.all_succeeded(), "all phases succeeded");
    check(gated_ran, "gated phase ran (gate passed)");

    // Now test with a gate that blocks:
    NLFixture f2;

    fall_n::AnalysisDirector director2;
    director2.add_phase("partial_load", [&](const fall_n::AnalysisState&) {
        NLA nl{f2.model.get()};
        nl.begin_incremental(10);
        nl.step_to(0.3);  // only reach p=0.3
        return nl.get_analysis_state();
    });

    bool blocked_ran = false;
    director2.add_phase(
        "blocked_phase",
        [&](const fall_n::AnalysisState&) {
            blocked_ran = true;
            return fall_n::AnalysisState{};
        },
        [](const fall_n::PhaseOutcome& prev) {
            return prev.state.time >= 0.9;  // p=0.3 < 0.9 → skip
        });

    auto report2 = director2.run();
    check(!blocked_ran, "gated phase was skipped (gate failed)");
    check(report2.phases.size() == 2, "both phases recorded");
    check(report2.phases[1].status == fall_n::PhaseStatus::Skipped,
          "blocked phase status == Skipped");
}


// =============================================================================
//  Test 8: DirectorReport summary queries
// =============================================================================

void test_report_summary() {
    std::cout << "\nTest 8: DirectorReport summary\n";

    NLFixture f1, f2;

    fall_n::AnalysisDirector director;
    director.set_continue_on_failure(true);

    director.add_phase("ok1", [&](const fall_n::AnalysisState&) {
        NLA nl{f1.model.get()};
        nl.solve_incremental(3);
        return nl.get_analysis_state();
    });

    // Failing phase
    director.add_phase("fail", [](const fall_n::AnalysisState&) {
        return fall_n::AnalysisState{};
    });

    director.add_phase("ok2", [&](const fall_n::AnalysisState&) {
        NLA nl{f2.model.get()};
        nl.solve_incremental(3);
        return nl.get_analysis_state();
    });

    auto report = director.run();
    check(report.phases.size() == 3, "3 phases in report");
    check(report.num_succeeded() == 2, "num_succeeded == 2");
    check(report.num_failed() == 1, "num_failed == 1");
    check(!report.all_succeeded(), "not all succeeded");
}


// =============================================================================
//  Test 9: Three-phase pipeline (preload → excite → redistribute)
// =============================================================================

void test_three_phase_pipeline() {
    std::cout << "\nTest 9: Three-phase pipeline\n";

    // All three phases share the same domain/model so vectors are compatible.
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

    // Keep solvers alive so borrowed Vec references remain valid.
    NLA  nl_pre{raw};
    DynA dyn{raw};
    NLA  nl_post{raw};

    fall_n::AnalysisDirector director;

    // Phase 1: static preload
    director.add_phase("preload", [&](const fall_n::AnalysisState&) {
        nl_pre.solve_incremental(5);
        return nl_pre.get_analysis_state();
    });

    // Phase 2: dynamic excitation
    director.add_phase("excitation", [&](const fall_n::AnalysisState& prev) {
        if (prev.displacement)
            dyn.set_initial_displacement(prev.displacement);
        dyn.set_time_step(0.01);
        dyn.step_n(3);
        return dyn.get_analysis_state();
    });

    // Phase 3: static redistribution
    director.add_phase("redistribution", [&](const fall_n::AnalysisState& prev) {
        check(prev.displacement != nullptr,
              "redistribution receives displacement");
        check(prev.velocity != nullptr,
              "redistribution receives velocity");

        nl_post.solve_incremental(3);
        return nl_post.get_analysis_state();
    });

    auto report = director.run();
    check(report.all_succeeded(), "all 3 phases succeeded");
    check(report.phases.size() == 3, "3 phases in report");

    // Verify phase names
    check(report.phases[0].name == "preload", "phase 0 name");
    check(report.phases[1].name == "excitation", "phase 1 name");
    check(report.phases[2].name == "redistribution", "phase 2 name");
}


// =============================================================================
//  Test 10: Phase timing — elapsed_seconds > 0
// =============================================================================

void test_phase_timing() {
    std::cout << "\nTest 10: Phase timing\n";

    NLFixture f;

    fall_n::AnalysisDirector director;
    director.add_phase("timed", [&](const fall_n::AnalysisState&) {
        NLA nl{f.model.get()};
        nl.solve_incremental(5);
        return nl.get_analysis_state();
    });

    auto report = director.run();
    check(report.phases[0].elapsed_seconds > 0.0,
          "elapsed_seconds > 0 for real computation");
    std::cout << "    elapsed: " << report.phases[0].elapsed_seconds << " s\n";
}


// =============================================================================
//  main
// =============================================================================

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    std::cout << std::string(55, '=') << "\n"
              << "  AnalysisDirector Verification (Phase D)\n"
              << std::string(55, '=') << "\n";

    test_empty_director();
    test_single_phase_static();
    test_two_phase_state_threading();
    test_static_then_dynamic();
    test_abort_on_failure();
    test_continue_on_failure();
    test_phase_gate();
    test_report_summary();
    test_three_phase_pipeline();
    test_phase_timing();

    std::cout << "\n" << std::string(55, '=') << "\n"
              << "  Summary: " << passed << " passed, " << failed << " failed\n"
              << std::string(55, '=') << "\n";

    PetscFinalize();
    return (failed > 0) ? 1 : 0;
}
