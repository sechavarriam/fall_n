// =============================================================================
//  test_steppable_solver.cpp — Phase C: SteppableSolver concept verification
// =============================================================================
//
//  Validates:
//
//    1. SteppableSolver concept satisfaction (NL + Dynamic, compile-time)
//    2. NL begin_incremental + step() — single-step advance
//    3. NL step_n(n) — advance n increments
//    4. NL step_to(p) — advance to control parameter
//    5. NL step_to + pause_at_times — breakpoint at specific p values
//    6. NL step_to + pause_on — custom condition
//    7. NL set_increment_size — runtime dp change
//    8. NL equivalence: step-by-step vs solve_incremental
//    9. NL get_analysis_state — snapshot
//   10. Dynamic get_analysis_state — snapshot
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
using LinA   = LinearAnalysis<Policy, continuum::SmallStrain>;
using NLA    = NonlinearAnalysis<Policy, continuum::SmallStrain>;
using NLA_TL = NonlinearAnalysis<Policy, continuum::TotalLagrangian>;
using NLA_UL = NonlinearAnalysis<Policy, continuum::UpdatedLagrangian>;
using ModelT = Model<Policy, continuum::SmallStrain, NDOF>;
using ModelTL = Model<Policy, continuum::TotalLagrangian, NDOF>;
using ModelUL = Model<Policy, continuum::UpdatedLagrangian, NDOF>;
using DynA   = DynamicAnalysis<Policy>;
using DynTL  = DynamicAnalysis<Policy, continuum::TotalLagrangian>;
using ArcTL  = fall_n::ArcLengthSolver<Policy, continuum::TotalLagrangian>;

using ContSmallElem = ContinuumElement<Policy, NDOF, continuum::SmallStrain>;
using ContTLElem    = ContinuumElement<Policy, NDOF, continuum::TotalLagrangian>;
using ContULElem    = ContinuumElement<Policy, NDOF, continuum::UpdatedLagrangian>;
using BeamCRElem    = BeamElement<TimoshenkoBeam3D, 3, beam::Corotational>;
using ShellSRElem   = MITC4Shell<>;
using ShellCRElem   = CorotationalMITC4Shell<>;

/// Create a fresh Model with unit cube, BCs, and forces.
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

/// Create a fixture for DynamicAnalysis (reuse from steppable_dynamic tests).
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
//  Test 1: SteppableSolver concept satisfaction (compile-time)
// =============================================================================

void test_concept_satisfaction() {
    std::cout << "\nTest 1: SteppableSolver concept satisfaction\n";

    // NonlinearAnalysis
    static_assert(fall_n::SteppableSolver<NLA>,
        "NonlinearAnalysis must satisfy SteppableSolver");
    check(true, "NonlinearAnalysis satisfies SteppableSolver");

    // DynamicAnalysis
    static_assert(fall_n::SteppableSolver<DynA>,
        "DynamicAnalysis must satisfy SteppableSolver");
    check(true, "DynamicAnalysis satisfies SteppableSolver");

    static_assert(fall_n::AnalysisRouteTaggedSolver<LinA>);
    static_assert(fall_n::AnalysisRouteTaggedSolver<NLA>);
    static_assert(fall_n::AnalysisRouteTaggedSolver<NLA_TL>);
    static_assert(fall_n::AnalysisRouteTaggedSolver<DynA>);
    static_assert(fall_n::AnalysisRouteTaggedSolver<DynTL>);
    static_assert(fall_n::AnalysisRouteTaggedSolver<ArcTL>);
    static_assert(fall_n::solver_analysis_route_kind_v<LinA> ==
                  fall_n::AnalysisRouteKind::linear_static);
    static_assert(fall_n::solver_analysis_route_kind_v<NLA> ==
                  fall_n::AnalysisRouteKind::nonlinear_incremental_newton);
    static_assert(fall_n::solver_analysis_route_kind_v<DynTL> ==
                  fall_n::AnalysisRouteKind::implicit_second_order_dynamics);
    static_assert(fall_n::solver_analysis_route_kind_v<ArcTL> ==
                  fall_n::AnalysisRouteKind::arc_length_continuation);
    static_assert(fall_n::solver_analysis_route_audit_scope_v<NLA_TL>.supports_checkpoint_restart);
    static_assert(!fall_n::solver_analysis_route_audit_scope_v<ArcTL>.supports_checkpoint_restart);
    check(true, "analysis solvers expose audited route tags");

    static_assert(fall_n::AuditedFiniteElementType<ContSmallElem>);
    static_assert(fall_n::AuditedFiniteElementType<ContTLElem>);
    static_assert(fall_n::AuditedFiniteElementType<ContULElem>);
    static_assert(fall_n::AuditedFiniteElementType<BeamCRElem>);
    static_assert(fall_n::AuditedFiniteElementType<ShellSRElem>);
    static_assert(fall_n::AuditedFiniteElementType<ShellCRElem>);

    static_assert(fall_n::NormativelySupportedSolverElementPair<ContSmallElem, LinA>);
    static_assert(fall_n::ReferenceLinearSolverElementPair<ContSmallElem, LinA>);
    static_assert(fall_n::NormativelySupportedSolverElementPair<ContTLElem, NLA_TL>);
    static_assert(fall_n::ReferenceGeometricNonlinearitySolverElementPair<ContTLElem, NLA_TL>);
    static_assert(fall_n::NormativelySupportedSolverElementPair<ContULElem, NLA_UL>);
    static_assert(!fall_n::ReferenceGeometricNonlinearitySolverElementPair<ContULElem, NLA_UL>);
    static_assert(!fall_n::NormativelySupportedSolverElementPair<BeamCRElem, NLA>);
    static_assert(fall_n::canonical_element_solver_audit_scope_v<BeamCRElem, NLA>.requires_scope_disclaimer());
    static_assert(fall_n::NormativelySupportedSolverElementPair<ShellSRElem, LinA>);
    static_assert(fall_n::ReferenceLinearSolverElementPair<ShellSRElem, LinA>);
    static_assert(!fall_n::NormativelySupportedSolverElementPair<ShellCRElem, NLA>);
    static_assert(fall_n::canonical_element_solver_audit_scope_v<ShellCRElem, NLA>.requires_scope_disclaimer());
    check(true, "element and solver types compose into audited computational scopes");

    static_assert(fall_n::AuditedComputationalModelType<ModelT>);
    static_assert(fall_n::AuditedComputationalModelType<ModelTL>);
    static_assert(fall_n::AuditedComputationalModelType<ModelUL>);
    static_assert(fall_n::SolverWithAuditedModelSlice<LinA>);
    static_assert(fall_n::SolverWithAuditedModelSlice<NLA_TL>);
    static_assert(fall_n::SolverWithAuditedModelSlice<DynTL>);
    static_assert(fall_n::SolverWithAuditedModelSlice<ArcTL>);
    static_assert(fall_n::NormativelySupportedModelSolverSlice<ModelT, LinA>);
    static_assert(fall_n::ReferenceLinearModelSolverSlice<ModelT, LinA>);
    static_assert(fall_n::NormativelySupportedModelSolverSlice<ModelTL, NLA_TL>);
    static_assert(fall_n::ReferenceGeometricNonlinearityModelSolverSlice<ModelTL, NLA_TL>);
    static_assert(fall_n::NormativelySupportedModelSolverSlice<ModelUL, NLA_UL>);
    static_assert(!fall_n::ReferenceGeometricNonlinearityModelSolverSlice<ModelUL, NLA_UL>);
    static_assert(!fall_n::NormativelySupportedModelSolverSlice<ModelTL, DynTL>);
    static_assert(fall_n::canonical_model_solver_slice_audit_scope_v<ModelTL, DynTL>.requires_scope_disclaimer());
    static_assert(!fall_n::NormativelySupportedModelSolverSlice<ModelTL, ArcTL>);
    static_assert(fall_n::canonical_model_solver_slice_audit_scope_v<ModelTL, ArcTL>.requires_scope_disclaimer());
    check(true, "model and solver types compose into audited computational slices");
}

// =============================================================================
//  Test 2: NL begin_incremental + step() — single-step advance
// =============================================================================

void test_nl_step() {
    std::cout << "\nTest 2: NL begin_incremental + step()\n";

    NLFixture f;
    NLA nl{f.model.get()};

    nl.begin_incremental(5);  // 5 steps → dp = 0.2

    bool ok = nl.step();
    check(ok, "first step converged");
    check(nl.current_step() == 1, "current_step() == 1");
    check(std::abs(nl.current_time() - 0.2) < 1e-12,
          "current_time() == 0.2 after first step");

    ok = nl.step();
    check(ok, "second step converged");
    check(nl.current_step() == 2, "current_step() == 2");
    check(std::abs(nl.current_time() - 0.4) < 1e-12,
          "current_time() == 0.4 after second step");
}

// =============================================================================
//  Test 3: NL step_n(n)
// =============================================================================

void test_nl_step_n() {
    std::cout << "\nTest 3: NL step_n(n)\n";

    NLFixture f;
    NLA nl{f.model.get()};

    nl.begin_incremental(10);  // dp = 0.1

    auto verdict = nl.step_n(5);
    check(verdict == fall_n::StepVerdict::Continue, "step_n(5) returned Continue");
    check(nl.current_step() == 5, "current_step() == 5");
    check(std::abs(nl.current_time() - 0.5) < 1e-12,
          "current_time() == 0.5 after 5 steps");

    verdict = nl.step_n(5);
    check(verdict == fall_n::StepVerdict::Continue, "step_n(5) again returned Continue");
    check(nl.current_step() == 10, "current_step() == 10");
    check(std::abs(nl.current_time() - 1.0) < 1e-12,
          "current_time() == 1.0 after 10 steps");
}

// =============================================================================
//  Test 4: NL step_to(p)
// =============================================================================

void test_nl_step_to() {
    std::cout << "\nTest 4: NL step_to(p)\n";

    NLFixture f;
    NLA nl{f.model.get()};

    nl.begin_incremental(10);  // dp = 0.1

    auto verdict = nl.step_to(0.3);
    check(verdict == fall_n::StepVerdict::Continue, "step_to(0.3) returned Continue");
    check(nl.current_step() == 3, "current_step() == 3");
    check(std::abs(nl.current_time() - 0.3) < 1e-12,
          "current_time() == 0.3");

    verdict = nl.step_to(1.0);
    check(verdict == fall_n::StepVerdict::Continue, "step_to(1.0) returned Continue");
    check(nl.current_step() == 10, "current_step() == 10");
    check(std::abs(nl.current_time() - 1.0) < 1e-12,
          "current_time() == 1.0");
}

// =============================================================================
//  Test 5: NL step_to + pause_at_times
// =============================================================================

void test_nl_pause_at_times() {
    std::cout << "\nTest 5: NL step_to + pause_at_times\n";

    NLFixture f;
    NLA nl{f.model.get()};

    nl.begin_incremental(10);  // dp = 0.1

    // Pause at p = 0.3 and p = 0.7
    auto dir = fall_n::step_director::pause_at_times<ModelT>({0.3, 0.7});

    auto verdict = nl.step_to(1.0, dir);
    check(verdict == fall_n::StepVerdict::Pause, "paused at p=0.3");
    check(std::abs(nl.current_time() - 0.3) < 1e-12,
          "current_time() == 0.3 at first pause");

    // Resume
    verdict = nl.step_to(1.0, dir);
    check(verdict == fall_n::StepVerdict::Pause, "paused at p=0.7");
    check(std::abs(nl.current_time() - 0.7) < 1e-12,
          "current_time() == 0.7 at second pause");

    // Resume to completion
    verdict = nl.step_to(1.0, dir);
    check(verdict == fall_n::StepVerdict::Continue, "completed to p=1.0");
    check(std::abs(nl.current_time() - 1.0) < 1e-12,
          "current_time() == 1.0");
}

// =============================================================================
//  Test 6: NL step_to + pause_on (custom condition)
// =============================================================================

void test_nl_pause_on() {
    std::cout << "\nTest 6: NL step_to + pause_on (custom condition)\n";

    NLFixture f;
    NLA nl{f.model.get()};

    nl.begin_incremental(10);  // dp = 0.1

    // Pause when step count reaches 4
    auto dir = fall_n::step_director::pause_on(
        [](const fall_n::StepEvent& ev, [[maybe_unused]] const ModelT&) {
            return ev.step >= 4;
        });

    auto verdict = nl.step_to(1.0, dir);
    check(verdict == fall_n::StepVerdict::Pause, "paused on custom condition");
    check(nl.current_step() == 4, "paused at step 4");
}

// =============================================================================
//  Test 7: NL set_increment_size — runtime dp change
// =============================================================================

void test_nl_set_increment_size() {
    std::cout << "\nTest 7: NL set_increment_size\n";

    NLFixture f;
    NLA nl{f.model.get()};

    nl.begin_incremental(10);  // dp = 0.1
    check(std::abs(nl.get_increment_size() - 0.1) < 1e-14,
          "initial dp == 0.1");

    // Take 2 steps at dp = 0.1  →  p = 0.2
    nl.step_n(2);

    // Change to dp = 0.2
    nl.set_increment_size(0.2);
    check(std::abs(nl.get_increment_size() - 0.2) < 1e-14,
          "dp changed to 0.2");

    // Take 1 step at dp = 0.2  →  p = 0.4
    nl.step();
    check(std::abs(nl.current_time() - 0.4) < 1e-12,
          "current_time() == 0.4 after dp change");
    check(nl.current_step() == 3, "step count == 3");
}

// =============================================================================
//  Test 8: NL equivalence — step-by-step vs solve_incremental
// =============================================================================

void test_nl_equivalence() {
    std::cout << "\nTest 8: NL equivalence (step-by-step vs solve_incremental)\n";

    const int N = 5;

    // ── Batch solve ─────────────────────────────────────────────
    NLFixture f1;
    NLA nl1{f1.model.get()};
    bool ok1 = nl1.solve_incremental(N);
    check(ok1, "batch solve_incremental converged");

    PetscReal norm1;
    VecNorm(f1.model->state_vector(), NORM_2, &norm1);

    // ── Step-by-step solve ──────────────────────────────────────
    NLFixture f2;
    NLA nl2{f2.model.get()};
    nl2.begin_incremental(N);
    for (int i = 0; i < N; ++i) {
        bool ok = nl2.step();
        check(ok, ("step " + std::to_string(i + 1) + " converged").c_str());
    }

    PetscReal norm2;
    VecNorm(f2.model->state_vector(), NORM_2, &norm2);

    double diff = std::abs(norm1 - norm2);
    std::cout << "    batch norm = " << std::scientific << norm1
              << "  step norm = " << norm2
              << "  diff = " << diff << "\n";

    check(diff < 1e-10, "step-by-step matches solve_incremental");
}

// =============================================================================
//  Test 9: NL get_analysis_state
// =============================================================================

void test_nl_analysis_state() {
    std::cout << "\nTest 9: NL get_analysis_state\n";

    NLFixture f;
    NLA nl{f.model.get()};

    nl.begin_incremental(5);
    nl.step_n(3);

    auto state = nl.get_analysis_state();
    check(state.displacement != nullptr, "displacement is not null");
    check(state.velocity == nullptr, "velocity is null (static)");
    check(std::abs(state.time - 0.6) < 1e-12, "time == 0.6 (p)");
    check(state.step == 3, "step == 3");

    PetscReal norm;
    VecNorm(state.displacement, NORM_2, &norm);
    check(norm > 1e-12, "displacement is non-trivial");
}

// =============================================================================
//  Test 10: Dynamic get_analysis_state
// =============================================================================

void test_dynamic_analysis_state() {
    std::cout << "\nTest 10: Dynamic get_analysis_state\n";

    DynFixture f;
    using DynA = DynamicAnalysis<Policy>;
    DynA dyn{f.model.get()};

    // Apply a small initial displacement
    DM dm = f.model->get_plex();
    Vec u_local;
    DMGetLocalVector(dm, &u_local);
    VecSet(u_local, 0.0);

    // Push node 1 in x
    auto& node1 = f.domain.node(1);
    auto idx = node1.dof_index();
    double val = 0.001;
    VecSetValue(u_local, idx[0], val, INSERT_VALUES);
    VecAssemblyBegin(u_local); VecAssemblyEnd(u_local);

    Vec u_global;
    DMCreateGlobalVector(dm, &u_global);
    VecSet(u_global, 0.0);
    DMLocalToGlobal(dm, u_local, INSERT_VALUES, u_global);
    DMRestoreLocalVector(dm, &u_local);

    dyn.set_initial_displacement(u_global);
    VecDestroy(&u_global);

    // Step a few times
    dyn.step_n(3);

    auto state = dyn.get_analysis_state();
    check(state.displacement != nullptr, "displacement is not null");
    check(state.velocity != nullptr, "velocity is not null (dynamic)");
    check(state.step == 3, "step == 3");
    check(state.time > 0.0, "time > 0");
}


// =============================================================================
//  main
// =============================================================================

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    std::cout << std::string(55, '=') << "\n"
              << "  SteppableSolver Concept Verification (Phase C)\n"
              << std::string(55, '=') << "\n";

    test_concept_satisfaction();
    test_nl_step();
    test_nl_step_n();
    test_nl_step_to();
    test_nl_pause_at_times();
    test_nl_pause_on();
    test_nl_set_increment_size();
    test_nl_equivalence();
    test_nl_analysis_state();
    test_dynamic_analysis_state();

    std::cout << "\n" << std::string(55, '=') << "\n"
              << "  Summary: " << passed << " passed, " << failed << " failed\n"
              << std::string(55, '=') << "\n";

    PetscFinalize();
    return (failed > 0) ? 1 : 0;
}
