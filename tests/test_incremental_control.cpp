// =============================================================================
//  test_incremental_control.cpp — IncrementalControlPolicy verification
// =============================================================================
//
//  Validates the injectable control-scheme infrastructure:
//
//    1. LoadControl default (backward-compatible, pure load scaling)
//    2. LoadControl explicit (identical to default)
//    3. DisplacementControl single DOF (pushover-style)
//    4. DisplacementControl multi-DOF
//    5. CustomControl via make_control (lambda-based)
//    6. Concept satisfaction (static checks)
//
//  Requires PETSc runtime (PetscInitialize / PetscFinalize).
//
// =============================================================================

#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <array>

#include <petsc.h>

#include "header_files.hh"
#include "src/post-processing/StateQuery.hh"

// ── Constants ─────────────────────────────────────────────────────────────────

static constexpr std::size_t DIM  = 3;
static constexpr std::size_t NDOF = DIM;

static int passed = 0;
static int failed = 0;

static void check(bool cond, const char* msg) {
    if (cond) { std::cout << "  [PASS] " << msg << "\n"; ++passed; }
    else      { std::cout << "  [FAIL] " << msg << "\n"; ++failed; }
}

// ── Helper: create a single-hex unit cube domain ──────────────────────────────

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

// =============================================================================
//  Test 1: LoadControl default — backward compatibility
// =============================================================================

void test_load_control_default() {
    std::cout << "\nTest 1: LoadControl default (backward-compatible)\n";

    Domain<DIM> D;
    create_unit_cube(D);

    ContinuumIsotropicElasticMaterial elastic{200.0, 0.3};
    Material<Policy> mat{elastic, ElasticUpdate{}};

    Model<Policy, continuum::SmallStrain, NDOF> M{D, mat};
    M.fix_x(0.0);
    M.setup();

    const double f_per_node = 1.0 / 4.0;
    for (std::size_t id : {1ul, 3ul, 5ul, 7ul})
        M.apply_node_force(id, f_per_node, 0.0, 0.0);

    NonlinearAnalysis<Policy, continuum::SmallStrain> nl{&M};

    // Default overload — should behave exactly as before
    bool ok = nl.solve_incremental(3);
    check(ok, "solve_incremental(3) with default LoadControl converged");

    PetscReal norm;
    VecNorm(M.state_vector(), NORM_2, &norm);
    check(norm > 1e-12, "solution is non-trivial");
}

// =============================================================================
//  Test 2: LoadControl explicit — same result as default
// =============================================================================

void test_load_control_explicit() {
    std::cout << "\nTest 2: LoadControl explicit (identical result)\n";

    Domain<DIM> D;
    create_unit_cube(D);

    ContinuumIsotropicElasticMaterial elastic{200.0, 0.3};
    Material<Policy> mat{elastic, ElasticUpdate{}};

    Model<Policy, continuum::SmallStrain, NDOF> M{D, mat};
    M.fix_x(0.0);
    M.setup();

    const double f_per_node = 1.0 / 4.0;
    for (std::size_t id : {1ul, 3ul, 5ul, 7ul})
        M.apply_node_force(id, f_per_node, 0.0, 0.0);

    NonlinearAnalysis<Policy, continuum::SmallStrain> nl{&M};

    bool ok = nl.solve_incremental(3, 4, LoadControl{});
    check(ok, "solve_incremental(3, 4, LoadControl{}) converged");

    PetscReal norm;
    VecNorm(M.state_vector(), NORM_2, &norm);
    check(norm > 1e-12, "solution is non-trivial");
}

// =============================================================================
//  Test 3: DisplacementControl — single DOF pushover
// =============================================================================

void test_displacement_control_single() {
    std::cout << "\nTest 3: DisplacementControl single DOF\n";

    Domain<DIM> D;
    create_unit_cube(D);

    ContinuumIsotropicElasticMaterial elastic{200.0, 0.3};
    Material<Policy> mat{elastic, ElasticUpdate{}};

    Model<Policy, continuum::SmallStrain, NDOF> M{D, mat};

    // Fix left face (x = 0)
    M.fix_x(0.0);

    // Pre-constrain the controlled DOF (node 1, dof 0 = x-direction)
    // with initial value 0.0 — the actual value is set by the scheme.
    M.constrain_dof(1, 0, 0.0);

    M.setup();

    // No external forces — pure displacement control
    NonlinearAnalysis<Policy, continuum::SmallStrain> nl{&M};

    const double target_disp = 0.005;
    std::vector<double> imposed_vs_total_state_mismatch;
    nl.set_step_callback([&](int /*step*/, double p, const auto& model) {
        const double expected = p * target_disp;
        const double imposed = model.prescribed_value(1, 0);
        const double total_state =
            fall_n::query::nodal_dof_value(model, model.state_vector(), 1, 0);
        imposed_vs_total_state_mismatch.push_back(std::abs(imposed - total_state));
        check(std::abs(imposed - expected) < 1e-10,
              "step callback: imposed displacement matches incremental target");
        check(std::abs(total_state - expected) < 1e-10,
              "step callback: total state reflects imposed displacement");
    });
    bool ok = nl.solve_incremental(5, 4, DisplacementControl{1, 0, target_disp});
    check(ok, "DisplacementControl single-DOF converged");

    // Verify the state vector: node 1 should have ~target_disp in x
    PetscReal norm;
    VecNorm(M.state_vector(), NORM_2, &norm);
    check(norm > 1e-14, "state is non-trivial under displacement control");

    // The imposed_solution should contain the target displacement
    const PetscScalar* vals;
    VecGetArrayRead(M.imposed_solution(), &vals);

    auto& node = D.node(1);
    auto dof_idx = node.dof_index();
    double imposed_x = std::abs(vals[dof_idx[0]]);

    VecRestoreArrayRead(M.imposed_solution(), &vals);

    check(std::abs(imposed_x - target_disp) < 1e-10,
          "imposed value matches target displacement");

    const double total_state_x =
        fall_n::query::nodal_dof_value(M, M.state_vector(), 1, 0);
    check(std::abs(total_state_x - target_disp) < 1e-10,
          "committed total state matches imposed displacement");
    check(!imposed_vs_total_state_mismatch.empty(),
          "step callback recorded displacement-control state checks");
    check(std::ranges::all_of(
              imposed_vs_total_state_mismatch,
              [](double err) { return err < 1e-10; }),
          "all converged steps keep imposed and total constrained state aligned");
}

// =============================================================================
//  Test 4: DisplacementControl — multi-DOF
// =============================================================================

void test_displacement_control_multi() {
    std::cout << "\nTest 4: DisplacementControl multi-DOF\n";

    Domain<DIM> D;
    create_unit_cube(D);

    ContinuumIsotropicElasticMaterial elastic{200.0, 0.3};
    Material<Policy> mat{elastic, ElasticUpdate{}};

    Model<Policy, continuum::SmallStrain, NDOF> M{D, mat};

    // Fix left face
    M.fix_x(0.0);

    // Pre-constrain the controlled DOFs
    M.constrain_dof(1, 0, 0.0);
    M.constrain_dof(3, 0, 0.0);

    M.setup();

    const double target = 0.003;
    DisplacementControl dc{
        {{1, 0, target}, {3, 0, target}},
        0.0  // no proportional load
    };

    NonlinearAnalysis<Policy, continuum::SmallStrain> nl{&M};
    bool ok = nl.solve_incremental(4, 4, dc);
    check(ok, "DisplacementControl multi-DOF converged");

    PetscReal norm;
    VecNorm(M.state_vector(), NORM_2, &norm);
    check(norm > 1e-14, "state is non-trivial");
}

// =============================================================================
//  Test 4b: PETSc unknown excludes newly constrained displacement DOFs
// =============================================================================

void test_constrained_dof_reduces_global_unknown_size() {
    std::cout << "\nTest 4b: constrained DOF shrinks PETSc global unknown\n";

    Domain<DIM> D_ref;
    create_unit_cube(D_ref);
    ContinuumIsotropicElasticMaterial elastic{200.0, 0.3};
    Material<Policy> mat{elastic, ElasticUpdate{}};

    Model<Policy, continuum::SmallStrain, NDOF> M_ref{D_ref, mat};
    M_ref.fix_x(0.0);
    M_ref.setup();

    Vec u_ref = nullptr;
    DMCreateGlobalVector(M_ref.get_plex(), &u_ref);
    PetscInt n_ref = 0;
    VecGetSize(u_ref, &n_ref);
    VecDestroy(&u_ref);

    Domain<DIM> D_ctrl;
    create_unit_cube(D_ctrl);
    Model<Policy, continuum::SmallStrain, NDOF> M_ctrl{D_ctrl, mat};
    M_ctrl.fix_x(0.0);
    M_ctrl.constrain_dof(1, 0, 0.0);
    M_ctrl.setup();

    Vec u_ctrl = nullptr;
    DMCreateGlobalVector(M_ctrl.get_plex(), &u_ctrl);
    PetscInt n_ctrl = 0;
    VecGetSize(u_ctrl, &n_ctrl);
    VecDestroy(&u_ctrl);

    check(n_ref - n_ctrl == 1,
          "one additional prescribed DOF removes exactly one PETSc global unknown");
}

// =============================================================================
//  Test 4c: low-level imposed-value edits are visible after one finalization
// =============================================================================

void test_batched_imposed_value_finalization() {
    std::cout << "\nTest 4c: batched imposed-value finalization\n";

    Domain<DIM> D;
    create_unit_cube(D);

    ContinuumIsotropicElasticMaterial elastic{200.0, 0.3};
    Material<Policy> mat{elastic, ElasticUpdate{}};

    Model<Policy, continuum::SmallStrain, NDOF> M{D, mat};
    M.fix_x(0.0);
    M.constrain_dof(1, 0, 0.0);
    M.constrain_dof(3, 0, 0.0);
    M.setup();

    M.set_imposed_value_unassembled(1, 0, 0.0015);
    M.set_imposed_value_unassembled(3, 0, 0.0025);
    M.finalize_imposed_solution();

    const auto imposed_1 =
        fall_n::query::nodal_dof_value(M, M.imposed_solution(), 1, 0);
    const auto imposed_3 =
        fall_n::query::nodal_dof_value(M, M.imposed_solution(), 3, 0);

    check(std::abs(imposed_1 - 0.0015) < 1e-12,
          "batched finalization materializes first imposed DOF");
    check(std::abs(imposed_3 - 0.0025) < 1e-12,
          "batched finalization materializes second imposed DOF");
}

// =============================================================================
//  Test 5: CustomControl via make_control (lambda)
// =============================================================================

void test_custom_control() {
    std::cout << "\nTest 5: CustomControl via make_control (lambda)\n";

    Domain<DIM> D;
    create_unit_cube(D);

    ContinuumIsotropicElasticMaterial elastic{200.0, 0.3};
    Material<Policy> mat{elastic, ElasticUpdate{}};

    Model<Policy, continuum::SmallStrain, NDOF> M{D, mat};
    M.fix_x(0.0);
    M.setup();

    const double f_per_node = 1.0 / 4.0;
    for (std::size_t id : {1ul, 3ul, 5ul, 7ul})
        M.apply_node_force(id, f_per_node, 0.0, 0.0);

    NonlinearAnalysis<Policy, continuum::SmallStrain> nl{&M};

    // Custom: sqrt ramping of the load
    auto scheme = make_control(
        [](double p, Vec f_full, Vec f_ext, auto* /*model*/) {
            VecCopy(f_full, f_ext);
            VecScale(f_ext, std::sqrt(p));
        }
    );

    bool ok = nl.solve_incremental(5, 4, scheme);
    check(ok, "CustomControl (sqrt ramp) converged");

    PetscReal norm;
    VecNorm(M.state_vector(), NORM_2, &norm);
    check(norm > 1e-12, "solution is non-trivial");
}

// =============================================================================
//  Test 6: typed small-residual acceptance policy
// =============================================================================

void test_small_residual_acceptance_policy() {
    std::cout << "\nTest 6: typed small-residual acceptance policy\n";

    auto profile = fall_n::make_newton_backtracking_profile();
    profile.atol = 1.0e-10;
    profile.small_residual_acceptance = {
        .absolute_function_norm_threshold = 0.0,
        .profile_atol_multiplier = 100.0,
        .accept_diverged_line_search = true,
        .accept_diverged_tr_delta = false,
        .accept_diverged_dtol = false};

    const auto accepted = fall_n::assess_nonlinear_solve_attempt(
        profile, SNES_DIVERGED_LINE_SEARCH, 4.0e-9);
    check(accepted.accepted,
          "small-residual policy accepts line-search divergence when ||F|| is tiny");
    check(accepted.accepted_by_small_residual_policy,
          "small-residual policy reports an override when it accepts a negative SNES reason");
    check(std::abs(accepted.accepted_function_norm_threshold - 1.0e-8) < 1.0e-18,
          "small-residual policy resolves the threshold from the profile atol");

    const auto rejected = fall_n::assess_nonlinear_solve_attempt(
        profile, SNES_DIVERGED_DTOL, 4.0e-9);
    check(!rejected.accepted,
          "small-residual policy does not accept unlisted negative SNES reasons");

    const auto rejected_large_residual = fall_n::assess_nonlinear_solve_attempt(
        profile, SNES_DIVERGED_LINE_SEARCH, 1.0e-4);
    check(!rejected_large_residual.accepted,
          "small-residual policy still rejects materially large residuals");
}

// =============================================================================
//  Test 7: Trial residual/tangent evaluation API
// =============================================================================

void test_trial_residual_tangent_evaluation_api() {
    std::cout << "\nTest 7: trial residual/tangent evaluation API\n";

    Domain<DIM> D;
    create_unit_cube(D);

    ContinuumIsotropicElasticMaterial elastic{200.0, 0.3};
    Material<Policy> mat{elastic, ElasticUpdate{}};

    Model<Policy, continuum::SmallStrain, NDOF> M{D, mat};
    M.fix_x(0.0);
    M.setup();

    const double f_per_node = 1.0 / 4.0;
    for (std::size_t id : {1ul, 3ul, 5ul, 7ul})
        M.apply_node_force(id, f_per_node, 0.0, 0.0);

    NonlinearAnalysis<Policy, continuum::SmallStrain> nl{&M};
    nl.begin_incremental(4, 2, LoadControl{});

    auto trial_u = nl.clone_solution_vector();
    auto residual = nl.create_global_vector();
    auto tangent = nl.create_tangent_matrix();

    nl.apply_incremental_control_parameter(0.5);
    nl.evaluate_residual_at(trial_u.get(), residual.get());
    nl.evaluate_tangent_at(trial_u.get(), tangent.get());

    PetscReal residual_norm = 0.0;
    PetscReal tangent_norm = 0.0;
    VecNorm(residual.get(), NORM_2, &residual_norm);
    MatNorm(tangent.get(), NORM_FROBENIUS, &tangent_norm);

    check(residual_norm > 1.0e-12,
          "trial residual sees the externally applied half-load");
    check(tangent_norm > 1.0e-12,
          "trial tangent assembles through the same DMPlex algebra");
    check(std::abs(nl.current_time()) < 1.0e-14,
          "trial control evaluation does not advance the incremental clock");
}

// =============================================================================
//  Test 8: Concept satisfaction (compile-time checks)
// =============================================================================

void test_concept_satisfaction() {
    std::cout << "\nTest 8: Concept satisfaction (compile-time)\n";

    using ModelT = Model<Policy, continuum::SmallStrain, NDOF>;

    // LoadControl
    static_assert(IncrementalControlPolicy<LoadControl, ModelT>,
                  "LoadControl must satisfy IncrementalControlPolicy");

    // DisplacementControl
    static_assert(IncrementalControlPolicy<DisplacementControl, ModelT>,
                  "DisplacementControl must satisfy IncrementalControlPolicy");

    // CustomControl with lambda
    auto lam = [](double, Vec, Vec, auto*){};
    using CC = CustomControl<decltype(lam)>;
    static_assert(IncrementalControlPolicy<CC, ModelT>,
                  "CustomControl<lambda> must satisfy IncrementalControlPolicy");

    check(true, "LoadControl satisfies IncrementalControlPolicy");
    check(true, "DisplacementControl satisfies IncrementalControlPolicy");
    check(true, "CustomControl<lambda> satisfies IncrementalControlPolicy");
}

// =============================================================================
//  main
// =============================================================================

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    std::cout << "=== IncrementalControlPolicy Verification ===\n";

    test_load_control_default();
    test_load_control_explicit();
    test_displacement_control_single();
    test_displacement_control_multi();
    test_constrained_dof_reduces_global_unknown_size();
    test_batched_imposed_value_finalization();
    test_custom_control();
    test_small_residual_acceptance_policy();
    test_trial_residual_tangent_evaluation_api();
    test_concept_satisfaction();

    std::cout << "\n=== Summary: " << passed << " passed, "
              << failed << " failed ===\n";

    PetscFinalize();
    return (failed > 0) ? 1 : 0;
}
