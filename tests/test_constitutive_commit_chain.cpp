// =============================================================================
//  test_constitutive_commit_chain.cpp — Phase 1 verification
// =============================================================================
//
//  Validates the complete constitutive state protocol:
//
//    1. TrialCommittedState: commit_trial / revert_trial semantics
//    2. ConstitutiveState: conditional commit/revert propagation
//    3. Material<> type-erased commit via integrators
//    4. Plasticity commit chain (return mapping + state update)
//    5. SNES single-step solve with commit on convergence
//    6. SNES incremental solve with commit chain
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
#include "src/continuum/HyperelasticRelation.hh"

// ── Constants ─────────────────────────────────────────────────────────────────

static constexpr std::size_t DIM  = 3;
static constexpr std::size_t NDOF = DIM;

static int passed = 0;
static int failed = 0;

static void check(bool cond, const char* msg) {
    if (cond) { std::cout << "  [PASS] " << msg << "\n"; ++passed; }
    else      { std::cout << "  [FAIL] " << msg << "\n"; ++failed; }
}

// =============================================================================
//  Test 1: TrialCommittedState storage semantics
// =============================================================================

void test_trial_committed_state() {
    std::cout << "\nTest 1: TrialCommittedState<double> semantics\n";

    TrialCommittedState<double> state;

    // Initial: no trial, committed = 0
    check(!state.has_trial_value(), "initial: no trial value");
    check(state.committed_value() == 0.0, "initial: committed = 0");
    check(state.current_value() == 0.0, "initial: current = committed");

    // Update (sets trial)
    state.update(42.0);
    check(state.has_trial_value(), "after update: has trial");
    check(state.trial_value() == 42.0, "after update: trial = 42");
    check(state.committed_value() == 0.0, "after update: committed unchanged");
    check(state.current_value() == 42.0, "after update: current = trial");

    // Commit trial
    state.commit_trial();
    check(!state.has_trial_value(), "after commit: no trial");
    check(state.committed_value() == 42.0, "after commit: committed = 42");
    check(state.current_value() == 42.0, "after commit: current = 42");

    // Update again, then revert
    state.update(99.0);
    check(state.trial_value() == 99.0, "second update: trial = 99");
    check(state.committed_value() == 42.0, "second update: committed still 42");

    state.revert_trial();
    check(!state.has_trial_value(), "after revert: no trial");
    check(state.committed_value() == 42.0, "after revert: committed preserved");
    check(state.current_value() == 42.0, "after revert: current = committed");
}

// =============================================================================
//  Test 2: Elastic material — commit is no-op, tangent is constant
// =============================================================================

void test_elastic_commit() {
    std::cout << "\nTest 2: Elastic material commit (PassThroughIntegrator)\n";

    using Policy  = ThreeDimensionalMaterial;
    using StateT  = typename Policy::StateVariableT;
    using MatrixT = Eigen::Matrix<double, StateT::num_components,
                                          StateT::num_components>;

    ContinuumIsotropicElasticMaterial elastic_site{200.0, 0.3};
    Material<Policy> mat{elastic_site, PassThroughIntegrator{}};

    // Compute tangent at zero strain
    StateT zero_strain{};
    MatrixT C0 = mat.tangent(zero_strain);

    // Compute tangent at nonzero strain
    StateT strain1{};
    strain1[0] = 0.001;  // uniaxial e_xx
    MatrixT C1 = mat.tangent(strain1);

    // Tangent should be identical (path-independent)
    check((C0 - C1).norm() < 1e-12, "elastic tangent constant");

    // Commit and verify tangent unchanged
    mat.commit(strain1);
    MatrixT C2 = mat.tangent(zero_strain);
    check((C0 - C2).norm() < 1e-12, "tangent unchanged after commit");

    // Compute response
    auto sigma = mat.compute_response(strain1);
    double sigma_xx = sigma[0];
    check(sigma_xx > 0.0, "elastic stress > 0 for positive strain");
}

// =============================================================================
//  Test 3: Plasticity — commit changes internal state
// =============================================================================

void test_plasticity_commit() {
    std::cout << "\nTest 3: J2 Plasticity commit chain\n";

    using Policy = ThreeDimensionalMaterial;
    using StateT = typename Policy::StateVariableT;

    double E = 200.0, nu = 0.3, sigma_y = 0.250, H = 10.0;

    J2PlasticityRelation<Policy> plasticity(E, nu, sigma_y, H);

    // ── Step 1: apply elastic strain (below yield) ──
    StateT strain_elastic{};
    strain_elastic[0] = 0.0005;  // small uniaxial strain

    auto sigma_e = plasticity.compute_response(strain_elastic);
    check(sigma_e[0] > 0.0, "elastic response: sigma_xx > 0");

    // ── Step 2: update internal variables at elastic strain ──
    plasticity.update(strain_elastic);
    auto alpha_before = plasticity.internal_state();
    double eps_p_before = alpha_before.eps_bar_p();
    check(std::abs(eps_p_before) < 1e-14, "elastic step: no plastic strain");

    // ── Step 3: apply large strain (beyond yield) ──
    StateT strain_plastic{};
    strain_plastic[0] = 0.01;  // large uniaxial strain -> yielding

    (void)plasticity.compute_response(strain_plastic);
    plasticity.update(strain_plastic);
    auto alpha_after = plasticity.internal_state();
    double eps_p_after = alpha_after.eps_bar_p();
    check(eps_p_after > 1e-6, "plastic step: accumulated plastic strain > 0");

    // ── Step 4: verify plastic strain increased ──
    check(eps_p_after > eps_p_before,
          "plastic strain increased after yielding");
}

// =============================================================================
//  Test 4: Type-erased Material<> with plasticity
// =============================================================================

void test_type_erased_plasticity_commit() {
    std::cout << "\nTest 4: Type-erased Material<> with plasticity commit\n";

    using Policy = ThreeDimensionalMaterial;
    using StateT = typename Policy::StateVariableT;

    double E = 200.0, nu = 0.3, sigma_y = 0.250, H = 10.0;

    J2PlasticMaterial3D site{E, nu, sigma_y, H};

    // Type-erase through Material<>
    Material<Policy> mat(std::move(site), InelasticUpdate{});

    // ── Evaluate at elastic strain ──
    StateT strain_e{};
    strain_e[0] = 0.0005;
    auto sigma_e = mat.compute_response(strain_e);
    check(sigma_e[0] > 0.0, "type-erased elastic response ok");

    // ── Evaluate at plastic strain (side-effect free) ──
    StateT strain_p{};
    strain_p[0] = 0.01;
    (void)mat.compute_response(strain_p);

    // ── Commit the plastic strain ──
    mat.commit(strain_p);

    // After commit, unloading to the elastic strain should give a different
    // stress because eps_p != 0 now.
    auto sigma_after = mat.compute_response(strain_e);
    double stress_diff = std::abs(sigma_after[0] - sigma_e[0]);
    check(stress_diff > 1e-6,
          "after commit: unloading stress differs from virgin");
}

// =============================================================================
//  Helper: create unit cube domain
// =============================================================================

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

// =============================================================================
//  Test 5: SNES single-step solve — commit only on convergence
// =============================================================================

void test_snes_single_step_commit() {
    std::cout << "\nTest 5: SNES single-step solve with commit\n";

    using Policy = ThreeDimensionalMaterial;

    Domain<DIM> D;
    create_unit_cube(D);

    ContinuumIsotropicElasticMaterial elastic_site{200.0, 0.3};
    Material<Policy> mat{elastic_site, ElasticUpdate{}};

    Model<Policy, continuum::SmallStrain, NDOF> M{D, mat};
    M.fix_x(0.0);
    M.setup();

    // Apply moderate uniaxial tension on x=1 face
    const double f_per_node = 0.02 / 4.0;
    for (std::size_t id : {1ul, 3ul, 5ul, 7ul})
        M.apply_node_force(id, f_per_node, 0.0, 0.0);

    // Verify initial state is zero
    PetscReal init_norm;
    VecNorm(M.state_vector(), NORM_2, &init_norm);
    check(init_norm < 1e-12, "initial state is zero");

    // Solve
    NonlinearAnalysis<Policy, continuum::SmallStrain> nl{&M};
    bool converged = nl.solve();

    check(converged, "SNES converged");

    if (converged) {
        PetscReal final_norm;
        VecNorm(M.state_vector(), NORM_2, &final_norm);
        check(final_norm > 1e-12, "state updated after convergence");

        check(nl.num_iterations() <= 2,
              "linear problem converged in <= 2 iterations");
    }
}

// =============================================================================
//  Test 6: Incremental solve — verifies commit chain across multiple steps
// =============================================================================

void test_incremental_commit() {
    std::cout << "\nTest 6: Incremental solve commit chain (3 steps)\n";

    using Policy = ThreeDimensionalMaterial;

    Domain<DIM> D;
    create_unit_cube(D);

    ContinuumIsotropicElasticMaterial elastic_site{200.0, 0.3};
    Material<Policy> mat{elastic_site, ElasticUpdate{}};

    Model<Policy, continuum::SmallStrain, NDOF> M{D, mat};
    M.fix_x(0.0);
    M.setup();

    // Moderate uniaxial tension
    const double f_per_node = 1.0 / 4.0;
    for (std::size_t id : {1ul, 3ul, 5ul, 7ul})
        M.apply_node_force(id, f_per_node, 0.0, 0.0);

    NonlinearAnalysis<Policy, continuum::SmallStrain> nl{&M};
    bool all_ok = nl.solve_incremental(3);
    check(all_ok, "incremental solve converged (3 steps)");

    // Verify solution is non-trivial
    PetscReal norm;
    VecNorm(M.state_vector(), NORM_2, &norm);
    check(norm > 1e-12, "solution is non-trivial after 3 increments");
}

// =============================================================================
//  main
// =============================================================================

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    std::cout << "=== Phase 1 Verification: Constitutive State Protocol ===\n";

    test_trial_committed_state();
    test_elastic_commit();
    test_plasticity_commit();
    test_type_erased_plasticity_commit();
    test_snes_single_step_commit();
    test_incremental_commit();

    std::cout << "\n=== Summary: " << passed << " passed, "
              << failed << " failed ===\n";

    PetscFinalize();
    return (failed > 0) ? 1 : 0;
}
