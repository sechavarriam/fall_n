// =============================================================================
//  test_material_strategy.cpp
//
//  Integration test for the Material<> type-erasure with Strategy injection.
//  Verifies that:
//    1. ElasticUpdate correctly routes compute_response, tangent, commit
//    2. InelasticUpdate correctly routes through J2 return-mapping
//    3. commit() evolves internal variables
//    4. Copy semantics produce deep clones
//
//  Build:
//    Linked against Eigen, PETSc, VTK (via CMake test target).
//
//  No mesh/PETSc runtime required — pure material-level tests.
// =============================================================================

#include <utility>
#include <iostream>
#include <cassert>
#include <cmath>

#include "../src/materials/LinealElasticMaterial.hh"
#include "../src/materials/Material.hh"
#include "../src/materials/update_strategy/IntegrationStrategy.hh"

// Helper: approximate equality for Eigen objects
template <typename D1, typename D2>
bool approx_equal(const Eigen::MatrixBase<D1>& a,
                  const Eigen::MatrixBase<D2>& b,
                  double tol = 1e-10) {
    return (a - b).norm() < tol;
}

// ─── Test 1: Elastic material through type-erasure ───────────────────────────

void test_elastic_strategy() {
    std::cout << "Test 1: Elastic material with ElasticUpdate strategy\n";

    ContinuumIsotropicElasticMaterial mat_instance{200.0, 0.3};

    // Wrap in type-erasure with ElasticUpdate strategy
    Material<ThreeDimensionalMaterial> mat{mat_instance, ElasticUpdate{}};

    // Create a uniaxial-like test strain
    Strain<6> eps;
    eps.set_strain(
        (Eigen::Vector<double, 6>() << 0.001, -0.0003, -0.0003, 0.0, 0.0, 0.0)
            .finished());

    // Test compute_response through type-erasure → Strategy → Relation
    auto sigma = mat.compute_response(eps);
    std::cout << "  sigma = [";
    for (int i = 0; i < 6; ++i) std::cout << " " << sigma[i];
    std::cout << " ]\n";

    // Verify: σ = C·ε (the Strategy should not alter the result)
    auto C = mat.C();
    Eigen::Vector<double, 6> sigma_expected = C * eps.components();
    assert(approx_equal(sigma.components(), sigma_expected) &&
           "compute_response must equal C * eps for elastic material");

    // tangent(ε) must equal C() for elastic material
    auto C_t = mat.tangent(eps);
    assert(approx_equal(C, C_t) &&
           "tangent(eps) must equal C() for elastic material");

    // commit must be a no-op (no crash)
    mat.commit(eps);

    // State management through type-erasure
    mat.update_state(eps);
    const auto& state = mat.current_state();
    assert(approx_equal(state.components(), eps.components()) &&
           "state must be updated after update_state");

    std::cout << "  PASSED\n\n";
}

// ─── Test 2: Inelastic material through type-erasure ─────────────────────────

void test_inelastic_strategy() {
    std::cout << "Test 2: J2 Plasticity with InelasticUpdate strategy\n";

    // J2 with E=200, ν=0.3, σ_y0=0.250, H=10
    J2PlasticMaterial3D mat_instance{200.0, 0.3, 0.250, 10.0};

    // Wrap with InelasticUpdate strategy
    Material<ThreeDimensionalMaterial> mat{mat_instance, InelasticUpdate{}};

    // ── 2a: Elastic regime (small strain, below yield) ─────────

    std::cout << "  2a: Elastic regime (below yield)\n";

    Strain<6> eps_small;
    eps_small.set_strain(
        (Eigen::Vector<double, 6>() << 1e-6, -3e-7, -3e-7, 0.0, 0.0, 0.0)
            .finished());

    auto sigma_small = mat.compute_response(eps_small);
    (void)sigma_small; // verified implicitly through tangent check
    auto C_t_small   = mat.tangent(eps_small);
    auto C_elastic   = mat.C(); // At zero state → elastic tangent

    // In elastic regime, tangent should equal elastic tangent
    assert(approx_equal(C_t_small, C_elastic, 1e-6) &&
           "tangent must equal elastic tangent below yield");

    std::cout << "    PASSED\n";

    // ── 2b: Plastic regime (large strain, above yield) ─────────

    std::cout << "  2b: Plastic regime (above yield)\n";

    Strain<6> eps_large;
    eps_large.set_strain(
        (Eigen::Vector<double, 6>() << 0.01, -0.003, -0.003, 0.0, 0.0, 0.0)
            .finished());

    auto sigma_large = mat.compute_response(eps_large);
    auto C_t_large   = mat.tangent(eps_large);

    // In plastic regime, consistent tangent must differ from elastic tangent
    double tangent_diff = (C_t_large - C_elastic).norm();
    std::cout << "    ||C_ep - C_e|| = " << tangent_diff << "\n";
    assert(tangent_diff > 1e-6 &&
           "consistent tangent must differ from elastic in plastic regime");

    std::cout << "    sigma = [";
    for (int i = 0; i < 6; ++i) std::cout << " " << sigma_large[i];
    std::cout << " ]\n";
    std::cout << "    PASSED\n";

    // ── 2c: commit evolves internal state ───────────────────────

    std::cout << "  2c: Commit evolves internal variables\n";

    // Capture response BEFORE commit
    auto sigma_pre_commit = mat.compute_response(eps_large);

    // Commit — this calls Strategy.commit → material.update(ε)
    // which runs the return-mapping and updates α = {ε^p, ε̄^p}
    mat.commit(eps_large);

    // After commit, re-evaluating at the SAME total strain must give the
    // SAME stress — the return-mapping is self-consistent (idempotent) for
    // purely normal strains where Ce·n̂ = 2G·n̂ exactly.
    auto sigma_post_commit = mat.compute_response(eps_large);

    double stress_diff = (sigma_post_commit.components()
                        - sigma_pre_commit.components()).norm();
    std::cout << "    ||σ_after − σ_before|| = " << stress_diff << "\n";
    assert(stress_diff < 1e-10 &&
           "committed state must be self-consistent at same strain");

    // Verify state ACTUALLY evolved: compare committed material vs a FRESH
    // (un-committed) material at a NON-PROPORTIONAL strain.  For proportional
    // loading with linear isotropic hardening the return mapping is path-
    // independent, so we must change deviatoric direction to reveal the
    // committed plastic strain.
    Strain<6> eps_np;
    eps_np.set_strain(
        (Eigen::Vector<double, 6>() << 0.005, 0.005, -0.010, 0.0, 0.0, 0.0)
            .finished());

    Material<ThreeDimensionalMaterial> mat_fresh{
        J2PlasticMaterial3D{200.0, 0.3, 0.250, 10.0}, InelasticUpdate{}};
    auto sigma_fresh    = mat_fresh.compute_response(eps_np);
    auto sigma_commited = mat.compute_response(eps_np);
    double evolution_diff = (sigma_commited.components()
                           - sigma_fresh.components()).norm();
    std::cout << "    ||σ_committed − σ_fresh|| at ε_np = " << evolution_diff << "\n";
    assert(evolution_diff > 1e-10 &&
           "committed material must differ from fresh at non-proportional strain");

    // Second commit at a non-proportional strain — again self-consistent
    Strain<6> eps_larger;
    eps_larger.set_strain(
        (Eigen::Vector<double, 6>() << 0.02, -0.006, -0.006, 0.0, 0.0, 0.0)
            .finished());
    auto sigma_step2_a = mat.compute_response(eps_larger);
    mat.commit(eps_larger);
    auto sigma_step2_b = mat.compute_response(eps_larger);

    double stress_diff_2 = (sigma_step2_b.components()
                          - sigma_step2_a.components()).norm();
    std::cout << "    ||σ_step2_after − σ_step2_before|| = " << stress_diff_2 << "\n";
    assert(stress_diff_2 < 1e-10 &&
           "second commit also self-consistent at same strain");

    std::cout << "    PASSED\n\n";
}

// ─── Test 3: Copy semantics (deep clone through type-erasure) ────────────────

void test_copy_semantics() {
    std::cout << "Test 3: Deep clone through type-erasure\n";

    Material<ThreeDimensionalMaterial> mat1{
        ContinuumIsotropicElasticMaterial{200.0, 0.3}, ElasticUpdate{}};

    // Copy: should deep-clone via the Prototype pattern (clone())
    Material<ThreeDimensionalMaterial> mat2 = mat1;

    Strain<6> eps;
    eps.set_strain(
        (Eigen::Vector<double, 6>() << 0.001, 0.0, 0.0, 0.0, 0.0, 0.0)
            .finished());

    // Both copies should produce identical responses
    auto sigma1 = mat1.compute_response(eps);
    auto sigma2 = mat2.compute_response(eps);
    assert(approx_equal(sigma1.components(), sigma2.components()) &&
           "copies must produce identical responses");

    // Modify the copy's state — original must be unaffected
    mat2.update_state(eps);

    const auto& s1 = mat1.current_state();
    const auto& s2 = mat2.current_state();

    assert(s1.components().norm() < 1e-15 &&
           "original must be unaffected after modifying copy");
    assert(s2.components().norm() > 1e-15 &&
           "copy must reflect state update");

    std::cout << "  PASSED\n\n";
}

// ─── Test 4: Strategy type safety ────────────────────────────────────────────

void test_strategy_type_safety() {
    std::cout << "Test 4: Strategy type safety (mixed scenarios)\n";

    // ElasticUpdate with elastic material → OK
    {
        Material<ThreeDimensionalMaterial> mat{
            ContinuumIsotropicElasticMaterial{200.0, 0.3}, ElasticUpdate{}};
        Strain<6> eps;
        eps.set_strain(
            (Eigen::Vector<double, 6>() << 0.001, 0.0, 0.0, 0.0, 0.0, 0.0)
                .finished());
        [[maybe_unused]] auto sigma = mat.compute_response(eps);
        mat.commit(eps); // ElasticUpdate::commit is a no-op
        std::cout << "  ElasticUpdate + Elastic material: OK\n";
    }

    // InelasticUpdate with inelastic material → OK
    {
        Material<ThreeDimensionalMaterial> mat{
            J2PlasticMaterial3D{200.0, 0.3, 0.250, 10.0}, InelasticUpdate{}};
        Strain<6> eps;
        eps.set_strain(
            (Eigen::Vector<double, 6>() << 0.01, -0.003, -0.003, 0.0, 0.0, 0.0)
                .finished());
        [[maybe_unused]] auto sigma = mat.compute_response(eps);
        mat.commit(eps); // InelasticUpdate::commit calls material.update(k)
        std::cout << "  InelasticUpdate + J2 material: OK\n";
    }

    // ElasticUpdate with inelastic material → Strategy is no-op commit
    // (valid: you can use an inelastic model without committing state)
    {
        Material<ThreeDimensionalMaterial> mat{
            J2PlasticMaterial3D{200.0, 0.3, 0.250, 10.0}, ElasticUpdate{}};
        Strain<6> eps;
        eps.set_strain(
            (Eigen::Vector<double, 6>() << 0.01, -0.003, -0.003, 0.0, 0.0, 0.0)
                .finished());
        [[maybe_unused]] auto sigma = mat.compute_response(eps);
        mat.commit(eps); // ElasticUpdate::commit is no-op (state not evolved)
        std::cout << "  ElasticUpdate + J2 material (no commit): OK\n";
    }

    std::cout << "  PASSED\n\n";
}

// ─── Test 5: Borrowed non-owning type-erasure views ─────────────────────────

void test_non_owning_material_views() {
    std::cout << "Test 5: Non-owning MaterialRef / MaterialConstRef\n";

    Material<ThreeDimensionalMaterial> owner{
        J2PlasticMaterial3D{200.0, 0.3, 0.250, 10.0}, InelasticUpdate{}};

    Strain<6> eps;
    eps.set_strain(
        (Eigen::Vector<double, 6>() << 0.01, -0.003, -0.003, 0.0, 0.0, 0.0)
            .finished());

    auto owner_sigma = owner.compute_response(eps);

    MaterialConstRef<ThreeDimensionalMaterial> cref{owner};
    auto cref_sigma = cref.compute_response(eps);
    assert(approx_equal(owner_sigma.components(), cref_sigma.components()) &&
           "MaterialConstRef must observe the same constitutive response");

    MaterialRef<ThreeDimensionalMaterial> ref{owner};
    ref.commit(eps);

    Strain<6> eps_np;
    eps_np.set_strain(
        (Eigen::Vector<double, 6>() << 0.005, 0.005, -0.010, 0.0, 0.0, 0.0)
            .finished());

    auto sigma_after_ref_commit = owner.compute_response(eps_np);
    Material<ThreeDimensionalMaterial> fresh{
        J2PlasticMaterial3D{200.0, 0.3, 0.250, 10.0}, InelasticUpdate{}};
    auto sigma_fresh = fresh.compute_response(eps_np);
    assert((sigma_after_ref_commit.components() - sigma_fresh.components()).norm() > 1e-10 &&
           "MaterialRef must mutate the borrowed owner state");

    Material<ThreeDimensionalMaterial> clone_from_view{cref};
    clone_from_view.update_state(eps_np);
    assert((clone_from_view.current_state().components() - owner.current_state().components()).norm() > 1e-10 &&
           "Cloning from MaterialConstRef must produce independent owning state");

    // Direct borrowed view over a concrete material + strategy without
    // first materializing a heap-owning Material<> wrapper.
    ContinuumIsotropicElasticMaterial typed_material{200.0, 0.3};
    ElasticUpdate typed_strategy{};
    MaterialConstRef<ThreeDimensionalMaterial> typed_cref{typed_material, typed_strategy};
    auto typed_sigma = typed_cref.compute_response(eps);
    auto typed_expected = typed_material.compute_response(eps);
    assert(approx_equal(typed_sigma.components(), typed_expected.components()) &&
           "Borrowed view over concrete material + strategy must be valid");

    std::cout << "  PASSED\n\n";
}

// =============================================================================

int main() {
    std::cout << "============================================\n";
    std::cout << "  Material Strategy Integration Test Suite\n";
    std::cout << "============================================\n\n";

    test_elastic_strategy();
    test_inelastic_strategy();
    test_copy_semantics();
    test_strategy_type_safety();
    test_non_owning_material_views();

    std::cout << "============================================\n";
    std::cout << "  All tests PASSED\n";
    std::cout << "============================================\n";

    return 0;
}
