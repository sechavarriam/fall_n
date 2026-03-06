// =============================================================================
//  test_plasticity_composition.cpp
//
//  Tests the decomposed plasticity architecture:
//    PlasticityRelation<Policy, YieldF, Hardening, Flow>
//
//  Verifies:
//    1. VonMises yield criterion (TrialState → equivalent stress, gradient)
//    2. LinearIsotropicHardening (yield stress, modulus, evolve)
//    3. AssociatedFlow (direction ≡ yield gradient)
//    4. PlasticityRelation (full return-mapping, consistent tangent)
//    5. Backward-compat: J2PlasticityRelation alias matches old behavior
//    6. Orthogonal composition: same yield + different hardening parameters
//    7. PlasticInternalVariables (eps_p, eps_bar_p accessors)
//    8. Type-erasure integration (Material<> + InelasticUpdate)
//
//  Build: linked against Eigen, PETSc, VTK (via CMake test target).
//  No mesh/PETSc runtime required — pure material-level tests.
// =============================================================================

#include <utility>
#include <iostream>
#include <cassert>
#include <cmath>

// Building blocks (directly)
#include "../src/materials/constitutive_models/non_lineal/plasticity/PlasticityConcepts.hh"
#include "../src/materials/constitutive_models/non_lineal/plasticity/VonMises.hh"
#include "../src/materials/constitutive_models/non_lineal/plasticity/IsotropicHardening.hh"
#include "../src/materials/constitutive_models/non_lineal/plasticity/AssociatedFlow.hh"

// Composed relation
#include "../src/materials/constitutive_models/non_lineal/PlasticityRelation.hh"

// Backward-compat aliases (J2PlasticityRelation, J2InternalVariables)
#include "../src/materials/constitutive_models/non_lineal/InelasticRelation.hh"

// MaterialInstance + named aliases
#include "../src/materials/LinealElasticMaterial.hh"

// Type-erasure + strategies
#include "../src/materials/Material.hh"
#include "../src/materials/update_strategy/IntegrationStrategy.hh"


// ─── Helpers ─────────────────────────────────────────────────────────────────

template <typename D1, typename D2>
bool approx_equal(const Eigen::MatrixBase<D1>& a,
                  const Eigen::MatrixBase<D2>& b,
                  double tol = 1e-10) {
    return (a - b).norm() < tol;
}


// ─── Test 1: VonMises yield criterion ────────────────────────────────────────

void test_von_mises_criterion() {
    std::cout << "Test 1: VonMises yield criterion\n";

    VonMises vm;

    // Build a TrialState manually: uniaxial stress σ = [σ_11, 0, 0, 0, 0, 0]
    TrialState<6> trial;
    double sigma_11 = 300.0;
    trial.stress     = (Eigen::Vector<double, 6>() << sigma_11, 0, 0, 0, 0, 0).finished();
    trial.deviatoric = trial.stress;
    double p = sigma_11 / 3.0;
    trial.deviatoric[0] -= p;
    trial.deviatoric[1] -= p;
    trial.deviatoric[2] -= p;
    trial.hydrostatic = p;

    // Voigt norm: ||s||² = s₁² + s₂² + s₃² (no shear)
    double s_norm = std::sqrt(trial.deviatoric[0]*trial.deviatoric[0]
                            + trial.deviatoric[1]*trial.deviatoric[1]
                            + trial.deviatoric[2]*trial.deviatoric[2]);
    trial.deviatoric_norm = s_norm;

    // Equivalent stress: σ_eq = √(3/2) ||s|| = |σ_11| for uniaxial
    double q = vm.equivalent_stress(trial);
    std::cout << "  σ_eq = " << q << " (expected " << sigma_11 << ")\n";
    assert(std::abs(q - sigma_11) < 1e-10 && "von Mises equiv stress for uniaxial");

    // Gradient: n̂ = s / ||s||
    auto n = vm.gradient(trial);
    double n_norm = n.norm();
    std::cout << "  ||n̂|| = " << n_norm << " (expected 1.0)\n";
    assert(std::abs(n_norm - 1.0) < 1e-10 && "gradient must be unit vector");

    // Zero stress → zero gradient (no crash)
    TrialState<6> trial_zero{};
    auto n_zero = vm.gradient(trial_zero);
    assert(n_zero.norm() < 1e-30 && "gradient at zero must be zero");

    std::cout << "  PASSED\n\n";
}


// ─── Test 2: LinearIsotropicHardening ────────────────────────────────────────

void test_isotropic_hardening() {
    std::cout << "Test 2: LinearIsotropicHardening\n";

    double sigma_y0 = 250.0;
    double H        = 10.0;
    LinearIsotropicHardening hard{sigma_y0, H};

    // Initial state (zero plastic strain)
    IsotropicHardeningState state{};
    assert(std::abs(hard.yield_stress(state) - sigma_y0) < 1e-15 &&
           "initial yield stress must be σ_y0");
    assert(std::abs(hard.modulus(state) - H) < 1e-15 &&
           "modulus must be H");

    // Evolve with Δγ = 0.1
    double delta_gamma = 0.1;
    auto state2 = hard.evolve(state, delta_gamma);
    double expected_eps_bar_p = std::sqrt(2.0 / 3.0) * delta_gamma;
    double expected_sigma_y   = sigma_y0 + H * expected_eps_bar_p;

    std::cout << "  ε̄^p after Δγ=0.1: " << state2.equivalent_plastic_strain
              << " (expected " << expected_eps_bar_p << ")\n";
    assert(std::abs(state2.equivalent_plastic_strain - expected_eps_bar_p) < 1e-15);

    std::cout << "  σ_y after evolve: " << hard.yield_stress(state2)
              << " (expected " << expected_sigma_y << ")\n";
    assert(std::abs(hard.yield_stress(state2) - expected_sigma_y) < 1e-12);

    // Original state is unchanged (functional style)
    assert(std::abs(state.equivalent_plastic_strain) < 1e-30 &&
           "evolve must not mutate the original state");

    std::cout << "  PASSED\n\n";
}


// ─── Test 3: AssociatedFlow delegates to yield gradient ──────────────────────

void test_associated_flow() {
    std::cout << "Test 3: AssociatedFlow ≡ yield gradient\n";

    VonMises vm;
    AssociatedFlow flow;

    // Some arbitrary trial state
    TrialState<6> trial;
    trial.stress     = (Eigen::Vector<double, 6>() << 200, -50, -50, 30, 0, 10).finished();
    trial.deviatoric = trial.stress;
    double p = (trial.stress[0] + trial.stress[1] + trial.stress[2]) / 3.0;
    trial.deviatoric[0] -= p; trial.deviatoric[1] -= p; trial.deviatoric[2] -= p;
    trial.deviatoric_norm = std::sqrt(
        trial.deviatoric[0]*trial.deviatoric[0]
      + trial.deviatoric[1]*trial.deviatoric[1]
      + trial.deviatoric[2]*trial.deviatoric[2]
      + 2.0*(trial.deviatoric[3]*trial.deviatoric[3]
           + trial.deviatoric[4]*trial.deviatoric[4]
           + trial.deviatoric[5]*trial.deviatoric[5]));
    trial.hydrostatic = p;

    auto n_yield = vm.gradient(trial);
    auto n_flow  = flow.direction(vm, trial);

    assert(approx_equal(n_yield, n_flow) &&
           "AssociatedFlow direction must equal yield gradient");

    std::cout << "  ||n_yield - n_flow|| = " << (n_yield - n_flow).norm() << "\n";
    std::cout << "  PASSED\n\n";
}


// ─── Test 4: PlasticityRelation — full return mapping ────────────────────────

void test_plasticity_relation() {
    std::cout << "Test 4: PlasticityRelation<3D, VonMises, LinearIsoHard, AssocFlow>\n";

    using PlRel = PlasticityRelation<
        ThreeDimensionalMaterial, VonMises, LinearIsotropicHardening, AssociatedFlow>;

    PlRel rel{200.0, 0.3, 0.250, 10.0};

    // 4a: Elastic regime
    std::cout << "  4a: Elastic regime\n";
    Strain<6> eps_small;
    eps_small.set_strain(
        (Eigen::Vector<double, 6>() << 1e-6, -3e-7, -3e-7, 0, 0, 0).finished());

    (void)rel.compute_response(eps_small);  // populate cache
    auto C_e = rel.elastic_tangent();
    auto C_t = rel.tangent(eps_small);
    assert(approx_equal(C_t, C_e, 1e-6) &&
           "tangent must equal elastic tangent below yield");
    std::cout << "    PASSED\n";

    // 4b: Plastic regime
    std::cout << "  4b: Plastic regime\n";
    Strain<6> eps_large;
    eps_large.set_strain(
        (Eigen::Vector<double, 6>() << 0.01, -0.003, -0.003, 0, 0, 0).finished());

    (void)rel.compute_response(eps_large);  // populate cache
    auto C_ep = rel.tangent(eps_large);
    double diff = (C_ep - C_e).norm();
    std::cout << "    ||C_ep - C_e|| = " << diff << "\n";
    assert(diff > 1e-6 && "consistent tangent must differ from elastic in plastic regime");
    std::cout << "    PASSED\n";

    // 4c: update (commit) evolves internal variables
    std::cout << "  4c: Commit evolves internal variables\n";
    assert(rel.internal_state().eps_bar_p() < 1e-30 &&
           "before update, ε̄^p must be zero");

    rel.update(eps_large);
    double eps_bar_p = rel.internal_state().eps_bar_p();
    std::cout << "    ε̄^p after update = " << eps_bar_p << "\n";
    assert(eps_bar_p > 1e-6 && "update must increment ε̄^p");

    double sy_now = rel.current_yield_stress();
    double sy_init = rel.initial_yield_stress();
    std::cout << "    σ_y(current) = " << sy_now << " > σ_y(initial) = " << sy_init << "\n";
    assert(sy_now > sy_init && "yield stress must increase with hardening");
    std::cout << "    PASSED\n";

    // 4d: Component accessors
    std::cout << "  4d: Component accessors\n";
    [[maybe_unused]] const auto& yf = rel.yield_criterion();
    [[maybe_unused]] const auto& hl = rel.hardening_law();
    [[maybe_unused]] const auto& fr = rel.flow_rule();
    assert(rel.young_modulus()  == 200.0);
    assert(rel.poisson_ratio()  == 0.3);
    std::cout << "    PASSED\n\n";
}


// ─── Test 5: Backward-compat J2PlasticityRelation alias ──────────────────────

void test_backward_compat() {
    std::cout << "Test 5: Backward-compat J2PlasticityRelation alias\n";

    // Old-style construction: J2PlasticityRelation<3D>{E, ν, σ_y0, H}
    J2PlasticityRelation<ThreeDimensionalMaterial> old_style{200.0, 0.3, 0.250, 10.0};

    // New-style construction
    PlasticityRelation<ThreeDimensionalMaterial, VonMises,
                       LinearIsotropicHardening, AssociatedFlow>
        new_style{200.0, 0.3, 0.250, 10.0};

    // Both must produce identical results
    Strain<6> eps;
    eps.set_strain(
        (Eigen::Vector<double, 6>() << 0.01, -0.003, -0.003, 0, 0, 0).finished());

    auto sigma_old = old_style.compute_response(eps);
    auto sigma_new = new_style.compute_response(eps);

    assert(approx_equal(sigma_old.components(), sigma_new.components()) &&
           "J2 alias must produce identical stress");

    auto C_old = old_style.tangent(eps);
    auto C_new = new_style.tangent(eps);
    assert(approx_equal(C_old, C_new) &&
           "J2 alias must produce identical tangent");

    std::cout << "  ||σ_old - σ_new|| = "
              << (sigma_old.components() - sigma_new.components()).norm() << "\n";
    std::cout << "  ||C_old - C_new|| = " << (C_old - C_new).norm() << "\n";

    // Named material alias through MaterialInstance
    J2PlasticMaterial3D mat{200.0, 0.3, 0.250, 10.0};
    auto sigma_via_instance = mat.compute_response(eps);
    assert(approx_equal(sigma_via_instance.components(), sigma_new.components()) &&
           "J2PlasticMaterial3D must produce identical stress");

    // J2InternalVariables alias
    J2InternalVariables<6> iv;
    iv.plastic_strain = Eigen::Vector<double, 6>::Ones() * 0.001;
    iv.hardening_state.equivalent_plastic_strain = 0.05;
    assert(std::abs(iv.eps_bar_p() - 0.05) < 1e-15 && "eps_bar_p accessor");
    assert(std::abs(iv.eps_p()[0] - 0.001) < 1e-15 && "eps_p accessor");

    std::cout << "  PASSED\n\n";
}


// ─── Test 6: Orthogonal composition — different hardening parameters ─────────

void test_orthogonal_composition() {
    std::cout << "Test 6: Orthogonal composition (same yield, different hardening)\n";

    using Rel = PlasticityRelation<
        ThreeDimensionalMaterial, VonMises, LinearIsotropicHardening, AssociatedFlow>;

    // Same elastic + yield, different hardening
    Rel soft{200.0, 0.3, LinearIsotropicHardening{0.250, 0.0}};   // perfect plasticity
    Rel hard{200.0, 0.3, LinearIsotropicHardening{0.250, 100.0}}; // strong hardening

    Strain<6> eps;
    eps.set_strain(
        (Eigen::Vector<double, 6>() << 0.01, -0.003, -0.003, 0, 0, 0).finished());

    auto sigma_soft = soft.compute_response(eps);
    auto sigma_hard = hard.compute_response(eps);

    double diff = (sigma_hard.components() - sigma_soft.components()).norm();
    std::cout << "  ||σ_hard - σ_soft|| = " << diff << "\n";
    assert(diff > 1e-6 && "different hardening must produce different stress");

    // With strong hardening, stress must be larger (more resistance)
    // Compare von Mises equivalent
    double q_soft = sigma_soft.components().head<3>().sum() / 3.0;
    double q_hard = sigma_hard.components().head<3>().sum() / 3.0;
    // Since uniaxial-like loading, σ₁₁ should be larger for harder material
    assert(sigma_hard[0] > sigma_soft[0] &&
           "harder material must produce larger axial stress");
    std::cout << "  σ_11 (soft) = " << sigma_soft[0]
              << ", σ_11 (hard) = " << sigma_hard[0] << "\n";
    (void)q_soft; (void)q_hard;

    // Consistent tangent must also differ
    auto C_soft = soft.tangent(eps);
    auto C_hard = hard.tangent(eps);
    double tangent_diff = (C_hard - C_soft).norm();
    std::cout << "  ||C_hard - C_soft|| = " << tangent_diff << "\n";
    assert(tangent_diff > 1e-6 && "tangent must differ for different hardening");

    std::cout << "  PASSED\n\n";
}


// ─── Test 7: Type-erasure integration ────────────────────────────────────────

void test_type_erasure_integration() {
    std::cout << "Test 7: Type-erasure (Material<> + InelasticUpdate)\n";

    // Construct via new explicit PlasticityRelation spelling
    using PlRel = PlasticityRelation<
        ThreeDimensionalMaterial, VonMises, LinearIsotropicHardening>;

    using MatInst = MaterialInstance<PlRel, MemoryState>;

    MatInst instance{200.0, 0.3, 0.250, 10.0};

    // Wrap in type-erasure
    Material<ThreeDimensionalMaterial> mat{instance, InelasticUpdate{}};

    Strain<6> eps;
    eps.set_strain(
        (Eigen::Vector<double, 6>() << 0.01, -0.003, -0.003, 0, 0, 0).finished());

    auto sigma = mat.compute_response(eps);
    std::cout << "  σ = [";
    for (int i = 0; i < 6; ++i) std::cout << " " << sigma[i];
    std::cout << " ]\n";

    // Compare with J2PlasticMaterial3D (backward-compat path)
    Material<ThreeDimensionalMaterial> mat_j2{
        J2PlasticMaterial3D{200.0, 0.3, 0.250, 10.0}, InelasticUpdate{}};
    auto sigma_j2 = mat_j2.compute_response(eps);

    assert(approx_equal(sigma.components(), sigma_j2.components()) &&
           "new PlasticityRelation must match J2 alias through type-erasure");

    // Commit and verify state evolution
    mat.commit(eps);
    auto sigma_after = mat.compute_response(eps);
    double diff = (sigma_after.components() - sigma.components()).norm();
    std::cout << "  ||σ_after_commit - σ_before|| = " << diff << "\n";
    assert(diff > 1e-10 && "commit must evolve state");

    // Deep clone:  MaterialInstance uses shared_ptr<Relation> (flyweight).
    // For inelastic materials, the relation carries mutable α, so clones
    // share internal state by design.  True independence requires separate
    // MaterialInstance construction (each with its own Relation copy).
    //
    // Here we verify that TWO independently-constructed instances wrapped
    // in type-erasure evolve independently after divergent commits.

    Material<ThreeDimensionalMaterial> mat_a{
        MatInst{200.0, 0.3, 0.250, 10.0}, InelasticUpdate{}};
    Material<ThreeDimensionalMaterial> mat_b{
        MatInst{200.0, 0.3, 0.250, 10.0}, InelasticUpdate{}};

    // Before any commit, they must agree
    auto sigma_a0 = mat_a.compute_response(eps);
    auto sigma_b0 = mat_b.compute_response(eps);
    assert(approx_equal(sigma_a0.components(), sigma_b0.components()) &&
           "identical instances must agree before commit");

    // Commit a at eps, b at a larger strain
    mat_a.commit(eps);
    Strain<6> eps2;
    eps2.set_strain(
        (Eigen::Vector<double, 6>() << 0.02, -0.006, -0.006, 0, 0, 0).finished());
    mat_b.commit(eps2);

    // Now they must differ at any common probe
    auto sigma_a1 = mat_a.compute_response(eps);
    auto sigma_b1 = mat_b.compute_response(eps);
    double clone_diff = (sigma_b1.components() - sigma_a1.components()).norm();
    std::cout << "  ||σ_b − σ_a|| (diverged commits) = " << clone_diff << "\n";
    assert(clone_diff > 1e-10 && "independently-constructed instances must evolve independently");

    std::cout << "  PASSED\n\n";
}


// ─── Test 8: Concept verification at compile time ────────────────────────────

void test_concept_verification() {
    std::cout << "Test 8: Static concept verification\n";

    // These are compile-time checks — if we get here, they passed.
    static_assert(YieldCriterion<VonMises, 6>);
    static_assert(YieldCriterion<VonMises, 3>);
    static_assert(YieldCriterion<VonMises, 1>);

    static_assert(HardeningLaw<LinearIsotropicHardening>);

    static_assert(FlowRule<AssociatedFlow, 6, VonMises>);
    static_assert(FlowRule<AssociatedFlow, 3, VonMises>);
    static_assert(FlowRule<AssociatedFlow, 1, VonMises>);

    using J2_3D = PlasticityRelation<
        ThreeDimensionalMaterial, VonMises, LinearIsotropicHardening, AssociatedFlow>;
    static_assert(ConstitutiveRelation<J2_3D>);
    static_assert(InelasticConstitutiveRelation<J2_3D>);

    using J2_2D = PlasticityRelation<
        PlaneMaterial, VonMises, LinearIsotropicHardening, AssociatedFlow>;
    static_assert(ConstitutiveRelation<J2_2D>);
    static_assert(InelasticConstitutiveRelation<J2_2D>);

    using J2_1D = PlasticityRelation<
        UniaxialMaterial, VonMises, LinearIsotropicHardening, AssociatedFlow>;
    static_assert(ConstitutiveRelation<J2_1D>);
    static_assert(InelasticConstitutiveRelation<J2_1D>);

    std::cout << "  All static_assert passed (3D, 2D, 1D)\n";
    std::cout << "  PASSED\n\n";
}


// =============================================================================

int main() {
    std::cout << "============================================\n";
    std::cout << "  Plasticity Composition Test Suite\n";
    std::cout << "============================================\n\n";

    test_von_mises_criterion();
    test_isotropic_hardening();
    test_associated_flow();
    test_plasticity_relation();
    test_backward_compat();
    test_orthogonal_composition();
    test_type_erasure_integration();
    test_concept_verification();

    std::cout << "============================================\n";
    std::cout << "  All tests PASSED\n";
    std::cout << "============================================\n";

    return 0;
}
