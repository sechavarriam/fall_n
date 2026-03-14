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
//    7. YieldFunction as an explicit compile-time customization point
//    8. ReturnAlgorithm as an explicit compile-time customization point
//    9. Type-erasure integration (Material<> + InelasticUpdate)
//   10. External algorithmic-state path for PlasticityRelation
//   11. Consistency residual/Jacobian as explicit customization points
//   12. Generic local nonlinear problem / solver split
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
#include "../src/materials/local_problem/LocalLinearSolver.hh"
#include "../src/materials/local_problem/LocalStepControl.hh"
#include "../src/materials/local_problem/LocalNonlinearProblem.hh"
#include "../src/materials/local_problem/NewtonLocalSolver.hh"
#include "../src/materials/constitutive_models/non_lineal/plasticity/ScalarConsistencyProblem.hh"


// ─── Helpers ─────────────────────────────────────────────────────────────────

template <typename D1, typename D2>
bool approx_equal(const Eigen::MatrixBase<D1>& a,
                  const Eigen::MatrixBase<D2>& b,
                  double tol = 1e-10) {
    return (a - b).norm() < tol;
}

template <int OffsetMilliUnits>
struct OffsetYieldFunction {
    template <std::size_t N, typename YieldCriterionT, typename HardeningT>
    [[nodiscard]] double value(const YieldCriterionT& yield,
                               const TrialState<N>& trial,
                               const HardeningT& hardening,
                               const typename HardeningT::StateT& state) const {
        return StandardYieldFunction{}.value(yield, trial, hardening, state)
             - static_cast<double>(OffsetMilliUnits) * 1.0e-3;
    }
};

struct HalfStepReturnAlgorithm {
    template <typename Relation>
    [[nodiscard]] auto integrate(
        const Relation& relation,
        const typename Relation::StrainVectorT& total_strain,
        const typename Relation::InternalVariablesT& alpha) const
        -> typename Relation::ReturnMapResultT
    {
        auto trial = relation.elastic_predictor(total_strain, alpha);
        auto eff = relation.effective_trial(trial, alpha);
        const double f_trial = relation.evaluate_yield_function(eff, alpha);

        if (f_trial <= 0.0) {
            return relation.make_elastic_result(trial, alpha);
        }

        const double H_mod = relation.hardening_modulus(alpha);
        const double delta_gamma =
            0.5 * relation.consistency_increment(f_trial, H_mod);
        const auto n_hat = relation.flow_direction(eff);

        typename Relation::ReturnMapResultT out{};
        out.stress = relation.corrected_stress(trial, delta_gamma, n_hat);
        out.tangent = relation.consistent_tangent(eff, delta_gamma, H_mod, n_hat);
        out.alpha_new = relation.evolve_internal_variables(alpha, delta_gamma, n_hat);
        out.plastic = true;
        return out;
    }
};

template <int OffsetMilliUnits>
struct OffsetConsistencyResidual {
    template <typename Relation>
    [[nodiscard]] double value(
        const Relation& relation,
        const typename Relation::TrialStateT& trial,
        const typename Relation::InternalVariablesT& alpha,
        double delta_gamma,
        const typename Relation::StrainVectorT& flow_direction) const
    {
        return StandardConsistencyResidual{}.value(
                   relation, trial, alpha, delta_gamma, flow_direction)
             - static_cast<double>(OffsetMilliUnits) * 1.0e-3;
    }
};

struct DummyLocalRelation {};

struct HalfStepControl {
    template <typename Problem, typename Relation, typename ContextT>
    [[nodiscard]] auto compute(
        const Problem&,
        const Relation&,
        const ContextT&,
        const typename Problem::UnknownT&,
        const typename Problem::UnknownT& delta_unknown,
        double) const
        -> typename Problem::UnknownT
    {
        return 0.5 * delta_unknown;
    }
};

struct AffineTwoEquationProblem {
    using UnknownT = Eigen::Vector2d;
    using ResidualT = Eigen::Vector2d;
    using JacobianT = Eigen::Matrix2d;

    struct ContextT {
        Eigen::Vector2d rhs{Eigen::Vector2d::Zero()};
    };

    [[nodiscard]] UnknownT initial_guess(
        const DummyLocalRelation&,
        const ContextT&) const
    {
        return UnknownT::Zero();
    }

    [[nodiscard]] ResidualT residual(
        const DummyLocalRelation&,
        const ContextT& context,
        const UnknownT& x) const
    {
        ResidualT r;
        r[0] = x[0] + x[1] - context.rhs[0];
        r[1] = x[0] - x[1] - context.rhs[1];
        return r;
    }

    [[nodiscard]] JacobianT jacobian(
        const DummyLocalRelation&,
        const ContextT&,
        const UnknownT&) const
    {
        JacobianT J;
        J << 1.0, 1.0,
             1.0, -1.0;
        return J;
    }

    [[nodiscard]] double residual_norm(const ResidualT& r) const
    {
        return r.norm();
    }
};


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


// ─── Test 7: YieldFunction customization point ──────────────────────────────

void test_yield_function_customization() {
    std::cout << "Test 7: YieldFunction customization point\n";

    using StandardRel = PlasticityRelation<
        ThreeDimensionalMaterial, VonMises, LinearIsotropicHardening, AssociatedFlow>;
    using ShiftedRel = PlasticityRelation<
        ThreeDimensionalMaterial,
        VonMises,
        LinearIsotropicHardening,
        AssociatedFlow,
        OffsetYieldFunction<50>>;

    static_assert(YieldFunctionPolicy<StandardYieldFunction, 6, VonMises, LinearIsotropicHardening>);
    static_assert(YieldFunctionPolicy<OffsetYieldFunction<50>, 6, VonMises, LinearIsotropicHardening>);

    StandardRel standard{200.0, 0.3, 0.250, 10.0};
    ShiftedRel shifted{200.0, 0.3, 0.250, 10.0};

    Strain<6> eps;
    eps.set_strain(
        (Eigen::Vector<double, 6>() << 0.0013, -0.00039, -0.00039, 0, 0, 0).finished());

    auto sigma_standard = standard.compute_response(eps);
    auto sigma_shifted = shifted.compute_response(eps);
    auto C_standard = standard.tangent(eps);
    auto C_shifted = shifted.tangent(eps);

    const double stress_diff =
        (sigma_standard.components() - sigma_shifted.components()).norm();
    const double tangent_diff = (C_standard - C_shifted).norm();

    std::cout << "  ||σ_standard - σ_shifted|| = " << stress_diff << "\n";
    std::cout << "  ||C_standard - C_shifted|| = " << tangent_diff << "\n";

    assert(stress_diff > 1e-6 &&
           "a custom YieldFunction must alter the constitutive response when it changes overstress");
    assert(tangent_diff > 1e-6 &&
           "a custom YieldFunction must alter the algorithmic tangent when it changes yielding");

    [[maybe_unused]] const auto& yf = shifted.yield_function();
    std::cout << "  PASSED\n\n";
}


// ─── Test 8: ReturnAlgorithm customization point ────────────────────────────

void test_return_algorithm_customization() {
    std::cout << "Test 8: ReturnAlgorithm customization point\n";

    using StandardRel = PlasticityRelation<
        ThreeDimensionalMaterial, VonMises, LinearIsotropicHardening, AssociatedFlow>;
    using HalfStepRel = PlasticityRelation<
        ThreeDimensionalMaterial,
        VonMises,
        LinearIsotropicHardening,
        AssociatedFlow,
        StandardYieldFunction,
        HalfStepReturnAlgorithm>;

    static_assert(ReturnAlgorithmPolicy<StandardRadialReturnAlgorithm, StandardRel>);
    static_assert(ReturnAlgorithmPolicy<HalfStepReturnAlgorithm, HalfStepRel>);

    StandardRel standard{200.0, 0.3, 0.250, 10.0};
    HalfStepRel half_step{200.0, 0.3, 0.250, 10.0};

    Strain<6> eps;
    eps.set_strain(
        (Eigen::Vector<double, 6>() << 0.01, -0.003, -0.003, 0, 0, 0).finished());

    auto sigma_standard = standard.compute_response(eps);
    auto sigma_half_step = half_step.compute_response(eps);
    auto C_standard = standard.tangent(eps);
    auto C_half_step = half_step.tangent(eps);

    const double stress_diff =
        (sigma_standard.components() - sigma_half_step.components()).norm();
    const double tangent_diff = (C_standard - C_half_step).norm();

    std::cout << "  ||σ_standard - σ_half_step|| = " << stress_diff << "\n";
    std::cout << "  ||C_standard - C_half_step|| = " << tangent_diff << "\n";

    assert(stress_diff > 1e-6 &&
           "a custom ReturnAlgorithm must alter the constitutive response when it changes the correction strategy");
    assert(tangent_diff > 1e-6 &&
           "a custom ReturnAlgorithm must alter the algorithmic tangent when it changes the local solve");

    [[maybe_unused]] const auto& ra = half_step.return_algorithm();
    std::cout << "  PASSED\n\n";
}


// ─── Test 9: Type-erasure integration ────────────────────────────────────────

void test_type_erasure_integration() {
    std::cout << "Test 9: Type-erasure (Material<> + InelasticUpdate)\n";

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

    // Commit and verify self-consistency: re-evaluating at the same
    // purely-normal strain must return the same corrected stress
    // (return-mapping idempotent when Ce·n̂ = 2G·n̂ for normal strains).
    mat.commit(eps);
    auto sigma_after = mat.compute_response(eps);
    double diff = (sigma_after.components() - sigma.components()).norm();
    std::cout << "  ||σ_after_commit - σ_before|| = " << diff << "\n";
    assert(diff < 1e-10 && "committed state must be self-consistent at same strain");

    // Verify that internal state actually evolved: compare committed
    // material vs a fresh one at a NON-PROPORTIONAL strain (for proportional
    // loading + linear hardening the return-mapping is path-independent).
    Strain<6> eps_np;
    eps_np.set_strain(
        (Eigen::Vector<double, 6>() << 0.005, 0.005, -0.010, 0, 0, 0).finished());
    Material<ThreeDimensionalMaterial> mat_fresh{MatInst{200.0, 0.3, 0.250, 10.0}, InelasticUpdate{}};
    auto sigma_committed = mat.compute_response(eps_np);
    auto sigma_fresh_val = mat_fresh.compute_response(eps_np);
    double state_diff = (sigma_committed.components()
                       - sigma_fresh_val.components()).norm();
    std::cout << "  ||σ_committed - σ_fresh|| at ε_np = " << state_diff << "\n";
    assert(state_diff > 1e-10 && "committed material must differ from fresh at non-proportional strain");

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


// ─── Test 10: External algorithmic-state path ───────────────────────────────

void test_external_algorithmic_state_path() {
    std::cout << "Test 10: External algorithmic-state path\n";

    using PlRel = PlasticityRelation<
        ThreeDimensionalMaterial, VonMises, LinearIsotropicHardening, AssociatedFlow>;

    static_assert(ExternallyStateDrivenConstitutiveRelation<PlRel>);

    PlRel rel{200.0, 0.3, 0.250, 10.0};
    typename PlRel::InternalVariablesT alpha{};

    Strain<6> eps_large;
    eps_large.set_strain(
        (Eigen::Vector<double, 6>() << 0.01, -0.003, -0.003, 0, 0, 0).finished());

    // Commit through the external-state path only.
    rel.commit(alpha, eps_large);

    assert(alpha.eps_bar_p() > 1e-6 &&
           "external algorithmic state must evolve under commit(alpha, eps)");
    assert(rel.internal_state().eps_bar_p() < 1e-30 &&
           "explicit-state commit must not mutate the relation-owned compatibility state");

    // A typed constitutive site should now store the algorithmic state at the
    // site level, not inside the relation.
    using MatInst = MaterialInstance<PlRel, CommittedState>;
    MatInst site{200.0, 0.3, 0.250, 10.0};
    site.update(eps_large);

    assert(site.internal_state().eps_bar_p() > 1e-6 &&
           "MaterialInstance must expose the externally stored algorithmic state");
    assert(site.constitutive_relation().internal_state().eps_bar_p() < 1e-30 &&
           "site update must leave the embedded compatibility state in the relation untouched");

    Strain<6> eps_np;
    eps_np.set_strain(
        (Eigen::Vector<double, 6>() << 0.005, 0.005, -0.010, 0, 0, 0).finished());

    auto sigma_site = site.compute_response(eps_np);
    MatInst fresh_site{200.0, 0.3, 0.250, 10.0};
    auto sigma_fresh = fresh_site.compute_response(eps_np);

    double diff = (sigma_site.components() - sigma_fresh.components()).norm();
    std::cout << "  ||σ_site - σ_fresh|| after explicit-state commit = " << diff << "\n";
    assert(diff > 1e-10 &&
           "site-level explicit algorithmic state must affect subsequent constitutive response");

    std::cout << "  PASSED\n\n";
}


// ─── Test 11: Concept verification at compile time ───────────────────────────

void test_concept_verification() {
    std::cout << "Test 11: Static concept verification\n";

    // These are compile-time checks — if we get here, they passed.
    static_assert(YieldCriterion<VonMises, 6>);
    static_assert(YieldCriterion<VonMises, 3>);
    static_assert(YieldCriterion<VonMises, 1>);
    static_assert(YieldFunctionPolicy<StandardYieldFunction, 6, VonMises, LinearIsotropicHardening>);

    static_assert(HardeningLaw<LinearIsotropicHardening>);

    static_assert(FlowRule<AssociatedFlow, 6, VonMises>);
    static_assert(FlowRule<AssociatedFlow, 3, VonMises>);
    static_assert(FlowRule<AssociatedFlow, 1, VonMises>);

    using J2_3D = PlasticityRelation<
        ThreeDimensionalMaterial, VonMises, LinearIsotropicHardening, AssociatedFlow>;
    static_assert(ConstitutiveRelation<J2_3D>);
    static_assert(InelasticConstitutiveRelation<J2_3D>);
    static_assert(ReturnAlgorithmPolicy<StandardRadialReturnAlgorithm, J2_3D>);
    static_assert(ConsistencyResidualPolicy<StandardConsistencyResidual, J2_3D>);
    static_assert(ConsistencyJacobianPolicy<
        StandardConsistencyJacobian, StandardConsistencyResidual, J2_3D>);
    static_assert(ConsistencyJacobianPolicy<
        FiniteDifferenceConsistencyJacobian<>, StandardConsistencyResidual, J2_3D>);
    static_assert(LocalNonlinearProblem<
        ScalarConsistencyProblem<>,
        J2_3D,
        ScalarConsistencyProblem<>::ContextT<J2_3D>>);
    static_assert(LocalNonlinearProblem<
        AffineTwoEquationProblem,
        DummyLocalRelation,
        AffineTwoEquationProblem::ContextT>);
    static_assert(LocalNonlinearSolverPolicy<
        NewtonLocalSolver<>,
        AffineTwoEquationProblem,
        DummyLocalRelation,
        AffineTwoEquationProblem::ContextT>);
    static_assert(LocalLinearSolvePolicy<
        DefaultLocalLinearSolver,
        Eigen::Vector2d,
        Eigen::Matrix2d,
        Eigen::Vector2d>);
    static_assert(LocalStepControlPolicy<
        FullStepControl,
        AffineTwoEquationProblem,
        DummyLocalRelation,
        AffineTwoEquationProblem::ContextT>);
    static_assert(LocalStepControlPolicy<
        BacktrackingResidualStepControl<>,
        AffineTwoEquationProblem,
        DummyLocalRelation,
        AffineTwoEquationProblem::ContextT>);
    static_assert(LocalStepControlPolicy<
        HalfStepControl,
        AffineTwoEquationProblem,
        DummyLocalRelation,
        AffineTwoEquationProblem::ContextT>);

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

void test_newton_return_algorithm_matches_standard_radial() {
    std::cout << "Test 12: Newton consistency algorithm matches standard radial return\n";

    using StandardRel = PlasticityRelation<
        ThreeDimensionalMaterial, VonMises, LinearIsotropicHardening, AssociatedFlow>;
    using NewtonRel = PlasticityRelation<
        ThreeDimensionalMaterial,
        VonMises,
        LinearIsotropicHardening,
        AssociatedFlow,
        StandardYieldFunction,
        NewtonConsistencyReturnAlgorithm<>>;
    using NewtonFDRel = PlasticityRelation<
        ThreeDimensionalMaterial,
        VonMises,
        LinearIsotropicHardening,
        AssociatedFlow,
        StandardYieldFunction,
        NewtonConsistencyReturnAlgorithm<
            StandardConsistencyResidual,
            FiniteDifferenceConsistencyJacobian<>>>;
    using GenericLocalRel = PlasticityRelation<
        ThreeDimensionalMaterial,
        VonMises,
        LinearIsotropicHardening,
        AssociatedFlow,
        StandardYieldFunction,
        LocalNonlinearReturnAlgorithm<>>;

    static_assert(ReturnAlgorithmPolicy<NewtonConsistencyReturnAlgorithm<>, NewtonRel>);
    static_assert(ReturnAlgorithmPolicy<
        NewtonConsistencyReturnAlgorithm<
            StandardConsistencyResidual,
            FiniteDifferenceConsistencyJacobian<>>,
        NewtonFDRel>);
    static_assert(ReturnAlgorithmPolicy<LocalNonlinearReturnAlgorithm<>, GenericLocalRel>);

    StandardRel standard{200.0, 0.3, 0.250, 10.0};
    NewtonRel newton{200.0, 0.3, 0.250, 10.0};
    NewtonFDRel newton_fd{200.0, 0.3, 0.250, 10.0};
    GenericLocalRel generic_local{200.0, 0.3, 0.250, 10.0};

    Strain<6> eps;
    eps.set_strain(
        (Eigen::Vector<double, 6>() << 0.01, -0.003, -0.003, 0, 0, 0).finished());

    auto sigma_standard = standard.compute_response(eps);
    auto sigma_newton = newton.compute_response(eps);
    auto sigma_newton_fd = newton_fd.compute_response(eps);
    auto sigma_generic_local = generic_local.compute_response(eps);

    auto C_standard = standard.tangent(eps);
    auto C_newton = newton.tangent(eps);
    auto C_newton_fd = newton_fd.tangent(eps);
    auto C_generic_local = generic_local.tangent(eps);

    const double stress_err =
        (sigma_standard.components() - sigma_newton.components()).norm();
    const double stress_fd_err =
        (sigma_standard.components() - sigma_newton_fd.components()).norm();
    const double stress_generic_err =
        (sigma_standard.components() - sigma_generic_local.components()).norm();
    const double tangent_err = (C_standard - C_newton).norm();
    const double tangent_fd_err = (C_standard - C_newton_fd).norm();
    const double tangent_generic_err = (C_standard - C_generic_local).norm();

    std::cout << "  ||σ_standard - σ_newton||    = " << stress_err << "\n";
    std::cout << "  ||σ_standard - σ_newton_fd|| = " << stress_fd_err << "\n";
    std::cout << "  ||σ_standard - σ_generic||   = " << stress_generic_err << "\n";
    std::cout << "  ||C_standard - C_newton||    = " << tangent_err << "\n";
    std::cout << "  ||C_standard - C_newton_fd|| = " << tangent_fd_err << "\n";
    std::cout << "  ||C_standard - C_generic||   = " << tangent_generic_err << "\n";

    assert(stress_err < 1e-9 &&
           "Newton consistency solve must reproduce the current radial-return stress");
    assert(stress_fd_err < 1e-8 &&
           "finite-difference Jacobian must remain a viable local solve for the current model");
    assert(stress_generic_err < 1e-9 &&
           "the generic local-problem return algorithm must reproduce the current radial-return stress");
    assert(tangent_err < 1e-8 &&
           "Newton consistency solve must reproduce the current algorithmic tangent");
    assert(tangent_fd_err < 1e-6 &&
           "finite-difference Jacobian must produce a compatible tangent");
    assert(tangent_generic_err < 1e-8 &&
           "the generic local-problem return algorithm must reproduce the current algorithmic tangent");

    std::cout << "  PASSED\n\n";
}

void test_consistency_residual_customization() {
    std::cout << "Test 13: Consistency residual customization point\n";

    using NewtonRel = PlasticityRelation<
        ThreeDimensionalMaterial,
        VonMises,
        LinearIsotropicHardening,
        AssociatedFlow,
        StandardYieldFunction,
        NewtonConsistencyReturnAlgorithm<>>;

    using ShiftedResidualRel = PlasticityRelation<
        ThreeDimensionalMaterial,
        VonMises,
        LinearIsotropicHardening,
        AssociatedFlow,
        StandardYieldFunction,
        NewtonConsistencyReturnAlgorithm<
            OffsetConsistencyResidual<5>,
            FiniteDifferenceConsistencyJacobian<>>>;

    static_assert(ConsistencyResidualPolicy<OffsetConsistencyResidual<5>, ShiftedResidualRel>);

    NewtonRel standard{200.0, 0.3, 0.250, 10.0};
    ShiftedResidualRel shifted{200.0, 0.3, 0.250, 10.0};

    Strain<6> eps;
    eps.set_strain(
        (Eigen::Vector<double, 6>() << 0.01, -0.003, -0.003, 0, 0, 0).finished());

    auto sigma_standard = standard.compute_response(eps);
    auto sigma_shifted = shifted.compute_response(eps);
    auto C_standard = standard.tangent(eps);
    auto C_shifted = shifted.tangent(eps);

    const double stress_diff =
        (sigma_standard.components() - sigma_shifted.components()).norm();
    const double tangent_diff = (C_standard - C_shifted).norm();

    std::cout << "  ||σ_standard - σ_shiftedResidual|| = " << stress_diff << "\n";
    std::cout << "  ||C_standard - C_shiftedResidual|| = " << tangent_diff << "\n";

    assert(stress_diff > 1e-6 &&
           "changing the consistency residual must alter the local constitutive response");
    assert(tangent_diff > 1e-6 &&
           "changing the consistency residual must alter the algorithmic tangent");

    std::cout << "  PASSED\n\n";
}

void test_generic_local_nonlinear_solver_vector_problem() {
    std::cout << "Test 14: Generic local nonlinear solver solves a fixed-size vector problem\n";

    static_assert(LocalNonlinearProblem<
        AffineTwoEquationProblem,
        DummyLocalRelation,
        AffineTwoEquationProblem::ContextT>);

    DummyLocalRelation relation{};
    AffineTwoEquationProblem problem{};
    NewtonLocalSolver<4> solver{};
    AffineTwoEquationProblem::ContextT context;
    context.rhs << 3.0, 1.0;

    const auto result = solver.solve(problem, relation, context);

    std::cout << "  converged      = " << result.converged << "\n";
    std::cout << "  iterations     = " << result.iterations << "\n";
    std::cout << "  residual_norm  = " << result.residual_norm << "\n";
    std::cout << "  solution       = [" << result.solution[0]
              << ", " << result.solution[1] << "]\n";

    assert(result.converged &&
           "the generic Newton local solver must converge on a well-posed 2x2 affine problem");
    assert(std::abs(result.solution[0] - 2.0) < 1e-12);
    assert(std::abs(result.solution[1] - 1.0) < 1e-12);

    std::cout << "  PASSED\n\n";
}

void test_newton_subpolicies_are_injectable() {
    std::cout << "Test 15: Newton sub-policies are injectable\n";

    DummyLocalRelation relation{};
    AffineTwoEquationProblem problem{};
    AffineTwoEquationProblem::ContextT context;
    context.rhs << 3.0, 1.0;

    NewtonLocalSolver<32, DefaultLocalLinearSolver, HalfStepControl> half_step_solver{};
    half_step_solver.tolerance_ = 1e-8;
    const auto half_step_result = half_step_solver.solve(problem, relation, context);

    std::cout << "  converged      = " << half_step_result.converged << "\n";
    std::cout << "  iterations     = " << half_step_result.iterations << "\n";
    std::cout << "  residual_norm  = " << half_step_result.residual_norm << "\n";
    std::cout << "  solution       = [" << half_step_result.solution[0]
              << ", " << half_step_result.solution[1] << "]\n";

    assert(half_step_result.converged &&
           "a custom step-control policy must remain a valid local solver configuration");
    assert(std::abs(half_step_result.solution[0] - 2.0) < 1e-6);
    assert(std::abs(half_step_result.solution[1] - 1.0) < 1e-6);
    assert(half_step_result.iterations > 0 &&
           "the damped path should require more than the immediate exact update of the affine problem");

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
    test_yield_function_customization();
    test_return_algorithm_customization();
    test_type_erasure_integration();
    test_external_algorithmic_state_path();
    test_concept_verification();
    test_newton_return_algorithm_matches_standard_radial();
    test_consistency_residual_customization();
    test_generic_local_nonlinear_solver_vector_problem();
    test_newton_subpolicies_are_injectable();

    std::cout << "============================================\n";
    std::cout << "  All tests PASSED\n";
    std::cout << "============================================\n";

    return 0;
}
