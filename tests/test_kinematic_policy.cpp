// =============================================================================
//  test_kinematic_policy.cpp — Tests for KinematicPolicy (Phase 2)
// =============================================================================
//
//  All tests use the gradient-based (_from_gradients) API of each policy,
//  which depends only on Eigen — no PETSc or ElementGeometry required.
//
//  Testing strategy:
//    1. Compile-time trait verification
//    2. SmallStrain B matrix structure for 1D / 2D / 3D
//    3. SmallStrain evaluate consistency
//    4. TotalLagrangian deformation gradient computation
//    5. TotalLagrangian B_NL(F=I) == SmallStrain B  (reduction test)
//    6. TotalLagrangian B_NL numerical derivative verification
//    7. TotalLagrangian evaluate consistency (B_NL·u == E_eng_voigt)
//    8. Geometric stiffness symmetry and structure
//    9. stress_voigt_to_matrix round-trip for all dims
//   10. GPKinematics struct integrity
//   11. CompatibleFormulation concept check
//   12. Manufactured-solution patch test (uniform strain)
//
// =============================================================================

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>

#include "src/analysis/AnalysisRouteAudit.hh"
#include "src/continuum/Continuum.hh"
#include "src/elements/BeamKinematicPolicy.hh"
#include "src/elements/MITCShellPolicy.hh"
#include "src/elements/ShellKinematicPolicy.hh"

using namespace continuum;

namespace {

constexpr bool approx(double a, double b, double tol = 1e-10) {
    return std::abs(a - b) <= tol;
}

int passed = 0, failed = 0;

void report(const char* name, bool ok) {
    if (ok) { ++passed; std::cout << "  PASS  " << name << "\n"; }
    else    { ++failed; std::cout << "  FAIL  " << name << "\n"; }
}

// Random number generator for manufactured tests
std::mt19937 rng{42};
std::uniform_real_distribution<double> dist{-0.05, 0.05};  // small displacements
double rand_small() { return dist(rng); }

} // anonymous namespace


// =============================================================================
//  1. Compile-time trait verification
// =============================================================================

void test_policy_traits() {
    // SmallStrain
    static_assert(SmallStrain::is_geometrically_linear);
    static_assert(!SmallStrain::needs_geometric_stiffness);
    static_assert(KinematicPolicyConcept<SmallStrain>);

    // TotalLagrangian
    static_assert(!TotalLagrangian::is_geometrically_linear);
    static_assert(TotalLagrangian::needs_geometric_stiffness);
    static_assert(KinematicPolicyConcept<TotalLagrangian>);

    // Additional policy slots
    static_assert(KinematicPolicyConcept<UpdatedLagrangian>);
    static_assert(KinematicPolicyConcept<Corotational>);
    static_assert(KinematicFormulationTraits<SmallStrain>::maturity == FormulationMaturity::implemented);
    static_assert(KinematicFormulationTraits<TotalLagrangian>::maturity == FormulationMaturity::implemented);
    static_assert(KinematicFormulationTraits<UpdatedLagrangian>::maturity == FormulationMaturity::partial);
    static_assert(KinematicFormulationTraits<Corotational>::maturity == FormulationMaturity::placeholder);
    static_assert(KinematicFormulationTraits<SmallStrain>::virtual_work_compatibility ==
                  VirtualWorkCompatibilityKind::linearized_equivalent);
    static_assert(KinematicFormulationTraits<TotalLagrangian>::virtual_work_compatibility ==
                  VirtualWorkCompatibilityKind::exact);
    static_assert(KinematicFormulationTraits<UpdatedLagrangian>::virtual_work_compatibility ==
                  VirtualWorkCompatibilityKind::exact);
    static_assert(KinematicFormulationTraits<Corotational>::virtual_work_compatibility ==
                  VirtualWorkCompatibilityKind::unaudited_placeholder);
    static_assert(KinematicFormulationTraits<TotalLagrangian>::audit_scope.is_reference_finite_kinematics_path());
    static_assert(KinematicFormulationTraits<UpdatedLagrangian>::audit_scope.requires_finite_kinematics_scope_disclaimer());
    static_assert(!KinematicFormulationTraits<SmallStrain>::audit_scope.supports_finite_kinematics);
    static_assert(!KinematicFormulationTraits<Corotational>::audit_scope.has_runtime_path());
    static_assert(KinematicFormulationTraits<TotalLagrangian>::family_audit_scope.is_reference_geometric_nonlinearity_path());
    static_assert(KinematicFormulationTraits<UpdatedLagrangian>::family_audit_scope.requires_geometric_nonlinearity_scope_disclaimer());
    static_assert(beam::BeamKinematicFormulationTraits<beam::Corotational>::audit_scope.is_reference_geometric_nonlinearity_path());
    static_assert(shell::ShellKinematicFormulationTraits<shell::Corotational>::audit_scope.requires_geometric_nonlinearity_scope_disclaimer());
    static_assert(FamilyNormativelySupportedKinematicPolicy<ElementFamilyKind::continuum_solid_3d, SmallStrain>);
    static_assert(FamilyNormativelySupportedKinematicPolicy<ElementFamilyKind::continuum_solid_3d, TotalLagrangian>);
    static_assert(FamilyNormativelySupportedKinematicPolicy<ElementFamilyKind::continuum_solid_3d, UpdatedLagrangian>);
    static_assert(!FamilyNormativelySupportedKinematicPolicy<ElementFamilyKind::continuum_solid_3d, Corotational>);
    static_assert(FamilyReferenceGeometricNonlinearityKinematicPolicy<ElementFamilyKind::continuum_solid_3d, TotalLagrangian>);
    static_assert(FamilyNormativelySupportedKinematicPolicy<ElementFamilyKind::beam_1d, beam::SmallRotation>);
    static_assert(FamilyNormativelySupportedKinematicPolicy<ElementFamilyKind::beam_1d, beam::Corotational>);
    static_assert(!FamilyNormativelySupportedKinematicPolicy<ElementFamilyKind::beam_1d, TotalLagrangian>);
    static_assert(FamilyReferenceGeometricNonlinearityKinematicPolicy<ElementFamilyKind::beam_1d, beam::Corotational>);
    static_assert(FamilyNormativelySupportedKinematicPolicy<ElementFamilyKind::shell_2d, shell::SmallRotation>);
    static_assert(FamilyNormativelySupportedKinematicPolicy<ElementFamilyKind::shell_2d, shell::Corotational>);
    static_assert(!FamilyNormativelySupportedKinematicPolicy<ElementFamilyKind::shell_2d, UpdatedLagrangian>);
    static_assert(!canonical_family_formulation_audit_scope(
        ElementFamilyKind::beam_1d, FormulationKind::total_lagrangian).supports_normatively());
    static_assert(!canonical_family_formulation_audit_scope(
        ElementFamilyKind::shell_2d, FormulationKind::updated_lagrangian).supports_normatively());
    static_assert(canonical_family_formulation_audit_row(ElementFamilyKind::continuum_solid_3d).size() == 4);
    static_assert(canonical_family_formulation_audit_table().size() == 12);
    static_assert(count_normatively_supported_family_formulations(ElementFamilyKind::continuum_solid_3d) == 3);
    static_assert(count_normatively_supported_family_formulations(ElementFamilyKind::beam_1d) == 2);
    static_assert(count_normatively_supported_family_formulations(ElementFamilyKind::shell_2d) == 2);
    static_assert(find_family_linear_reference_path(ElementFamilyKind::continuum_solid_3d).has_value());
    static_assert(find_family_geometric_nonlinearity_reference_path(ElementFamilyKind::continuum_solid_3d)
                      .has_value());
    static_assert(find_family_geometric_nonlinearity_reference_path(ElementFamilyKind::continuum_solid_3d)
                      ->formulation_kind == FormulationKind::total_lagrangian);
    static_assert(find_family_geometric_nonlinearity_reference_path(ElementFamilyKind::beam_1d)
                      .has_value());
    static_assert(find_family_geometric_nonlinearity_reference_path(ElementFamilyKind::beam_1d)
                      ->formulation_kind == FormulationKind::corotational);
    static_assert(!find_family_geometric_nonlinearity_reference_path(ElementFamilyKind::shell_2d)
                       .has_value());
    static_assert(ReferencePlacement<3>::from_configuration == ConfigurationKind::material_body);
    static_assert(ReferencePlacement<3>::to_configuration == ConfigurationKind::reference);
    static_assert(CurrentPlacement<3>::to_configuration == ConfigurationKind::current);
    static_assert(CorotatedPlacement<3>::to_configuration == ConfigurationKind::corotated);
    static_assert(canonical_analysis_route_audit_scope(
                      fall_n::AnalysisRouteKind::nonlinear_incremental_newton)
                      .supports_checkpoint_restart);
    static_assert(canonical_analysis_route_audit_scope(
                      fall_n::AnalysisRouteKind::implicit_second_order_dynamics)
                      .supports_inertial_terms);
    static_assert(canonical_analysis_route_audit_scope(
                      fall_n::AnalysisRouteKind::arc_length_continuation)
                      .requires_scope_disclaimer());
    static_assert(fall_n::canonical_family_formulation_analysis_route_row(
                      ElementFamilyKind::continuum_solid_3d,
                      FormulationKind::total_lagrangian).size() == 4);
    static_assert(fall_n::canonical_family_formulation_analysis_route_table().size() == 48);
    static_assert(fall_n::count_normatively_supported_analysis_routes(
                      ElementFamilyKind::continuum_solid_3d,
                      FormulationKind::total_lagrangian) == 1);
    static_assert(fall_n::count_runtime_declared_analysis_routes(
                      ElementFamilyKind::continuum_solid_3d,
                      FormulationKind::total_lagrangian) == 3);
    static_assert(canonical_family_formulation_analysis_route_audit_scope(
                      ElementFamilyKind::continuum_solid_3d,
                      FormulationKind::total_lagrangian,
                      fall_n::AnalysisRouteKind::nonlinear_incremental_newton)
                      .is_reference_route_for_scope());
    static_assert(canonical_family_formulation_analysis_route_audit_scope(
                      ElementFamilyKind::continuum_solid_3d,
                      FormulationKind::total_lagrangian,
                      fall_n::AnalysisRouteKind::implicit_second_order_dynamics)
                      .requires_scope_disclaimer());
    static_assert(canonical_family_formulation_analysis_route_audit_scope(
                      ElementFamilyKind::beam_1d,
                      FormulationKind::corotational,
                      fall_n::AnalysisRouteKind::nonlinear_incremental_newton)
                      .has_runtime_path);
    static_assert(!canonical_family_formulation_analysis_route_audit_scope(
                       ElementFamilyKind::beam_1d,
                       FormulationKind::corotational,
                       fall_n::AnalysisRouteKind::nonlinear_incremental_newton)
                       .supports_normatively());
    static_assert(canonical_family_formulation_analysis_route_audit_scope(
                      ElementFamilyKind::shell_2d,
                      FormulationKind::small_strain,
                      fall_n::AnalysisRouteKind::linear_static)
                      .is_reference_route_for_scope());

    report("policy_traits_SmallStrain", true);
    report("policy_traits_TotalLagrangian", true);
    report("policy_traits_audit_status", true);
    report("policy_traits_family_audit_table_helpers", true);
    report("policy_traits_analysis_route_audit_helpers", true);
}


// =============================================================================
//  2. SmallStrain B matrix structure
// =============================================================================

void test_small_strain_B_1D() {
    // 2-node bar element:  N₁ = (1-ξ)/2,  N₂ = (1+ξ)/2
    // For a unit bar: dN₁/dx = -1/L,  dN₂/dx = 1/L,  L=2
    Eigen::Matrix<double, Eigen::Dynamic, 1> grad(2, 1);
    grad(0, 0) = -0.5;  // dN₁/dx
    grad(1, 0) =  0.5;  // dN₂/dx

    auto B = SmallStrain::compute_B_from_gradients<1>(grad, 1);

    report("SS_B_1D_size", B.rows() == 1 && B.cols() == 2);
    report("SS_B_1D_values", approx(B(0,0), -0.5) && approx(B(0,1), 0.5));
}

void test_small_strain_B_2D() {
    // 3-node triangular element (constant strain triangle)
    // Assume gradients in a unit right triangle:
    //  dN₁/dx₁ = -1,  dN₁/dx₂ = -1
    //  dN₂/dx₁ =  1,  dN₂/dx₂ =  0
    //  dN₃/dx₁ =  0,  dN₃/dx₂ =  1
    Eigen::Matrix<double, Eigen::Dynamic, 2> grad(3, 2);
    grad << -1.0, -1.0,
             1.0,  0.0,
             0.0,  1.0;

    auto B = SmallStrain::compute_B_from_gradients<2>(grad, 2);

    // B should be 3×6:  Voigt {ε₁₁, ε₂₂, γ₁₂}
    report("SS_B_2D_size", B.rows() == 3 && B.cols() == 6);

    // Row 0 (ε₁₁): B(0, 2I) = dN_I/dx₁
    report("SS_B_2D_eps11", approx(B(0,0), -1.0) && approx(B(0,2), 1.0)
                         && approx(B(0,4),  0.0));

    // Row 1 (ε₂₂): B(1, 2I+1) = dN_I/dx₂
    report("SS_B_2D_eps22", approx(B(1,1), -1.0) && approx(B(1,3), 0.0)
                         && approx(B(1,5),  1.0));

    // Row 2 (γ₁₂): B(2, 2I) = dN_I/dx₂,  B(2, 2I+1) = dN_I/dx₁
    report("SS_B_2D_gamma12", approx(B(2,0), -1.0) && approx(B(2,1), -1.0)
                           && approx(B(2,2),  0.0) && approx(B(2,3),  1.0)
                           && approx(B(2,4),  1.0) && approx(B(2,5),  0.0));
}

void test_small_strain_B_3D() {
    // 4-node tetrahedral element
    Eigen::Matrix<double, Eigen::Dynamic, 3> grad(4, 3);
    grad << -1.0, -1.0, -1.0,   // Node 1
             1.0,  0.0,  0.0,   // Node 2
             0.0,  1.0,  0.0,   // Node 3
             0.0,  0.0,  1.0;   // Node 4

    auto B = SmallStrain::compute_B_from_gradients<3>(grad, 3);

    // B should be 6×12
    report("SS_B_3D_size", B.rows() == 6 && B.cols() == 12);

    // Spot-check node 2 (I=1):  k = 1*3 = 3
    // Row 0 (ε₁₁): B(0, 3) = dN₂/dx₁ = 1.0
    report("SS_B_3D_node2_eps11", approx(B(0, 3), 1.0));
    // Row 5 (γ₁₂): B(5, 3) = dN₂/dx₂ = 0,  B(5, 4) = dN₂/dx₁ = 1
    report("SS_B_3D_node2_gamma12", approx(B(5, 3), 0.0) && approx(B(5, 4), 1.0));
}


// =============================================================================
//  3. SmallStrain evaluate consistency
// =============================================================================

void test_small_strain_evaluate_2D() {
    // Triangular element with uniform strain ε₁₁ = 0.01
    Eigen::Matrix<double, Eigen::Dynamic, 2> grad(3, 2);
    grad << -1.0, -1.0,
             1.0,  0.0,
             0.0,  1.0;

    // Displacement: u = 0.01 · x₁  →  u_x increases linearly
    //  Node 1 at (0,0): u₁ = (0, 0)
    //  Node 2 at (1,0): u₂ = (0.01, 0)
    //  Node 3 at (0,1): u₃ = (0, 0)
    Eigen::VectorXd u_e(6);
    u_e << 0.0, 0.0,  0.01, 0.0,  0.0, 0.0;

    auto gp = SmallStrain::evaluate_from_gradients<2>(grad, 2, u_e);

    // ε₁₁ = du₁/dx₁ = 0.01
    report("SS_eval_2D_eps11", approx(gp.strain_voigt(0), 0.01, 1e-12));
    // ε₂₂ = du₂/dx₂ = 0
    report("SS_eval_2D_eps22", approx(gp.strain_voigt(1), 0.0, 1e-12));
    // γ₁₂ = du₁/dx₂ + du₂/dx₁ = 0
    report("SS_eval_2D_gamma12", approx(gp.strain_voigt(2), 0.0, 1e-12));
    // F should be identity for SmallStrain
    report("SS_eval_2D_F_is_I", gp.F.approx_equal(Tensor2<2>::identity(), 1e-15));
    report("SS_eval_2D_detF_1", approx(gp.detF, 1.0, 1e-15));
}


// =============================================================================
//  4. TotalLagrangian deformation gradient
// =============================================================================

void test_TL_deformation_gradient_2D() {
    // Right triangle, gradients as above
    Eigen::Matrix<double, Eigen::Dynamic, 2> grad(3, 2);
    grad << -1.0, -1.0,
             1.0,  0.0,
             0.0,  1.0;

    // Simple shear: u_x = γ · x₂ with γ = 0.1
    //  Node 1 at (0,0): u₁ = (0, 0)
    //  Node 2 at (1,0): u₂ = (0, 0)
    //  Node 3 at (0,1): u₃ = (0.1, 0)
    Eigen::VectorXd u_e(6);
    u_e << 0.0, 0.0,  0.0, 0.0,  0.1, 0.0;

    auto F = TotalLagrangian::compute_F_from_gradients<2>(grad, u_e);

    // F = I + ∂u/∂X:
    //   F₁₁ = 1 + ∂u_x/∂X₁ = 1 + 0 = 1
    //   F₁₂ = ∂u_x/∂X₂ = γ = 0.1
    //   F₂₁ = ∂u_y/∂X₁ = 0
    //   F₂₂ = 1 + ∂u_y/∂X₂ = 1
    report("TL_F_2D_11", approx(F(0,0), 1.0, 1e-12));
    report("TL_F_2D_12", approx(F(0,1), 0.1, 1e-12));
    report("TL_F_2D_21", approx(F(1,0), 0.0, 1e-12));
    report("TL_F_2D_22", approx(F(1,1), 1.0, 1e-12));
    report("TL_F_2D_det", approx(F.determinant(), 1.0, 1e-12));
}

void test_TL_deformation_gradient_3D() {
    // Tet gradients
    Eigen::Matrix<double, Eigen::Dynamic, 3> grad(4, 3);
    grad << -1.0, -1.0, -1.0,
             1.0,  0.0,  0.0,
             0.0,  1.0,  0.0,
             0.0,  0.0,  1.0;

    // Uniaxial extension: u_x = λ·X₁,  λ = 0.05
    //  Node 1 at (0,0,0): u = (0, 0, 0)
    //  Node 2 at (1,0,0): u = (0.05, 0, 0)
    //  Node 3 at (0,1,0): u = (0, 0, 0)
    //  Node 4 at (0,0,1): u = (0, 0, 0)
    Eigen::VectorXd u_e(12);
    u_e << 0.0, 0.0, 0.0,   0.05, 0.0, 0.0,   0.0, 0.0, 0.0,   0.0, 0.0, 0.0;

    auto F = TotalLagrangian::compute_F_from_gradients<3>(grad, u_e);

    // F = diag(1.05, 1, 1)
    report("TL_F_3D_11", approx(F(0,0), 1.05, 1e-12));
    report("TL_F_3D_22", approx(F(1,1), 1.0, 1e-12));
    report("TL_F_3D_33", approx(F(2,2), 1.0, 1e-12));
    report("TL_F_3D_off_diag", approx(F(0,1), 0.0, 1e-12)
                             && approx(F(0,2), 0.0, 1e-12)
                             && approx(F(1,0), 0.0, 1e-12)
                             && approx(F(2,0), 0.0, 1e-12));
    report("TL_F_3D_det", approx(F.determinant(), 1.05, 1e-12));
}


// =============================================================================
//  5. B_NL(F=I) == SmallStrain B
// =============================================================================

void test_TL_BNL_reduces_to_linear_2D() {
    Eigen::Matrix<double, Eigen::Dynamic, 2> grad(3, 2);
    grad << -1.0, -1.0,
             1.0,  0.0,
             0.0,  1.0;

    auto B_ss  = SmallStrain::compute_B_from_gradients<2>(grad, 2);
    auto B_nl  = TotalLagrangian::compute_B_NL_from_gradients<2>(
                     grad, 2, Tensor2<2>::identity());

    double max_diff = (B_ss - B_nl).cwiseAbs().maxCoeff();
    report("TL_BNL_eq_B_linear_2D", max_diff < 1e-15);
}

void test_TL_BNL_reduces_to_linear_3D() {
    Eigen::Matrix<double, Eigen::Dynamic, 3> grad(4, 3);
    grad << -1.0, -1.0, -1.0,
             1.0,  0.0,  0.0,
             0.0,  1.0,  0.0,
             0.0,  0.0,  1.0;

    auto B_ss  = SmallStrain::compute_B_from_gradients<3>(grad, 3);
    auto B_nl  = TotalLagrangian::compute_B_NL_from_gradients<3>(
                     grad, 3, Tensor2<3>::identity());

    double max_diff = (B_ss - B_nl).cwiseAbs().maxCoeff();
    report("TL_BNL_eq_B_linear_3D", max_diff < 1e-15);
}

void test_TL_BNL_reduces_to_linear_1D() {
    Eigen::Matrix<double, Eigen::Dynamic, 1> grad(2, 1);
    grad(0, 0) = -0.5;
    grad(1, 0) =  0.5;

    auto B_ss  = SmallStrain::compute_B_from_gradients<1>(grad, 1);
    auto B_nl  = TotalLagrangian::compute_B_NL_from_gradients<1>(
                     grad, 1, Tensor2<1>::identity());

    double max_diff = (B_ss - B_nl).cwiseAbs().maxCoeff();
    report("TL_BNL_eq_B_linear_1D", max_diff < 1e-15);
}


// =============================================================================
//  6. B_NL numerical derivative verification
// =============================================================================
//
//  Verify B_NL = ∂(E_eng_voigt)/∂u using finite differences.
//

void test_TL_BNL_numerical_derivative_2D() {
    Eigen::Matrix<double, Eigen::Dynamic, 2> grad(3, 2);
    grad << -1.0, -1.0,
             1.0,  0.0,
             0.0,  1.0;

    // Random displacement
    Eigen::VectorXd u_e(6);
    for (int i = 0; i < 6; ++i) u_e(i) = rand_small();

    auto F = TotalLagrangian::compute_F_from_gradients<2>(grad, u_e);
    auto B_NL = TotalLagrangian::compute_B_NL_from_gradients<2>(grad, 2, F);

    // Numerical derivative of E_eng_voigt w.r.t. each DOF
    const double h = 1e-7;
    double max_err = 0.0;

    for (int dof = 0; dof < 6; ++dof) {
        Eigen::VectorXd u_p = u_e;
        u_p(dof) += h;
        auto F_p = TotalLagrangian::compute_F_from_gradients<2>(grad, u_p);
        auto E_p = strain::green_lagrange(F_p).voigt_engineering();

        Eigen::VectorXd u_m = u_e;
        u_m(dof) -= h;
        auto F_m = TotalLagrangian::compute_F_from_gradients<2>(grad, u_m);
        auto E_m = strain::green_lagrange(F_m).voigt_engineering();

        Eigen::Vector<double, 3> dE_num = (E_p - E_m) / (2.0 * h);

        for (int r = 0; r < 3; ++r) {
            double err = std::abs(B_NL(r, dof) - dE_num(r));
            max_err = std::max(max_err, err);
        }
    }
    report("TL_BNL_num_deriv_2D", max_err < 1e-6);
}

void test_TL_BNL_numerical_derivative_3D() {
    Eigen::Matrix<double, Eigen::Dynamic, 3> grad(4, 3);
    grad << -1.0, -1.0, -1.0,
             1.0,  0.0,  0.0,
             0.0,  1.0,  0.0,
             0.0,  0.0,  1.0;

    Eigen::VectorXd u_e(12);
    for (int i = 0; i < 12; ++i) u_e(i) = rand_small();

    auto F = TotalLagrangian::compute_F_from_gradients<3>(grad, u_e);
    auto B_NL = TotalLagrangian::compute_B_NL_from_gradients<3>(grad, 3, F);

    const double h = 1e-7;
    double max_err = 0.0;

    for (int dof = 0; dof < 12; ++dof) {
        Eigen::VectorXd u_p = u_e, u_m = u_e;
        u_p(dof) += h;
        u_m(dof) -= h;

        auto E_p = strain::green_lagrange(
            TotalLagrangian::compute_F_from_gradients<3>(grad, u_p)).voigt_engineering();
        auto E_m = strain::green_lagrange(
            TotalLagrangian::compute_F_from_gradients<3>(grad, u_m)).voigt_engineering();

        Eigen::Vector<double, 6> dE_num = (E_p - E_m) / (2.0 * h);

        for (int r = 0; r < 6; ++r) {
            double err = std::abs(B_NL(r, dof) - dE_num(r));
            max_err = std::max(max_err, err);
        }
    }
    report("TL_BNL_num_deriv_3D", max_err < 1e-6);
}

void test_TL_BNL_numerical_derivative_1D() {
    Eigen::Matrix<double, Eigen::Dynamic, 1> grad(2, 1);
    grad(0, 0) = -0.5;
    grad(1, 0) =  0.5;

    Eigen::VectorXd u_e(2);
    u_e << 0.02, 0.05;  // Specific values for 1D

    auto F = TotalLagrangian::compute_F_from_gradients<1>(grad, u_e);
    auto B_NL = TotalLagrangian::compute_B_NL_from_gradients<1>(grad, 1, F);

    const double h = 1e-7;
    double max_err = 0.0;

    for (int dof = 0; dof < 2; ++dof) {
        Eigen::VectorXd u_p = u_e, u_m = u_e;
        u_p(dof) += h;
        u_m(dof) -= h;

        auto E_p = strain::green_lagrange(
            TotalLagrangian::compute_F_from_gradients<1>(grad, u_p)).voigt_engineering();
        auto E_m = strain::green_lagrange(
            TotalLagrangian::compute_F_from_gradients<1>(grad, u_m)).voigt_engineering();

        Eigen::Vector<double, 1> dE_num = (E_p - E_m) / (2.0 * h);

        double err = std::abs(B_NL(0, dof) - dE_num(0));
        max_err = std::max(max_err, err);
    }
    report("TL_BNL_num_deriv_1D", max_err < 1e-6);
}


// =============================================================================
//  7. TotalLagrangian evaluate consistency
// =============================================================================

void test_TL_evaluate_consistency_2D() {
    Eigen::Matrix<double, Eigen::Dynamic, 2> grad(3, 2);
    grad << -1.0, -1.0,
             1.0,  0.0,
             0.0,  1.0;

    Eigen::VectorXd u_e(6);
    for (int i = 0; i < 6; ++i) u_e(i) = rand_small();

    auto gp = TotalLagrangian::evaluate_from_gradients<2>(grad, 2, u_e);

    // Verify: B_NL · u_e should equal the strain_voigt
    [[maybe_unused]] Eigen::Vector<double, 3> E_from_B = gp.B * u_e;
    Eigen::Vector<double, 3> E_direct = gp.strain_voigt;

    // They should NOT be identical in general (E is nonlinear in u).
    // However, evaluate computes E directly from green_lagrange(F),
    // while B_NL·u only equals E to first order. So we verify:
    //   evaluate uses green_lagrange, not B*u.
    auto E_expected = strain::green_lagrange(gp.F).voigt_engineering();
    double err = (E_direct - E_expected).cwiseAbs().maxCoeff();
    report("TL_eval_E_matches_GL_2D", err < 1e-14);

    // Also verify detF
    report("TL_eval_detF_2D", approx(gp.detF, gp.F.determinant(), 1e-14));
}

void test_TL_evaluate_consistency_3D() {
    Eigen::Matrix<double, Eigen::Dynamic, 3> grad(4, 3);
    grad << -1.0, -1.0, -1.0,
             1.0,  0.0,  0.0,
             0.0,  1.0,  0.0,
             0.0,  0.0,  1.0;

    Eigen::VectorXd u_e(12);
    for (int i = 0; i < 12; ++i) u_e(i) = rand_small();

    auto gp = TotalLagrangian::evaluate_from_gradients<3>(grad, 3, u_e);

    auto E_expected = strain::green_lagrange(gp.F).voigt_engineering();
    double err = (gp.strain_voigt - E_expected).cwiseAbs().maxCoeff();
    report("TL_eval_E_matches_GL_3D", err < 1e-14);
    report("TL_eval_detF_3D", approx(gp.detF, gp.F.determinant(), 1e-14));
}


// =============================================================================
//  8. Geometric stiffness tests
// =============================================================================

void test_geometric_stiffness_symmetry_2D() {
    Eigen::Matrix<double, Eigen::Dynamic, 2> grad(3, 2);
    grad << -1.0, -1.0,
             1.0,  0.0,
             0.0,  1.0;

    // Random symmetric stress
    Eigen::Matrix2d S;
    S << 100.0, 30.0,
          30.0, 50.0;

    auto K_sigma = TotalLagrangian::compute_geometric_stiffness_from_gradients<2>(
        grad, 2, S);

    // K_σ must be symmetric
    double asym = (K_sigma - K_sigma.transpose()).cwiseAbs().maxCoeff();
    report("K_sigma_symmetric_2D", asym < 1e-12);

    // K_σ should be 6×6
    report("K_sigma_size_2D", K_sigma.rows() == 6 && K_sigma.cols() == 6);
}

void test_geometric_stiffness_symmetry_3D() {
    Eigen::Matrix<double, Eigen::Dynamic, 3> grad(4, 3);
    grad << -1.0, -1.0, -1.0,
             1.0,  0.0,  0.0,
             0.0,  1.0,  0.0,
             0.0,  0.0,  1.0;

    Eigen::Matrix3d S;
    S << 200.0,  30.0, -10.0,
          30.0, 100.0,  20.0,
         -10.0,  20.0, 150.0;

    auto K_sigma = TotalLagrangian::compute_geometric_stiffness_from_gradients<3>(
        grad, 3, S);

    double asym = (K_sigma - K_sigma.transpose()).cwiseAbs().maxCoeff();
    report("K_sigma_symmetric_3D", asym < 1e-12);
    report("K_sigma_size_3D", K_sigma.rows() == 12 && K_sigma.cols() == 12);
}

void test_geometric_stiffness_block_structure_2D() {
    // K_σ[I·d + a, J·d + b] = δ_{ab} · (g_Iᵀ · S · g_J)
    // So the (I,J) block is a scalar times the identity
    Eigen::Matrix<double, Eigen::Dynamic, 2> grad(3, 2);
    grad << -1.0, -1.0,
             1.0,  0.0,
             0.0,  1.0;

    Eigen::Matrix2d S;
    S << 100.0, 30.0,
          30.0, 50.0;

    auto K_sigma = TotalLagrangian::compute_geometric_stiffness_from_gradients<2>(
        grad, 2, S);

    // For nodes I=0, J=1: block is K_sigma(0:2, 2:4)
    // Should be scalar * I₂
    double s01 = K_sigma(0, 2);  // g_0ᵀ·S·g_1
    report("K_sigma_block_structure_2D",
        approx(K_sigma(0, 2), s01) && approx(K_sigma(1, 3), s01)
     && approx(K_sigma(0, 3), 0.0, 1e-12) && approx(K_sigma(1, 2), 0.0, 1e-12));
}

void test_geometric_stiffness_zero_stress() {
    // Zero stress → zero K_σ
    Eigen::Matrix<double, Eigen::Dynamic, 2> grad(3, 2);
    grad << -1.0, -1.0,
             1.0,  0.0,
             0.0,  1.0;

    Eigen::Matrix2d S = Eigen::Matrix2d::Zero();

    auto K_sigma = TotalLagrangian::compute_geometric_stiffness_from_gradients<2>(
        grad, 2, S);

    report("K_sigma_zero_for_zero_stress", K_sigma.cwiseAbs().maxCoeff() < 1e-15);
}

void test_geometric_stiffness_numerical_2D() {
    // Verify K_σ numerically: K_σ = d/du [ B_NL(F(u))ᵀ · S_voigt ]
    // where S_voigt is held constant (initial stress stiffness).
    Eigen::Matrix<double, Eigen::Dynamic, 2> grad(3, 2);
    grad << -1.0, -1.0,
             1.0,  0.0,
             0.0,  1.0;

    Eigen::VectorXd u_e(6);
    for (int i = 0; i < 6; ++i) u_e(i) = rand_small();

    // Compute stress at current state (F used implicitly by B_NL below)
    [[maybe_unused]] auto F = TotalLagrangian::compute_F_from_gradients<2>(grad, u_e);
    Eigen::Matrix2d S;
    S << 100.0, 25.0,
          25.0, 80.0;

    // Voigt stress (tensor notation, no factor 2)
    Eigen::Vector3d S_voigt;
    S_voigt << S(0,0), S(1,1), S(0,1);

    // Analytical K_σ
    auto K_sigma_a = TotalLagrangian::compute_geometric_stiffness_from_gradients<2>(
        grad, 2, S);

    // Numerical K_σ via finite differences on B_NLᵀ·S
    const double h = 1e-7;
    Eigen::MatrixXd K_sigma_n = Eigen::MatrixXd::Zero(6, 6);

    for (int j = 0; j < 6; ++j) {
        Eigen::VectorXd u_p = u_e, u_m = u_e;
        u_p(j) += h;
        u_m(j) -= h;

        auto F_p = TotalLagrangian::compute_F_from_gradients<2>(grad, u_p);
        auto F_m = TotalLagrangian::compute_F_from_gradients<2>(grad, u_m);

        auto B_p = TotalLagrangian::compute_B_NL_from_gradients<2>(grad, 2, F_p);
        auto B_m = TotalLagrangian::compute_B_NL_from_gradients<2>(grad, 2, F_m);

        Eigen::VectorXd f_p = B_p.transpose() * S_voigt;
        Eigen::VectorXd f_m = B_m.transpose() * S_voigt;

        K_sigma_n.col(j) = (f_p - f_m) / (2.0 * h);
    }

    double max_err = (K_sigma_a - K_sigma_n).cwiseAbs().maxCoeff();
    report("K_sigma_numerical_2D", max_err < 1e-5);
}


// =============================================================================
//  9. stress_voigt_to_matrix round-trip
// =============================================================================

void test_stress_voigt_to_matrix_1D() {
    Eigen::Vector<double, 1> S_v;
    S_v << 42.0;
    auto S = TotalLagrangian::stress_voigt_to_matrix<1>(S_v);
    report("stress_voigt_matrix_1D", approx(S(0,0), 42.0));
}

void test_stress_voigt_to_matrix_2D() {
    Eigen::Vector3d S_v;
    S_v << 100.0, 50.0, 25.0;  // {S₁₁, S₂₂, S₁₂}
    auto S = TotalLagrangian::stress_voigt_to_matrix<2>(S_v);

    report("stress_voigt_matrix_2D",
        approx(S(0,0), 100.0) && approx(S(1,1), 50.0)
     && approx(S(0,1), 25.0)  && approx(S(1,0), 25.0));
}

void test_stress_voigt_to_matrix_3D() {
    Eigen::Vector<double, 6> S_v;
    S_v << 100.0, 200.0, 300.0, 23.0, 13.0, 12.0;
    auto S = TotalLagrangian::stress_voigt_to_matrix<3>(S_v);

    report("stress_voigt_matrix_3D_diag",
        approx(S(0,0), 100.0) && approx(S(1,1), 200.0) && approx(S(2,2), 300.0));
    report("stress_voigt_matrix_3D_off",
        approx(S(1,2), 23.0) && approx(S(2,1), 23.0)
     && approx(S(0,2), 13.0) && approx(S(2,0), 13.0)
     && approx(S(0,1), 12.0) && approx(S(1,0), 12.0));
}


// =============================================================================
//  10. GPKinematics struct integrity
// =============================================================================

void test_gpkinematics_struct() {
    GPKinematics<2> gp2;
    gp2.F = Tensor2<2>::identity();
    gp2.detF = 1.0;
    report("GPK_2D_N", GPKinematics<2>::N == 3);
    report("GPK_3D_N", GPKinematics<3>::N == 6);
    report("GPK_1D_N", GPKinematics<1>::N == 1);
    report("GPK_detF_default", approx(gp2.detF, 1.0));
}


// =============================================================================
//  11. CompatibleFormulation concept
// =============================================================================

void test_compatible_formulation() {
    // Continuum legacy alias
    static_assert(CompatibleFormulation<SmallStrain, SmallStrain>);
    static_assert(CompatibleFormulation<TotalLagrangian, TotalLagrangian>);
    static_assert(!CompatibleFormulation<Corotational, Corotational>);

    // Family-aware constraints are now the normative compile-time path.
    static_assert(FamilyNormativelySupportedKinematicPolicy<ElementFamilyKind::beam_1d, beam::Corotational>);
    static_assert(!FamilyNormativelySupportedKinematicPolicy<ElementFamilyKind::beam_1d, UpdatedLagrangian>);
    static_assert(FamilyNormativelySupportedKinematicPolicy<ElementFamilyKind::shell_2d, shell::SmallRotation>);
    static_assert(!FamilyNormativelySupportedKinematicPolicy<ElementFamilyKind::shell_2d, TotalLagrangian>);
    report("compatible_formulation_concept", true);
}


// =============================================================================
//  12. Manufactured-solution patch test (uniform Green-Lagrange)
// =============================================================================
//
//  A patch of elements under uniform strain should produce the same
//  Green-Lagrange strain E at every Gauss point, and B_NL·u_e should
//  approximate E to first order.
//

void test_TL_uniform_extension_3D() {
    // Single tet under uniform uniaxial extension: λ = 1.1 (10% stretch)
    // u_x = (λ - 1) · X₁ = 0.1·X₁
    Eigen::Matrix<double, Eigen::Dynamic, 3> grad(4, 3);
    grad << -1.0, -1.0, -1.0,
             1.0,  0.0,  0.0,
             0.0,  1.0,  0.0,
             0.0,  0.0,  1.0;

    // Nodes at (0,0,0), (1,0,0), (0,1,0), (0,0,1)
    // u = 0.1·X₁ in x-direction only
    Eigen::VectorXd u_e(12);
    u_e << 0.0, 0.0, 0.0,   // Node 1: X=(0,0,0)
           0.1, 0.0, 0.0,   // Node 2: X=(1,0,0)
           0.0, 0.0, 0.0,   // Node 3: X=(0,1,0)
           0.0, 0.0, 0.0;   // Node 4: X=(0,0,1)

    auto gp = TotalLagrangian::evaluate_from_gradients<3>(grad, 3, u_e);

    // F = diag(1.1, 1, 1)
    report("TL_patch_F11", approx(gp.F(0,0), 1.1, 1e-12));

    // E₁₁ = ½(F₁₁² - 1) = ½(1.21 - 1) = 0.105
    // E₂₂ = E₃₃ = 0
    report("TL_patch_E11", approx(gp.strain_voigt(0), 0.105, 1e-12));
    report("TL_patch_E22", approx(gp.strain_voigt(1), 0.0, 1e-12));
    report("TL_patch_E33", approx(gp.strain_voigt(2), 0.0, 1e-12));
    report("TL_patch_shears_zero",
        approx(gp.strain_voigt(3), 0.0, 1e-12)
     && approx(gp.strain_voigt(4), 0.0, 1e-12)
     && approx(gp.strain_voigt(5), 0.0, 1e-12));

    report("TL_patch_detF", approx(gp.detF, 1.1, 1e-12));
}

void test_TL_simple_shear_2D() {
    // Simple shear: F = [[1, γ], [0, 1]] with γ = 0.2
    // E = ½(FᵀF - I) = ½([[1, γ], [γ, 1+γ²]] - I) = [[0, γ/2], [γ/2, γ²/2]]
    Eigen::Matrix<double, Eigen::Dynamic, 2> grad(3, 2);
    grad << -1.0, -1.0,
             1.0,  0.0,
             0.0,  1.0;

    const double gamma = 0.2;
    // u_x = γ · X₂,  u_y = 0
    // Node 1 (0,0): u=(0,0),  Node 2 (1,0): u=(0,0),  Node 3 (0,1): u=(γ,0)
    Eigen::VectorXd u_e(6);
    u_e << 0.0, 0.0,  0.0, 0.0,  gamma, 0.0;

    auto gp = TotalLagrangian::evaluate_from_gradients<2>(grad, 2, u_e);

    // E₁₁ = 0, E₂₂ = γ²/2, 2E₁₂ = 2·γ/2 = γ
    report("TL_shear_E11", approx(gp.strain_voigt(0), 0.0, 1e-12));
    report("TL_shear_E22", approx(gp.strain_voigt(1), gamma*gamma/2.0, 1e-12));
    report("TL_shear_2E12", approx(gp.strain_voigt(2), gamma, 1e-12));
}


// =============================================================================
//  13. SmallStrain vs TotalLagrangian in the small deformation limit
// =============================================================================

void test_small_deformation_limit_2D() {
    // For very small displacements, TL and SS should give the same strain
    Eigen::Matrix<double, Eigen::Dynamic, 2> grad(3, 2);
    grad << -1.0, -1.0,
             1.0,  0.0,
             0.0,  1.0;

    // Very small displacement: 1e-8 scale
    Eigen::VectorXd u_e(6);
    u_e << 0.0, 0.0,  1e-8, 0.0,  0.0, 5e-9;

    auto gp_ss = SmallStrain::evaluate_from_gradients<2>(grad, 2, u_e);
    auto gp_tl = TotalLagrangian::evaluate_from_gradients<2>(grad, 2, u_e);

    // Strains should be very close (differ by O(u²))
    double max_diff = (gp_ss.strain_voigt - gp_tl.strain_voigt).cwiseAbs().maxCoeff();
    report("small_deformation_limit_2D", max_diff < 1e-14);
}

void test_small_deformation_limit_3D() {
    Eigen::Matrix<double, Eigen::Dynamic, 3> grad(4, 3);
    grad << -1.0, -1.0, -1.0,
             1.0,  0.0,  0.0,
             0.0,  1.0,  0.0,
             0.0,  0.0,  1.0;

    Eigen::VectorXd u_e(12);
    u_e << 0.0, 0.0, 0.0,  1e-8, 0.0, 0.0,  0.0, 2e-9, 0.0,  0.0, 0.0, 3e-9;

    auto gp_ss = SmallStrain::evaluate_from_gradients<3>(grad, 3, u_e);
    auto gp_tl = TotalLagrangian::evaluate_from_gradients<3>(grad, 3, u_e);

    double max_diff = (gp_ss.strain_voigt - gp_tl.strain_voigt).cwiseAbs().maxCoeff();
    report("small_deformation_limit_3D", max_diff < 1e-14);
}


// =============================================================================
//  14. Objectivity test: rigid rotation should give zero strain
// =============================================================================

void test_TL_rigid_rotation_zero_strain_2D() {
    // 45° rigid rotation applied to a triangle
    const double theta = M_PI / 4.0;
    const double c = std::cos(theta), s = std::sin(theta);

    Eigen::Matrix<double, Eigen::Dynamic, 2> grad(3, 2);
    grad << -1.0, -1.0,
             1.0,  0.0,
             0.0,  1.0;

    // Nodes at (0,0), (1,0), (0,1)
    // After rotation: new_pos = R · X
    //   Node 1 (0,0) → (0,0), u = (0,0)
    //   Node 2 (1,0) → (c, s), u = (c-1, s)
    //   Node 3 (0,1) → (-s, c), u = (-s, c-1)
    Eigen::VectorXd u_e(6);
    u_e << 0.0, 0.0,  c-1.0, s,  -s, c-1.0;

    auto gp = TotalLagrangian::evaluate_from_gradients<2>(grad, 2, u_e);

    // F should be the rotation matrix R
    report("TL_rotation_F_det1", approx(gp.detF, 1.0, 1e-10));

    // Green-Lagrange strain should be zero (E = ½(RᵀR - I) = 0)
    double max_strain = gp.strain_voigt.cwiseAbs().maxCoeff();
    report("TL_rigid_rotation_zero_E_2D", max_strain < 1e-10);
}

void test_TL_rigid_rotation_zero_strain_3D() {
    // 30° rotation about z-axis applied to a tet
    const double theta = M_PI / 6.0;
    const double c = std::cos(theta), s = std::sin(theta);

    Eigen::Matrix<double, Eigen::Dynamic, 3> grad(4, 3);
    grad << -1.0, -1.0, -1.0,
             1.0,  0.0,  0.0,
             0.0,  1.0,  0.0,
             0.0,  0.0,  1.0;

    // Nodes: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
    // R_z(θ) = [[c,-s,0],[s,c,0],[0,0,1]]
    Eigen::VectorXd u_e(12);
    u_e << 0.0, 0.0, 0.0,          // Node 1 (0,0,0) → (0,0,0)
           c-1.0, s, 0.0,          // Node 2 (1,0,0) → (c,s,0)
           -s, c-1.0, 0.0,         // Node 3 (0,1,0) → (-s,c,0)
           0.0, 0.0, 0.0;          // Node 4 (0,0,1) → (0,0,1)

    auto gp = TotalLagrangian::evaluate_from_gradients<3>(grad, 3, u_e);

    report("TL_rotation_F_det1_3D", approx(gp.detF, 1.0, 1e-10));

    double max_strain = gp.strain_voigt.cwiseAbs().maxCoeff();
    report("TL_rigid_rotation_zero_E_3D", max_strain < 1e-10);
}


// =============================================================================
//  main
// =============================================================================

int main() {
    std::cout << "\n══════════════════════════════════════════════════════\n"
              << "  KinematicPolicy Tests (Phase 2)\n"
              << "══════════════════════════════════════════════════════\n\n";

    // 1. Compile-time traits
    test_policy_traits();

    // 2. SmallStrain B matrix
    std::cout << "\n── SmallStrain B matrix ──\n";
    test_small_strain_B_1D();
    test_small_strain_B_2D();
    test_small_strain_B_3D();

    // 3. SmallStrain evaluate
    std::cout << "\n── SmallStrain evaluate ──\n";
    test_small_strain_evaluate_2D();

    // 4. TotalLagrangian deformation gradient
    std::cout << "\n── TotalLagrangian F ──\n";
    test_TL_deformation_gradient_2D();
    test_TL_deformation_gradient_3D();

    // 5. B_NL(F=I) reduction
    std::cout << "\n── B_NL(F=I) == B_linear ──\n";
    test_TL_BNL_reduces_to_linear_1D();
    test_TL_BNL_reduces_to_linear_2D();
    test_TL_BNL_reduces_to_linear_3D();

    // 6. B_NL numerical derivatives
    std::cout << "\n── B_NL numerical derivative ──\n";
    test_TL_BNL_numerical_derivative_1D();
    test_TL_BNL_numerical_derivative_2D();
    test_TL_BNL_numerical_derivative_3D();

    // 7. TL evaluate consistency
    std::cout << "\n── TL evaluate consistency ──\n";
    test_TL_evaluate_consistency_2D();
    test_TL_evaluate_consistency_3D();

    // 8. Geometric stiffness
    std::cout << "\n── Geometric stiffness K_σ ──\n";
    test_geometric_stiffness_symmetry_2D();
    test_geometric_stiffness_symmetry_3D();
    test_geometric_stiffness_block_structure_2D();
    test_geometric_stiffness_zero_stress();
    test_geometric_stiffness_numerical_2D();

    // 9. stress_voigt_to_matrix
    std::cout << "\n── stress_voigt_to_matrix ──\n";
    test_stress_voigt_to_matrix_1D();
    test_stress_voigt_to_matrix_2D();
    test_stress_voigt_to_matrix_3D();

    // 10. GPKinematics struct
    std::cout << "\n── GPKinematics struct ──\n";
    test_gpkinematics_struct();

    // 11. CompatibleFormulation
    test_compatible_formulation();

    // 12. Manufactured solutions
    std::cout << "\n── Manufactured solutions ──\n";
    test_TL_uniform_extension_3D();
    test_TL_simple_shear_2D();

    // 13. Small deformation limit
    std::cout << "\n── Small deformation limit ──\n";
    test_small_deformation_limit_2D();
    test_small_deformation_limit_3D();

    // 14. Objectivity (rigid rotation)
    std::cout << "\n── Objectivity (rigid rotation → zero strain) ──\n";
    test_TL_rigid_rotation_zero_strain_2D();
    test_TL_rigid_rotation_zero_strain_3D();

    // Summary
    std::cout << "\n══════════════════════════════════════════════════════\n"
              << "  Results: " << passed << " passed, " << failed << " failed"
              << " (total " << (passed + failed) << ")\n"
              << "══════════════════════════════════════════════════════\n\n";

    return failed > 0 ? 1 : 0;
}
