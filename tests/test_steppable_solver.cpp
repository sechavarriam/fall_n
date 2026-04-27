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
#include "src/analysis/AnalysisRouteCatalog.hh"
#include "src/analysis/ComputationalModelSliceCatalog.hh"
#include "src/analysis/ComputationalSliceMatrixCatalog.hh"
#include "src/analysis/ComputationalVariationalSliceCatalog.hh"

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
using BeamSRElem    = BeamElement<TimoshenkoBeam3D, 3, beam::SmallRotation>;
using ShellSRElem   = MITC4Shell<>;
using ShellCRElem   = CorotationalMITC4Shell<>;
using BeamCRPolicy  = SingleElementPolicy<BeamCRElem>;
using BeamSRPolicy  = SingleElementPolicy<BeamSRElem>;
using ShellSRPolicy = SingleElementPolicy<ShellSRElem>;
using ShellCRPolicy = SingleElementPolicy<ShellCRElem>;
using BeamSRModel   = Model<TimoshenkoBeam3D, beam::SmallRotation, 6, BeamSRPolicy>;
using BeamCRModel   = Model<TimoshenkoBeam3D, beam::Corotational, 6, BeamCRPolicy>;
using ShellSRModel  = Model<MindlinReissnerShell3D, shell::SmallRotation, 6, ShellSRPolicy>;
using ShellCRModel  = Model<MindlinReissnerShell3D, shell::Corotational, 6, ShellCRPolicy>;
using BeamSRLin     = LinearAnalysis<TimoshenkoBeam3D, beam::SmallRotation, 6, BeamSRPolicy>;
using BeamCRNLA     = NonlinearAnalysis<TimoshenkoBeam3D, beam::Corotational, 6, BeamCRPolicy>;
using ShellSRLin    = LinearAnalysis<MindlinReissnerShell3D, shell::SmallRotation, 6, ShellSRPolicy>;
using ShellCRNLA    = NonlinearAnalysis<MindlinReissnerShell3D, shell::Corotational, 6, ShellCRPolicy>;
static constexpr auto representative_family_formulation_analysis_route_audit_table =
    fall_n::canonical_representative_family_formulation_analysis_route_audit_table_v;
static constexpr auto representative_model_solver_slice_audit_table =
    fall_n::canonical_representative_model_solver_slice_audit_table_v;
static constexpr auto representative_computational_slice_matrix =
    fall_n::canonical_representative_computational_slice_matrix_v;
static constexpr auto representative_computational_variational_slice_matrix =
    fall_n::canonical_representative_computational_variational_slice_matrix_v;

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

struct ArcFixture {
    Domain<DIM>                               domain;
    ContinuumIsotropicElasticMaterial         mat_site{200.0, 0.3};
    Material<Policy>                          mat{mat_site, ElasticUpdate{}};
    std::unique_ptr<ModelTL>                  model;

    ArcFixture() {
        create_unit_cube(domain);
        model = std::make_unique<ModelTL>(domain, mat);
        model->fix_x(0.0);
        model->setup();

        const double f_per_node = 1.0 / 4.0;
        for (std::size_t id : {1ul, 3ul, 5ul, 7ul})
            model->apply_node_force(id, f_per_node, 0.0, 0.0);
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
    static_assert(representative_family_formulation_analysis_route_audit_table.size() == 9);
    static_assert(fall_n::canonical_representative_analysis_route_support_count_v<
                      fall_n::AnalysisRouteSupportLevel::reference_baseline> == 2);
    static_assert(fall_n::canonical_representative_analysis_route_support_count_v<
                      fall_n::AnalysisRouteSupportLevel::implemented> == 2);
    static_assert(fall_n::canonical_representative_analysis_route_support_count_v<
                      fall_n::AnalysisRouteSupportLevel::partial> == 4);
    static_assert(fall_n::canonical_representative_analysis_route_support_count_v<
                      fall_n::AnalysisRouteSupportLevel::interface_declared> == 1);
    static_assert(fall_n::canonical_representative_analysis_route_support_count_v<
                      fall_n::AnalysisRouteSupportLevel::unavailable> == 0);
    static_assert(
        fall_n::canonical_representative_analysis_routes_requiring_scope_disclaimer_v ==
        5);
    static_assert(representative_family_formulation_analysis_route_audit_table[1]
                      .audit_scope.is_reference_route_for_scope());
    static_assert(representative_family_formulation_analysis_route_audit_table[2]
                      .audit_scope.requires_scope_disclaimer());
    static_assert(representative_family_formulation_analysis_route_audit_table[3]
                      .audit_scope.requires_scope_disclaimer());
    static_assert(representative_family_formulation_analysis_route_audit_table[6]
                      .audit_scope.requires_scope_disclaimer());
    static_assert(representative_family_formulation_analysis_route_audit_table[8]
                      .audit_scope.requires_scope_disclaimer());
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
    static_assert(fall_n::canonical_model_solver_slice_support_level_v<ModelT, LinA> ==
                  fall_n::ComputationalModelSliceSupportLevel::reference_linear);
    static_assert(fall_n::NormativelySupportedModelSolverSlice<ModelTL, NLA_TL>);
    static_assert(fall_n::ReferenceGeometricNonlinearityModelSolverSlice<ModelTL, NLA_TL>);
    static_assert(fall_n::canonical_model_solver_slice_support_level_v<ModelTL, NLA_TL> ==
                  fall_n::ComputationalModelSliceSupportLevel::reference_geometric_nonlinearity);
    static_assert(fall_n::NormativelySupportedModelSolverSlice<ModelUL, NLA_UL>);
    static_assert(!fall_n::ReferenceGeometricNonlinearityModelSolverSlice<ModelUL, NLA_UL>);
    static_assert(fall_n::canonical_model_solver_slice_support_level_v<ModelUL, NLA_UL> ==
                  fall_n::ComputationalModelSliceSupportLevel::normative);
    static_assert(!fall_n::NormativelySupportedModelSolverSlice<ModelTL, DynTL>);
    static_assert(fall_n::canonical_model_solver_slice_support_level_v<ModelTL, DynTL> ==
                  fall_n::ComputationalModelSliceSupportLevel::unsupported_or_disclaimed);
    static_assert(fall_n::canonical_model_solver_slice_audit_scope_v<ModelTL, DynTL>.requires_scope_disclaimer());
    static_assert(!fall_n::NormativelySupportedModelSolverSlice<ModelTL, ArcTL>);
    static_assert(fall_n::canonical_model_solver_slice_support_level_v<ModelTL, ArcTL> ==
                  fall_n::ComputationalModelSliceSupportLevel::unsupported_or_disclaimed);
    static_assert(fall_n::canonical_model_solver_slice_audit_scope_v<ModelTL, ArcTL>.requires_scope_disclaimer());
    static_assert(fall_n::NormativelySupportedModelSolverSlice<BeamSRModel, BeamSRLin>);
    static_assert(fall_n::ReferenceLinearModelSolverSlice<BeamSRModel, BeamSRLin>);
    static_assert(fall_n::canonical_model_solver_slice_support_level_v<BeamSRModel, BeamSRLin> ==
                  fall_n::ComputationalModelSliceSupportLevel::reference_linear);
    static_assert(!fall_n::NormativelySupportedModelSolverSlice<BeamCRModel, BeamCRNLA>);
    static_assert(fall_n::canonical_model_solver_slice_support_level_v<BeamCRModel, BeamCRNLA> ==
                  fall_n::ComputationalModelSliceSupportLevel::unsupported_or_disclaimed);
    static_assert(fall_n::canonical_model_solver_slice_audit_scope_v<BeamCRModel, BeamCRNLA>.requires_scope_disclaimer());
    static_assert(fall_n::NormativelySupportedModelSolverSlice<ShellSRModel, ShellSRLin>);
    static_assert(fall_n::ReferenceLinearModelSolverSlice<ShellSRModel, ShellSRLin>);
    static_assert(fall_n::canonical_model_solver_slice_support_level_v<ShellSRModel, ShellSRLin> ==
                  fall_n::ComputationalModelSliceSupportLevel::reference_linear);
    static_assert(!fall_n::NormativelySupportedModelSolverSlice<ShellCRModel, ShellCRNLA>);
    static_assert(fall_n::canonical_model_solver_slice_support_level_v<ShellCRModel, ShellCRNLA> ==
                  fall_n::ComputationalModelSliceSupportLevel::unsupported_or_disclaimed);
    static_assert(fall_n::canonical_model_solver_slice_audit_scope_v<ShellCRModel, ShellCRNLA>.requires_scope_disclaimer());
    static_assert(representative_model_solver_slice_audit_table.size() == 9);
    static_assert(fall_n::canonical_representative_model_solver_slice_support_count_v<
                      fall_n::ComputationalModelSliceSupportLevel::reference_linear> == 3);
    static_assert(fall_n::canonical_representative_model_solver_slice_support_count_v<
                      fall_n::ComputationalModelSliceSupportLevel::reference_geometric_nonlinearity> == 1);
    static_assert(fall_n::canonical_representative_model_solver_slice_support_count_v<
                      fall_n::ComputationalModelSliceSupportLevel::normative> == 1);
    static_assert(fall_n::canonical_representative_model_solver_slice_support_count_v<
                      fall_n::ComputationalModelSliceSupportLevel::unsupported_or_disclaimed> == 4);
    static_assert(fall_n::canonical_representative_model_solver_slices_requiring_scope_disclaimer_v == 5);
    static_assert(representative_model_solver_slice_audit_table[1].support_level() ==
                  fall_n::ComputationalModelSliceSupportLevel::reference_geometric_nonlinearity);
    static_assert(representative_model_solver_slice_audit_table[2].audit_scope.requires_scope_disclaimer());
    static_assert(representative_model_solver_slice_audit_table[3].audit_scope.requires_scope_disclaimer());
    static_assert(representative_model_solver_slice_audit_table[8].audit_scope.requires_scope_disclaimer());
    static_assert(representative_computational_slice_matrix.size() == 9);
    static_assert(fall_n::canonical_representative_computational_slice_matrix_support_count_v<
                      fall_n::ComputationalModelSliceSupportLevel::reference_linear> == 3);
    static_assert(fall_n::canonical_representative_computational_slice_matrix_support_count_v<
                      fall_n::ComputationalModelSliceSupportLevel::reference_geometric_nonlinearity> == 1);
    static_assert(fall_n::canonical_representative_computational_slice_matrix_support_count_v<
                      fall_n::ComputationalModelSliceSupportLevel::normative> == 1);
    static_assert(fall_n::canonical_representative_computational_slice_matrix_support_count_v<
                      fall_n::ComputationalModelSliceSupportLevel::unsupported_or_disclaimed> == 4);
    static_assert(
        fall_n::canonical_representative_computational_slice_matrix_scope_disclaimer_count_v ==
        5);
    static_assert(representative_computational_slice_matrix[1].family_support_level() ==
                  continuum::FamilyFormulationSupportLevel::reference_baseline);
    static_assert(representative_computational_slice_matrix[1].route_support_level() ==
                  fall_n::AnalysisRouteSupportLevel::reference_baseline);
    static_assert(representative_computational_slice_matrix[1].slice_support_level() ==
                  fall_n::ComputationalModelSliceSupportLevel::reference_geometric_nonlinearity);
    static_assert(representative_computational_slice_matrix[3].route_support_level() ==
                  fall_n::AnalysisRouteSupportLevel::interface_declared);
    static_assert(representative_computational_slice_matrix[3].requires_scope_disclaimer());
    static_assert(representative_computational_slice_matrix[6].family_support_level() ==
                  continuum::FamilyFormulationSupportLevel::reference_baseline);
    static_assert(representative_computational_slice_matrix[6].requires_scope_disclaimer());
    check(true, "model and solver types compose into audited computational slices");

    static_assert(representative_computational_variational_slice_matrix.size() == 9);
    static_assert(
        fall_n::canonical_representative_computational_variational_slice_scope_disclaimer_count_v ==
        5);
    static_assert(
        fall_n::canonical_representative_structural_reduction_variational_slice_count_v == 4);
    static_assert(
        fall_n::canonical_representative_effective_operator_predictor_variational_slice_count_v ==
        1);
    static_assert(representative_computational_variational_slice_matrix[1]
                      .audit_scope.has_normative_variational_slice());
    static_assert(representative_computational_variational_slice_matrix[1]
                      .audit_scope.global_residual_operator ==
                  fall_n::GlobalResidualOperatorKind::incremental_static_equilibrium);
    static_assert(representative_computational_variational_slice_matrix[1]
                      .audit_scope.global_tangent_operator ==
                  fall_n::GlobalTangentOperatorKind::monolithic_incremental_consistent_tangent);
    static_assert(representative_computational_variational_slice_matrix[1]
                      .audit_scope.incremental_state_management ==
                  fall_n::IncrementalStateManagementKind::checkpointable_converged_step_commit);
    static_assert(representative_computational_variational_slice_matrix[1]
                      .audit_scope.integrates_on_reference_like_domain());
    static_assert(representative_computational_variational_slice_matrix[2]
                      .audit_scope.global_residual_operator ==
                  fall_n::GlobalResidualOperatorKind::incremental_static_equilibrium);
    static_assert(representative_computational_variational_slice_matrix[2]
                      .audit_scope.discrete_variational_semantics.integrates_on_current_like_domain());
    static_assert(representative_computational_variational_slice_matrix[2]
                      .requires_scope_disclaimer());
    static_assert(representative_computational_variational_slice_matrix[3]
                      .audit_scope.global_residual_operator ==
                  fall_n::GlobalResidualOperatorKind::second_order_dynamic_equilibrium);
    static_assert(representative_computational_variational_slice_matrix[3]
                      .audit_scope.global_tangent_operator ==
                  fall_n::GlobalTangentOperatorKind::effective_mass_damping_stiffness_tangent);
    static_assert(representative_computational_variational_slice_matrix[3]
                      .audit_scope.augments_with_inertial_terms);
    static_assert(representative_computational_variational_slice_matrix[3]
                      .requires_scope_disclaimer());
    static_assert(representative_computational_variational_slice_matrix[4]
                      .audit_scope.global_residual_operator ==
                  fall_n::GlobalResidualOperatorKind::arc_length_augmented_equilibrium);
    static_assert(representative_computational_variational_slice_matrix[4]
                      .audit_scope.global_tangent_operator ==
                  fall_n::GlobalTangentOperatorKind::bordered_arc_length_tangent);
    static_assert(representative_computational_variational_slice_matrix[4]
                      .audit_scope.augments_with_continuation_constraint);
    static_assert(representative_computational_variational_slice_matrix[5]
                      .audit_scope.is_structural_reduction_path());
    static_assert(representative_computational_variational_slice_matrix[5]
                      .audit_scope.integrates_on_reference_like_domain());
    static_assert(representative_computational_variational_slice_matrix[6]
                      .audit_scope.is_structural_reduction_path());
    static_assert(representative_computational_variational_slice_matrix[6]
                      .audit_scope.admits_effective_operator_predictor_injection);
    static_assert(representative_computational_variational_slice_matrix[7]
                      .audit_scope.discrete_variational_semantics.stress_carrier ==
                  continuum::DiscreteStressCarrierKind::shell_section_resultants);
    static_assert(representative_computational_variational_slice_matrix[8]
                      .audit_scope.discrete_variational_semantics.integration_domain ==
                  continuum::DiscreteIntegrationDomainKind::corotated_surface);
    check(true, "computational slices now expose audited discrete variational semantics");
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

void test_nl_step_to_clamps_requested_target() {
    std::cout << "\nTest 7b: NL step_to clamps requested target under large dp\n";

    NLFixture f;
    NLA nl{f.model.get()};

    nl.begin_incremental(10);  // nominal dp = 0.1
    nl.set_increment_size(0.25);

    const auto verdict = nl.step_to(0.3);
    check(verdict == fall_n::StepVerdict::Continue,
          "step_to(0.3) returned Continue under oversized dp");
    check(std::abs(nl.current_time() - 0.3) < 1e-12,
          "step_to(0.3) does not overshoot the requested target");
    check(nl.current_step() == 2,
          "step_to(0.3) advances in as many logical steps as needed");
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
//  Test 11: Arc-length continuation hook smoke test
// =============================================================================

void test_arc_length_hook_smoke() {
    std::cout << "\nTest 11: ArcLengthSolver hook smoke test\n";

    ArcFixture f;
    ArcTL arc{f.model.get()};
    arc.set_arc_length(1.0e-3);
    arc.set_psi(1.0);
    arc.set_max_iterations(20);
    arc.set_tolerances(1.0e-8, 1.0e-10);

    int local_residual_calls = 0;
    int global_residual_calls = 0;
    int jacobian_calls = 0;

    arc.set_residual_hook(
        [&](Vec, Vec, DM) { ++local_residual_calls; });
    arc.set_global_residual_hook(
        [&](Vec, Vec, DM) { ++global_residual_calls; });
    arc.set_jacobian_hook(
        [&](Vec, Mat, DM) { ++jacobian_calls; });

    const auto result = arc.solve_step();
    check(result.converged, "arc-length step converged on elastic TL cube");
    check(std::abs(result.lambda) > 0.0,
          "arc-length step advances the load parameter");
    check(local_residual_calls > 0,
          "arc-length local residual hook is executed");
    check(global_residual_calls > 0,
          "arc-length global residual hook is executed");
    check(jacobian_calls > 0,
          "arc-length Jacobian hook is executed");
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
    test_nl_step_to_clamps_requested_target();
    test_nl_equivalence();
    test_nl_analysis_state();
    test_dynamic_analysis_state();
    test_arc_length_hook_smoke();

    std::cout << "\n" << std::string(55, '=') << "\n"
              << "  Summary: " << passed << " passed, " << failed << " failed\n"
              << std::string(55, '=') << "\n";

    PetscFinalize();
    return (failed > 0) ? 1 : 0;
}
