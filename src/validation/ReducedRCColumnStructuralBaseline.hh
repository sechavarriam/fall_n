#ifndef FALL_N_REDUCED_RC_COLUMN_STRUCTURAL_BASELINE_HH
#define FALL_N_REDUCED_RC_COLUMN_STRUCTURAL_BASELINE_HH

// =============================================================================
//  ReducedRCColumnStructuralBaseline.hh
// =============================================================================
//
//  Clean runtime entry point for the Phase-3 reduced structural column
//  baseline. This API intentionally lives outside the legacy cyclic-validation
//  driver so the validation reboot can use an auditable, modular surface.
//
//  Current scope:
//    - TimoshenkoBeamN<N>
//    - compile-time beam-axis quadrature family (Gauss / Lobatto / Radau)
//    - small-strain beam formulation
//    - lateral displacement control with optional axial compression force
//    - optional equilibrated axial-preload stage held constant during the
//      lateral branch
//
//  Future scope:
//    - corotational TimoshenkoBeamN family
//    - finite-kinematics beam families if/when they become real runtime paths
//
// =============================================================================

#include "src/numerics/numerical_integration/BeamAxisQuadrature.hh"
#include "src/analysis/LocalModelTaxonomy.hh"
#include "src/validation/TableCyclicValidationAPI.hh"
#include "src/validation/ReducedRCColumnReferenceSpec.hh"
#include "src/validation/ReducedRCColumnSolveControl.hh"

#include <cstddef>
#include <string>
#include <vector>

namespace fall_n::validation_reboot {

enum class ReducedRCColumnStructuralMaterialMode {
    nonlinear,
    elasticized
};

[[nodiscard]] constexpr const char*
to_string(ReducedRCColumnStructuralMaterialMode mode) noexcept
{
    switch (mode) {
        case ReducedRCColumnStructuralMaterialMode::nonlinear:
            return "nonlinear";
        case ReducedRCColumnStructuralMaterialMode::elasticized:
            return "elasticized";
    }
    return "unknown_reduced_rc_column_structural_material_mode";
}

struct ReducedRCColumnStructuralRunSpec {
    ReducedRCColumnStructuralMaterialMode material_mode{
        ReducedRCColumnStructuralMaterialMode::nonlinear};
    std::size_t beam_nodes{3};
    std::size_t structural_element_count{1};
    BeamAxisQuadratureFamily beam_axis_quadrature_family{
        BeamAxisQuadratureFamily::GaussLegendre};
    bool clamp_top_bending_rotation{false};
    bool prescribe_top_bending_rotation_from_drift{false};
    double top_bending_rotation_drift_ratio{0.0};
    double axial_compression_force_mn{0.0};
    bool use_equilibrated_axial_preload_stage{true};
    int axial_preload_steps{4};
    ReducedRCColumnContinuationKind continuation_kind{
        ReducedRCColumnContinuationKind::
            monolithic_incremental_displacement_control};
    ReducedRCColumnSolverPolicyKind solver_policy_kind{
        ReducedRCColumnSolverPolicyKind::canonical_newton_profile_cascade};
    int continuation_segment_substep_factor{1};
    bool write_hysteresis_csv{true};
    bool write_section_response_csv{true};
    bool write_section_fiber_history_csv{true};
    bool write_element_tangent_audit_csv{false};
    bool print_progress{true};
    ReducedRCColumnReferenceSpec reference_spec{};

    [[nodiscard]] bool has_axial_compression() const noexcept
    {
        return axial_compression_force_mn > 0.0;
    }

    [[nodiscard]] bool prescribes_top_bending_rotation() const noexcept
    {
        return clamp_top_bending_rotation ||
               prescribe_top_bending_rotation_from_drift;
    }

    [[nodiscard]] bool uses_equilibrated_axial_preload_stage() const noexcept
    {
        return has_axial_compression() &&
               use_equilibrated_axial_preload_stage &&
               axial_preload_steps > 0;
    }

    [[nodiscard]] bool uses_segmented_continuation() const noexcept
    {
        return continuation_kind ==
                   ReducedRCColumnContinuationKind::
                       segmented_incremental_displacement_control ||
               continuation_kind ==
                   ReducedRCColumnContinuationKind::
                       reversal_guarded_incremental_displacement_control;
    }
};

[[nodiscard]] inline fall_n::LocalModelTaxonomy
describe_reduced_rc_column_structural_local_model(
    const ReducedRCColumnStructuralRunSpec&) noexcept
{
    return {
        .discretization_kind =
            fall_n::LocalModelDiscretizationKind::structural_section_surrogate,
        .fracture_representation_kind =
            fall_n::LocalFractureRepresentationKind::none,
        .reinforcement_representation_kind =
            fall_n::LocalReinforcementRepresentationKind::
                constitutive_section_fibers,
        .maturity_kind =
            fall_n::LocalModelMaturityKind::promoted_baseline,
        .supports_discrete_crack_geometry = false,
        .requires_enriched_dofs = false,
        .requires_skeleton_trace_unknowns = false,
        .suitable_for_future_multiscale_local_model = false,
        .notes =
            "Reduced-column structural surrogate based on TimoshenkoBeamN and "
            "fiber-section constitutive response."};
}

struct ReducedRCColumnSectionResponseRecord {
    int step{0};
    double p{0.0};
    double drift{0.0};
    std::size_t section_gp{0};
    double xi{0.0};
    double axial_strain{0.0};
    double curvature_y{0.0};
    double curvature_z{0.0};
    double axial_force{0.0};
    double moment_y{0.0};
    double moment_z{0.0};
    // Axial tangent is left raw. The benchmark keeps both:
    //   1. the direct bending entry Kyy/Kzz,
    //   2. the axial-force-condensed effective tangent dM/dkappa|N=const.
    //
    // This is important because the current fall_n vs OpenSees discrepancy is
    // now localized mostly in the axial-flexural coupling and in the
    // corresponding Schur-complement reduction, not in the gross moment or
    // curvature histories.
    double tangent_ea{0.0};
    double tangent_eiy{0.0};
    double tangent_eiz{0.0};
    double tangent_eiy_direct_raw{0.0};
    double tangent_eiz_direct_raw{0.0};
    double raw_tangent_k00{0.0};
    double raw_tangent_k0y{0.0};
    double raw_tangent_ky0{0.0};
    double raw_tangent_kyy{0.0};
    double fd_tangent_k00{0.0};
    double fd_tangent_k0y{0.0};
    double fd_tangent_ky0{0.0};
    double fd_tangent_kyy{0.0};
    double rel_error_k00{0.0};
    double rel_error_k0y{0.0};
    double rel_error_ky0{0.0};
    double rel_error_kyy{0.0};
};

struct ReducedRCColumnStructuralControlStateRecord {
    int runtime_step{0};
    int step{0};
    double p{0.0};
    double runtime_p{0.0};
    double target_drift{0.0};
    double actual_tip_lateral_displacement{0.0};
    double actual_tip_lateral_total_state_displacement{0.0};
    double imposed_vs_total_state_tip_displacement_error{0.0};
    double prescribed_top_bending_rotation{0.0};
    double actual_top_bending_rotation{0.0};
    double imposed_vs_total_state_top_bending_rotation_error{0.0};
    double top_axial_displacement{0.0};
    double base_shear{0.0};
    double base_axial_reaction{0.0};
    bool preload_equilibrated{false};
    int target_increment_direction{0};
    int actual_increment_direction{0};
    int protocol_branch_id{0};
    int reversal_index{0};
    int branch_step_index{0};
    int accepted_substep_count{0};
    int max_bisection_level{0};
    double newton_iterations{0.0};
    double newton_iterations_per_substep{0.0};
    int solver_profile_attempt_count{0};
    std::string solver_profile_label{};
    std::string solver_snes_type{};
    std::string solver_linesearch_type{};
    std::string solver_ksp_type{};
    std::string solver_pc_type{};
    int last_snes_reason{0};
    double last_function_norm{0.0};
    bool accepted_by_small_residual_policy{false};
    double accepted_function_norm_threshold{0.0};
    bool converged{true};
};

struct ReducedRCColumnStructuralSectionFiberRecord {
    int step{0};
    double p{0.0};
    double drift{0.0};
    std::size_t section_gp{0};
    double xi{0.0};
    double axial_strain{0.0};
    double curvature_y{0.0};
    bool zero_curvature_anchor{false};
    std::size_t fiber_index{0};
    double y{0.0};
    double z{0.0};
    double area{0.0};
    RCSectionZoneKind zone{RCSectionZoneKind::cover_top};
    RCSectionMaterialRole material_role{
        RCSectionMaterialRole::unconfined_concrete};
    double strain_xx{0.0};
    double stress_xx{0.0};
    double tangent_xx{0.0};
    double axial_force_contribution{0.0};
    double moment_y_contribution{0.0};
    double raw_tangent_k00_contribution{0.0};
    double raw_tangent_k0y_contribution{0.0};
    double raw_tangent_kyy_contribution{0.0};
    int history_state_code{0};
    double history_min_strain{0.0};
    double history_min_stress{0.0};
    double history_closure_strain{0.0};
    double history_max_tensile_strain{0.0};
    double history_max_tensile_stress{0.0};
    double history_committed_strain{0.0};
    double history_committed_stress{0.0};
    bool history_cracked{false};
};

struct ReducedRCColumnStructuralElementTangentAuditRecord {
    int runtime_step{0};
    int step{0};
    double p{0.0};
    double runtime_p{0.0};
    double drift{0.0};
    bool failed_attempt{false};
    double fd_step_reference{0.0};
    double displacement_inf_norm{0.0};
    double internal_force_norm{0.0};
    double tangent_frobenius_norm{0.0};
    double fd_tangent_frobenius_norm{0.0};
    double tangent_fd_rel_error{0.0};
    double max_column_rel_error{0.0};
    int worst_column_index{0};
    int worst_column_node{0};
    int worst_column_local_dof{0};
    double worst_column_tangent_norm{0.0};
    double worst_column_fd_norm{0.0};
    double top_control_column_rel_error{0.0};
    double top_bending_rotation_column_rel_error{0.0};
};

struct ReducedRCColumnStructuralSectionTangentAuditRecord {
    int runtime_step{0};
    int step{0};
    double p{0.0};
    double runtime_p{0.0};
    double drift{0.0};
    bool failed_attempt{false};
    std::size_t section_gp{0};
    double xi{0.0};
    double axial_strain{0.0};
    double curvature_y{0.0};
    double curvature_z{0.0};
    double fd_step_reference{0.0};
    double tangent_frobenius_norm{0.0};
    double fd_tangent_frobenius_norm{0.0};
    double tangent_fd_rel_error{0.0};
    double max_column_rel_error{0.0};
    int worst_column_index{0};
    double axial_column_rel_error{0.0};
    double curvature_y_column_rel_error{0.0};
    double shear_z_column_rel_error{0.0};
    double raw_tangent_k00{0.0};
    double raw_tangent_k0y{0.0};
    double raw_tangent_ky0{0.0};
    double raw_tangent_kyy{0.0};
    double fd_tangent_k00{0.0};
    double fd_tangent_k0y{0.0};
    double fd_tangent_ky0{0.0};
    double fd_tangent_kyy{0.0};
    double rel_error_k00{0.0};
    double rel_error_k0y{0.0};
    double rel_error_ky0{0.0};
    double rel_error_kyy{0.0};
};

struct ReducedRCColumnStructuralTimingSummary {
    double total_wall_seconds{0.0};
    double solve_wall_seconds{0.0};
    double output_write_wall_seconds{0.0};
};

struct ReducedRCColumnStructuralRunResult {
    std::vector<table_cyclic_validation::StepRecord> hysteresis_records{};
    std::vector<ReducedRCColumnSectionResponseRecord> section_response_records{};
    std::vector<ReducedRCColumnStructuralControlStateRecord> control_state_records{};
    std::vector<ReducedRCColumnStructuralSectionFiberRecord> fiber_history_records{};
    std::vector<ReducedRCColumnStructuralElementTangentAuditRecord>
        element_tangent_audit_records{};
    std::vector<ReducedRCColumnStructuralSectionTangentAuditRecord>
        section_tangent_audit_records{};
    ReducedRCColumnStructuralTimingSummary timing{};
    bool completed_successfully{false};
    bool has_failed_attempt_control_state{false};
    ReducedRCColumnStructuralControlStateRecord failed_attempt_control_state{};
    bool has_failed_attempt_element_tangent_audit{false};
    ReducedRCColumnStructuralElementTangentAuditRecord
        failed_attempt_element_tangent_audit{};
    bool has_failed_attempt_section_tangent_audit{false};
    std::vector<ReducedRCColumnStructuralSectionTangentAuditRecord>
        failed_attempt_section_tangent_audit_records{};
    std::vector<ReducedRCColumnSectionResponseRecord>
        failed_attempt_section_response_records{};
    std::vector<ReducedRCColumnStructuralSectionFiberRecord>
        failed_attempt_fiber_history_records{};

    [[nodiscard]] bool has_section_response_observable() const noexcept
    {
        return !section_response_records.empty();
    }
};

[[nodiscard]] ReducedRCColumnStructuralRunResult
run_reduced_rc_column_small_strain_beam_case_result(
    const ReducedRCColumnStructuralRunSpec& spec,
    const std::string& out_dir,
    const table_cyclic_validation::CyclicValidationRunConfig& cfg);

[[nodiscard]] std::vector<table_cyclic_validation::StepRecord>
run_reduced_rc_column_small_strain_beam_case(
    const ReducedRCColumnStructuralRunSpec& spec,
    const std::string& out_dir,
    const table_cyclic_validation::CyclicValidationRunConfig& cfg);

} // namespace fall_n::validation_reboot

#endif // FALL_N_REDUCED_RC_COLUMN_STRUCTURAL_BASELINE_HH
