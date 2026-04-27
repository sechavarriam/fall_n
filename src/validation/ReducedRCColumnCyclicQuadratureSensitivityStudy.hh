#ifndef FALL_N_REDUCED_RC_COLUMN_CYCLIC_QUADRATURE_SENSITIVITY_STUDY_HH
#define FALL_N_REDUCED_RC_COLUMN_CYCLIC_QUADRATURE_SENSITIVITY_STUDY_HH

// =============================================================================
//  ReducedRCColumnCyclicQuadratureSensitivityStudy.hh
// =============================================================================
//
//  Cyclic quadrature-family sensitivity study for the reduced
//  reinforced-concrete column reboot.
//
//  The monotonic quadrature study already answers how the base-side
//  moment-curvature observable changes across Gauss / Lobatto / Radau station
//  placement when the response is read on the positive branch. The cyclic
//  frontier is stricter:
//
//      "Once unloading and reloading are present, and once the continuation
//       policy is declared explicitly, does the same family-to-family spread
//       remain bounded at fixed beam-node count?"
//
//  This artifact keeps that question separate from the monotonic family-spread
//  bundle and from the continuation-sensitivity study. The comparison remains
//  internal: each quadrature family is contrasted against a declared reference
//  family at the same beam-node count, under the same displacement-driven
//  cyclic protocol and the same declared continuation policy.
//
// =============================================================================

#include "src/validation/ReducedRCColumnStructuralBaseline.hh"
#include "src/validation/ReducedRCColumnStructuralMatrixCatalog.hh"

#include <cstddef>
#include <string>
#include <vector>

namespace fall_n::validation_reboot {

struct ReducedRCColumnCyclicQuadratureSensitivityRunSpec {
    ReducedRCColumnStructuralRunSpec structural_spec{};
    table_cyclic_validation::CyclicValidationRunConfig structural_protocol{};
    std::vector<std::size_t> beam_nodes_filter{};
    std::vector<BeamAxisQuadratureFamily> quadrature_filter{};
    BeamAxisQuadratureFamily reference_family{
        BeamAxisQuadratureFamily::GaussLegendre};

    bool include_only_phase3_runtime_baseline{true};
    bool write_case_outputs{true};
    bool write_csv{true};
    bool print_progress{true};

    double relative_error_floor{1.0e-12};
    double protocol_alignment_tolerance{1.0e-12};
    double secant_activation_curvature_tolerance{1.0e-10};
    double representative_terminal_return_relative_spread_tolerance{0.10};
    double representative_max_rel_moment_history_spread_tolerance{0.15};
    double representative_max_rel_tangent_history_spread_tolerance{0.25};
    double representative_max_rel_secant_history_spread_tolerance{0.15};
    double representative_max_rel_turning_point_moment_spread_tolerance{0.12};
    double representative_max_rel_axial_force_history_spread_tolerance{0.10};
};

struct ReducedRCColumnCyclicQuadratureSensitivityCaseRow {
    std::size_t beam_nodes{0};
    BeamAxisQuadratureFamily beam_axis_quadrature_family{
        BeamAxisQuadratureFamily::GaussLegendre};
    continuum::FormulationKind formulation_kind{
        continuum::FormulationKind::small_strain};
    ReducedRCColumnContinuationKind continuation_kind{
        ReducedRCColumnContinuationKind::
            monolithic_incremental_displacement_control};
    int continuation_segment_substep_factor{1};
    std::string case_id{};
    std::string reference_case_id{};
    std::string case_out_dir{};
    std::string scope_label{};
    std::string rationale_label{};

    bool execution_ok{false};
    std::string error_message{};

    std::size_t history_point_count{0};
    std::size_t turning_point_count{0};
    double controlling_station_xi{0.0};
    double reference_controlling_station_xi{0.0};
    double abs_station_xi_shift{0.0};
    double terminal_return_moment_y{0.0};
    double reference_terminal_return_moment_y{0.0};
    double rel_terminal_return_moment_spread{0.0};
    double max_rel_moment_history_spread{0.0};
    double rms_rel_moment_history_spread{0.0};
    double max_rel_tangent_history_spread{0.0};
    double max_rel_secant_history_spread{0.0};
    double max_rel_turning_point_moment_spread{0.0};
    double max_rel_axial_force_history_spread{0.0};

    bool terminal_return_within_representative_tolerance{false};
    bool moment_history_within_representative_tolerance{false};
    bool tangent_history_within_representative_tolerance{false};
    bool secant_history_within_representative_tolerance{false};
    bool turning_point_within_representative_tolerance{false};
    bool axial_force_history_within_representative_tolerance{false};

    [[nodiscard]] bool representative_internal_cyclic_family_spread_passes()
        const noexcept
    {
        return terminal_return_within_representative_tolerance &&
               moment_history_within_representative_tolerance &&
               tangent_history_within_representative_tolerance &&
               turning_point_within_representative_tolerance &&
               axial_force_history_within_representative_tolerance;
    }
};

struct ReducedRCColumnCyclicQuadratureSensitivityNodeRow {
    std::size_t beam_nodes{0};
    std::size_t case_count{0};
    std::size_t completed_case_count{0};
    std::size_t representative_pass_count{0};
    double min_rel_terminal_return_moment_spread{0.0};
    double max_rel_terminal_return_moment_spread{0.0};
    double avg_rel_terminal_return_moment_spread{0.0};
    double min_max_rel_moment_history_spread{0.0};
    double max_max_rel_moment_history_spread{0.0};
    double avg_max_rel_moment_history_spread{0.0};
    double min_max_rel_tangent_history_spread{0.0};
    double max_max_rel_tangent_history_spread{0.0};
    double avg_max_rel_tangent_history_spread{0.0};
    double min_max_rel_secant_history_spread{0.0};
    double max_max_rel_secant_history_spread{0.0};
    double avg_max_rel_secant_history_spread{0.0};
    double min_max_rel_turning_point_moment_spread{0.0};
    double max_max_rel_turning_point_moment_spread{0.0};
    double avg_max_rel_turning_point_moment_spread{0.0};
    double min_max_rel_axial_force_history_spread{0.0};
    double max_max_rel_axial_force_history_spread{0.0};
    double avg_max_rel_axial_force_history_spread{0.0};
    double max_abs_station_xi_shift{0.0};
};

struct ReducedRCColumnCyclicQuadratureSensitivityFamilyRow {
    BeamAxisQuadratureFamily beam_axis_quadrature_family{
        BeamAxisQuadratureFamily::GaussLegendre};
    std::size_t case_count{0};
    std::size_t completed_case_count{0};
    std::size_t representative_pass_count{0};
    double min_rel_terminal_return_moment_spread{0.0};
    double max_rel_terminal_return_moment_spread{0.0};
    double avg_rel_terminal_return_moment_spread{0.0};
    double min_max_rel_moment_history_spread{0.0};
    double max_max_rel_moment_history_spread{0.0};
    double avg_max_rel_moment_history_spread{0.0};
    double min_max_rel_tangent_history_spread{0.0};
    double max_max_rel_tangent_history_spread{0.0};
    double avg_max_rel_tangent_history_spread{0.0};
    double min_max_rel_secant_history_spread{0.0};
    double max_max_rel_secant_history_spread{0.0};
    double avg_max_rel_secant_history_spread{0.0};
    double min_max_rel_turning_point_moment_spread{0.0};
    double max_max_rel_turning_point_moment_spread{0.0};
    double avg_max_rel_turning_point_moment_spread{0.0};
    double min_max_rel_axial_force_history_spread{0.0};
    double max_max_rel_axial_force_history_spread{0.0};
    double avg_max_rel_axial_force_history_spread{0.0};
    double max_abs_station_xi_shift{0.0};
};

struct ReducedRCColumnCyclicQuadratureSensitivityReferenceRow {
    std::size_t beam_nodes{0};
    BeamAxisQuadratureFamily reference_family{
        BeamAxisQuadratureFamily::GaussLegendre};
    ReducedRCColumnContinuationKind continuation_kind{
        ReducedRCColumnContinuationKind::
            monolithic_incremental_displacement_control};
    int continuation_segment_substep_factor{1};
    std::string reference_case_id{};
    std::size_t compared_case_count{0};
    std::size_t history_point_count{0};
    std::size_t turning_point_count{0};
    double reference_controlling_station_xi{0.0};
    double reference_terminal_return_moment_y{0.0};
};

struct ReducedRCColumnCyclicQuadratureSensitivitySummary {
    std::size_t total_case_count{0};
    std::size_t completed_case_count{0};
    std::size_t failed_case_count{0};
    std::size_t representative_pass_count{0};

    double worst_rel_terminal_return_moment_spread{0.0};
    double worst_max_rel_moment_history_spread{0.0};
    double worst_max_rel_tangent_history_spread{0.0};
    double worst_max_rel_secant_history_spread{0.0};
    double worst_max_rel_turning_point_moment_spread{0.0};
    double worst_max_rel_axial_force_history_spread{0.0};
    double worst_abs_station_xi_shift{0.0};

    std::string worst_terminal_return_case_id{};
    std::string worst_moment_history_case_id{};
    std::string worst_tangent_history_case_id{};
    std::string worst_secant_history_case_id{};
    std::string worst_turning_point_case_id{};
    std::string worst_axial_force_history_case_id{};
    std::string worst_station_shift_case_id{};

    [[nodiscard]] bool all_cases_completed() const noexcept
    {
        return completed_case_count == total_case_count && failed_case_count == 0;
    }

    [[nodiscard]] bool all_completed_cases_pass_representative_internal_cyclic_family_spread()
        const noexcept
    {
        return completed_case_count > 0 &&
               representative_pass_count == completed_case_count;
    }
};

struct ReducedRCColumnCyclicQuadratureSensitivityResult {
    std::vector<ReducedRCColumnCyclicQuadratureSensitivityCaseRow> case_rows{};
    std::vector<ReducedRCColumnCyclicQuadratureSensitivityNodeRow> node_rows{};
    std::vector<ReducedRCColumnCyclicQuadratureSensitivityFamilyRow> family_rows{};
    std::vector<ReducedRCColumnCyclicQuadratureSensitivityReferenceRow>
        reference_rows{};
    ReducedRCColumnCyclicQuadratureSensitivitySummary summary{};

    [[nodiscard]] bool empty() const noexcept { return case_rows.empty(); }
};

[[nodiscard]] ReducedRCColumnCyclicQuadratureSensitivityResult
run_reduced_rc_column_cyclic_quadrature_sensitivity_study(
    const ReducedRCColumnCyclicQuadratureSensitivityRunSpec& spec,
    const std::string& out_dir);

} // namespace fall_n::validation_reboot

#endif // FALL_N_REDUCED_RC_COLUMN_CYCLIC_QUADRATURE_SENSITIVITY_STUDY_HH
