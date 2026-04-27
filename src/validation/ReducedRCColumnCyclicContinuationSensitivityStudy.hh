#ifndef FALL_N_REDUCED_RC_COLUMN_CYCLIC_CONTINUATION_SENSITIVITY_STUDY_HH
#define FALL_N_REDUCED_RC_COLUMN_CYCLIC_CONTINUATION_SENSITIVITY_STUDY_HH

// =============================================================================
//  ReducedRCColumnCyclicContinuationSensitivityStudy.hh
// =============================================================================
//
//  Continuation-policy sensitivity study for the reduced reinforced-concrete
//  column reboot.
//
//  The cyclic node-refinement study already answers, for a fixed continuation
//  policy, how the base-side observable drifts against the highest-N internal
//  reference inside each quadrature family. That still leaves one numerical
//  question open before the structural validation chapter can interpret the
//  remaining cyclic frontier honestly:
//
//      "How much of the residual cyclic drift is caused by the continuation
//       policy itself, especially near reversal, and how much remains after a
//       more guarded displacement-control schedule is used?"
//
//  This artifact keeps that question separate from both the baseline runtime
//  driver and the node-refinement study. The benchmark semantics remain
//  displacement-driven: the candidate policy may refine the path near
//  reversals, but it is not allowed to silently change the target protocol.
//  An arc-length path is therefore kept as an explicit future candidate rather
//  than being smuggled into the current reduced-column baseline.
//
// =============================================================================

#include "src/validation/ReducedRCColumnCyclicNodeRefinementStudy.hh"

#include <cstddef>
#include <string>
#include <vector>

namespace fall_n::validation_reboot {

struct ReducedRCColumnCyclicContinuationSensitivityRunSpec {
    ReducedRCColumnCyclicNodeRefinementRunSpec baseline_spec{};
    ReducedRCColumnCyclicNodeRefinementRunSpec candidate_spec{};
    bool write_csv{true};
    bool print_progress{true};
};

struct ReducedRCColumnCyclicContinuationSensitivityCaseRow {
    std::size_t beam_nodes{0};
    BeamAxisQuadratureFamily beam_axis_quadrature_family{
        BeamAxisQuadratureFamily::GaussLegendre};
    continuum::FormulationKind formulation_kind{
        continuum::FormulationKind::small_strain};
    std::string case_id{};
    std::string scope_label{};
    std::string rationale_label{};

    ReducedRCColumnContinuationKind baseline_continuation_kind{
        ReducedRCColumnContinuationKind::
            monolithic_incremental_displacement_control};
    int baseline_continuation_segment_substep_factor{1};
    bool baseline_execution_ok{false};
    bool baseline_representative_internal_cyclic_refinement_passes{false};
    std::size_t baseline_history_point_count{0};
    std::size_t baseline_turning_point_count{0};
    double baseline_rel_terminal_return_moment_drift{0.0};
    double baseline_max_rel_moment_history_drift{0.0};
    double baseline_max_rel_tangent_history_drift{0.0};
    double baseline_max_rel_secant_history_drift{0.0};
    double baseline_max_rel_turning_point_moment_drift{0.0};
    double baseline_max_rel_axial_force_history_drift{0.0};
    double baseline_abs_station_xi_shift{0.0};

    ReducedRCColumnContinuationKind candidate_continuation_kind{
        ReducedRCColumnContinuationKind::
            monolithic_incremental_displacement_control};
    int candidate_continuation_segment_substep_factor{1};
    bool candidate_execution_ok{false};
    bool candidate_representative_internal_cyclic_refinement_passes{false};
    std::size_t candidate_history_point_count{0};
    std::size_t candidate_turning_point_count{0};
    double candidate_rel_terminal_return_moment_drift{0.0};
    double candidate_max_rel_moment_history_drift{0.0};
    double candidate_max_rel_tangent_history_drift{0.0};
    double candidate_max_rel_secant_history_drift{0.0};
    double candidate_max_rel_turning_point_moment_drift{0.0};
    double candidate_max_rel_axial_force_history_drift{0.0};
    double candidate_abs_station_xi_shift{0.0};

    double delta_rel_terminal_return_moment_drift{0.0};
    double delta_max_rel_moment_history_drift{0.0};
    double delta_max_rel_tangent_history_drift{0.0};
    double delta_max_rel_secant_history_drift{0.0};
    double delta_max_rel_turning_point_moment_drift{0.0};
    double delta_max_rel_axial_force_history_drift{0.0};
    double delta_abs_station_xi_shift{0.0};

    bool candidate_improves_terminal_return_drift{false};
    bool candidate_improves_moment_history_drift{false};
    bool candidate_improves_tangent_history_drift{false};
    bool candidate_improves_secant_history_drift{false};
    bool candidate_improves_turning_point_drift{false};
    bool candidate_improves_axial_force_drift{false};
    bool candidate_improves_station_shift{false};
};

struct ReducedRCColumnCyclicContinuationSensitivitySummaryRow {
    std::size_t beam_nodes{0};
    std::size_t case_count{0};
    std::size_t baseline_completed_case_count{0};
    std::size_t candidate_completed_case_count{0};
    std::size_t baseline_representative_pass_count{0};
    std::size_t candidate_representative_pass_count{0};
    std::size_t candidate_additional_representative_pass_count{0};
    std::size_t candidate_lost_representative_pass_count{0};
    std::size_t candidate_improves_terminal_return_count{0};
    std::size_t candidate_improves_moment_history_count{0};
    std::size_t candidate_improves_tangent_history_count{0};
    std::size_t candidate_improves_secant_history_count{0};
    std::size_t candidate_improves_turning_point_count{0};
    std::size_t candidate_improves_axial_force_count{0};
    std::size_t candidate_improves_station_shift_count{0};
    double max_abs_delta_rel_terminal_return_moment_drift{0.0};
    double max_abs_delta_max_rel_moment_history_drift{0.0};
    double max_abs_delta_max_rel_tangent_history_drift{0.0};
    double max_abs_delta_max_rel_secant_history_drift{0.0};
    double max_abs_delta_max_rel_turning_point_moment_drift{0.0};
    double max_abs_delta_max_rel_axial_force_history_drift{0.0};
    double max_abs_delta_abs_station_xi_shift{0.0};
};

struct ReducedRCColumnCyclicContinuationSensitivitySummary {
    std::size_t total_case_count{0};
    std::size_t compared_case_count{0};
    std::size_t baseline_completed_case_count{0};
    std::size_t candidate_completed_case_count{0};
    std::size_t baseline_representative_pass_count{0};
    std::size_t candidate_representative_pass_count{0};
    std::size_t candidate_additional_representative_pass_count{0};
    std::size_t candidate_lost_representative_pass_count{0};
    std::size_t candidate_improves_terminal_return_count{0};
    std::size_t candidate_improves_moment_history_count{0};
    std::size_t candidate_improves_tangent_history_count{0};
    std::size_t candidate_improves_secant_history_count{0};
    std::size_t candidate_improves_turning_point_count{0};
    std::size_t candidate_improves_axial_force_count{0};
    std::size_t candidate_improves_station_shift_count{0};
    double max_abs_delta_rel_terminal_return_moment_drift{0.0};
    std::string max_abs_delta_terminal_return_case_id{};
    double max_abs_delta_max_rel_moment_history_drift{0.0};
    std::string max_abs_delta_moment_history_case_id{};
    double max_abs_delta_max_rel_tangent_history_drift{0.0};
    std::string max_abs_delta_tangent_history_case_id{};
    double max_abs_delta_max_rel_secant_history_drift{0.0};
    std::string max_abs_delta_secant_history_case_id{};
    double max_abs_delta_max_rel_turning_point_moment_drift{0.0};
    std::string max_abs_delta_turning_point_case_id{};
    double max_abs_delta_max_rel_axial_force_history_drift{0.0};
    std::string max_abs_delta_axial_force_case_id{};
    double max_abs_delta_abs_station_xi_shift{0.0};
    std::string max_abs_delta_station_shift_case_id{};

    [[nodiscard]] bool baseline_and_candidate_all_cases_completed() const noexcept
    {
        return baseline_completed_case_count == total_case_count &&
               candidate_completed_case_count == total_case_count;
    }
};

struct ReducedRCColumnCyclicContinuationSensitivityResult {
    ReducedRCColumnCyclicNodeRefinementResult baseline_result{};
    ReducedRCColumnCyclicNodeRefinementResult candidate_result{};
    std::vector<ReducedRCColumnCyclicContinuationSensitivityCaseRow> case_rows{};
    std::vector<ReducedRCColumnCyclicContinuationSensitivitySummaryRow> summary_rows{};
    ReducedRCColumnCyclicContinuationSensitivitySummary summary{};

    [[nodiscard]] bool empty() const noexcept { return case_rows.empty(); }
};

[[nodiscard]] ReducedRCColumnCyclicContinuationSensitivityResult
run_reduced_rc_column_cyclic_continuation_sensitivity_study(
    const ReducedRCColumnCyclicContinuationSensitivityRunSpec& spec,
    const std::string& out_dir);

} // namespace fall_n::validation_reboot

#endif // FALL_N_REDUCED_RC_COLUMN_CYCLIC_CONTINUATION_SENSITIVITY_STUDY_HH
