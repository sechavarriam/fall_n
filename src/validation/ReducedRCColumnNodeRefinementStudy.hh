#ifndef FALL_N_REDUCED_RC_COLUMN_NODE_REFINEMENT_STUDY_HH
#define FALL_N_REDUCED_RC_COLUMN_NODE_REFINEMENT_STUDY_HH

// =============================================================================
//  ReducedRCColumnNodeRefinementStudy.hh
// =============================================================================
//
//  Internal node-refinement study for the reduced reinforced-concrete column
//  reboot.
//
//  The preload-consistent moment-curvature closure matrix already answered an
//  important internal-consistency question:
//
//      "Does each runtime-ready structural slice remain consistent with the
//       independent section baseline?"
//
//  The next methodological question is different:
//
//      "How does the structural base-side response change as the reduced
//       TimoshenkoBeamN<N> family is refined from N=2 to N=10?"
//
//  This header freezes that question as a dedicated runtime artifact. The
//  study remains explicitly internal: it measures stabilization against the
//  highest runtime-ready reference slice available today inside each beam-axis
//  quadrature family. It does not claim external physical validation by
//  itself, but it does remove one major source of ambiguity before external
//  benchmark comparison begins.
//
// =============================================================================

#include "src/validation/ReducedRCColumnMomentCurvatureClosure.hh"
#include "src/validation/ReducedRCColumnStructuralMatrixCatalog.hh"

#include <cstddef>
#include <string>
#include <vector>

namespace fall_n::validation_reboot {

struct ReducedRCColumnNodeRefinementRunSpec {
    ReducedRCColumnMomentCurvatureClosureRunSpec closure_spec{};
    std::vector<std::size_t> beam_nodes_filter{};
    std::vector<BeamAxisQuadratureFamily> quadrature_filter{};

    bool include_only_phase3_runtime_baseline{true};
    bool write_case_outputs{true};
    bool write_csv{true};
    bool print_progress{true};

    double relative_error_floor{1.0e-12};
    double representative_terminal_moment_relative_drift_tolerance{0.05};
    double representative_max_rel_moment_drift_tolerance{0.10};
    double representative_max_rel_tangent_drift_tolerance{0.15};
    double representative_max_rel_secant_drift_tolerance{0.10};
};

struct ReducedRCColumnNodeRefinementCaseRow {
    std::size_t beam_nodes{0};
    BeamAxisQuadratureFamily beam_axis_quadrature_family{
        BeamAxisQuadratureFamily::GaussLegendre};
    continuum::FormulationKind formulation_kind{
        continuum::FormulationKind::small_strain};
    std::string case_id{};
    std::string reference_case_id{};
    std::string case_out_dir{};
    std::string scope_label{};
    std::string rationale_label{};

    bool execution_ok{false};
    std::string error_message{};

    std::size_t positive_branch_point_count{0};
    std::size_t overlap_point_count{0};
    double compared_max_curvature_y{0.0};
    double reference_max_curvature_y{0.0};
    double terminal_structural_moment_y{0.0};
    double reference_terminal_structural_moment_y{0.0};
    double rel_terminal_moment_drift{0.0};
    double max_rel_moment_drift{0.0};
    double rms_rel_moment_drift{0.0};
    double max_rel_tangent_drift{0.0};
    double max_rel_secant_drift{0.0};

    bool terminal_moment_within_representative_tolerance{false};
    bool moment_drift_within_representative_tolerance{false};
    bool tangent_drift_within_representative_tolerance{false};
    bool secant_drift_within_representative_tolerance{false};

    [[nodiscard]] bool representative_internal_refinement_passes() const noexcept
    {
        return terminal_moment_within_representative_tolerance &&
               moment_drift_within_representative_tolerance &&
               tangent_drift_within_representative_tolerance &&
               secant_drift_within_representative_tolerance;
    }
};

struct ReducedRCColumnNodeRefinementSummaryRow {
    std::size_t beam_nodes{0};
    std::size_t case_count{0};
    std::size_t completed_case_count{0};
    std::size_t representative_pass_count{0};
    double min_rel_terminal_moment_drift{0.0};
    double max_rel_terminal_moment_drift{0.0};
    double avg_rel_terminal_moment_drift{0.0};
    double min_max_rel_moment_drift{0.0};
    double max_max_rel_moment_drift{0.0};
    double avg_max_rel_moment_drift{0.0};
    double min_max_rel_tangent_drift{0.0};
    double max_max_rel_tangent_drift{0.0};
    double avg_max_rel_tangent_drift{0.0};
    double min_max_rel_secant_drift{0.0};
    double max_max_rel_secant_drift{0.0};
    double avg_max_rel_secant_drift{0.0};
};

struct ReducedRCColumnNodeRefinementReferenceRow {
    BeamAxisQuadratureFamily beam_axis_quadrature_family{
        BeamAxisQuadratureFamily::GaussLegendre};
    std::size_t reference_beam_nodes{0};
    std::string reference_case_id{};
    std::size_t compared_case_count{0};
    double reference_max_curvature_y{0.0};
    double reference_terminal_structural_moment_y{0.0};
};

struct ReducedRCColumnNodeRefinementSummary {
    std::size_t total_case_count{0};
    std::size_t completed_case_count{0};
    std::size_t failed_case_count{0};
    std::size_t representative_pass_count{0};

    double worst_rel_terminal_moment_drift{0.0};
    double worst_max_rel_moment_drift{0.0};
    double worst_max_rel_tangent_drift{0.0};
    double worst_max_rel_secant_drift{0.0};

    std::string worst_terminal_moment_case_id{};
    std::string worst_moment_case_id{};
    std::string worst_tangent_case_id{};
    std::string worst_secant_case_id{};

    [[nodiscard]] bool all_cases_completed() const noexcept
    {
        return completed_case_count == total_case_count && failed_case_count == 0;
    }

    [[nodiscard]] bool all_completed_cases_pass_representative_internal_refinement()
        const noexcept
    {
        return completed_case_count > 0 &&
               representative_pass_count == completed_case_count;
    }
};

struct ReducedRCColumnNodeRefinementResult {
    std::vector<ReducedRCColumnNodeRefinementCaseRow> case_rows{};
    std::vector<ReducedRCColumnNodeRefinementSummaryRow> summary_rows{};
    std::vector<ReducedRCColumnNodeRefinementReferenceRow> reference_rows{};
    ReducedRCColumnNodeRefinementSummary summary{};

    [[nodiscard]] bool empty() const noexcept { return case_rows.empty(); }
};

[[nodiscard]] ReducedRCColumnNodeRefinementResult
run_reduced_rc_column_node_refinement_study(
    const ReducedRCColumnNodeRefinementRunSpec& spec,
    const std::string& out_dir);

} // namespace fall_n::validation_reboot

#endif // FALL_N_REDUCED_RC_COLUMN_NODE_REFINEMENT_STUDY_HH
