#ifndef FALL_N_REDUCED_RC_COLUMN_QUADRATURE_SENSITIVITY_STUDY_HH
#define FALL_N_REDUCED_RC_COLUMN_QUADRATURE_SENSITIVITY_STUDY_HH

// =============================================================================
//  ReducedRCColumnQuadratureSensitivityStudy.hh
// =============================================================================

#include "src/validation/ReducedRCColumnMomentCurvatureClosure.hh"
#include "src/validation/ReducedRCColumnStructuralMatrixCatalog.hh"

#include <cstddef>
#include <string>
#include <vector>

namespace fall_n::validation_reboot {

struct ReducedRCColumnQuadratureSensitivityRunSpec {
    ReducedRCColumnMomentCurvatureClosureRunSpec closure_spec{};
    std::vector<std::size_t> beam_nodes_filter{};
    std::vector<BeamAxisQuadratureFamily> quadrature_filter{};
    BeamAxisQuadratureFamily reference_family{
        BeamAxisQuadratureFamily::GaussLegendre};

    bool include_only_phase3_runtime_baseline{true};
    bool write_case_outputs{true};
    bool write_csv{true};
    bool print_progress{true};

    double relative_error_floor{1.0e-12};
    double representative_terminal_moment_relative_spread_tolerance{0.05};
    double representative_max_rel_moment_spread_tolerance{0.10};
    double representative_max_rel_tangent_spread_tolerance{0.15};
    double representative_max_rel_secant_spread_tolerance{0.10};
};

struct ReducedRCColumnQuadratureSensitivityCaseRow {
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

    double controlling_station_xi{0.0};
    double reference_controlling_station_xi{0.0};
    double abs_station_xi_shift{0.0};

    double compared_max_curvature_y{0.0};
    double reference_max_curvature_y{0.0};
    double terminal_structural_moment_y{0.0};
    double reference_terminal_structural_moment_y{0.0};
    double rel_terminal_moment_spread{0.0};
    double max_rel_moment_spread{0.0};
    double rms_rel_moment_spread{0.0};
    double max_rel_tangent_spread{0.0};
    double max_rel_secant_spread{0.0};

    bool terminal_moment_within_representative_tolerance{false};
    bool moment_spread_within_representative_tolerance{false};
    bool tangent_spread_within_representative_tolerance{false};
    bool secant_spread_within_representative_tolerance{false};

    [[nodiscard]] bool representative_internal_sensitivity_passes() const noexcept
    {
        return terminal_moment_within_representative_tolerance &&
               moment_spread_within_representative_tolerance &&
               tangent_spread_within_representative_tolerance &&
               secant_spread_within_representative_tolerance;
    }
};

struct ReducedRCColumnQuadratureSensitivityNodeRow {
    std::size_t beam_nodes{0};
    std::size_t case_count{0};
    std::size_t completed_case_count{0};
    std::size_t representative_pass_count{0};
    double min_rel_terminal_moment_spread{0.0};
    double max_rel_terminal_moment_spread{0.0};
    double avg_rel_terminal_moment_spread{0.0};
    double min_max_rel_moment_spread{0.0};
    double max_max_rel_moment_spread{0.0};
    double avg_max_rel_moment_spread{0.0};
    double min_max_rel_tangent_spread{0.0};
    double max_max_rel_tangent_spread{0.0};
    double avg_max_rel_tangent_spread{0.0};
    double min_max_rel_secant_spread{0.0};
    double max_max_rel_secant_spread{0.0};
    double avg_max_rel_secant_spread{0.0};
    double max_abs_station_xi_shift{0.0};
};

struct ReducedRCColumnQuadratureSensitivityFamilyRow {
    BeamAxisQuadratureFamily beam_axis_quadrature_family{
        BeamAxisQuadratureFamily::GaussLegendre};
    std::size_t case_count{0};
    std::size_t completed_case_count{0};
    std::size_t representative_pass_count{0};
    double min_rel_terminal_moment_spread{0.0};
    double max_rel_terminal_moment_spread{0.0};
    double avg_rel_terminal_moment_spread{0.0};
    double min_max_rel_moment_spread{0.0};
    double max_max_rel_moment_spread{0.0};
    double avg_max_rel_moment_spread{0.0};
    double min_max_rel_tangent_spread{0.0};
    double max_max_rel_tangent_spread{0.0};
    double avg_max_rel_tangent_spread{0.0};
    double min_max_rel_secant_spread{0.0};
    double max_max_rel_secant_spread{0.0};
    double avg_max_rel_secant_spread{0.0};
    double max_abs_station_xi_shift{0.0};
};

struct ReducedRCColumnQuadratureSensitivityReferenceRow {
    std::size_t beam_nodes{0};
    BeamAxisQuadratureFamily reference_family{
        BeamAxisQuadratureFamily::GaussLegendre};
    std::string reference_case_id{};
    std::size_t compared_case_count{0};
    double reference_max_curvature_y{0.0};
    double reference_terminal_structural_moment_y{0.0};
    double reference_controlling_station_xi{0.0};
};

struct ReducedRCColumnQuadratureSensitivitySummary {
    std::size_t total_case_count{0};
    std::size_t completed_case_count{0};
    std::size_t failed_case_count{0};
    std::size_t representative_pass_count{0};

    double worst_rel_terminal_moment_spread{0.0};
    double worst_max_rel_moment_spread{0.0};
    double worst_max_rel_tangent_spread{0.0};
    double worst_max_rel_secant_spread{0.0};
    double worst_abs_station_xi_shift{0.0};

    std::string worst_terminal_moment_case_id{};
    std::string worst_moment_case_id{};
    std::string worst_tangent_case_id{};
    std::string worst_secant_case_id{};
    std::string worst_station_shift_case_id{};

    [[nodiscard]] bool all_cases_completed() const noexcept
    {
        return completed_case_count == total_case_count && failed_case_count == 0;
    }

    [[nodiscard]] bool all_completed_cases_pass_representative_internal_sensitivity()
        const noexcept
    {
        return completed_case_count > 0 &&
               representative_pass_count == completed_case_count;
    }
};

struct ReducedRCColumnQuadratureSensitivityResult {
    std::vector<ReducedRCColumnQuadratureSensitivityCaseRow> case_rows{};
    std::vector<ReducedRCColumnQuadratureSensitivityNodeRow> node_rows{};
    std::vector<ReducedRCColumnQuadratureSensitivityFamilyRow> family_rows{};
    std::vector<ReducedRCColumnQuadratureSensitivityReferenceRow> reference_rows{};
    ReducedRCColumnQuadratureSensitivitySummary summary{};

    [[nodiscard]] bool empty() const noexcept { return case_rows.empty(); }
};

[[nodiscard]] ReducedRCColumnQuadratureSensitivityResult
run_reduced_rc_column_quadrature_sensitivity_study(
    const ReducedRCColumnQuadratureSensitivityRunSpec& spec,
    const std::string& out_dir);

} // namespace fall_n::validation_reboot

#endif // FALL_N_REDUCED_RC_COLUMN_QUADRATURE_SENSITIVITY_STUDY_HH
