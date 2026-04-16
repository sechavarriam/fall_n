#ifndef FALL_N_REDUCED_RC_COLUMN_MOMENT_CURVATURE_CLOSURE_MATRIX_HH
#define FALL_N_REDUCED_RC_COLUMN_MOMENT_CURVATURE_CLOSURE_MATRIX_HH

// =============================================================================
//  ReducedRCColumnMomentCurvatureClosureMatrix.hh
// =============================================================================
//
//  Matrix-level structural-versus-section closure sweep for the reduced
//  reinforced-concrete column reboot.
//
//  The representative closure artifact is already useful, but it is not enough
//  to answer the harder validation question:
//
//      "How does the structural-versus-section moment-curvature closure evolve
//       over the whole runtime-ready TimoshenkoBeamN<N> family and over the
//       explicit beam-axis quadrature families?"
//
//  This header lifts that question into a dedicated runtime artifact. The
//  matrix sweep remains outside the numerical hot path and is intentionally
//  separate from the baselines themselves, so that:
//
//    - structural slices can evolve independently,
//    - the section baseline can evolve independently,
//    - and the closure metric can evolve independently.
//
// =============================================================================

#include "src/validation/ReducedRCColumnMomentCurvatureClosure.hh"
#include "src/validation/ReducedRCColumnStructuralMatrixCatalog.hh"

#include <cstddef>
#include <string>
#include <vector>

namespace fall_n::validation_reboot {

struct ReducedRCColumnMomentCurvatureClosureMatrixRunSpec {
    ReducedRCColumnMomentCurvatureClosureRunSpec closure_spec{};
    std::vector<std::size_t> beam_nodes_filter{};
    std::vector<BeamAxisQuadratureFamily> quadrature_filter{};

    bool include_only_phase3_runtime_baseline{true};
    bool write_case_outputs{true};
    bool write_matrix_csv{true};
    bool print_progress{true};
};

struct ReducedRCColumnMomentCurvatureClosureMatrixCaseRow {
    std::size_t beam_nodes{0};
    BeamAxisQuadratureFamily beam_axis_quadrature_family{
        BeamAxisQuadratureFamily::GaussLegendre};
    continuum::FormulationKind formulation_kind{
        continuum::FormulationKind::small_strain};
    std::string case_id{};
    std::string case_out_dir{};
    std::string scope_label{};
    std::string rationale_label{};

    bool execution_ok{false};
    std::string error_message{};

    std::size_t positive_branch_point_count{0};
    double structural_max_curvature_y{0.0};
    double section_baseline_max_curvature_y{0.0};
    double max_rel_axial_force_error{0.0};
    double max_rel_moment_error{0.0};
    double rms_rel_moment_error{0.0};
    double max_rel_tangent_error{0.0};
    double max_rel_secant_error{0.0};

    bool moment_within_representative_tolerance{false};
    bool tangent_within_representative_tolerance{false};
    bool secant_within_representative_tolerance{false};
    bool axial_force_within_representative_tolerance{false};
    bool representative_closure_passes{false};
};

struct ReducedRCColumnMomentCurvatureClosureNodeSpreadRow {
    std::size_t beam_nodes{0};
    std::size_t case_count{0};
    std::size_t completed_case_count{0};
    std::size_t representative_pass_count{0};
    double min_max_rel_moment_error{0.0};
    double max_max_rel_moment_error{0.0};
    double avg_max_rel_moment_error{0.0};
    double min_max_rel_tangent_error{0.0};
    double max_max_rel_tangent_error{0.0};
    double avg_max_rel_tangent_error{0.0};
    double min_max_rel_secant_error{0.0};
    double max_max_rel_secant_error{0.0};
    double avg_max_rel_secant_error{0.0};
};

struct ReducedRCColumnMomentCurvatureClosureQuadratureSpreadRow {
    BeamAxisQuadratureFamily beam_axis_quadrature_family{
        BeamAxisQuadratureFamily::GaussLegendre};
    std::size_t case_count{0};
    std::size_t completed_case_count{0};
    std::size_t representative_pass_count{0};
    double min_max_rel_moment_error{0.0};
    double max_max_rel_moment_error{0.0};
    double avg_max_rel_moment_error{0.0};
    double min_max_rel_tangent_error{0.0};
    double max_max_rel_tangent_error{0.0};
    double avg_max_rel_tangent_error{0.0};
    double min_max_rel_secant_error{0.0};
    double max_max_rel_secant_error{0.0};
    double avg_max_rel_secant_error{0.0};
};

struct ReducedRCColumnMomentCurvatureClosureMatrixSummary {
    std::size_t total_case_count{0};
    std::size_t completed_case_count{0};
    std::size_t failed_case_count{0};
    std::size_t representative_pass_count{0};

    double worst_max_rel_axial_force_error{0.0};
    double worst_max_rel_moment_error{0.0};
    double worst_max_rel_tangent_error{0.0};
    double worst_max_rel_secant_error{0.0};

    std::string worst_axial_force_case_id{};
    std::string worst_moment_case_id{};
    std::string worst_tangent_case_id{};
    std::string worst_secant_case_id{};

    [[nodiscard]] bool all_cases_completed() const noexcept
    {
        return completed_case_count == total_case_count && failed_case_count == 0;
    }

    [[nodiscard]] bool all_completed_cases_pass_representative_closure() const noexcept
    {
        return completed_case_count > 0 &&
               representative_pass_count == completed_case_count;
    }
};

struct ReducedRCColumnMomentCurvatureClosureMatrixResult {
    std::vector<ReducedRCColumnMomentCurvatureClosureMatrixCaseRow> case_rows{};
    std::vector<ReducedRCColumnMomentCurvatureClosureNodeSpreadRow> node_spread_rows{};
    std::vector<ReducedRCColumnMomentCurvatureClosureQuadratureSpreadRow>
        quadrature_spread_rows{};
    ReducedRCColumnMomentCurvatureClosureMatrixSummary summary{};

    [[nodiscard]] bool empty() const noexcept { return case_rows.empty(); }
};

[[nodiscard]] ReducedRCColumnMomentCurvatureClosureMatrixResult
run_reduced_rc_column_moment_curvature_closure_matrix(
    const ReducedRCColumnMomentCurvatureClosureMatrixRunSpec& spec,
    const std::string& out_dir);

} // namespace fall_n::validation_reboot

#endif // FALL_N_REDUCED_RC_COLUMN_MOMENT_CURVATURE_CLOSURE_MATRIX_HH
